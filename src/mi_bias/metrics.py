"""Bias metrics: legacy thesis compatibility plus paired continuation scoring."""

from __future__ import annotations

import torch
import torch.nn.functional as F


THESIS_MALE_WORDS = [
    "he", "him", "his", "man", "men", "boy", "boys", "male", "father", "son",
    "husband", "brother", "gentleman", "sir", "king", "prince", "himself", "mr",
    "businessman", "uncle",
]
THESIS_FEMALE_WORDS = [
    "she", "her", "hers", "woman", "women", "girl", "girls", "female", "mother",
    "daughter", "wife", "sister", "lady", "madam", "queen", "princess", "herself",
    "mrs", "businesswoman", "aunt",
]


def thesis_legacy_l2(logits: torch.Tensor, male_token_ids, female_token_ids) -> torch.Tensor:
    """Reproduce the prior first-token, index-paired log-probability L2 score.

    Keep this only for direct replication. It does not account for multi-token
    words and assumes the two lists are meaningfully aligned by index.
    """
    log_probs = F.log_softmax(logits[..., -1, :], dim=-1)
    male = torch.as_tensor(male_token_ids, device=log_probs.device, dtype=torch.long)
    female = torch.as_tensor(female_token_ids, device=log_probs.device, dtype=torch.long)
    paired_length = min(male.numel(), female.numel())
    if paired_length == 0:
        raise ValueError("Legacy score requires non-empty male and female token lists")
    return torch.linalg.vector_norm(
        log_probs[..., male[:paired_length]] - log_probs[..., female[:paired_length]], dim=-1
    )


def thesis_gender_token_ids(model):
    """Get the first token IDs for the exact word lists in the prior thesis."""
    def ids(words):
        result = []
        for word in words:
            tokens = model.to_tokens(f" {word}", prepend_bos=False).reshape(-1)
            if tokens.numel():
                result.append(int(tokens[0].item()))
        return list(dict.fromkeys(result))

    male, female = ids(THESIS_MALE_WORDS), ids(THESIS_FEMALE_WORDS)
    if not male or not female:
        raise ValueError("Tokenizer produced an empty thesis gender token list")
    return male, female


def score_thesis_legacy_l2(model, prompt: str, male_token_ids, female_token_ids, hooks=None):
    """Score one prompt with the previous thesis's first-token L2 definition."""
    tokens = model.to_tokens(prompt, prepend_bos=True).to(model.cfg.device)
    with torch.inference_mode():
        logits = (model.run_with_hooks(tokens, fwd_hooks=hooks)
                  if hooks else model(tokens))
    return float(thesis_legacy_l2(logits, male_token_ids, female_token_ids).item())


def paired_log_odds_bias(
    male_continuation_logprobs: torch.Tensor,
    female_continuation_logprobs: torch.Tensor,
) -> torch.Tensor:
    """Absolute log odds between matched male/female continuation sets.

    Inputs contain sequence log-likelihoods (one value per continuation), so
    completions may be multi-token and are normalized by their token count by
    the caller before passing them here.
    """
    if male_continuation_logprobs.shape != female_continuation_logprobs.shape:
        raise ValueError("Male and female continuation scores must be pair-aligned")
    male_mass = torch.logsumexp(male_continuation_logprobs, dim=-1)
    female_mass = torch.logsumexp(female_continuation_logprobs, dim=-1)
    return torch.abs(male_mass - female_mass)


def continuation_logprobs(model, prompt: str, continuations: list[str], hooks=None):
    """Score complete continuation strings, including multi-token candidates.

    Candidate strings should include their leading whitespace when required by
    the tokenizer (for example, ``" he"``). A token-boundary merge across the
    prompt/completion boundary is rejected instead of silently mis-scored.
    """
    if not continuations:
        raise ValueError("At least one continuation is required")
    prompt_tokens = model.to_tokens(prompt, prepend_bos=True)
    scores = []
    for continuation in continuations:
        full_tokens = model.to_tokens(prompt + continuation, prepend_bos=True)
        prefix_length = prompt_tokens.shape[1]
        if full_tokens.shape[1] <= prefix_length or not torch.equal(
            full_tokens[:, :prefix_length].cpu(), prompt_tokens.cpu()
        ):
            raise ValueError(
                f"Continuation {continuation!r} changes tokenization of the prompt; "
                "use a boundary-safe prompt/completion pair"
            )
        full_tokens = full_tokens.to(model.cfg.device)
        with torch.inference_mode():
            logits = (model.run_with_hooks(full_tokens, fwd_hooks=hooks)
                      if hooks else model(full_tokens))
        log_probs = F.log_softmax(logits[0, prefix_length - 1:-1], dim=-1)
        targets = full_tokens[0, prefix_length:]
        token_scores = log_probs.gather(1, targets[:, None]).squeeze(1)
        scores.append(token_scores.sum() / max(int(targets.numel()), 1))
    return torch.stack(scores)


def score_prompt_bias(model, prompt: str, male_continuations: list[str],
                      female_continuations: list[str], hooks=None):
    """Return paired continuation scores plus the token-aware aggregate bias."""
    if len(male_continuations) != len(female_continuations):
        raise ValueError("Provide equally sized, meaningfully paired continuation lists")
    male = continuation_logprobs(model, prompt, male_continuations, hooks)
    female = continuation_logprobs(model, prompt, female_continuations, hooks)
    return {
        "bias_score": paired_log_odds_bias(male, female),
        "male_logprobs": male,
        "female_logprobs": female,
    }


def _bootstrap_means(values_t: torch.Tensor, *, groups=None, seed: int,
                     samples: int) -> torch.Tensor:
    if groups is None:
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randint(values_t.numel(), (samples, values_t.numel()), generator=generator)
        return values_t[indices].mean(dim=1)
    if len(groups) != values_t.numel():
        raise ValueError("Provide one bootstrap group per observation")
    grouped: dict[str, list[torch.Tensor]] = {}
    for value, group in zip(values_t, groups):
        grouped.setdefault(str(group), []).append(value.reshape(1))
    group_values = [torch.cat(items) for items in grouped.values()]
    if not group_values:
        raise ValueError("Cluster bootstrap requires at least one group")
    generator = torch.Generator().manual_seed(seed)
    draws = torch.randint(len(group_values), (samples, len(group_values)), generator=generator)
    means = []
    for draw in draws:
        sample_values = torch.cat([group_values[int(index)] for index in draw])
        means.append(sample_values.mean())
    return torch.stack(means)


def _bootstrap_relative_reductions(before_t: torch.Tensor, after_t: torch.Tensor,
                                   *, groups=None, seed: int, samples: int) -> torch.Tensor:
    if groups is None:
        generator = torch.Generator().manual_seed(seed)
        draws = torch.randint(before_t.numel(), (samples, before_t.numel()), generator=generator)
        return torch.stack([
            100 * (before_t[index].mean() - after_t[index].mean())
            / before_t[index].mean().clamp_min(1e-12)
            for index in draws
        ])
    if len(groups) != before_t.numel():
        raise ValueError("Provide one bootstrap group per observation")
    grouped: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        grouped.setdefault(str(group), []).append(index)
    group_indices = [torch.as_tensor(indices, dtype=torch.long) for indices in grouped.values()]
    generator = torch.Generator().manual_seed(seed)
    draws = torch.randint(len(group_indices), (samples, len(group_indices)), generator=generator)
    values = []
    for draw in draws:
        index = torch.cat([group_indices[int(group_index)] for group_index in draw])
        before_mean, after_mean = before_t[index].mean(), after_t[index].mean()
        values.append(100 * (before_mean-after_mean) / before_mean.clamp_min(1e-12))
    return torch.stack(values)


def summarize_paired_changes(before, after, *, groups=None, seed: int = 42,
                             bootstrap_samples: int = 5000) -> dict[str, float | int | list[float] | str]:
    """Summarize paired per-example bias changes without discarding raw values."""
    before_t = torch.as_tensor(before, dtype=torch.float64)
    after_t = torch.as_tensor(after, dtype=torch.float64)
    if before_t.shape != after_t.shape or before_t.numel() == 0:
        raise ValueError("Before/after scores must be non-empty and pair-aligned")
    delta = before_t - after_t
    bootstrap_means = _bootstrap_means(
        delta, groups=groups, seed=seed, samples=bootstrap_samples
    )
    interval = torch.quantile(bootstrap_means, torch.tensor([0.025, 0.975], dtype=torch.float64))
    relative_samples = _bootstrap_relative_reductions(
        before_t, after_t, groups=groups, seed=seed + 1, samples=bootstrap_samples
    )
    relative_interval = torch.quantile(
        relative_samples, torch.tensor([0.025, 0.975], dtype=torch.float64)
    )
    return {
        "n": int(delta.numel()),
        "mean_before": float(before_t.mean()),
        "mean_after": float(after_t.mean()),
        "mean_absolute_reduction": float(delta.mean()),
        "relative_reduction_percent": float(
            100 * delta.mean() / before_t.mean().clamp_min(1e-12)
        ),
        "relative_reduction_bootstrap_95ci": [
            float(relative_interval[0]), float(relative_interval[1])
        ],
        "improved_count": int((delta > 0).sum()),
        "mean_reduction_bootstrap_95ci": [float(interval[0]), float(interval[1])],
        "bootstrap_seed": int(seed),
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_unit": "group" if groups is not None else "prompt",
    }


def bootstrap_mean_interval(values, *, groups=None, seed: int = 42,
                            samples: int = 5000) -> list[float]:
    """Return a deterministic percentile interval, optionally resampling groups."""
    values_t = torch.as_tensor(values, dtype=torch.float64)
    if values_t.ndim != 1 or values_t.numel() == 0 or samples < 1:
        raise ValueError("Expected a non-empty one-dimensional sample and positive samples")
    means = _bootstrap_means(values_t, groups=groups, seed=seed, samples=samples)
    bounds = torch.quantile(means, torch.tensor([0.025, 0.975], dtype=torch.float64))
    return [float(bounds[0]), float(bounds[1])]
