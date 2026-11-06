"""Bias metrics: legacy thesis compatibility plus paired continuation scoring."""

from __future__ import annotations

import torch
import torch.nn.functional as F


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


def summarize_paired_changes(before, after, *, seed: int = 42,
                             bootstrap_samples: int = 5000) -> dict[str, float | int | list[float]]:
    """Summarize paired per-example bias changes without discarding raw values."""
    before_t = torch.as_tensor(before, dtype=torch.float64)
    after_t = torch.as_tensor(after, dtype=torch.float64)
    if before_t.shape != after_t.shape or before_t.numel() == 0:
        raise ValueError("Before/after scores must be non-empty and pair-aligned")
    delta = before_t - after_t
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(
        delta.numel(), (bootstrap_samples, delta.numel()), generator=generator
    )
    bootstrap_means = delta[indices].mean(dim=1)
    interval = torch.quantile(bootstrap_means, torch.tensor([0.025, 0.975], dtype=torch.float64))
    return {
        "n": int(delta.numel()),
        "mean_before": float(before_t.mean()),
        "mean_after": float(after_t.mean()),
        "mean_absolute_reduction": float(delta.mean()),
        "relative_reduction_percent": float(
            100 * delta.mean() / before_t.mean().clamp_min(1e-12)
        ),
        "improved_count": int((delta > 0).sum()),
        "mean_reduction_bootstrap_95ci": [float(interval[0]), float(interval[1])],
        "bootstrap_seed": int(seed),
        "bootstrap_samples": int(bootstrap_samples),
    }


def bootstrap_mean_interval(values, *, seed: int = 42, samples: int = 5000) -> list[float]:
    """Return a deterministic percentile bootstrap interval for a sample mean."""
    values_t = torch.as_tensor(values, dtype=torch.float64)
    if values_t.ndim != 1 or values_t.numel() == 0 or samples < 1:
        raise ValueError("Expected a non-empty one-dimensional sample and positive samples")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(values_t.numel(), (samples, values_t.numel()), generator=generator)
    means = values_t[indices].mean(dim=1)
    bounds = torch.quantile(means, torch.tensor([0.025, 0.975], dtype=torch.float64))
    return [float(bounds[0]), float(bounds[1])]
