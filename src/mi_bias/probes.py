"""Prompt-group-safe probes, CAA directions, and covariance-aware LEACE."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .types import PromptPair


@dataclass
class LayerRepresentation:
    activations: torch.Tensor
    labels: torch.Tensor
    groups: list[str]
    label_values: list[str]


def collect_pair_activations(model, pairs: list[PromptPair], layers: list[int]):
    """Collect last-token residual activations without padding prompt pairs."""
    labels = sorted({
        str(label)
        for pair in pairs
        for label in (pair.clean_label, pair.corrupted_label)
        if label is not None
    })
    label_ids = {label: index for index, label in enumerate(labels)}
    hook_names = [f"blocks.{layer}.hook_resid_pre" for layer in layers]
    result = {layer: {"x": [], "y": [], "groups": []} for layer in layers}
    for pair in pairs:
        if pair.clean_label is None or pair.corrupted_label is None:
            raise ValueError(
                f"Pair {pair.pair_id} needs explicit clean_label and corrupted_label "
                "before it can train a probe or intervention direction"
            )
        for text, label in ((pair.clean, pair.clean_label),
                            (pair.corrupted, pair.corrupted_label)):
            tokens = model.to_tokens(text, prepend_bos=True).to(model.cfg.device)
            with torch.inference_mode():
                _, cache = model.run_with_cache(tokens, names_filter=hook_names)
            for layer in layers:
                name = f"blocks.{layer}.hook_resid_pre"
                result[layer]["x"].append(cache[name][0, -1].detach().cpu())
                result[layer]["y"].append(label_ids[str(label)])
                result[layer]["groups"].append(pair.group_id)
    return {
        layer: LayerRepresentation(
            torch.stack(values["x"]),
            torch.as_tensor(values["y"]),
            values["groups"],
            labels,
        )
        for layer, values in result.items()
    }


def fit_grouped_probe(data: LayerRepresentation, validation_fraction=0.2, seed=42):
    """Train and score a logistic probe with group-disjoint train/test examples."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from sklearn.model_selection import GroupShuffleSplit

    labels = np.asarray(data.labels)
    if len(np.unique(labels)) != 2:
        raise ValueError("Logistic gender/bias probe currently requires exactly two labels")
    groups = np.asarray(data.groups)
    splitter = GroupShuffleSplit(n_splits=1, test_size=validation_fraction, random_state=seed)
    train, test = next(splitter.split(data.activations.numpy(), labels, groups))
    classifier = LogisticRegression(max_iter=2000, random_state=seed)
    classifier.fit(data.activations.numpy()[train], labels[train])
    predicted = classifier.predict(data.activations.numpy()[test])
    metrics = {
        "accuracy": float(accuracy_score(labels[test], predicted)),
        "f1_macro": float(f1_score(labels[test], predicted, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(labels[test], predicted, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(labels[test], predicted, average="macro", zero_division=0)),
        "n_train": int(len(train)), "n_test": int(len(test)),
        "n_groups_train": int(len(set(groups[train]))),
        "n_groups_test": int(len(set(groups[test]))),
    }
    direction = torch.as_tensor(classifier.coef_[0], dtype=torch.float32)
    direction = direction / direction.norm().clamp_min(1e-12)
    return classifier, direction, metrics


def compute_caa_direction(data: LayerRepresentation, positive_label=1):
    """Return the normalized mean activation difference for a binary label."""
    positive = data.activations[data.labels == positive_label]
    negative = data.activations[data.labels != positive_label]
    if positive.numel() == 0 or negative.numel() == 0:
        raise ValueError("CAA requires examples from both label classes")
    delta = positive.mean(dim=0) - negative.mean(dim=0)
    return delta / delta.norm().clamp_min(1e-12)


def compute_leace_transform(
    activations: torch.Tensor, labels: torch.Tensor, rtol: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit a covariance-aware affine LEACE eraser for a categorical concept.

    Returns ``(matrix, bias)`` so transformed row vectors are ``x @ matrix.T + bias``.
    The fitted transform is intentionally a separate API from the legacy
    rank-one mean-difference projection.
    """
    x = activations.to(dtype=torch.float64)
    y = labels.reshape(-1)
    if x.ndim != 2 or x.shape[0] != y.numel() or x.shape[0] < 2:
        raise ValueError("Expected activations [samples, features] and one label per row")
    classes, inverse = torch.unique(y, return_inverse=True)
    z = torch.nn.functional.one_hot(inverse, num_classes=len(classes)).to(x.dtype)
    x_mean, z_mean = x.mean(0), z.mean(0)
    xc, zc = x - x_mean, z - z_mean
    covariance = xc.T @ xc / max(x.shape[0] - 1, 1)
    cross_covariance = xc.T @ zc / max(x.shape[0] - 1, 1)

    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    floor = max(float(eigenvalues.max().clamp_min(0)) * rtol, torch.finfo(x.dtype).eps)
    inverse_sqrt = (eigenvectors * eigenvalues.clamp_min(floor).rsqrt()) @ eigenvectors.T
    whitened_cross = inverse_sqrt @ cross_covariance
    u, singular_values, _ = torch.linalg.svd(whitened_cross, full_matrices=False)
    rank = int((singular_values > singular_values.max().clamp_min(0) * rtol).sum())
    basis = u[:, :rank]
    projector = basis @ basis.T if rank else torch.zeros_like(covariance)
    inverse_whitener = torch.linalg.pinv(inverse_sqrt, rtol=rtol)
    matrix = torch.eye(x.shape[1], dtype=x.dtype) - inverse_whitener @ projector @ inverse_sqrt
    bias = x_mean - matrix @ x_mean
    return matrix.to(activations.dtype), bias.to(activations.dtype)
