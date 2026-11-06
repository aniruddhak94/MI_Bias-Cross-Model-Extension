"""Inference-time CAA, LEACE, hybrid, and residual-hook composition."""

from __future__ import annotations

from collections.abc import Mapping

import torch


def build_residual_hooks(
    steering_vectors: Mapping[int, torch.Tensor] | None = None,
    leace_transforms: Mapping[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
    *,
    caa_strength: float = 0.0,
    leace_alpha: float = 0.0,
    expected_d_model: int | None = None,
    n_layers: int | None = None,
):
    """Build TransformerLens hooks for CAA, LEACE, or their hybrid.

    At a target layer, LEACE is blended first and CAA is subtracted second,
    matching the ordering used in the prior thesis experiment.
    """
    steering_vectors = steering_vectors or {}
    leace_transforms = leace_transforms or {}
    layers = sorted(set(steering_vectors) | set(leace_transforms))
    hooks = []
    for layer in layers:
        if layer < 0 or (n_layers is not None and layer >= n_layers):
            raise ValueError(f"Intervention layer {layer} is outside the loaded model")
        vector = steering_vectors.get(layer)
        transform = leace_transforms.get(layer)
        if vector is not None and vector.ndim != 1:
            raise ValueError(f"CAA vector for layer {layer} must be one-dimensional")
        if expected_d_model is not None and vector is not None and vector.shape[0] != expected_d_model:
            raise ValueError(f"CAA vector for layer {layer} does not match model width")
        if transform is not None:
            matrix, bias = transform
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise ValueError(f"LEACE matrix for layer {layer} must be square")
            if bias.ndim != 1 or bias.shape[0] != matrix.shape[0]:
                raise ValueError(f"LEACE bias for layer {layer} must match its feature dimension")
            if expected_d_model is not None and matrix.shape[0] != expected_d_model:
                raise ValueError(f"LEACE transform for layer {layer} does not match model width")
            if vector is not None and vector.shape[0] != matrix.shape[0]:
                raise ValueError(f"CAA and LEACE dimensions disagree at layer {layer}")

        def make_hook(vector, transform):
            def hook(activation, **_kwargs):
                result = activation
                if transform is not None and leace_alpha:
                    matrix, bias = (part.to(activation.device, activation.dtype) for part in transform)
                    projected = torch.matmul(result, matrix.T) + bias
                    result = result + leace_alpha * (projected - result)
                if vector is not None and caa_strength:
                    result = result - caa_strength * vector.to(
                        activation.device, activation.dtype
                    ).view(1, 1, -1)
                return result
            return hook

        hooks.append((f"blocks.{layer}.hook_resid_pre", make_hook(vector, transform)))
    return hooks


def run_with_intervention(model, tokens, hooks):
    """Run an inference pass under the supplied hook set."""
    with torch.inference_mode():
        return model.run_with_hooks(tokens, fwd_hooks=hooks) if hooks else model(tokens)
