"""Model loading helpers with explicit model and tokenizer revisions."""

from __future__ import annotations

from dataclasses import dataclass
import gc
from typing import Any


@dataclass
class LoadedModel:
    model: Any
    tokenizer: Any
    model_name: str
    model_revision: str | None
    tokenizer_revision: str | None
    device: str
    dtype: str


def load_hooked_model(
    model_name: str,
    *,
    revision: str | None = None,
    device: str = "auto",
    dtype: str = "auto",
    token: str | None = None,
) -> LoadedModel:
    """Load a pinned Hugging Face checkpoint into TransformerLens.

    Transformers performs the revision-pinned download first; the resulting
    weights are then wrapped by TransformerLens for hook-based experiments.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformer_lens import HookedTransformer

    resolved_device = (
        "cuda" if device == "auto" and torch.cuda.is_available()
        else "cpu" if device == "auto" else device
    )
    resolved_dtype = (
        "float16" if dtype == "auto" and resolved_device.startswith("cuda")
        else "float32" if dtype == "auto" else dtype
    )
    if resolved_dtype not in {"float16", "bfloat16", "float32"}:
        raise ValueError("dtype must be auto, float16, bfloat16, or float32")
    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[resolved_dtype]
    if resolved_dtype == "bfloat16" and resolved_device.startswith("cuda"):
        if not torch.cuda.is_bf16_supported():
            raise ValueError("The selected CUDA GPU does not support bfloat16")
    common = {"revision": revision, "token": token}
    tokenizer = AutoTokenizer.from_pretrained(model_name, **common)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, **common
    )
    hooked = HookedTransformer.from_pretrained(
        model_name, hf_model=hf_model, tokenizer=tokenizer, device=resolved_device,
        dtype=resolved_dtype,
    )
    hooked.eval()
    resolved_model_revision = getattr(hf_model.config, "_commit_hash", None) or revision
    resolved_tokenizer_revision = (
        getattr(tokenizer, "init_kwargs", {}).get("_commit_hash") or revision
    )
    del hf_model
    gc.collect()
    if resolved_device.startswith("cuda"):
        torch.cuda.empty_cache()
    return LoadedModel(
        hooked, hooked.tokenizer, model_name, resolved_model_revision,
        resolved_tokenizer_revision, resolved_device, resolved_dtype
    )


def architecture_metadata(model: Any) -> dict[str, Any]:
    """Return architecture dimensions used to interpret layer-indexed results."""
    cfg = model.cfg
    return {
        "model_name": getattr(cfg, "model_name", None),
        "n_layers": int(cfg.n_layers),
        "n_heads": int(cfg.n_heads),
        "d_model": int(cfg.d_model),
        "d_head": int(cfg.d_head),
        "device": str(cfg.device),
    }
