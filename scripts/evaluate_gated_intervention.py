#!/usr/bin/env python3
"""Compare always-on and gated intervention on labeled held-out prompts."""

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mi_bias.data import load_prompt_pairs
from mi_bias.guardrails import validate_bias_metric_pairs
from mi_bias.interventions import build_residual_hooks
from mi_bias.metrics import (
    score_prompt_bias, score_thesis_legacy_l2, summarize_paired_changes,
    thesis_gender_token_ids,
)
from mi_bias.models import load_hooked_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("test_data", type=Path,
                        help="JSON/CSV with candidate arrays and bias_relevant labels")
    parser.add_argument("transforms", type=Path)
    parser.add_argument("gate", type=Path, help="joblib artifact from train_prompt_gate.py")
    parser.add_argument("--caa-strength", type=float, default=0.0)
    parser.add_argument("--leace-alpha", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=Path("results/gated_intervention.json"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    args = parser.parse_args()

    import joblib
    import torch
    gate = joblib.load(args.gate)
    transforms = torch.load(args.transforms, map_location="cpu", weights_only=False)
    pairs = load_prompt_pairs(args.test_data)
    validate_bias_metric_pairs(pairs)
    if any(label not in (0, 1, "0", "1")
           for pair in pairs for label in (pair.clean_label, pair.corrupted_label)):
        raise ValueError("Held-out pairs need binary clean_label/corrupted_label for gate evaluation")
    loaded = load_hooked_model(
        transforms["model_name"], revision=transforms.get("model_revision"),
        device=args.device, dtype=transforms.get("dtype", "auto"), token=args.hf_token
    )
    saved_revision = transforms.get("model_revision")
    if saved_revision and loaded.model_revision and saved_revision != loaded.model_revision:
        raise ValueError("Loaded model revision does not match the fitted transform artifact")
    model = loaded.model
    hooks = build_residual_hooks(
        transforms["caa_directions"], transforms["leace_transforms"],
        caa_strength=args.caa_strength, leace_alpha=args.leace_alpha,
        expected_d_model=int(model.cfg.d_model),
        n_layers=int(model.cfg.n_layers),
    )
    prompts = [pair.clean for pair in pairs]
    gate_decisions = gate.predict_sensitive(prompts)
    before, static_after, gated_after, legacy_before, legacy_static, legacy_gated = [], [], [], [], [], []
    per_prompt = []
    thesis_token_ids = thesis_gender_token_ids(model)
    gate_eval_prompts, gold = [], []
    for pair in pairs:
        gate_eval_prompts.extend([pair.clean, pair.corrupted])
        gold.extend([pair.clean_label, pair.corrupted_label])
    if any(label not in (0, 1, "0", "1") for label in gold):
        raise ValueError("Held-out pairs need binary clean_label/corrupted_label for gate evaluation")
    gate_eval_gold = [int(label) for label in gold]
    gate_eval_predictions = gate.predict_sensitive(gate_eval_prompts)
    for pair, apply_intervention in zip(pairs, gate_decisions):
        male = pair.metadata["male_continuations"]
        female = pair.metadata["female_continuations"]
        baseline = score_prompt_bias(model, pair.clean, male, female)
        treated = score_prompt_bias(model, pair.clean, male, female, hooks=hooks)
        b = float(baseline["bias_score"].item())
        s = float(treated["bias_score"].item())
        g = s if apply_intervention else b
        legacy_b = score_thesis_legacy_l2(model, pair.clean, *thesis_token_ids)
        legacy_s = score_thesis_legacy_l2(
            model, pair.clean, *thesis_token_ids, hooks=hooks
        )
        legacy_g = legacy_s if apply_intervention else legacy_b
        before.append(b)
        static_after.append(s)
        gated_after.append(g)
        legacy_before.append(legacy_b)
        legacy_static.append(legacy_s)
        legacy_gated.append(legacy_g)
        per_prompt.append({
            "pair_id": pair.pair_id,
            "group_id": pair.group_id,
            "bias_relevant_gold": pair.clean_label,
            "gate_score": float(gate.classifier.predict_proba(
                gate.vectorizer.transform([pair.clean])
            )[0, 1]),
            "intervention_applied": apply_intervention,
            "bias_before": b,
            "bias_always_on": s,
            "bias_gated": g,
            "thesis_legacy_l2_before": legacy_b,
            "thesis_legacy_l2_always_on": legacy_s,
            "thesis_legacy_l2_gated": legacy_g,
        })

    report = {
        "model_name": transforms["model_name"],
        "model_revision": transforms.get("model_revision"),
        "tokenizer_revision": transforms.get("tokenizer_revision"),
        "dtype": transforms.get("dtype"),
        "gate_threshold": gate.threshold,
        "summary": {
            "gate_off": summarize_paired_changes(
                before, before, groups=[pair.group_id for pair in pairs]
            ),
            "always_on": summarize_paired_changes(
                before, static_after, groups=[pair.group_id for pair in pairs]
            ),
            "gated": summarize_paired_changes(
                before, gated_after, groups=[pair.group_id for pair in pairs]
            ),
            "n_prompts_intervened": sum(gate_decisions),
            "intervention_rate": sum(gate_decisions) / max(len(gate_decisions), 1),
        },
        "thesis_legacy_l2_summary": {
            "gate_off": summarize_paired_changes(
                legacy_before, legacy_before, groups=[pair.group_id for pair in pairs]
            ),
            "always_on": summarize_paired_changes(
                legacy_before, legacy_static, groups=[pair.group_id for pair in pairs]
            ),
            "gated": summarize_paired_changes(
                legacy_before, legacy_gated, groups=[pair.group_id for pair in pairs]
            ),
        },
        "per_prompt": per_prompt,
    }
    if gate_eval_gold:
        from sklearn.metrics import confusion_matrix, precision_score, recall_score
        tn, fp, fn, tp = confusion_matrix(
            gate_eval_gold, gate_eval_predictions, labels=[0, 1]
        ).ravel()
        report["gate_test_metrics"] = {
            "precision": float(precision_score(
                gate_eval_gold, gate_eval_predictions, zero_division=0
            )),
            "recall": float(recall_score(
                gate_eval_gold, gate_eval_predictions, zero_division=0
            )),
            "false_positive_rate": float(fp / max(fp + tn, 1)),
            "false_negative_rate": float(fn / max(fn + tp, 1)),
            "n_prompt_contexts": len(gate_eval_gold),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved gate comparison: {args.output.resolve()}")


if __name__ == "__main__":
    main()
