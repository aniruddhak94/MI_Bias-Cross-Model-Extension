#!/usr/bin/env python3
"""Create per-model thesis figures and cautious cross-model comparisons."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {"baseline": "#1B4F72", "static": "#27AE60", "gated": "#E67E22"}
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.22,
    "figure.dpi": 160, "savefig.dpi": 240, "savefig.bbox": "tight",
})


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def save(fig, folder: Path, stem: str):
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{stem}.png")
    fig.savefig(folder / f"{stem}.pdf")
    plt.close(fig)
    print(f"Saved {folder / stem}.png and .pdf")


def bootstrap_mean(values, rng, n_boot=5000, groups=None):
    x = np.asarray(values, dtype=float)
    if len(x) == 0:
        return (np.nan, np.nan)
    if groups is None:
        idx = rng.integers(0, len(x), size=(n_boot, len(x)))
        means = x[idx].mean(axis=1)
    else:
        groups = np.asarray(groups, dtype=str)
        if len(groups) != len(x):
            raise ValueError("Provide one bootstrap group per prompt")
        clusters = [x[groups == group] for group in np.unique(groups)]
        draws = rng.integers(0, len(clusters), size=(n_boot, len(clusters)))
        means = np.asarray([
            np.concatenate([clusters[i] for i in draw]).mean() for draw in draws
        ])
    return tuple(np.quantile(means, [0.025, 0.975]))


def paired_arrays(static):
    rows = static.get("per_prompt", [])
    before = np.asarray([float(row.get("counterfactual_bias_before", row["bias_before"])) for row in rows])
    after = np.asarray([float(row.get("counterfactual_bias_after", row["bias_after"])) for row in rows])
    return rows, before, after


def plot_model(slug, label, baseline, static, gated, artifacts_root, results_root, output_root, seed):
    folder = output_root / "by_model" / slug
    rows, before, after = paired_arrays(static)
    if not len(rows):
        print(f"Skipped {slug}: held-out intervention has no per-prompt rows")
        return None
    ids = [row.get("pair_id") for row in rows]
    base_rows = {row.get("pair_id"): row for row in (baseline or {}).get("per_prompt", [])}
    if baseline and any(key not in base_rows for key in ids):
        raise ValueError(f"Baseline/test prompt IDs differ for {slug}")
    if baseline:
        before_from_baseline = np.asarray([float(base_rows[key].get(
            "counterfactual_bias_score", base_rows[key]["bias_score"]
        )) for key in ids])
        if not np.allclose(before, before_from_baseline, rtol=1e-5, atol=1e-6):
            print(f"WARNING: baseline and static pre-scores differ for {slug}; figures use baseline JSON values")
        before = before_from_baseline
    rng = np.random.default_rng(seed)
    group_ids = [row.get("group_id", row.get("pair_id")) for row in rows]
    reductions = before - after
    labels = ["Baseline", "Static"]
    values = [before, after]
    colors = [COLORS["baseline"], COLORS["static"]]
    if gated:
        gate_rows = {row.get("pair_id"): row for row in gated.get("per_prompt", [])}
        if all(key in gate_rows for key in ids) and all("bias_gated" in gate_rows[key] for key in ids):
            gate_key = "counterfactual_bias_gated" if all(
                "counterfactual_bias_gated" in gate_rows[key] for key in ids
            ) else "bias_gated"
            values.append(np.asarray([float(gate_rows[key][gate_key]) for key in ids]))
            labels.append("Gated")
            colors.append(COLORS["gated"])

    means = [float(x.mean()) for x in values]
    intervals = [bootstrap_mean(x, rng, groups=group_ids) for x in values]
    errors = np.asarray([[m - ci[0] for m, ci in zip(means, intervals)],
                         [ci[1] - m for m, ci in zip(means, intervals)]])
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    bars = ax.bar(labels, means, color=colors, alpha=0.88, width=0.6)
    ax.errorbar(range(len(means)), means, yerr=errors, fmt="none", ecolor="#17202A", capsize=5)
    for bar, value in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, value, f"{value:.3f}", ha="center", va="bottom")
    ax.set_ylabel("Mean absolute counterfactual stereotype effect (log-odds)")
    ax.set_title(f"{label}: held-out counterfactual bias (95% group-bootstrap CI, n={len(before)})")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    save(fig, folder, f"{slug}_heldout_bias")

    ci = bootstrap_mean(reductions, rng, groups=group_ids)
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ax.errorbar(reductions.mean(), 0,
                xerr=[[reductions.mean()-ci[0]], [ci[1]-reductions.mean()]],
                fmt="o", color=COLORS["static"], capsize=6, markersize=8)
    ax.axvline(0, color="#C0392B", linestyle="--", label="No mean reduction")
    ax.set_yticks([])
    ax.set_xlabel("Paired bias reduction (+ means lower bias)")
    ax.set_title(f"{label}: mean reduction, 95% paired bootstrap CI\nImproved {int((reductions > 0).sum())}/{len(reductions)} prompts")
    ax.legend()
    fig.tight_layout()
    save(fig, folder, f"{slug}_paired_reduction")

    thesis_summary = (static or {}).get("thesis_legacy_l2_summary")
    thesis_rows = (static or {}).get("per_prompt", [])
    if thesis_summary and thesis_rows and all(
        "thesis_legacy_l2_before" in row and "thesis_legacy_l2_after" in row
        for row in thesis_rows
    ):
        legacy_before = np.asarray([row["thesis_legacy_l2_before"] for row in thesis_rows], dtype=float)
        legacy_after = np.asarray([row["thesis_legacy_l2_after"] for row in thesis_rows], dtype=float)
        legacy_delta = legacy_before - legacy_after
        legacy_ci = bootstrap_mean(legacy_delta, rng, groups=group_ids)
        fig, ax = plt.subplots(figsize=(7.4, 3.8))
        ax.errorbar(legacy_delta.mean(), 0,
                    xerr=[[legacy_delta.mean()-legacy_ci[0]], [legacy_ci[1]-legacy_delta.mean()]],
                    fmt="o", color="#8E44AD", capsize=6, markersize=8)
        ax.axvline(0, color="#C0392B", linestyle="--", label="No mean reduction")
        ax.set_yticks([])
        ax.set_xlabel("Historical first-token L2 reduction (+ means lower bias)")
        ax.set_title(f"{label}: thesis-compatible metric, occupation-cluster bootstrap CI\nn={len(legacy_delta)}")
        ax.legend()
        fig.tight_layout()
        save(fig, folder, f"{slug}_thesis_legacy_l2")

    per_prompt_reduction = np.sort(reductions)
    fig, ax = plt.subplots(figsize=(8.5, max(4.5, min(10, len(rows) * 0.13))))
    ax.barh(np.arange(len(rows)), per_prompt_reduction, color=np.where(
        per_prompt_reduction > 0, COLORS["static"], "#C0392B"))
    ax.axvline(0, color="#17202A", linewidth=1)
    ax.set_xlabel("Per-prompt held-out bias reduction (+ = lower bias)")
    ax.set_ylabel("Test prompts, sorted by paired change")
    ax.set_title(f"{label}: individual test-prompt changes")
    fig.tight_layout()
    save(fig, folder, f"{slug}_per_prompt_changes")

    group_indices = [np.flatnonzero(np.asarray(group_ids) == group)
                     for group in np.unique(np.asarray(group_ids, dtype=str))]
    group_draws = rng.integers(0, len(group_indices), size=(5000, len(group_indices)))
    relative_samples = []
    for draw in group_draws:
        sample = np.concatenate([group_indices[i] for i in draw])
        denom = before[sample].mean()
        relative_samples.append(100 * reductions[sample].mean() / max(denom, 1e-12))
    relative_ci = np.quantile(relative_samples, [0.025, 0.975])

    # The layerwise probe figure is model-specific because layer indices cannot
    # be compared directly across architectures.
    direction_json = read_json(artifacts_root / f"{slug}_directions.json")
    probe_metrics = (direction_json or {}).get("probe_metrics", {})
    if probe_metrics:
        points = sorted((int(layer), float(metrics["accuracy"]),
                         float(metrics.get("f1_macro", metrics["accuracy"])))
                        for layer, metrics in probe_metrics.items())
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.plot([p[0] for p in points], [p[1]*100 for p in points], "o-", label="Accuracy")
        ax.plot([p[0] for p in points], [p[2]*100 for p in points], "s--", label="Macro F1")
        ax.axhline(50, color="#95A5A6", linestyle=":", label="Binary chance")
        ax.set_xlabel("Residual layer (model-specific index)")
        ax.set_ylabel("Probe score (%)")
        ax.set_ylim(0, 105)
        ax.set_title(f"{label}: layerwise probe diagnostics")
        ax.legend()
        fig.tight_layout()
        save(fig, folder, f"{slug}_probe_layers")

    gate_fit = read_json(artifacts_root / f"{slug}_gate.json")
    gate_test = (gated or {}).get("gate_test_metrics", {})
    if gate_fit and (gate_fit.get("validation_metrics") or gate_test):
        metrics = ["precision", "recall", "false_positive_rate", "false_negative_rate"]
        fit_values = gate_fit.get("validation_metrics", {})
        x = np.arange(len(metrics)); width = 0.38
        fig, ax = plt.subplots(figsize=(8.5, 4.7))
        if fit_values:
            ax.bar(x-width/2, [fit_values.get(m, np.nan) for m in metrics], width, label="Validation")
        if gate_test:
            ax.bar(x+width/2, [gate_test.get(m, np.nan) for m in metrics], width, label="Held-out test")
        ax.set_xticks(x, [m.replace("_", " ").title() for m in metrics], rotation=12)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("Rate")
        ax.set_title(f"{label}: gate diagnostics (threshold={gate_fit.get('threshold', 'n/a')})")
        ax.legend()
        fig.tight_layout()
        save(fig, folder, f"{slug}_gate_diagnostics")

    candidates = []
    for path in sorted(results_root.glob(f"validation_{slug}_caa_*_leace_*.json")):
        data = read_json(path)
        match = re.fullmatch(rf"validation_{re.escape(slug)}_caa_([-+0-9.eE]+)_leace_([-+0-9.eE]+)", path.stem)
        if match and data and data.get("summary"):
            candidates.append((float(match.group(1)), float(match.group(2)),
                               float(data["summary"]["relative_reduction_percent"])))
    if candidates:
        caas = sorted({item[0] for item in candidates})
        alphas = sorted({item[1] for item in candidates})
        matrix = np.full((len(caas), len(alphas)), np.nan)
        for caa, alpha, value in candidates:
            matrix[caas.index(caa), alphas.index(alpha)] = value
        fig, ax = plt.subplots(figsize=(max(6, len(alphas)*1.4), max(4, len(caas)*0.8)))
        im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", origin="lower")
        ax.set_xticks(range(len(alphas)), [f"{v:g}" for v in alphas])
        ax.set_yticks(range(len(caas)), [f"{v:g}" for v in caas])
        ax.set_xlabel("LEACE blend")
        ax.set_ylabel("CAA strength")
        ax.set_title(f"{label}: validation counterfactual relative reduction (%)")
        for i in range(len(caas)):
            for j in range(len(alphas)):
                if np.isfinite(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i,j]:.1f}%", ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=ax, label="Relative bias reduction (%)")
        fig.tight_layout()
        save(fig, folder, f"{slug}_validation_grid")

    capability = read_json(results_root / f"{slug}_capability.json")
    capability_summary = (capability or {}).get("summary", {})
    if capability_summary:
        cap_before = float(capability_summary["perplexity_before"])
        cap_after = float(capability_summary["perplexity_after"])
        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        bars = ax.bar(["Baseline", "Static"], [cap_before, cap_after],
                      color=[COLORS["baseline"], COLORS["static"]], width=0.58)
        for bar, val in zip(bars, [cap_before, cap_after]):
            ax.text(bar.get_x()+bar.get_width()/2, val, f"{val:.2f}", ha="center", va="bottom")
        ax.set_ylabel("Token-weighted perplexity")
        ax.set_title(f"{label}: {capability.get('dataset_label', 'capability')}\nLanguage-modeling metric only")
        fig.tight_layout()
        save(fig, folder, f"{slug}_capability_perplexity")
    return {
        "slug": slug, "label": label, "n": len(before),
        "baseline_mean": float(before.mean()), "static_mean": float(after.mean()),
        "mean_reduction": float(reductions.mean()),
        "reduction_ci": [float(ci[0]), float(ci[1])],
        "relative_reduction_percent": float(100 * reductions.mean() / max(before.mean(), 1e-12)),
        "relative_reduction_ci": [float(relative_ci[0]), float(relative_ci[1])],
        "improved_count": int((reductions > 0).sum()),
        "gated_mean": float(values[2].mean()) if len(values) > 2 else None,
        "capability_perplexity_change_percent": (
            float(capability_summary["perplexity_change_percent"])
            if capability_summary else None
        ),
        "thesis_legacy_l2_mean_baseline": (
            float(thesis_summary["mean_before"]) if thesis_summary else None
        ),
        "thesis_legacy_l2_mean_static": (
            float(thesis_summary["mean_after"]) if thesis_summary else None
        ),
        "thesis_legacy_l2_reduction_percent": (
            float(thesis_summary["relative_reduction_percent"])
            if thesis_summary else None
        ),
        "thesis_legacy_l2_reduction_ci": (
            thesis_summary.get("relative_reduction_bootstrap_95ci")
            if thesis_summary else None
        ),
    }


def plot_cross_model(summaries, output_root):
    if len(summaries) < 2:
        print("Need at least two completed models for cross-model plots")
        return
    labels = [item["label"] for item in summaries]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 5))
    means = np.asarray([item["relative_reduction_percent"] for item in summaries])
    lower = np.asarray([item["relative_reduction_ci"][0] for item in summaries])
    upper = np.asarray([item["relative_reduction_ci"][1] for item in summaries])
    ax.errorbar(x, means, yerr=[means-lower, upper-means], fmt="o", capsize=6,
                color="#21618C", markersize=8, linewidth=1.8)
    ax.axhline(0, color="#C0392B", linestyle="--", label="No mean reduction")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Relative held-out bias reduction (%)")
    ax.set_title("Cross-model paired intervention comparison\n95% within-model bootstrap intervals")
    ax.legend()
    fig.tight_layout()
    save(fig, output_root, "cross_model_relative_reduction")

    legacy = [item for item in summaries
              if item.get("thesis_legacy_l2_reduction_percent") is not None]
    if legacy:
        labels_legacy = [item["label"] for item in legacy]
        lx = np.arange(len(legacy))
        lmeans = np.asarray([item["thesis_legacy_l2_reduction_percent"] for item in legacy])
        lleft = np.asarray([item["thesis_legacy_l2_reduction_ci"][0] for item in legacy])
        lright = np.asarray([item["thesis_legacy_l2_reduction_ci"][1] for item in legacy])
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.errorbar(lx, lmeans, yerr=[lmeans-lleft, lright-lmeans], fmt="o", capsize=6)
        ax.axhline(0, color="#C0392B", linestyle="--", label="No mean reduction")
        ax.set_xticks(lx, labels_legacy)
        ax.set_ylabel("Held-out historical L2 bias reduction (%)")
        ax.set_title("Thesis-compatible metric across models\nOccupation-cluster bootstrap intervals")
        ax.legend()
        fig.tight_layout()
        save(fig, output_root, "cross_model_thesis_legacy_l2")

    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.36
    base = [item["baseline_mean"] for item in summaries]
    treated = [item["static_mean"] for item in summaries]
    ax.bar(x-width/2, base, width, color=COLORS["baseline"], label="Baseline")
    ax.bar(x+width/2, treated, width, color=COLORS["static"], label="Static intervention")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Mean token-aware paired bias score")
    ax.set_title("Model-specific baseline and intervention scores\nInterpret absolute levels with tokenizer and model differences in mind")
    ax.legend()
    fig.tight_layout()
    save(fig, output_root, "cross_model_bias_means")

    capability = [item for item in summaries
                  if item.get("capability_perplexity_change_percent") is not None]
    if capability:
        fig, ax = plt.subplots(figsize=(8.5, 5))
        for item in capability:
            ax.scatter(item["relative_reduction_percent"],
                       item["capability_perplexity_change_percent"], s=90)
            ax.annotate(item["label"], (item["relative_reduction_percent"],
                                       item["capability_perplexity_change_percent"]),
                        xytext=(5, 4), textcoords="offset points")
        ax.axhline(0, color="#7F8C8D", linestyle="--")
        ax.axvline(0, color="#7F8C8D", linestyle="--")
        ax.set_xlabel("Relative held-out bias reduction (%)")
        ax.set_ylabel("Change in held-out perplexity (%)")
        ax.set_title("Bias–language-modeling trade-off\nCoLA perplexity is not task accuracy")
        fig.tight_layout()
        save(fig, output_root, "cross_model_bias_capability_tradeoff")

    table = {
        "comparison_note": "Primary relative reductions use paired held-out counterfactual stereotype-effect scores with occupation-cluster bootstrap intervals; absolute scores may differ with model/tokenizer behavior.",
        "models": summaries,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "cross_model_summary.json").write_text(json.dumps(table, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/cross_model"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures/cross_model"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/cross_model"))
    parser.add_argument("--models", default="gpt2_small,gpt2_large,pythia_1b")
    parser.add_argument("--labels", default="GPT-2 small,GPT-2 Large,Pythia-1B")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    slugs, labels = args.models.split(","), args.labels.split(",")
    if len(slugs) != len(labels):
        parser.error("--models and --labels must have the same number of comma-separated entries")
    summaries = []
    for slug, label in zip(slugs, labels):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", slug):
            parser.error(f"Unsafe model slug: {slug}")
        baseline = read_json(args.results_dir / f"{slug}_baseline_test.json")
        static = read_json(args.results_dir / f"{slug}_static_test.json")
        gated = read_json(args.results_dir / f"{slug}_gated_test.json")
        if static is None:
            print(f"Skipping {slug}: missing static held-out JSON")
            continue
        summaries.append(plot_model(slug, label, baseline, static, gated,
                                    args.artifacts_dir, args.results_dir,
                                    args.output_dir, args.seed))
    summaries = [item for item in summaries if item]
    plot_cross_model(summaries, args.output_dir)
    print(f"Processed {len(summaries)} models; figures at {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
