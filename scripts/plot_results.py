#!/usr/bin/env python3
"""Generate thesis-style PNG and PDF figures from saved experiment outputs."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PALETTE = {
    "primary": "#1B4F72",
    "secondary": "#2E86C1",
    "accent": "#E67E22",
    "success": "#27AE60",
    "danger": "#E74C3C",
    "light": "#AED6F1",
    "dark": "#1C2833",
    "muted": "#95A5A6",
    "grid": "#D5DBDB",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.facecolor": "#FDFEFE",
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.color": PALETTE["grid"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 160,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
})


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_figure(fig, output_dir: Path, stem: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png")
    fig.savefig(output_dir / f"{stem}.pdf")
    plt.close(fig)
    print(f"Saved {stem}.png and {stem}.pdf")


def bootstrap_mean_ci(values, seed=42, samples=5000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return np.quantile(means, [0.025, 0.975])


def short_label(prompt, pair_id):
    if prompt:
        words = prompt.replace("\n", " ").split()
        text = " ".join(words[:5])
        return text[:40] + ("…" if len(text) > 40 else "")
    return str(pair_id)


def plot_bias_summary(baseline, static, gated, output_dir):
    baseline_rows = baseline.get("per_prompt", [])
    static_rows = static.get("per_prompt", [])
    base_by_id = {row["pair_id"]: float(row["bias_score"]) for row in baseline_rows}
    static_by_id = {row["pair_id"]: row for row in static_rows}
    pair_ids = [key for key in base_by_id if key in static_by_id]
    before = np.asarray([base_by_id[key] for key in pair_ids], dtype=float)
    after = np.asarray([float(static_by_id[key]["bias_after"]) for key in pair_ids], dtype=float)
    if not len(pair_ids):
        print("Skipped bias summary: baseline/static per-prompt IDs did not match")
        return

    means = [before.mean(), after.mean()]
    labels = ["Baseline", "Static intervention"]
    if gated and gated.get("summary"):
        gated_rows = {row["pair_id"]: row for row in gated.get("per_prompt", [])}
        gated_values = [float(gated_rows[key]["bias_gated"]) for key in pair_ids if key in gated_rows]
        if len(gated_values) == len(pair_ids):
            means.append(float(np.mean(gated_values)))
            labels.append("Gated intervention")

    intervals = [bootstrap_mean_ci(before), bootstrap_mean_ci(after)]
    if len(means) == 3:
        gated_values = np.asarray(gated_values, dtype=float)
        intervals.append(bootstrap_mean_ci(gated_values))
    yerr = np.asarray([
        [mean - ci[0] for mean, ci in zip(means, intervals)],
        [ci[1] - mean for mean, ci in zip(means, intervals)],
    ])

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [PALETTE["primary"], PALETTE["success"], PALETTE["accent"]][:len(means)]
    bars = ax.bar(labels, means, color=colors, alpha=0.85, width=0.62)
    ax.errorbar(np.arange(len(means)), means, yerr=yerr, fmt="none",
                ecolor=PALETTE["dark"], capsize=5, linewidth=1.4)
    for bar, value in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Mean paired continuation bias score")
    ax.set_title(f"Held-out bias comparison (n={len(pair_ids)}; 95% bootstrap CIs)",
                 fontweight="bold")
    ax.tick_params(axis="x", rotation=12)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    save_figure(fig, output_dir, "bias_summary")


def plot_per_prompt(static, output_dir):
    rows = static.get("per_prompt", [])
    if not rows:
        print("Skipped per-prompt plot: no per_prompt rows in static result")
        return
    rows = sorted(rows, key=lambda row: float(row["reduction"]))
    before = np.asarray([float(row["bias_before"]) for row in rows])
    after = np.asarray([float(row["bias_after"]) for row in rows])
    reduction = before - after
    labels = [short_label(row.get("clean_prompt"), row.get("pair_id")) for row in rows]
    y = np.arange(len(rows))

    height = max(6, len(rows) * 0.22)
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(15, height), sharey=True,
        gridspec_kw={"width_ratios": [3, 1], "wspace": 0.08},
    )
    left.barh(y + 0.18, before, height=0.34, color="#F1948A", label="Before")
    left.barh(y - 0.18, after, height=0.34, color="#82E0AA", label="After")
    left.set_yticks(y)
    left.set_yticklabels(labels, fontsize=7)
    left.set_xlabel("Paired continuation bias")
    left.set_title("Before and after by prompt", fontweight="bold")
    left.legend(loc="lower right")
    left.invert_yaxis()

    colors = [PALETTE["success"] if val > 0 else PALETTE["danger"] for val in reduction]
    right.barh(y, reduction, color=colors, height=0.62)
    right.axvline(0, color=PALETTE["dark"], linewidth=1)
    right.set_xlabel("Reduction (+ = lower bias)")
    right.set_title("Paired change", fontweight="bold")
    right.tick_params(axis="y", left=False, labelleft=False)
    fig.suptitle("Per-prompt held-out intervention results", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, output_dir, "per_prompt_bias_comparison")


def plot_bias_distribution(static, output_dir):
    rows = static.get("per_prompt", [])
    if not rows:
        return
    before = np.asarray([float(row["bias_before"]) for row in rows])
    after = np.asarray([float(row["bias_after"]) for row in rows])
    reduction = before - after
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    violin = axes[0].violinplot([before, after], positions=[1, 2], showmeans=True,
                                showmedians=True, widths=0.65)
    for body, color in zip(violin["bodies"], [PALETTE["danger"], PALETTE["success"]]):
        body.set_facecolor(color)
        body.set_alpha(0.6)
    axes[0].set_xticks([1, 2], ["Before", "After"])
    axes[0].set_ylabel("Paired continuation bias score")
    axes[0].set_title("Bias score distributions", fontweight="bold")
    axes[1].hist(reduction, bins=min(20, max(6, len(reduction) // 3)),
                 color=PALETTE["secondary"], edgecolor="white", alpha=0.85)
    axes[1].axvline(0, color=PALETTE["dark"], linewidth=1.2)
    axes[1].axvline(reduction.mean(), color=PALETTE["accent"], linestyle="--",
                    label=f"Mean = {reduction.mean():.3f}")
    axes[1].set_xlabel("Paired bias reduction (+ = lower bias)")
    axes[1].set_ylabel("Prompt count")
    axes[1].set_title("Distribution of per-prompt changes", fontweight="bold")
    axes[1].legend()
    fig.tight_layout()
    save_figure(fig, output_dir, "bias_distribution")


def plot_validation_grid(results_dir, output_dir):
    candidates = []
    for path in sorted(results_dir.glob("validation_caa_*_leace_*.json")):
        result = load_json(path)
        if not result or "summary" not in result:
            continue
        match = re.fullmatch(r"validation_caa_([-+0-9.eE]+)_leace_([-+0-9.eE]+)", path.stem)
        if not match:
            continue
        caa, alpha = map(float, match.groups())
        candidates.append((caa, alpha, float(result["summary"]["relative_reduction_percent"])))
    if not candidates:
        print("Skipped validation grid: no validation_caa_*_leace_*.json files")
        return
    caas = sorted({row[0] for row in candidates})
    alphas = sorted({row[1] for row in candidates})
    matrix = np.full((len(caas), len(alphas)), np.nan)
    for caa, alpha, reduction in candidates:
        matrix[caas.index(caa), alphas.index(alpha)] = reduction
    fig, ax = plt.subplots(figsize=(max(6, len(alphas) * 1.6), max(4, len(caas) * 1.0)))
    image = ax.imshow(matrix, cmap="Greens", aspect="auto", origin="lower")
    ax.set_xticks(range(len(alphas)), [f"{value:g}" for value in alphas])
    ax.set_yticks(range(len(caas)), [f"{value:g}" for value in caas])
    ax.set_xlabel("LEACE alpha")
    ax.set_ylabel("CAA strength")
    ax.set_title("Validation relative bias reduction (%)", fontweight="bold")
    for i in range(len(caas)):
        for j in range(len(alphas)):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.1f}%", ha="center", va="center",
                        color="black", fontweight="bold")
    fig.colorbar(image, ax=ax, label="Relative reduction (%)")
    fig.tight_layout()
    save_figure(fig, output_dir, "validation_grid_heatmap")


def plot_probe_layers(artifacts_dir, output_dir):
    probe_path = artifacts_dir / "gpt2_small_directions.json"
    data = load_json(probe_path)
    metrics = data.get("probe_metrics", {}) if data else {}
    if not metrics:
        print("Skipped probe plot: no probe_metrics in directions JSON")
        return
    layers = sorted((int(layer), value) for layer, value in metrics.items())
    x = [item[0] for item in layers]
    accuracy = [float(item[1]["accuracy"]) * 100 for item in layers]
    f1 = [float(item[1]["f1_macro"]) * 100 for item in layers]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, accuracy, "o-", color=PALETTE["primary"], linewidth=2,
            label="Accuracy")
    ax.plot(x, f1, "s--", color=PALETTE["accent"], linewidth=2,
            label="Macro F1")
    ax.axhline(50, color=PALETTE["muted"], linestyle=":", label="Binary chance reference")
    ax.set_xticks(x, [f"L{layer}" for layer in x])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Score (%)")
    ax.set_title("Probe performance by residual layer", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, output_dir, "probe_accuracy_layers")


def plot_gate_diagnostics(artifacts_dir, gated, output_dir):
    fit = load_json(artifacts_dir / "gpt2_small_gate.json") or {}
    validation = fit.get("validation_metrics", {})
    test = (gated or {}).get("gate_test_metrics", {})
    if not validation and not test:
        print("Skipped gate plot: no saved gate metrics")
        return
    intervention_rate = (gated or {}).get("summary", {}).get("intervention_rate")
    labels = ["Precision", "Recall", "False positive rate", "False negative rate"]
    val_values = [validation.get("precision"), validation.get("recall"),
                  validation.get("false_positive_rate"), validation.get("false_negative_rate")]
    test_values = [test.get("precision"), test.get("recall"),
                   test.get("false_positive_rate"), test.get("false_negative_rate")]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(labels))
    width = 0.36
    if any(value is not None for value in val_values):
        ax.bar(x - width / 2, [np.nan if value is None else value for value in val_values],
               width, color=PALETTE["secondary"], label="Gate validation")
    if any(value is not None for value in test_values):
        ax.bar(x + width / 2, [np.nan if value is None else value for value in test_values],
               width, color=PALETTE["accent"], label="Held-out gate test")
    ax.set_xticks(x, labels, rotation=12, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Rate")
    title = "Prompt gate error and detection rates"
    if intervention_rate is not None:
        title += f" (test intervention rate={float(intervention_rate):.0%})"
    ax.set_title(title, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, output_dir, "gate_diagnostics")


def plot_eap_edges(results_dir, output_dir, top_k=20):
    edge_path = results_dir / "eap_gpt2" / "gpt2_gender_GSS1_M1_top_edges.txt"
    if not edge_path.exists():
        print("Skipped EAP edge plot: no saved top_edges.txt")
        return
    parsed = []
    pattern = re.compile(r"^\s*\d+\.\s+(.+?): score=([-+0-9.eE]+)")
    for line in edge_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(line)
        if match:
            parsed.append((match.group(1), float(match.group(2))))
    if not parsed:
        print("Skipped EAP edge plot: no parseable edge scores")
        return
    parsed = parsed[:top_k]
    parsed.reverse()
    labels = [row[0] for row in parsed]
    scores = [row[1] for row in parsed]
    colors = [PALETTE["secondary"] if score >= 0 else PALETTE["accent"] for score in scores]
    fig, ax = plt.subplots(figsize=(10, max(5, len(labels) * 0.28)))
    ax.barh(np.arange(len(labels)), scores, color=colors, alpha=0.88)
    ax.set_yticks(np.arange(len(labels)), labels, fontsize=8)
    ax.axvline(0, color=PALETTE["dark"], linewidth=1)
    ax.set_xlabel("EAP attribution score (signed)")
    ax.set_title(f"Top {len(labels)} EAP edges by absolute attribution", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, output_dir, "eap_top_edges")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures"))
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    baseline = load_json(args.results_dir / "gpt2_small_baseline_test.json")
    static = load_json(args.results_dir / "gpt2_small_static_test.json")
    gated = load_json(args.results_dir / "gpt2_small_gated_test.json")
    if baseline and static:
        plot_bias_summary(baseline, static, gated, args.output_dir)
        plot_per_prompt(static, args.output_dir)
        plot_bias_distribution(static, args.output_dir)
    else:
        print("Skipped held-out bias figures: baseline/static test JSON is missing")

    plot_validation_grid(args.results_dir, args.output_dir)
    plot_probe_layers(args.artifacts_dir, args.output_dir)
    plot_gate_diagnostics(args.artifacts_dir, gated, args.output_dir)
    plot_eap_edges(args.results_dir, args.output_dir, top_k=args.top_k)
    print(f"Figures saved in: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
