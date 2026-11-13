# Cross-Model Mechanistic Bias Intervention Roadmap

## Research sequence

1. **Replication and audit:** reproduce the thesis's GPT-2 small setup using `Dataset/thesis_gender_bias_200.json`; preserve the historical L2 score alongside token-aware continuation scoring, and reconcile model/tokenizer revisions and saved artifacts before claiming replication.
2. **Comparable baselines:** measure gender and demographic bias on matched prompt sets, first with GPT-2 small, then GPT-2 Large, then Pythia-1B for the default Kaggle study. Llama-2 remains an optional future run for a larger-memory environment. Keep model revisions, tokenizers, data versions, and per-prompt outputs attached to every result.
3. **Mechanism analysis:** use EAP to rank edges; validate top-ranked edges with direct activation patching. Train layerwise probes with complete prompt groups held out together. Treat probe scores as decodability evidence, not causal evidence.
4. **Static interventions:** compare edge ablation, CAA, covariance-aware LEACE, and their hybrid. Fit directions and projections on training data; tune strengths on validation data; report locked settings on held-out data.
5. **Dynamic intervention gate:** train a prompt classifier on explicit bias-relevant versus neutral labels. Select its threshold on validation data with recall prioritized. Compare gate-off, always-on, static, and gated conditions; report false-positive and false-negative rates.
6. **Thesis evaluation:** report paired bias changes, capability impact, per-prompt outcomes, confidence intervals, seed variability, and the selected operating point for each model and bias type.

## Recommended package layout

```text
src/mi_bias/
  data.py          # prompt pairs, labels, group-safe splits
  models.py        # checkpoint-pinned model loading and architecture metadata
  metrics.py       # thesis-compatible and tokenization-aware bias metrics
  discovery.py     # EAP adapters and direct patching verification
  probes.py        # grouped probes, CAA directions, covariance-aware LEACE
  interventions.py # CAA, LEACE, hybrid hook construction
  gate.py          # sensitivity classifier and threshold selection
  evaluation.py    # task-appropriate capability metrics
  runs.py          # reproducible run folders and manifests
scripts/           # experiment entry points
configs/           # replication and cross-model JSON configurations
runs/<run-id>/     # config, manifest, per-example metrics, figures
```

## Reproducibility and acceptance

- Every run records the code revision, model/tokenizer revision, seed, environment, dataset hashes, and resolved configuration.
- Report paired bootstrap confidence intervals for mean bias change and repeat selected configurations across the configured seeds.
- Preserve the old thesis figures as reported claims and compare them against a fresh run. If the fresh run differs, document the exact cause or report the result as unreproduced.
- EAP rankings receive direct-patching checks; probes use group-disjoint splits; intervention settings are selected on validation data and reported on held-out data.
- Report CoLA perplexity only as language-modeling loss. If claiming CoLA task performance, report its classification metrics including Matthews correlation. If using CoNLL, report exact entity-span F1.
- Compare gate-off, always-on, static, and gated outcomes; report task-level capability metrics where a task-specific evaluator is available.
- GPT-2 small is the replication anchor; GPT-2 Large and Pythia-1B are the default extension models. Learned vectors and layer indices are model-specific.

## Current execution status

The scripts accept model checkpoints by name and refit model-specific artifacts, so the workflow is structured for cross-model experiments. The results reviewed so far are from GPT-2 small only; GPT-2 Large and Pythia-1B have not yet been run. A configured model list is a plan, not evidence of completed cross-model results.

The split preflight (`scripts/validate_splits.py`) blocks pair-ID, group-ID, and exact normalized prompt overlap between train, validation, and test. Bias-evaluation scripts validate the matched continuation arrays before loading model weights. These checks reduce common experiment errors; they do not establish that labels or prompt templates are scientifically valid.

## Initial replication reference

The thesis reports a 10.07% bias reduction, 10.15% CoLA perplexity increase, and improvement on 193/200 prompts with CAA strength 3.0, LEACE blend 0.3, and layers 6, 8, 9, 10, and 11. The saved prior-run JSON instead records a baseline bias of 10.8151 and baseline perplexity of 160.52, so these values must be reconciled before the headline result is reused.
