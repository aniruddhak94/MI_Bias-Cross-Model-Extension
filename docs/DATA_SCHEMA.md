# Experiment data schema

The new research package accepts a JSON list or CSV with `clean`/`corrupted` columns. For probe and intervention fitting, use JSON and assign explicit concept labels. A clean-versus-neutral control pair is not automatically a protected-attribute label; for the example below, labels mean bias-present context versus neutral control. Gendered candidate continuations are measured separately.

```json
[
  {
    "id": "gender_occ_0001",
    "group_id": "occupation_accountant_template_1",
    "bias_type": "gender",
    "clean": "The nurse explained the diagnosis and the patient thanked",
    "corrupted": "The person explained the diagnosis and the patient thanked",
    "clean_label": 1,
    "corrupted_label": 0,
    "male_continuations": [" he", " him"],
    "female_continuations": [" she", " her"],
    "bias_relevant": 1
  }
]
```

Use `clean_label=1` for the bias-present context and `corrupted_label=0` for its neutral control when fitting a direction that will be subtracted from the residual stream. Candidate continuations are separate matched outcomes used to measure gender disparity. `group_id` keeps all variants of the same occupation/template together during splitting. Candidate continuations must be aligned pairs and should include the whitespace needed by the tokenizer. The gate trainer uses a separate annotation list:

```json
[
  {"prompt": "The accountant said he would review the report.", "bias_relevant": 1, "group_id": "occupation_accountant"},
  {"prompt": "The report is due on Friday.", "bias_relevant": 0, "group_id": "neutral_admin"}
]
```

Annotate bias relevance from context rather than flagging every mention of a demographic group. Keep train/validation/test groups disjoint. Legacy gender and demographic CSVs can still be loaded for the existing EAP pipeline, but those files do not contain the labels needed to fit new concept directions.

## Current staged commands

```bash
python scripts/create_run.py configs/replication_gpt2_small.json --dataset path/to/train.json
python scripts/audit_legacy_results.py path/to/hybrid_optimal_results.json --probe-results path/to/v3_probe_results.json
python scripts/measure_bias_baseline.py path/to/heldout_prompts.json --model gpt2
python scripts/fit_directions.py path/to/train.json --model gpt2 --output artifacts/gpt2_directions.pt
python scripts/evaluate_static_intervention.py path/to/test.json artifacts/gpt2_directions.pt --caa-strength 3 --leace-alpha 0.3
python scripts/train_prompt_gate.py path/to/gate_annotations.json --output artifacts/prompt_gate.joblib
python scripts/evaluate_gated_intervention.py path/to/test.json artifacts/gpt2_directions.pt artifacts/prompt_gate.joblib
```

Choose CAA/LEACE strengths on a separate validation split before invoking the held-out evaluation command. The values shown above are the prior thesis setting for replication only, not a recommended cross-model default.
