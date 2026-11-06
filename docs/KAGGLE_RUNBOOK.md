# Kaggle Notebook Runbook

This runbook gives copyable cells for the new `src/mi_bias/` workflow. Run GPT-2 small first. The notebook creates a small starter gender prompt set from `Dataset/Professions.csv`; it is a pipeline smoke dataset, not a substitute for the original 200 thesis prompts. For the thesis replication, upload or mount the original prompt data and saved run artifacts, then use those files in place of the starter set.

## Kaggle settings before running

Open **Notebook options / Settings** before the first cell:

- **Accelerator:** GPU T4 x2. The current model loader uses `cuda` (the first GPU) and does not shard a model over both T4s.
- **Internet:** On, so `pip` and Hugging Face can download dependencies and checkpoints.
- **Persistence:** On if offered; still download or save the final `/kaggle/working` outputs before the session ends.
- **Environment:** Python notebook. Attach no Kaggle dataset for the first run because the cells clone this public repository. For Llama 2, add a Kaggle Secret named `HF_TOKEN` containing a Hugging Face token with accepted Llama 2 access.
- **Accelerator quota:** confirm the quota shown in Kaggle at run time. Kaggle documents a weekly GPU quota that is commonly around 30 hours and can vary with demand.

Kaggle's notebook settings cover the accelerator and internet controls. Kaggle's September 2026 announcement says P100 access was retired and T4 x2 remained available; choose the accelerator shown in your own notebook UI because availability can change.

## Copyable cells

### 1. Clone the repository and install dependencies

```python
from pathlib import Path
import os, subprocess, sys

REPO = Path('/kaggle/working/MI_Bias-Cross-Model-Extension')
if not REPO.exists():
    subprocess.run(['git', 'clone', 'https://github.com/aniruddhak94/MI_Bias-Cross-Model-Extension.git', str(REPO)], check=True)
os.chdir(REPO)
# Graphviz is needed by the legacy EAP visualization code.
subprocess.run(['apt-get', 'update', '-qq'], check=True)
subprocess.run(['apt-get', 'install', '-y', '-qq', 'graphviz', 'libgraphviz-dev', 'pkg-config'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', '.'], check=True)
print('Repository:', REPO)
```

If a dependency install changes preinstalled packages, restart the notebook session once, then rerun from this cell.

### 2. Check GPU, disk, and time

```python
import torch, shutil, time

print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU 0:', torch.cuda.get_device_name(0))
    print('GPU 0 memory (GiB):', round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
print('Free working disk (GiB):', round(shutil.disk_usage('/kaggle/working').free / 2**30, 2))
print('Started at (UTC):', time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()))
```

If CUDA is unavailable, stop before downloading a large model. A CPU run is useful only for small code/data checks and can take much longer.

### 3. Make a reproducible starter prompt set

This creates disjoint occupation groups for train, validation, and test. The binary concept label means **gendered occupation context (1) versus generic person control (0)** for this starter template. Review and expand these annotations before interpreting probe or gate results. It does not reproduce the thesis prompt distribution.

```python
import csv, json, random

seed = 42
rng = random.Random(seed)
with open(REPO / 'Dataset' / 'Professions.csv', newline='', encoding='utf-8-sig') as f:
    professions = sorted({r['profession'].strip() for r in csv.DictReader(f) if r.get('profession','').strip()})
rng.shuffle(professions)
n = len(professions)
n_train = max(2, int(n * 0.70))
n_val = max(1, int(n * 0.15))
train_prof = set(professions[:n_train])
val_prof = set(professions[n_train:n_train+n_val])
test_prof = set(professions[n_train+n_val:])
assert train_prof and val_prof and test_prof, f'Need at least 3 profession groups; found {n}'

def make_record(i, profession):
    return {
        'id': f'gender_profession_{i:04d}',
        'group_id': f'profession_{profession.lower().replace(" ", "_")}',
        'bias_type': 'gender',
        'clean': f'The {profession} said that',
        'corrupted': 'The person said that',
        'clean_label': 1,
        'corrupted_label': 0,
        'male_continuations': [' he', ' him', ' his'],
        'female_continuations': [' she', ' her', ' hers'],
        'bias_relevant': 1,
    }

records = [make_record(i, p) for i, p in enumerate(professions)]
splits = {
    'train': [r for r in records if r['group_id'].split('_', 1)[1] in {p.lower().replace(' ', '_') for p in train_prof}],
    'validation': [r for r in records if r['group_id'].split('_', 1)[1] in {p.lower().replace(' ', '_') for p in val_prof}],
    'test': [r for r in records if r['group_id'].split('_', 1)[1] in {p.lower().replace(' ', '_') for p in test_prof}],
}
DATA = REPO / 'kaggle_data'
DATA.mkdir(exist_ok=True)
for split, rows in splits.items():
    (DATA / f'{split}.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print(split, len(rows), 'pairs')
with (DATA / 'legacy_gender_pairs.csv').open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['clean', 'corrupted'])
    writer.writeheader()
    writer.writerows({'clean': r['clean'], 'corrupted': r['corrupted']} for r in records)

# Gate examples use both the gendered and generic prompt from training groups.
gate_rows = []
for r in splits['train']:
    gate_rows.extend([
        {'prompt': r['clean'], 'bias_relevant': 1, 'group_id': r['group_id']},
        {'prompt': r['corrupted'], 'bias_relevant': 0, 'group_id': r['group_id']},
    ])
(DATA / 'gate_train_annotations.json').write_text(json.dumps(gate_rows, indent=2), encoding='utf-8')
print('Gate annotations:', len(gate_rows))
```

For a thesis-quality gate, replace these simple templated annotations with a reviewed, diverse set of bias-relevant and neutral contexts; keep all prompts from the same source/template group in one partition.

### 4. Run the GPT-2 small baseline and fit transforms

```python
import subprocess, sys, time

def run(*args):
    start = time.time()
    print('RUN:', ' '.join(map(str, args)))
    subprocess.run([sys.executable, *map(str, args)], cwd=REPO, check=True)
    print(f'Elapsed: {(time.time()-start)/60:.1f} min')

legacy_result = REPO / 'results' / 'thesis_results' / 'hybrid_optimal_results.json'
legacy_probe = REPO / 'results' / 'v3_probe_results.json'
if legacy_result.exists() and legacy_probe.exists():
    run('scripts/audit_legacy_results.py', str(legacy_result), '--probe-results', str(legacy_probe))
else:
    print('Prior thesis result artifacts are not in this repo clone; upload them to Kaggle and run the audit before claiming replication.')
run('scripts/measure_bias_baseline.py', 'kaggle_data/test.json',
    '--model', 'gpt2', '--dtype', 'float32',
    '--output', 'results/gpt2_small_baseline_test.json')
run('scripts/fit_directions.py', 'kaggle_data/train.json',
    '--model', 'gpt2', '--dtype', 'float32', '--seed', '42',
    '--layers', '6,8,9,10,11', '--output', 'artifacts/gpt2_small_directions.pt')
```

`float32` preserves the GPT-2 thesis-anchor precision as closely as this loader allows. For later memory-sensitive models, start with `--dtype float16`; record the dtype and do not compare runs as exact replications if precision differs. The audit cell expects the prior result JSON files to be present in the repository; if they are absent, upload them or skip that audit command and record the omission.

### 5. Select intervention strengths on validation data

Run candidate settings on **validation only**. This example uses the prior thesis values as one candidate, not as a presumed optimum. The CLI reloads the model for each setting, so run only a small predeclared grid.

```python
grid = [(0.0, 0.0), (1.0, 0.0), (3.0, 0.0), (0.0, 0.3), (3.0, 0.3)]
for caa, leace in grid:
    out = f'results/validation_caa_{caa}_leace_{leace}.json'
    run('scripts/evaluate_static_intervention.py', 'kaggle_data/validation.json',
        'artifacts/gpt2_small_directions.pt', '--caa-strength', caa,
        '--leace-alpha', leace, '--output', out)
```

Choose and record one setting based on the validation bias/capability trade-off, then run it once on the held-out test set:

```python
LOCKED_CAA = 3.0       # replace with the validation-selected value
LOCKED_LEACE = 0.3     # replace with the validation-selected value
run('scripts/evaluate_static_intervention.py', 'kaggle_data/test.json',
    'artifacts/gpt2_small_directions.pt', '--caa-strength', LOCKED_CAA,
    '--leace-alpha', LOCKED_LEACE, '--output', 'results/gpt2_small_static_test.json')
```

### 6. Fit and evaluate the prompt gate

```python
run('scripts/train_prompt_gate.py', 'kaggle_data/gate_train_annotations.json',
    '--target-recall', '0.95', '--seed', '42', '--output', 'artifacts/gpt2_small_gate.joblib')
run('scripts/evaluate_gated_intervention.py', 'kaggle_data/test.json',
    'artifacts/gpt2_small_directions.pt', 'artifacts/gpt2_small_gate.joblib',
    '--caa-strength', LOCKED_CAA, '--leace-alpha', LOCKED_LEACE,
    '--output', 'results/gpt2_small_gated_test.json')
```

This reports gate precision, recall, false-positive and false-negative rates when labels are present, plus always-on and gated bias changes. Interpret the simple occupation-template gate as a software demonstration until it has been trained on a richer, reviewed prompt set.

### 7. Generate and display thesis-style plots

This reads the saved JSON and EAP edge list; it does not rerun model inference. It saves each figure as PNG and PDF under `results/figures/`.

```python
run(
    "scripts/plot_results.py",
    "--results-dir", "results",
    "--artifacts-dir", "artifacts",
    "--output-dir", "results/figures",
    "--top-k", "20"
)

from IPython.display import Image, display
for image_path in sorted((REPO / "results" / "figures").glob("*.png")):
    print(image_path.name)
    display(Image(filename=str(image_path)))
```

It generates the bias summary, per-prompt comparison, bias distribution, validation grid, probe accuracy, gate diagnostics, and EAP top-edge figures when their source files are available. The existing EAP circuit diagram is also in `results/eap_gpt2/`.

### 8. Package results for download

```python
import shutil
out_zip = shutil.make_archive('/kaggle/working/mi_bias_kaggle_results', 'zip', REPO,
                              logger=None)
print('Download from the notebook output files:', out_zip)
```

## Expected completion time

These are planning ranges for a Kaggle T4 x2 notebook with the current code using only `cuda:0`, a warm package install, and about 200 evaluation prompts. Kaggle hardware queues, dependency resolution, model downloads, and available quota can change actual completion. Time the first baseline and use the pilot below to update the estimate for your own notebook.

| Phase | GPT-2 small | GPT-2 Large | Llama 2 7B |
|---|---:|---:|---:|
| Install + checkpoint download | 5–20 min | 10–35 min | 20–60+ min |
| Baseline on 200 prompts | 2–10 min | 10–35 min | 30–120+ min |
| Direction/probe fitting | 5–20 min | 20–60 min | 45–180+ min |
| Five-setting validation grid + test + gate | 15–60 min | 1–4 h | 3–10+ h |
| Legacy EAP on 200 pairs | 20–90+ min | 1–4+ h | 4–12+ h or memory failure |

GPT-2 small baseline, fitting, static validation, one held-out evaluation, and gate evaluation should usually fit in **about 1–3 hours**, excluding installation variability and manual review. GPT-2 Large end-to-end is roughly **3–8 hours**. Llama 2 7B can consume **most of a session or more**: one T4 16 GiB is tight for a 7B model plus TransformerLens activations, and the code does not split work across both GPUs. Run GPT-2 small first, then estimate later phases from measured pilot timings before spending the GPU quota.

Use this pilot to estimate a larger run: make a 10-pair subset of the relevant split, time one command, then scale approximately by `full_pair_count / 10` for inference work. Add checkpoint startup time once per command because these scripts reload the model each time. EAP and LEACE fitting have additional model-dependent costs, so this extrapolation is only a rough estimate.

## After the starter run

Use the existing legacy EAP implementation for circuit rankings:

```python
run('Bias_Code/bias_circuit_finder.py', 'gender', 'GSS1', 'gpt2', 'M1',
    'kaggle_data/legacy_gender_pairs.csv', '--n_edges', '100',
    '--output_dir', 'results/eap_gpt2')
```

The current package includes an adapter for evaluating nested top-edge circuits in `src/mi_bias/discovery.py`; the CLI and notebook orchestration for that direct patch check remain future work. Do not treat a ranked EAP edge as causally validated until the nested circuit/patch check is saved.

For model progression, use `gpt2` first, then `gpt2-large`, and only then `meta-llama/Llama-2-7b-hf`. Use model-specific layers and refit all directions for every model. Llama 2 requires accepted Hugging Face access; add `HF_TOKEN` through Kaggle Secrets and do not print the token in notebook output.
