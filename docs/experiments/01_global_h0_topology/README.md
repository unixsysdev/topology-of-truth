# 01 Global H0/H1 Topology (Negative)

## Objective

Test whether global topological summaries of hidden-state trajectories predict correctness.

Initial intuition: correct reasoning might have a cleaner global topological signature.
This track is the direct test of that idea.

## Inputs

- GSM8K examples from configured dataset loader.
- Model generations and hidden states from Qwen3.5 0.8B/2B pilots.
- Legacy first-pass pipeline scripts now live in:
  - `phase_a_global_h0_topology/scripts/`

## Artifact generation pipeline

### Step 1: generate baseline smoke artifacts

```bash
python3 scripts/run_five_question.py \
  --model-id Qwen/Qwen3.5-0.8B \
  --artifact-dir artifacts/five_question
```

For single-question integrity checks:

```bash
python3 scripts/run_one_question.py \
  --model-id Qwen/Qwen3.5-0.8B \
  --artifact-dir artifacts/one_question
```

Historical paired pilot artifacts used in this track are committed directly:

- `artifacts/paired_qwen35_08b_10/`
- `artifacts/paired_qwen35_2b_10/`

### Step 2: dynamic H0 analysis

```bash
python3 scripts/analyze_paired_dynamic.py \
  --artifact-08b artifacts/paired_qwen35_08b_10 \
  --artifact-2b artifacts/paired_qwen35_2b_10 \
  --out-dir artifacts/paired_dynamic_qwen35
```

Outputs:

- `artifacts/paired_dynamic_qwen35/dynamic_features.parquet`
- `artifacts/paired_dynamic_qwen35/static_vs_dynamic_separability_2b.csv`
- trajectory figures in the same folder

### Step 3: windowed correction check

```bash
python3 scripts/analyze_2b_window_corrected.py \
  --artifact-dir artifacts/paired_qwen35_2b_10 \
  --out-dir artifacts/qwen35_2b_window_corrected
```

Outputs:

- `artifacts/qwen35_2b_window_corrected/classifier_auc.csv`
- variant diagnostics and feature tables in the same folder

## Reported results (source-of-truth files)

- `artifacts/paired_dynamic_qwen35/static_vs_dynamic_separability_2b.csv`
- `artifacts/qwen35_2b_window_corrected/classifier_auc.csv`

Observed failure pattern:

- small models were failure-heavy,
- many wrong traces were dominated by long/non-convergent generation,
- dynamic and fixed-window topology features stayed weak or degenerate,
- added trajectory/uncertainty style summaries did not recover a robust global correctness scalar.

## Conclusion

This effort did not verify a reliable global ``truth meter'' from these summaries.
It is retained as a negative baseline that motivated later pivots.
