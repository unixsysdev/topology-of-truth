# 02 Convergence Failure (Validated)

## Objective

Measure fixed-decoding within-question behavior and test whether token-cap termination is associated with error.

This track was the follow-up after global topology failed to provide a stable correctness signal.

## Inputs

- GSM8K question slice via `topology_paper/configs/gsm8k_qwen35.yaml`.
- Qwen3.5-2B with fixed decoding.

## Artifact generation pipeline

### Step 1: fixed-config within-question run

```bash
python3 scripts/run_multi_sample_questions.py \
  --model-id Qwen/Qwen3.5-2B \
  --question-start 0 \
  --question-count 20 \
  --samples-per-question 5 \
  --temperature 0.7 \
  --top-p 0.95 \
  --max-new-tokens 384 \
  --artifact-dir artifacts/qwen35_2b_within_question
```

Primary outputs:

- `artifacts/qwen35_2b_within_question/run_manifest.csv`
- per-question sample folders containing:
  - `sample.json`
  - `hidden.npy`
  - `logits_summary.npz`

### Step 2: within-question basin/reference analysis

```bash
python3 scripts/analyze_within_question_basins.py \
  --artifact-dir artifacts/qwen35_2b_within_question \
  --out-dir artifacts/qwen35_2b_within_question/analysis
```

Outputs:

- `aggregate_pairwise.csv`
- `aggregate_reference.csv`
- `question_summary.csv`
- sign-test files under the same folder

### Step 3: capped-sample sensitivity rerun

```bash
python3 scripts/run_cap_sensitivity.py \
  --model-id Qwen/Qwen3.5-2B \
  --source-artifact-dir artifacts/qwen35_2b_within_question \
  --out-dir artifacts/qwen35_2b_cap_sensitivity_640 \
  --temperature 0.7 \
  --top-p 0.95 \
  --max-new-tokens 640
```

Primary output:

- `artifacts/qwen35_2b_cap_sensitivity_640/run_manifest.csv`

### Step 4: sensitivity summary

```bash
python3 scripts/analyze_cap_sensitivity.py \
  --sensitivity-dir artifacts/qwen35_2b_cap_sensitivity_640 \
  --out-dir artifacts/qwen35_2b_cap_sensitivity_640/analysis
```

Outputs:

- `transition_summary.csv`
- `per_question_transition_summary.csv`

## Reported results (source-of-truth files)

- `artifacts/qwen35_2b_within_question/run_manifest.csv`
- `artifacts/qwen35_2b_cap_sensitivity_640/analysis/transition_summary.csv`

Key counts from those files:

- \(n=180\) fixed-config samples, with strong cap/error association.
- matched capped rerun \(n=87\): partial rescue but large residual failure.

## Conclusion

Non-convergence is the dominant signal in this branch.
Question-conditioned basin effects were only weak/underpowered here and did not scale cleanly under augmentation.
