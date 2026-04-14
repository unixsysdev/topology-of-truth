# Topology of Truth (Consolidated Main)

This repository is now one merged branch with all experiments, artifacts, and paper outputs.

## Project Status (short)

- The original global-topology hypothesis did **not** verify.
- A fixed-decoding GSM8K branch found a strong **convergence-failure** effect.
- A procedural micro-world branch found a replicated **representation--decoder dissociation** for `Unknown`.

## The 3 Experiment Tracks

1. **Old global H0/H1 topology (negative)**  
   Details: [docs/experiments/01_global_h0_topology/README.md](/home/marcel/Work/topology/docs/experiments/01_global_h0_topology/README.md)

2. **Convergence failure on fixed GSM8K decoding (positive)**  
   Details: [docs/experiments/02_convergence_failure/README.md](/home/marcel/Work/topology/docs/experiments/02_convergence_failure/README.md)

3. **Procedural micro-world semantics (main positive)**  
   Details: [docs/experiments/03_micro_world_semantics/README.md](/home/marcel/Work/topology/docs/experiments/03_micro_world_semantics/README.md)

Index: [docs/experiments/README.md](/home/marcel/Work/topology/docs/experiments/README.md)

## Key Results

### Track 1 (negative)

- Dynamic/topology separability stayed weak (`AUC ~0.5-0.6`).
- Corrected fixed-window topology did not beat controls (`AUC 0.20` vs controls `0.60`).

### Track 2 (convergence)

From `artifacts/qwen35_2b_within_question/run_manifest.csv`:

- `n=180`, correct/wrong = `88/92`
- capped = `87` (`75` wrong, `12` correct)
- eos = `93` (`17` wrong, `76` correct)
- cap->wrong AUC = `0.839`

From `artifacts/qwen35_2b_cap_sensitivity_640/analysis/transition_summary.csv`:

- matched capped reruns = `87`
- wrong->correct at longer budget = `27`
- wrong->still wrong = `48`

### Track 3 (micro-world)

Decoder baseline (`artifacts/micro_world_v1/comparison_decoder_qwen_gemma.csv`):

- Qwen3.5-2B Unknown recall = `0.000`
- Qwen3.5-4B Unknown recall = `0.000`
- Gemma-3-4B-it Unknown recall = `0.0125`

Verdict-token probe (`artifacts/micro_world_v1/comparison_probe_states_qwen_gemma.csv`):

- Qwen3.5-2B Unknown recall = `0.7375`
- Qwen3.5-4B Unknown recall = `0.1292`
- Gemma-3-4B-it Unknown recall = `0.5625`

## Paper and CI

- Paper source: [paper/main.tex](/home/marcel/Work/topology/paper/main.tex)
- Compiled PDF: [paper/paper.pdf](/home/marcel/Work/topology/paper/paper.pdf)
- Figure generator: [paper/scripts/make_figures.py](/home/marcel/Work/topology/paper/scripts/make_figures.py)
- CI workflow: [.github/workflows/paper.yml](/home/marcel/Work/topology/.github/workflows/paper.yml)
- Release bundle script: [scripts/package_release_bundle.py](/home/marcel/Work/topology/scripts/package_release_bundle.py)

## Reproducibility

- Version lock: [repro/requirements.lock.txt](/home/marcel/Work/topology/repro/requirements.lock.txt)
- Rebuild instructions: [repro/README.md](/home/marcel/Work/topology/repro/README.md)

## Artifact policy

All committed artifacts are retained in-repo under `artifacts/` and `outputs/`.
Release packaging additionally provides `release/paper_release_bundle.tar.gz` for a compact reproducible subset.
