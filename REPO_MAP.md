# Repository Map

This file maps the paper claims to concrete directories and entry points.

## Top-level Layout

- `README.md` — front-door summary and links.
- `CLAIMS_AND_EVIDENCE.md` — claim-by-claim evidence table.
- `docs/experiments/` — per-phase reproducibility docs.
- `paper/` — manuscript (`main.tex`), PDF, figure pipeline.
- `artifacts/` — committed run outputs used in analysis.
- `outputs/` — additional generated outputs from legacy runs.
- `repro/` — environment lock and rebuild notes.
- `scripts/` — active analysis/generation scripts for Phases A/B/C.
- `phase_a_global_h0_topology/scripts/` — legacy original root scripts (moved for clarity).
- `phase_a_global_h0_topology/figures_archive/` — preserved visuals from the original Phase A direction.

## Experiment Phases

### Phase A: Global Topology (Negative)

- Doc: `docs/experiments/01_global_h0_topology/README.md`
- Legacy scripts: `phase_a_global_h0_topology/scripts/`
- Main artifacts:
  - `artifacts/paired_dynamic_qwen35/`
  - `artifacts/qwen35_2b_window_corrected/`

### Phase B: Convergence Failure (Fixed Decoding)

- Doc: `docs/experiments/02_convergence_failure/README.md`
- Main scripts:
  - `scripts/run_multi_sample_questions.py`
  - `scripts/analyze_within_question_basins.py`
  - `scripts/run_cap_sensitivity.py`
  - `scripts/analyze_cap_sensitivity.py`
- Main artifacts:
  - `artifacts/qwen35_2b_within_question/`
  - `artifacts/qwen35_2b_cap_sensitivity_640/`

### Phase C: Micro-World Semantics (Main Result)

- Doc: `docs/experiments/03_micro_world_semantics/README.md`
- Main scripts:
  - `scripts/generate_micro_world_dataset.py`
  - `scripts/run_micro_world_inference.py`
  - `scripts/analyze_micro_world_geometry.py`
  - `scripts/run_micro_world_probe.py`
  - `scripts/analyze_micro_world_label_logits.py`
  - `scripts/run_micro_world_layer_sweep_probe.py`
- Main artifacts:
  - `artifacts/micro_world_v1/dataset/`
  - `artifacts/micro_world_v1/generations/`
  - `artifacts/micro_world_v1/comparison_*.csv`
  - `artifacts/micro_world_v1/layer_sweep_*`
  - `artifacts/micro_world_v1/label_logits_*`

## Paper Assets

- TeX source: `paper/main.tex`
- Compiled PDF: `paper/paper.pdf`
- Figure script: `paper/scripts/make_figures.py`
- Generated figures: `paper/figures/`
- CI workflow: `.github/workflows/paper.yml`
- Release bundler: `scripts/package_release_bundle.py`

## Legacy Script Location

Legacy first-direction scripts previously kept at repo root are now under:

- `phase_a_global_h0_topology/scripts/`

This includes:

- `audit_report.py`
- `trace_generator.py`
- `batch_verifier.py`
- `topological_engine.py`
- `analyze_results.py`
- `visualizer.py`
- `visualize_comparisons.py`
