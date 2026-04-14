# Reproducibility Notes

This folder records the versioned environment and deterministic rebuild commands for the consolidated paper branch.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r repro/requirements.lock.txt
```

## Regenerate paper figures

```bash
python3 paper/scripts/make_figures.py
```

## Compile manuscript

```bash
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

## Data provenance

All metrics and plots in the manuscript are generated from committed CSV/manifest outputs in:

- `artifacts/qwen35_2b_within_question/`
- `artifacts/qwen35_2b_cap_sensitivity_640/analysis/`
- `artifacts/micro_world_v1/`
