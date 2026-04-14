# Paper Draft

This directory contains a submission-style draft of the project report and the figure pipeline used by the manuscript.

## Files

- `main.tex` — manuscript source.
- `paper.pdf` — compiled manuscript.
- `figures/` — generated figures consumed by `main.tex`.
- `scripts/make_figures.py` — deterministic figure generation from committed CSV artifacts.

## Rebuild

From repo root:

```bash
python3 paper/scripts/make_figures.py
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The GitHub workflow `.github/workflows/paper.yml` runs the same steps and uploads a release bundle.
