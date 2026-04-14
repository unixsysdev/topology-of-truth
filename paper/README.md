# Paper Draft

This directory contains the submission draft aligned with the current repository claim:
the micro-world representation--decoder dissociation for `Unknown` is the main positive result; the GSM8K global-topology branch is a documented negative result.

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

See also:

- `../CLAIMS_AND_EVIDENCE.md`
- `../REPO_MAP.md`
- `../PAPER_STATUS.md`
