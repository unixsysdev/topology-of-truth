# Paper Status

Current manuscript path:

- Source: `paper/main.tex`
- PDF: `paper/paper.pdf`

Current framing:

1. Phase A global topology branch is reported as a negative replication.
2. Phase B fixed-decoding branch is reported as a convergence-failure result.
3. Phase C micro-world branch is the main positive result:
   representation--decoder dissociation for `Unknown`.

What is included in the current draft:

- Probe equation and protocol.
- World-level geometry gap definition.
- Verdict-step label-logit metrics.
- Layer-sweep formulation.
- Post-hoc readout intervention pilot.
- Pre-readout latent residual steering pilot.
- Nonlinear probe sensitivity (shallow MLP).
- Worked micro-world example table.
- Exact artifact paths and reproduction commands.

Build:

```bash
python3 paper/scripts/make_figures.py
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```
