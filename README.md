# Topology Repository

This repo contains three completed experiment phases. The front-door claim is:

> On a held-out procedural micro-world task, `Unknown` / non-entailment is recoverable from verdict-region hidden states, while decoder outputs under-express it.

The old “global topology of truth” headline did not survive.

## Start Here

- Paper source: [paper/main.tex](paper/main.tex)
- Paper PDF: [paper/paper.pdf](paper/paper.pdf)
- Claim-to-file mapping: [CLAIMS_AND_EVIDENCE.md](CLAIMS_AND_EVIDENCE.md)
- Repository file map: [REPO_MAP.md](REPO_MAP.md)
- Experiment index: [docs/experiments/README.md](docs/experiments/README.md)

## Three Phases

1. Phase A (negative): global H0/H1 topology on GSM8K traces  
   [docs/experiments/01_global_h0_topology/README.md](docs/experiments/01_global_h0_topology/README.md)

2. Phase B (positive): fixed-decoding convergence failure on GSM8K  
   [docs/experiments/02_convergence_failure/README.md](docs/experiments/02_convergence_failure/README.md)

3. Phase C (main positive): procedural micro-world representation–decoder dissociation  
   [docs/experiments/03_micro_world_semantics/README.md](docs/experiments/03_micro_world_semantics/README.md)

## Main Numbers

- Phase A: corrected topology-only AUC = `0.20` (controls = `0.60`)
- Phase B: cap→wrong AUC = `0.839`; OR ≈ `26.40`
- Phase C: decoder Unknown recall (`Qwen2B=0.000`, `Qwen4B=0.000`, `Gemma-it=0.0125`) while verdict-token probe Unknown recall (`0.7375`, `0.1292`, `0.5625`)

## Repo Notes

- Legacy root Python scripts were moved to `phase_a_global_h0_topology/scripts/`.
- All committed artifacts are kept under `artifacts/` and `outputs/`.
- `.npz` artifacts are committed (not ignored).

## Reproducibility

- Locked deps: [repro/requirements.lock.txt](repro/requirements.lock.txt)
- Rebuild guide: [repro/README.md](repro/README.md)
- CI build/release: [.github/workflows/paper.yml](.github/workflows/paper.yml)
