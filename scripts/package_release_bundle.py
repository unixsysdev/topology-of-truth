from __future__ import annotations

import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "release"
OUT_TAR = OUT_DIR / "paper_release_bundle.tar.gz"


FILES = [
    # Paper assets
    "paper/main.tex",
    "paper/paper.pdf",
    "paper/README.md",
    "paper/figures/fig1_convergence_cap_outcomes.png",
    "paper/figures/fig2_unknown_decoder_vs_probe.png",
    "paper/figures/fig3_gemma_unknown_logit_competitiveness.png",
    "paper/figures/fig4_layer_sweep_unknown_recall.png",
    "paper/figures/fig5_geometry_sign_tests.png",
    # Core summaries
    "artifacts/paired_dynamic_qwen35/static_vs_dynamic_separability_2b.csv",
    "artifacts/qwen35_2b_window_corrected/classifier_auc.csv",
    "artifacts/qwen35_2b_within_question/run_manifest.csv",
    "artifacts/qwen35_2b_cap_sensitivity_640/analysis/transition_summary.csv",
    "artifacts/micro_world_v1/comparison_decoder_qwen_gemma.csv",
    "artifacts/micro_world_v1/comparison_probe_states_qwen_gemma.csv",
    "artifacts/micro_world_v1/comparison_decoder_constrained_vs_unconstrained_qwen_gemma.csv",
    "artifacts/micro_world_v1/comparison_label_logits_gemma_it_vs_pt_basefmt.csv",
    "artifacts/micro_world_v1/comparison_layer_sweep_gemma_it_vs_pt_basefmt.csv",
    "artifacts/micro_world_v1/analysis_qwen35_2b_partial19/sign_test_summary.csv",
    "artifacts/micro_world_v1/analysis_gemma_3_4b_eval20_nothink/sign_test_summary.csv",
    # Repro docs
    "README.md",
    "CONVERGENCE_FAILURE_README.md",
    "MICRO_WORLD_EXPERIMENT_SPEC.md",
    "repro/README.md",
    "repro/requirements.lock.txt",
]


SAMPLE_FILES = [
    "artifacts/qwen35_2b_within_question/q_0001/sample_00/sample.json",
    "artifacts/qwen35_2b_cap_sensitivity_640/q_0003/sample_00/sample.json",
    "artifacts/micro_world_v1/generations/Qwen__Qwen3_5_2B_eval_full/examples/eval_world_000000_prop_0000_para_00/sample.json",
    "artifacts/micro_world_v1/generations/Gemma__gemma_3_4b_it_eval20_nothink/examples/eval_world_000000_prop_0000_para_00/sample.json",
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(OUT_TAR, "w:gz") as tf:
        for rel in FILES + SAMPLE_FILES:
            src = ROOT / rel
            if src.exists():
                tf.add(src, arcname=rel)
            else:
                print(f"[warn] missing: {rel}")
    print(f"wrote {OUT_TAR}")


if __name__ == "__main__":
    main()
