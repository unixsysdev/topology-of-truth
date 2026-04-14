from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "paper" / "figures"


def _ensure_dir() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def _plot_convergence_cap() -> None:
    manifest = pd.read_csv(
        ROOT / "artifacts" / "qwen35_2b_within_question" / "run_manifest.csv"
    )
    manifest = manifest[manifest["status"] == "ok"].copy()
    manifest["outcome"] = manifest["correct"].map({True: "Correct", False: "Wrong"})
    manifest["termination"] = manifest["stop_reason"].map(
        {"max_new_tokens": "Capped", "eos_token": "EOS"}
    )

    summary = (
        manifest.groupby(["termination", "outcome"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )

    pivot = summary.pivot(index="termination", columns="outcome", values="count").fillna(0)
    pivot = pivot.reindex(["EOS", "Capped"])

    ax = pivot.plot(
        kind="bar",
        figsize=(6.2, 4.0),
        color=["#2a9d8f", "#e76f51"],
        rot=0,
    )
    ax.set_ylabel("Count")
    ax.set_xlabel("")
    ax.set_title("Qwen3.5-2B 384-token run: correctness by termination")
    ax.legend(title="")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig1_convergence_cap_outcomes.png", dpi=220)
    plt.close()


def _plot_unknown_decoder_vs_probe() -> None:
    dec = pd.read_csv(ROOT / "artifacts" / "micro_world_v1" / "comparison_decoder_qwen_gemma.csv")
    probe = pd.read_csv(
        ROOT / "artifacts" / "micro_world_v1" / "comparison_probe_states_qwen_gemma.csv"
    )
    probe = probe[probe["state_key"] == "verdict_token"].copy()

    model_order = [
        "Qwen3.5-2B",
        "Qwen3.5-4B (no-think)",
        "Gemma-3-4B-it (no-think)",
    ]

    merged = dec.merge(
        probe[["model", "probe_unknown_recall"]],
        on="model",
        how="inner",
    )
    merged["model"] = pd.Categorical(merged["model"], categories=model_order, ordered=True)
    merged = merged.sort_values("model")

    x = range(len(merged))
    width = 0.38

    plt.figure(figsize=(7.2, 4.0))
    plt.bar(
        [i - width / 2 for i in x],
        merged["decoder_unknown_recall"],
        width=width,
        label="Decoder Unknown recall",
        color="#264653",
    )
    plt.bar(
        [i + width / 2 for i in x],
        merged["probe_unknown_recall"],
        width=width,
        label="Probe Unknown recall (verdict token)",
        color="#e9c46a",
    )
    plt.xticks(list(x), merged["model"], rotation=12, ha="right")
    plt.ylim(0, 0.85)
    plt.ylabel("Recall")
    plt.title("Unknown: decoder vs hidden-state probe")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig2_unknown_decoder_vs_probe.png", dpi=220)
    plt.close()


def _plot_gemma_logit_competitiveness() -> None:
    logits = pd.read_csv(
        ROOT
        / "artifacts"
        / "micro_world_v1"
        / "comparison_label_logits_gemma_it_vs_pt_basefmt.csv"
    )
    logits["run_label"] = logits["run"].map(
        {
            "gemma_3_4b_it_raw": "Gemma-3-4B-it",
            "gemma_3_4b_pt_basefmt": "Gemma-3-4B-pt (basefmt)",
        }
    )
    logits = logits.sort_values("run_label")

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8))

    axes[0].bar(
        logits["run_label"],
        logits["unknown_fail_mean_prob_unknown"],
        color=["#2a9d8f", "#f4a261"],
    )
    axes[0].set_ylim(0.0, 0.25)
    axes[0].set_title("Mean P(Unknown)\non gold-Unknown decoder failures")
    axes[0].tick_params(axis="x", rotation=12)

    axes[1].bar(
        logits["run_label"],
        logits["unknown_fail_mean_unknown_minus_best_nonunknown_logp"],
        color=["#2a9d8f", "#f4a261"],
    )
    axes[1].set_title("Mean logP(Unknown) - best non-Unknown\n(gold-Unknown decoder failures)")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].tick_params(axis="x", rotation=12)

    fig.suptitle("Verdict-step Unknown under-ranking in Gemma", y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3_gemma_unknown_logit_competitiveness.png", dpi=220)
    plt.close()


def _plot_layer_sweep() -> None:
    layer = pd.read_csv(
        ROOT
        / "artifacts"
        / "micro_world_v1"
        / "comparison_layer_sweep_gemma_it_vs_pt_basefmt.csv"
    )
    layer["run_label"] = layer["run"].map(
        {
            "gemma_3_4b_it_raw": "Gemma-3-4B-it",
            "gemma_3_4b_pt_basefmt": "Gemma-3-4B-pt (basefmt)",
        }
    )
    layer = layer[layer["state_kind"].isin(["prompt_last", "verdict_token"])].copy()

    pivot = layer.pivot(index="run_label", columns="state_kind", values="best_recall_unknown")
    pivot = pivot[["prompt_last", "verdict_token"]]

    ax = pivot.plot(
        kind="bar",
        figsize=(6.8, 4.0),
        color=["#8ecae6", "#219ebc"],
        rot=10,
    )
    ax.set_ylabel("Best probe Unknown recall")
    ax.set_xlabel("")
    ax.set_ylim(0.65, 0.86)
    ax.set_title("Layer sweep: best Unknown recall by state family")
    ax.legend(title="")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig4_layer_sweep_unknown_recall.png", dpi=220)
    plt.close()


def _plot_geometry_sign_tests() -> None:
    qwen = pd.read_csv(
        ROOT / "artifacts" / "micro_world_v1" / "analysis_qwen35_2b_partial19" / "sign_test_summary.csv"
    )
    gemma = pd.read_csv(
        ROOT
        / "artifacts"
        / "micro_world_v1"
        / "analysis_gemma_3_4b_eval20_nothink"
        / "sign_test_summary.csv"
    )
    keep = ["final_prompt", "verdict_span_mean", "verdict_token"]
    qwen = qwen[qwen["state_key"].isin(keep)].copy()
    gemma = gemma[gemma["state_key"].isin(keep)].copy()

    qwen["model"] = "Qwen3.5-2B"
    gemma["model"] = "Gemma-3-4B-it"
    both = pd.concat([qwen, gemma], ignore_index=True)
    both["ratio"] = both["worlds_positive_gap"] / both["n_valid_worlds"]

    pivot = both.pivot(index="state_key", columns="model", values="ratio")
    pivot = pivot.loc[["final_prompt", "verdict_span_mean", "verdict_token"]]

    ax = pivot.plot(
        kind="bar",
        figsize=(7.0, 3.8),
        color=["#577590", "#43aa8b"],
        rot=0,
    )
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Fraction of worlds with positive gap")
    ax.set_xlabel("")
    ax.set_title("Within-world geometry sign tests: different-label > same-label")
    ax.legend(title="")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig5_geometry_sign_tests.png", dpi=220)
    plt.close()


def main() -> None:
    _ensure_dir()
    _plot_convergence_cap()
    _plot_unknown_decoder_vs_probe()
    _plot_gemma_logit_competitiveness()
    _plot_layer_sweep()
    _plot_geometry_sign_tests()
    print(f"Wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
