from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from _common import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize capped-sample longer-budget sensitivity results.")
    parser.add_argument("--sensitivity-dir", default="artifacts/qwen35_2b_cap_sensitivity_640")
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    sensitivity_dir = Path(args.sensitivity_dir)
    out_dir = ensure_dir(args.out_dir or sensitivity_dir / "analysis")
    df = pd.read_csv(sensitivity_dir / "run_manifest.csv")
    df["source_wrong"] = ~df["source_correct"].astype(bool)
    df["rerun_wrong"] = ~df["correct"].astype(bool)
    df["rerun_capped"] = df["stop_reason"] == "max_new_tokens"
    df["rerun_eos"] = df["stop_reason"] == "eos_token"

    transition_rows = [
        count_row(df, "all"),
        count_row(df[df["source_wrong"]], "source_wrong"),
        count_row(df[df["source_correct"].astype(bool)], "source_correct"),
    ]
    pd.DataFrame(transition_rows).to_csv(out_dir / "transition_summary.csv", index=False)

    per_question = []
    for question_id, group in df.groupby("question_id", sort=True):
        per_question.append(count_row(group, question_id))
    pd.DataFrame(per_question).to_csv(out_dir / "per_question_transition_summary.csv", index=False)

    stop_by_correct = pd.crosstab(df["stop_reason"], df["correct"])
    stop_by_correct.to_csv(out_dir / "rerun_stop_by_correct.csv")

    source_by_rerun = pd.crosstab(df["source_correct"], df["correct"])
    source_by_rerun.to_csv(out_dir / "source_correct_by_rerun_correct.csv")


def count_row(df: pd.DataFrame, label: str) -> dict[str, Any]:
    source_wrong = df["source_wrong"]
    source_correct = df["source_correct"].astype(bool)
    rerun_correct = df["correct"].astype(bool)
    rerun_wrong = df["rerun_wrong"]
    rerun_capped = df["rerun_capped"]
    rerun_eos = df["rerun_eos"]
    return {
        "label": label,
        "n": len(df),
        "source_correct": int(source_correct.sum()),
        "source_wrong": int(source_wrong.sum()),
        "rerun_correct": int(rerun_correct.sum()),
        "rerun_wrong": int(rerun_wrong.sum()),
        "rerun_capped": int(rerun_capped.sum()),
        "rerun_eos": int(rerun_eos.sum()),
        "wrong_to_correct": int((source_wrong & rerun_correct).sum()),
        "wrong_to_wrong": int((source_wrong & rerun_wrong).sum()),
        "wrong_to_eos_correct": int((source_wrong & rerun_eos & rerun_correct).sum()),
        "wrong_to_eos_wrong": int((source_wrong & rerun_eos & rerun_wrong).sum()),
        "wrong_to_still_capped_correct": int((source_wrong & rerun_capped & rerun_correct).sum()),
        "wrong_to_still_capped_wrong": int((source_wrong & rerun_capped & rerun_wrong).sum()),
        "correct_to_correct": int((source_correct & rerun_correct).sum()),
        "correct_to_wrong": int((source_correct & rerun_wrong).sum()),
        "correct_to_eos_correct": int((source_correct & rerun_eos & rerun_correct).sum()),
        "correct_to_still_capped_correct": int((source_correct & rerun_capped & rerun_correct).sum()),
        "correct_to_still_capped_wrong": int((source_correct & rerun_capped & rerun_wrong).sum()),
    }


if __name__ == "__main__":
    main()
