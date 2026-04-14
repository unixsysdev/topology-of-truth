from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _common import reopen_json, write_json
from topology_paper.src.data.answer_extract import extract_pred_answer
from topology_paper.src.eval.correctness import score_generation


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify one saved sample and hidden-state artifact.")
    parser.add_argument("--artifact-dir", default="artifacts/one_question")
    parser.add_argument("--sample-name", default="sample_0001")
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    sample_path = artifact_dir / f"{args.sample_name}.json"
    hidden_path = artifact_dir / f"{args.sample_name}_hidden.npy"
    sample = reopen_json(sample_path)
    hidden = np.load(hidden_path)

    checks = {
        "sample_json_exists": sample_path.exists(),
        "hidden_exists": hidden_path.exists(),
        "token_count_matches_hidden": int(sample["token_count"]) == int(hidden.shape[0]),
        "hidden_dim_matches_metadata": list(hidden.shape) == list(sample["hidden_shape"]),
        "pred_answer_recomputed": extract_pred_answer(sample["generated_text"]),
    }
    recomputed = score_generation(sample["generated_text"], sample["gold_answer"])
    checks["correctness_matches"] = bool(recomputed["correct"]) == bool(sample["correct"])
    checks["pred_answer_matches"] = recomputed["pred_answer"] == sample["pred_answer"]

    sample["verification"] = checks
    write_json(sample_path, sample)

    if not all(v if isinstance(v, bool) else True for v in checks.values()):
        raise RuntimeError(f"Verification failed: {checks}")


if __name__ == "__main__":
    main()
