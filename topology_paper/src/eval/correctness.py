from __future__ import annotations

from topology_paper.src.data.answer_extract import answers_match, extract_pred_answer


def score_generation(generated_text: str, gold_answer: str) -> dict[str, str | bool | None]:
    pred = extract_pred_answer(generated_text)
    return {
        "pred_answer": pred,
        "correct": answers_match(pred, gold_answer),
    }

