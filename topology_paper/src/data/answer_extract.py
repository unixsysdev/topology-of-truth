from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation


BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
GSM8K_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")
NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def extract_boxed_answer(text: str) -> str | None:
    matches = BOXED_RE.findall(text or "")
    if matches:
        return clean_answer(matches[-1])
    return None


def extract_gsm8k_answer(text: str) -> str | None:
    matches = GSM8K_RE.findall(text or "")
    if matches:
        return clean_answer(matches[-1])
    return None


def extract_last_number(text: str) -> str | None:
    matches = NUMBER_RE.findall(text or "")
    if matches:
        return clean_answer(matches[-1])
    return None


def extract_pred_answer(text: str) -> str | None:
    return extract_boxed_answer(text) or extract_gsm8k_answer(text) or extract_last_number(text)


def clean_answer(answer: str | None) -> str | None:
    if answer is None:
        return None
    answer = answer.strip()
    answer = answer.replace(",", "")
    answer = answer.strip("$ ")
    return answer or None


def _decimal(value: str | None) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(clean_answer(value) or "")
    except (InvalidOperation, ValueError):
        return None


def answers_match(pred: str | None, gold: str | None) -> bool:
    pred_clean = clean_answer(pred)
    gold_clean = clean_answer(gold)
    pred_num = _decimal(pred_clean)
    gold_num = _decimal(gold_clean)
    if pred_num is not None and gold_num is not None:
        return pred_num == gold_num
    return pred_clean is not None and gold_clean is not None and pred_clean == gold_clean

