# Convergence Failure Pivot

This branch records the pivot away from the original "truth geometry" direction and toward a convergence-failure result.

## Status

The 384-token within-question artifact is frozen:

- Artifact root: `artifacts/qwen35_2b_within_question`
- Model: `Qwen/Qwen3.5-2B`
- Temperature: `0.7`
- Top-p: `0.95`
- Max new tokens: `384`
- Prompt and extractor: unchanged from the fixed setup

The matched capped-sample sensitivity run is separate:

- Artifact root: `artifacts/qwen35_2b_cap_sensitivity_640`
- Max new tokens: `640`
- Source: all capped rows from the frozen 384-token manifest
- Seeds and sample IDs: preserved from the source run

## Main Result

The dominant empirical signal is convergence failure.

In the 180-sample fixed 384-token artifact:

- Total samples: 180
- Correct / wrong: 88 / 92
- EOS / capped: 93 / 87
- Capped traces: 75 wrong / 12 correct
- Non-capped traces: 17 wrong / 76 correct
- Cap predicts wrong with AUC about 0.839
- Odds ratio for wrong given cap: about 27.94

This is much stronger than the hidden-state topology and basin effects tested so far.

## Sensitivity Result

The 640-token matched rerun tested whether the cap effect was just truncation.

For the 87 originally capped samples:

- 87 / 87 reran successfully
- At 640 tokens: 59 reached EOS, 28 still capped
- Original capped wrong: 75
- Wrong -> correct at 640: 27
- Wrong -> still wrong at 640: 48
- Wrong -> EOS correct: 24
- Wrong -> EOS wrong: 24
- Wrong -> still capped wrong: 24
- Wrong -> still capped correct: 3
- Original capped correct: 12
- Correct stayed correct: 11
- Correct became wrong: 1

Interpretation: the cap effect is both truncation and deeper non-convergence. A longer budget rescues a real fraction of capped wrong traces, but most originally capped wrong traces remain wrong even at 640 tokens. The still-capped-at-640 group remains especially error-prone.

## Negative Results From The Old Direction

The following directions should be demoted or framed as negative/secondary:

- Global topology summaries did not produce a reliable correctness signal.
- Sliding-window H0 collapsed under the original preprocessing and was not rescued enough to become primary.
- Teacher-forced reference proximity did not support a "truth reference basin" hypothesis.
- Answer-region geometry showed a small 3/3 CC < CW signal on eligible questions, but targeted augmentation did not increase mixed non-capped support.
- Same-wrong-answer clustering was underpowered after augmentation.

The honest conclusion is that this run does not support a broad "truth has geometry" claim.

## Revised Paper Shape

Working title:

`Convergence Failure as a Dominant Error Mode in Small-Model Math Reasoning`

Core claims:

1. Under fixed prompting and decoding, Qwen3.5-2B exhibits a strong converged vs non-converged split.
2. Hitting the token cap is strongly associated with wrong answers.
3. Increasing the token budget rescues some capped failures, so truncation matters.
4. Most capped wrong traces remain wrong at the larger budget, so cap is also an operational marker of deeper non-convergence.
5. Hidden-state topology, teacher-forced reference distance, and local answer-state geometry are negative or weak secondary findings under this protocol.

## Next Work

Do not scale the old topology experiment as the main story.

Better next steps:

- Write the convergence-failure result first.
- Treat hidden-state/topology failures as honest negative baselines.
- Add examples of capped traces that become correct at 640 and capped traces that remain wrong/capped at 640.
- If more compute is used, test a small grid of token budgets, for example `384`, `512`, `640`, and `768`, on the same capped-source set.
- Keep any geometry claims question-conditioned and secondary unless a future dataset produces many more mixed non-capped questions.
