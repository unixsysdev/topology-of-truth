# Dynamic Topology Paper Pipeline

This directory is a paper-focused pipeline for testing whether time-varying hidden-state topology predicts correctness better than static full-trace topology.

Default scope:

- Models: `Qwen/Qwen3.5-0.8B`, `Qwen/Qwen3.5-2B`
- Dataset: `openai/gsm8k`, `main`, `test`
- Hidden states: last hidden layer across all generated token positions
- Core features: prefix and sliding-window H0 persistent entropy, dust score, and H1 summaries for ablation
- Output root: `outputs/gsm8k_qwen35_small/`

Run a small smoke pilot:

```bash
HF_HUB_DISABLE_XET=1 \
LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64 \
python3 -m topology_paper.src.runs.run_all --config topology_paper/configs/gsm8k_qwen35.yaml --end 5
```

Run stages separately:

```bash
HF_HUB_DISABLE_XET=1 \
LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64 \
python3 -m topology_paper.src.runs.run_generation --config topology_paper/configs/gsm8k_qwen35.yaml --start 0 --end 200
HF_HUB_DISABLE_XET=1 \
LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64 \
python3 -m topology_paper.src.runs.run_topology --config topology_paper/configs/gsm8k_qwen35.yaml
HF_HUB_DISABLE_XET=1 \
LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64 \
python3 -m topology_paper.src.runs.run_stats --config topology_paper/configs/gsm8k_qwen35.yaml
```

On the current `llama-rocm-7.2` toolbox, disabling the `hf-xet` download backend avoids a stalled initial checkpoint fetch for `Qwen/Qwen3.5-0.8B`.

Trace schema:

```json
{
  "sample_id": "gsm8k_000123",
  "model_id": "Qwen/Qwen3.5-0.8B",
  "question": "...",
  "gold_answer": "42",
  "generated_text": "...",
  "pred_answer": "40",
  "correct": false,
  "n_generated_tokens": 87,
  "token_ids": [],
  "hidden_path": "outputs/gsm8k_qwen35_small/traces/.../last_layer.npy",
  "logits_path": "outputs/gsm8k_qwen35_small/traces/.../logits_summary.npz"
}
```

`logits_path` defaults to compact per-token summaries: entropy, max probability, chosen-token log probability, and top-2 probability margin. Set `extraction.save_full_logits: true` only when you explicitly want full vocabulary logits, because they are large for Qwen-scale vocabularies.

By default, the topology stage excludes traces that appear to have ended because they hit `generation.max_new_tokens`. New traces record `hit_max_new_tokens`, `terminated_by_eos`, and `stop_reason` in `trace.json`. Older traces are conservatively excluded when `n_generated_tokens == max_new_tokens`.

The generated feature tables are:

- `features/dynamic_features.parquet`: one row per prefix/window TDA computation
- `features/sample_features.parquet`: one row per sample/model with static and aggregate dynamic features
- `stats/stats_summary.json`: cross-validated logistic-regression ROC-AUCs for length, static H0, dynamic H0, and lexical-baseline comparisons
- `figs/*.png`: correctness-vs-H0 and dynamic trajectory plots
