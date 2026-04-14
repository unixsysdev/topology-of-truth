# Claims and Evidence

This table is the canonical mapping from research claims to exact artifacts and scripts.

| Claim | Supported | Evidence Files | Producing Script(s) |
|---|---|---|---|
| Global H0/H1 summaries robustly predict correctness on GSM8K traces | **No** | `artifacts/paired_dynamic_qwen35/static_vs_dynamic_separability_2b.csv`; `artifacts/qwen35_2b_window_corrected/classifier_auc.csv` | `scripts/analyze_paired_dynamic.py`; `scripts/analyze_2b_window_corrected.py` |
| Under fixed decoding, token-cap termination strongly predicts wrong answers | **Yes** | `artifacts/qwen35_2b_within_question/run_manifest.csv` | `scripts/run_multi_sample_questions.py` |
| Cap sensitivity: longer budget rescues some capped failures but many remain wrong | **Yes** | `artifacts/qwen35_2b_cap_sensitivity_640/analysis/transition_summary.csv`; `artifacts/qwen35_2b_cap_sensitivity_640/analysis/per_question_transition_summary.csv` | `scripts/run_cap_sensitivity.py`; `scripts/analyze_cap_sensitivity.py` |
| Micro-world decoder under-expresses `Unknown` | **Yes** | `artifacts/micro_world_v1/comparison_decoder_qwen_gemma.csv` | `scripts/run_micro_world_inference.py`; `scripts/analyze_micro_world_geometry.py` |
| Verdict-region hidden states recover `Unknown` better than decoder outputs | **Yes** | `artifacts/micro_world_v1/comparison_probe_states_qwen_gemma.csv` | `scripts/run_micro_world_probe.py` |
| Same-label vs different-label geometry gap is positive across worlds | **Yes** | `artifacts/micro_world_v1/analysis_qwen35_2b_partial19/sign_test_summary.csv`; `artifacts/micro_world_v1/analysis_gemma_3_4b_eval20_nothink/sign_test_summary.csv` | `scripts/analyze_micro_world_geometry.py` |
| Constrained decoding fixes Unknown collapse | **No** | `artifacts/micro_world_v1/comparison_decoder_constrained_vs_unconstrained_qwen_gemma.csv` | `scripts/run_micro_world_inference.py`; comparison scripts in `artifacts/micro_world_v1` |
| Gemma base-vs-instruct comparison after parse-confound repair still shows readout mismatch | **Yes** | `artifacts/micro_world_v1/comparison_gemma_base_prompt_rerun.csv`; `artifacts/micro_world_v1/comparison_probe_gemma_basefmt_vs_it_raw.csv` | `scripts/run_micro_world_inference.py`; `scripts/run_micro_world_probe.py` |
| Verdict-step label logits under-rank `Unknown` on gold-Unknown decoder failures | **Yes** | `artifacts/micro_world_v1/comparison_label_logits_gemma_it_vs_pt_basefmt.csv` | `scripts/analyze_micro_world_label_logits.py` |
| Layer sweeps show strong internal Unknown recoverability even when decoder fails | **Yes** | `artifacts/micro_world_v1/comparison_layer_sweep_gemma_it_vs_pt_basefmt.csv` | `scripts/run_micro_world_layer_sweep_probe.py` |

## Main Paper Claim

Supported by triangulation above:

> On a held-out procedural micro-world task with nonce lexicons and held-out templates, `Unknown` / non-entailment is recoverable from verdict-region hidden states while decoder outputs under-express it; this dissociation replicates across Qwen and Gemma and survives prompt-path, constrained-decoding, logit, and layer-sweep controls.
