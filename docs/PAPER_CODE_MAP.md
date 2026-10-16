# Paper-to-code map

Target manuscript: **IDEN-VLN: An Indonesian-English Code-Mixed Dataset for Long-Horizon Vision-and-Language Navigation**, current structure revision 15 October 2026.

| Paper component | Implementation | Packaged evidence | Boundary |
|---|---|---|---|
| Construction Figure 1, stage A | root `main.py`, `llm.py`, `task_gen.py`, `prompt/`; `research/data/run_downstream_batches.py` | `assets/construction-pipeline.png` | Generic adaptation and historical generation driver; generation needs original run inputs/model service. |
| Figure 1, stage B | root `dataset_gen.py`, `dataset.py`, `habitat_base/` | source export manifest | Inherited NavGen/Habitat implementation; raw assets external. |
| Figure 1, stage C | root `split_task.py`, `recognize_anything/`; `research/data/export_release_package.py` | compact task/segment JSONLs | RAM and scene metadata remain third-party components. |
| Dataset characteristics and splits | `research/data/export_release_package.py`; `tools/verify_release.py` | `dataset/` | Split verification does not prove language naturalness. |
| Rule/LLM textual grounding | `research/legacy/run_publishable_benchmarks.py` | manuscript grounding table sources | Historical source paths and model-server inputs required. |
| TF-IDF Logistic Regression/SVM | `research/legacy/run_classic_ml_strengthening.py` | manuscript classic-ML table sources | Preserve task-averaged versus micro-F1 definitions. |
| XLM-R grounding | `research/grounding/train_xlmr_grounding.py` | `results/grounding/aggregate_metrics.json` | Portable CLI; weights/dependencies obtained separately. |
| Canonical navigation Table 4 | `research/navigation/run_ordered_metrics_eval.py`, `goal_resolver.py`; `research/legacy/run_harmonized_agent_eval_fixed.py` | `results/stage_metric_audit_20260928/`; manuscript agent table | Two-stage cohort, horizon 150, strict <1 m; weights/source traces external. |
| Balanced100 | `build_level_balanced_bank.py`, `run_frozen_bank_eval.py`, `verify_frozen_bank_eval.py` in `research/navigation/` | `results/balanced100_eval_20261013_v1/verified_results.json` | 25 tasks per level 2-5; development/exposure overlap. |
| Matched-contract repair | `research/navigation/run_balanced100_matched_contract.py`; `research/snapshots/pipeline_repair_20261013_v3/` | matched-contract summary/protocol | Distinct diagnostic contract; not a replacement for historical results. |
| Approach demonstrations and bank expansion | `build_visible_approach_pilot.py`, `frozen_approach_resolver.py`, expansion/replay/freezer scripts | expanded-bank reviewed summary/protocol | Frozen goal construction and teacher success are not learned-policy performance. |
| Small-bank IL and text controls | `train_two_reviewed_il_overfit.py`, `spatial_clip_features.py`, matched-IL scripts | experiment recap | SequenceGRU and frozen CLIP dependencies; training fit does not imply generalization. |
| Recovery and optimization schedules | recovery collector/cache/trainer/evaluator; `research/snapshots/approach_clip8_human_recovery_*/` | recovery summaries and verification | Per-run sources/configuration control; generic helpers may differ. |
| STOP-counter ablation | `research/snapshots/approach_clip8_stop_counter_20261014_v1/` | local verification and experiment record | Counter uses prior executed actions; negative result retained. |

The snapshots and selected summaries document the experiments; they do not include every required checkpoint, cache, trajectory, or raw result. No new experiment was performed to create this map.
