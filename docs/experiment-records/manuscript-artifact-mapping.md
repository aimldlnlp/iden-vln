# Manuscript experiment-to-artifact mapping

Updated 15 October 2026. Internal identifiers are preserved here rather than in the paper narrative.

| Reader-facing description | Original identifier |
|---|---|
| Initial four-task imitation-learning pilot | `approach_train4_il_20261011_v1` |
| Expanded eight-task demonstration set | `visible_approach_framed_v2_expanded_bank_20261011_v1` |
| Matched initial/expanded training-set comparison | `approach_matched_il_20261011_v1` |
| Level-balanced 100-task evaluation | `balanced100_eval_20261013_v1` |
| Corrective-demonstration comparison, 1,200 updates | `approach_clip8_human_recovery_pair_20261014_v1` |
| Corrective-demonstration comparison, 3,600 updates at constant learning rate | `approach_clip8_human_recovery_budget3600_20261014_v1` |
| Corrective-demonstration comparison, 3,600 updates with reduced learning rate | `approach_clip8_human_recovery_lr_drop_20261014_v1` |
| Action-derived STOP-counter ablation | `approach_clip8_stop_counter_20261014_v1` |
| Frozen visibility- and framing-constrained approach-goal resolver | `visible_approach_framed_v2` |
| Historical AABB-based goal resolver | `aabb_v0` |
| Legacy release field storing normalized language entropy, not the standard M-index | `m_index_mean` |
| Right-turn action | `turn_right` |

## Artifact access

Experiment folders are retained under `remote_ordered_eval/` in the workspace and, for remote runs, under `/home/aimldl/LH-VLN/runs/`. This mapping is not a claim that every run contains the same artifact schema.
Use each run's frozen contract, checkpoint references, comparison summary, and verification records to resolve configurations and results. No experiment data is deleted or renamed by the manuscript cleanup.

## Archived manuscript presentation

The former complete diagnostic table source remains in `cie_cas_dc_template_20260925/supplementary/tables_agent_diagnostics_full.tex` as an uncompiled historical source. Its underlying experiment results remain unchanged.
The former failure-profile PDF is retained as an unreferenced artifact. OCS, premature STOP, maximum-step rate, and order violation are no longer presented in the compiled manuscript.

The pre-cleanup manuscript sources and PDF are saved under `revision_backups/presentation_cleanup_20261015/`.
