# IDEN-VLN Release Data Card

## Summary

IDEN-VLN is an Indonesian-English operational code-mixed long-horizon embodied navigation dataset package released as a compact research record package.

This release package contains:

| Artifact | Count / status |
|---|---:|
| High-level trajectory-success tasks | 2,500 |
| Step-task segments | 3,291 |
| Random split | included |
| Scene-held-out split | included |
| Selected benchmark summaries | see `../results/` |
| License | Deferred: source-derived rights review; see `LICENSE.md` |

## Intended Use

- Dataset and benchmark research for Indonesian-English code-mixed embodied navigation instructions.
- Textual grounding and instruction parsing benchmarks under random and scene-held-out splits.
- Reproducible baseline comparison using rule extraction, LLM extraction, TF-IDF Logistic Regression/SVM, and sequence-order metrics.
- Controlled downstream diagnostics for agent pipelines.

## Not Intended Use

- Do not use this package to claim a competitive learned visual navigation policy.
- Do not claim robust RL/PPO/MGDM embodied navigation success.
- Do not claim exhaustive human validation of all 2,500 tasks. The manuscript reports a limited five-rater audit of 200 tasks.
- Do not interpret scene-held-out results as zero-shot object, room, or language generalization.

## Files

| File / folder | Description |
|---|---|
| `iden_vln_tasks.jsonl` | High-level task records, one JSON object per task. |
| `iden_vln_step_tasks.jsonl` | Step-task segment records, one JSON object per segment. |
| `splits_random.json` | Random train/val/test task split assignments and summary. |
| `splits_scene_heldout.json` | Scene-held-out train/val/test task split assignments and summary. |
| `../results/` | Selected benchmark and diagnostic summaries; not every original output. |
| `release_manifest.json` | Export counts, source paths, split summary, and known limitations. |
| `release_checksums.json` | SHA-256 checksums and file sizes. |
| `../docs/REPRODUCIBILITY.md` | Environment and reproduction boundaries. |

## High-Level Task Schema

Each line in `iden_vln_tasks.jsonl` contains:

| Field | Meaning |
|---|---|
| `task_id` | Stable release identifier. |
| `source_run` | Source partition: `old`, `v2`, or `topup`. |
| `source_index` | Index within the source manifest. |
| `source_id` | Source manifest identifier when available. |
| `source_slug` | Portable source name/slug without raw trajectory assets. |
| `level` | Generated long-horizon level, usually 2-5. |
| `robot` | Robot condition, e.g. `stretch` or `spot`. |
| `scene` | Scene identifier. |
| `task_instruction` | Indonesian-English high-level instruction. |
| `subtask_list` | Symbolic subtask list. |
| `objects`, `object_names` | Object grounding metadata. |
| `region_ids`, `region_names` | Room/region grounding metadata. |
| `audit` | Automatic prompt/audit metadata where available. |
| `generator` | Generator label where available. |
| `validation_status` | `trajectory_success`. |
| `random_split` | Random split assignment. |
| `scene_heldout_split` | Scene-held-out split assignment. |

## Step-Task Schema

Each line in `iden_vln_step_tasks.jsonl` contains:

| Field | Meaning |
|---|---|
| `step_task_id` | Stable step segment identifier. |
| `parent_task_id` | Parent high-level task identifier. |
| `parent_source_slug` | Parent source slug extracted from trajectory reference. |
| `step_index_within_parent` | Segment index within the parent trajectory. |
| `global_step_index` | Global step index from the original step filename. |
| `trajectory_ref` | Portable relative trajectory reference from the generated step-task file. |
| `start`, `end` | Segment frame/index range. |
| `robot`, `scene` | Robot and scene metadata. |
| `target`, `region` | Segment target metadata. |
| `start_pos`, `start_yaw` | Start pose metadata where available. |
| `mapping` | Internal mapping metadata from trajectory-to-step conversion. |
| `step_instruction` | Step-level navigation instruction. |

## Split Policy

- `splits_random.json` uses deterministic hash ordering over task IDs and an 80/10/10 split.
- `splits_scene_heldout.json` assigns whole scenes to train/val/test partitions.
- Scene-held-out split has train/val/test scene overlap `0`.

## Limitations

- No CC BY dataset grant is applied. See `LICENSE.md` and `license_metadata.json` for HM3D-derived rights and reuse conditions.
- Code-mixing labels are automatic heuristics, not human linguistic annotation.
- Scene-held-out split is not zero-shot object, room, or language generalization because object/room/lexical overlap remains high.
- Failure-mode counts are qualitative file/log audit evidence, not mutually exclusive event-level statistics.
- Agent results are controlled downstream diagnostics; learned policies remain below expert/oracle metadata upper bound and have SR/OSR `0.0` in harmonized evaluation.
- Raw trajectory assets, simulator scenes, and Habitat/HM3D assets are not included in this compact release package.

## Citation / Manuscript Note

Use the manuscript framing in the parent `README.md`: dataset + benchmark + validation pipeline, with agent diagnostics as secondary/appendix evidence.

## Current license status

The source export manifest is a historical record. Current reuse status is recorded in [LICENSE.md](LICENSE.md) and [license_metadata.json](license_metadata.json). The owner-controlled code MIT grant does not cover these records or override provider terms.
