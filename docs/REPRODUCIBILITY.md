# Reproducibility status

## What is directly available

The compact JSONL dataset, both split files, source export manifest, package checksums, selected experiment sources, numerical table sources, and selected result/verification summaries are included. Run `python tools/verify_release.py` from the repository root for CPU-only integrity checks.

`research/grounding/train_xlmr_grounding.py` accepts `--data-root`, `--run-root`, `--split`, `--seed`, and optimization arguments. It reads the packaged task/split schema. Install compatible PyTorch and Transformers plus SentencePiece; obtain XLM-R weights through the model provider. The analysis requirements specify compatibility ranges, not a frozen environment that guarantees manuscript-identical numbers.

## Historical environment evidence

Observed notes record Python 3.9.23 in the Habitat environment and Python 3.12.3 in the system environment; scikit-learn versions 1.5.1 and 1.8.0 occurred in different environments. These must not be collapsed into a single supposedly tested installation. The inherited generator requirements contain an old Habitat-Sim pin; use the original experiment environment and protocol when reconstructing simulation runs.

## What requires additional assets and adaptation

Navigation sources retain the original `LH-VLN/runs/...` and `nav_gen/runs/...` layouts, often as absolute paths. Dynamic imports and snapshot references are not all resolved by static import closure. Sources alone are insufficient for end-to-end reruns.

Required external material can include:

- Authorized HM3D/Habitat scene meshes, semantic instances, NavMeshes, scene configuration, and original start states.
- Original trajectory frames, task configurations, feature caches, and training episode manifests.
- Policy checkpoints and CLIP/RAM weights. These are not uploaded in this package.
- The exact frozen goal bank, scorer, model settings, action mapping, seed, and input contract for the selected run.
- Original model-server settings for Qwen/Llama/Mistral parsing or generation.

Do not run historical builders/freezers into an existing output directory. Read their constants and use a new versioned output identity. Restoring a directory layout does not establish version parity; check source and checkpoint hashes too.

## Experiment boundaries

Canonical Table 4, Balanced100, matched-contract repair, and approach-goal IL are separate cohorts/contracts. Do not replace Table 4 with approach results. Retain invalid-reference coverage and failures in reported denominators. Do not treat accepted development subsets as the entire development cohort or move development scenes into training.

Semantic masks, geometry, and NavMeshes used by a teacher/scorer are not automatically permitted policy inputs. Audit each historical policy's actual information access. Manipulation wording is projected to navigation in these diagnostics; physical manipulation success is not established.

## Packaging verification

Packaging checks cover dataset integrity, split/parent relationships, source hashes, Python parsing, credential patterns, and README assets. They do not establish fresh GPU training, fresh simulator replay, exact score reproduction from a clean machine, or completed license clearance. Included experiment verification records describe the earlier runs; they are not newly executed simulator checks.

## Inherited optional vision helper

The root generator is the text-oriented adaptation. Its optional `vision_llm` helper retains an unfinished message placeholder and must not be presented as a validated VLM path. None of the package integrity checks exercises this optional helper. The text-only generation configuration is separate from learned navigation policies.
