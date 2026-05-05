# IDEN-VLN

IDEN-VLN is an attributed adaptation of the LH-VLN/NavGen data-generation pipeline for long-horizon Vision-and-Language Navigation (VLN) with Indonesian-English code-switched instructions.

This repository is not presented as a from-scratch implementation of LH-VLN/NavGen. It builds on the original LH-VLN/NavGen workflow and adapts it for an open-source LLM setting with Qwen2.5, while adding evaluation utilities for Indonesian-English code-switching behavior.

## Main Contributions

- Adapts the LH-VLN/NavGen pipeline for Indonesian-English long-horizon VLN data generation.
- Replaces GPT-oriented generation with OpenAI-compatible LLM endpoints, enabling Qwen2.5 and other open-source instruction models.
- Adds Indonesian-English code-switching evaluation, including token-level language identification and aggregate mixing metrics.
- Keeps intermediate artifacts traceable across task generation, Habitat-HM3D rollout, trajectory segmentation, RAM/RAM++ scene tagging, and step-level instruction synthesis.

## Pipeline Overview

The pipeline follows four main stages:

1. `gen_task`: generate a high-level multi-target task from scene objects, room context, robot constraints, and bilingual prompting.
2. `gen_traj`: execute the task in Habitat-Sim over HM3D scenes and record successful or failed rollouts.
3. `split_traj`: segment successful rollouts into shorter action spans and attach RAM/RAM++ scene tags.
4. `gen_step_task`: generate concise Indonesian-English navigation instructions for the segmented spans.

The default entry point is:

```powershell
python main.py --API_KEY dummy --API_BASE http://localhost:8000/v1 --MODEL_MAIN Qwen2.5-7B-Instruct --MODEL_MINI Qwen2.5-7B-Instruct
```

Use the model names expected by your serving backend. Any OpenAI-compatible chat-completions server can be used if it exposes `/v1/chat/completions`.

## Expected External Assets

The repository intentionally does not include large datasets or model weights. Place them outside the repository, using the default layout or overriding paths through CLI arguments:

- HM3D scene data: `../data/hm3d/`
- HM3D scene dataset config: `../data/hm3d/hm3d_annotated_basis.scene_dataset_config.json`
- RAM++ checkpoint: `../data/models/ram_plus_swin_large_14m.pth`

Relevant arguments:

```powershell
python main.py `
  --scene ../data/hm3d/ `
  --scene_dataset ../data/hm3d/hm3d_annotated_basis.scene_dataset_config.json `
  --ram_model ../data/models/ram_plus_swin_large_14m.pth
```

## Outputs

Runtime outputs are generated locally and ignored by Git:

- `task/`: high-level task configurations and rollout records.
- `step_task/`: generated step-level navigation instructions.
- `logs/`: evaluation logs and RAM/RAM++ processing logs.

These folders are excluded from the repository because they are experiment artifacts, not source code.

## Code-Switching Evaluation

The `eval/` folder contains Indonesian-English code-switching analysis utilities. The evaluation focuses on language distribution and switching behavior, such as:

- Indonesian vs. English token ratios.
- Code-Mixing Index (CMI).
- I-index and switch-point behavior.
- Supporting span and entropy-style metrics where configured.

The metrics are intended to evaluate whether generated instructions satisfy the intended Indonesian-English code-switching profile, not only whether they are navigable.

## Attribution

This repository adapts parts of the LH-VLN/NavGen pipeline by HCPLab-SYSU. Please cite and acknowledge the original project when using this repository:

- Original repository: https://github.com/HCPLab-SYSU/LH-VLN

Additional third-party components include Habitat-Sim/HM3D, RAM/RAM++, and Qwen2.5. See `NOTICE.md` for attribution details.

## License Note

The upstream LH-VLN/NavGen repository does not appear to provide a clear top-level license in the copied source context. For that reason, this repository does not add a new permissive top-level license. Clarify permissions with the upstream authors before public redistribution beyond academic review or private research use.

