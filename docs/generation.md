# IDEN-VLN: Dataset Generation Pipeline for Long-Horizon VLN with Indonesian-English Code-Switched Instructions

IDEN-VLN is a dataset-generation pipeline for long-horizon Vision-and-Language Navigation (LH-VLN) with Indonesian-English code-switched instructions. It adapts the LH-VLN/NavGen workflow into an open-source LLM setting and adds language-side evaluation for Indonesian-English code-switching quality.

This repository should be cited as an adaptation or extension of LH-VLN/NavGen, not as a from-scratch replacement for the original project. The IDEN-VLN contribution is the Indonesian-English generation and evaluation layer: Qwen2.5-compatible instruction generation, code-switched prompts, and code-switching metrics for bilingual output analysis.

## Contributions

- Indonesian-English LH-VLN task and step-instruction generation.
- Qwen2.5-oriented OpenAI-compatible LLM interface for task and instruction synthesis.
- Habitat-HM3D rollout validation for generated long-horizon navigation tasks.
- RAM/RAM++ scene-tag augmentation for segment-level visual context.
- Code-switching evaluation for bilingual balance, switching intensity, and language-distribution compliance.

## Pipeline

The pipeline has four operational stages:

1. `gen_task`: sample scene context and generate a high-level multi-target task with Indonesian-English prompting.
2. `gen_traj`: execute the task in Habitat-Sim over HM3D scenes and save successful or failed rollouts.
3. `split_traj`: segment successful rollouts into shorter action spans and attach RAM/RAM++ scene tags.
4. `gen_step_task`: synthesize concise Indonesian-English step-level navigation instructions from segmented trajectories.

Conceptually, the data flow is:

```text
HM3D scene context
  -> Qwen2.5 task generation
  -> Habitat rollout validation
  -> trajectory segmentation
  -> RAM/RAM++ scene tagging
  -> Indonesian-English instruction synthesis
  -> code-switching evaluation
```

## Usage

Serve Qwen2.5 or another instruction model through an OpenAI-compatible `/v1/chat/completions` endpoint, then run:

```powershell
python main.py `
  --API_KEY dummy `
  --API_BASE http://localhost:8000/v1 `
  --MODEL_MAIN Qwen2.5-7B-Instruct `
  --MODEL_STEP Qwen2.5-7B-Instruct
```

Model names must match the names exposed by your serving backend.

## External Assets

Large datasets and model weights are intentionally not included. Place them outside the repository and override paths when needed:

- HM3D scene data: `../data/hm3d/`
- HM3D scene dataset config: `../data/hm3d/hm3d_annotated_basis.scene_dataset_config.json`
- RAM++ checkpoint: `../data/models/ram_plus_swin_large_14m.pth`

Example:

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

## Evaluation

The `eval/` folder contains Indonesian-English code-switching metrics. These metrics evaluate language quality in addition to navigation validity:

- Indonesian and English token distribution.
- Code-Mixing Index (CMI).
- I-index and switch-point behavior.
- Span, entropy, burstiness, memory, and T-index style measurements where configured.

See `DATASET_CARD.md` for the intended dataset profile, limitations, and reporting notes.

## Attribution

IDEN-VLN adapts the LH-VLN/NavGen data-generation workflow by HCPLab-SYSU:

- Original repository: https://github.com/HCPLab-SYSU/LH-VLN

The repository also depends on Habitat-Sim/HM3D, RAM/RAM++, and Qwen2.5. See `NOTICE.md` and `CITATION.md` for attribution details.

## License Note

The upstream LH-VLN/NavGen repository does not appear to provide a clear top-level license in the available project context. This repository therefore does not add a new permissive top-level license. Verify permissions with the upstream authors before public redistribution.



Historical generator instructions above use paths relative to the repository root. Consult the main README for current dataset and paper-companion content.
