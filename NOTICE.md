# Notices and Attribution

IDEN-VLN is an attributed adaptation of the LH-VLN/NavGen data-generation pipeline for Indonesian-English code-switched long-horizon VLN.

## Upstream Pipeline

Parts of the data-generation workflow, including the long-horizon task-generation structure, Habitat-Sim rollout logic, trajectory segmentation approach, and NavGen-style intermediate artifacts, are adapted from:

- HCPLab-SYSU, LH-VLN
- Repository: https://github.com/HCPLab-SYSU/LH-VLN

This repository should be described as an adaptation or extension of LH-VLN/NavGen, not as a fully independent implementation created from scratch.

## IDEN-VLN Modifications

This adaptation adds or changes the following parts:

- OpenAI-compatible LLM configuration for Qwen2.5 and other open-source instruction models.
- Indonesian-English code-switched prompt usage.
- Code-switching evaluation utilities for Indonesian-English output analysis.
- English-facing documentation and explicit attribution for academic use.

## Third-Party Components

This repository includes or depends on additional third-party research software and datasets:

- Habitat-Sim and HM3D for embodied navigation simulation and scene data.
- RAM/RAM++ from Recognize Anything for scene-tag augmentation.
- Qwen2.5 for open-source instruction generation when used through a compatible serving endpoint.

The `recognize_anything/` subtree preserves the upstream RAM/RAM++ license and notice files where available.

## License Status

No new permissive top-level license is granted here because the upstream LH-VLN/NavGen source license is not clearly established in the available project context. Users should verify permissions before redistributing modified versions.

