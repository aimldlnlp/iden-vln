# Notices and Attribution

IDEN-VLN is an attributed adaptation of the LH-VLN/NavGen data-generation pipeline for long-horizon VLN with Indonesian-English code-switched instructions.

## Upstream Pipeline

Parts of the data-generation workflow, including the long-horizon task-generation structure, Habitat-Sim rollout logic, trajectory segmentation approach, and NavGen-style intermediate artifacts, are adapted from:

- HCPLab-SYSU, LH-VLN
- Repository: https://github.com/HCPLab-SYSU/LH-VLN

This repository should be described as an adaptation or extension of LH-VLN/NavGen, not as a fully independent implementation created from scratch.

## IDEN-VLN Modifications

This adaptation adds or changes the following project-facing parts:

- Qwen2.5-oriented OpenAI-compatible LLM configuration.
- Indonesian-English code-switched task and step-instruction prompts.
- Code-switching evaluation utilities for bilingual balance and switching behavior.
- English-facing documentation, dataset card, and explicit academic attribution.

## Third-Party Components

This repository includes or depends on additional third-party research software and datasets:

- Habitat-Sim and HM3D for embodied navigation simulation and scene data.
- RAM/RAM++ from Recognize Anything for scene-tag augmentation.
- Qwen2.5 for open-source instruction generation when used through a compatible serving endpoint.

The `recognize_anything/` subtree preserves the upstream RAM/RAM++ license and notice files where available.

## License Status

A scoped MIT grant now covers the original owner-controlled IDEN-VLN code contributions identified in `LICENSE` and `docs/licensing_manifest.json`. It does not cover inherited NavGen expressions or third-party components. Upstream LH-VLN/NavGen permission remains unresolved.

## Research companion package (16 October 2026)

Additional project experiment sources, compact dataset records, selected result summaries, and the manuscript pipeline schematic accompany the generation adaptation. See `docs/PROVENANCE.md` and `docs/source_inventory.json`. Their inclusion does not establish independent authorship of every component or grant a blanket license. No dataset CC BY grant is applied. Limited owner-controlled academic permission and provider conditions are documented in `dataset/LICENSE.md` and `dataset/TERMS_OF_USE.md`, with a private acceptance procedure in `docs/DATA_ACCESS.md`. Raw scene assets and model weights are excluded.

## Source data terms

Materials containing HM3D data or derived information remain subject to the [Matterport Academic Use Agreement](https://matterport.com/legal/matterport-end-user-license-agreement-academic-use-model-data). This source notice is not replaced by the project code license. See `docs/LICENSE_SCOPE.md` for the current review and remaining questions.
