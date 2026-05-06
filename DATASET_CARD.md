# IDEN-VLN Dataset Card

## Dataset Summary

IDEN-VLN is designed to support long-horizon Vision-and-Language Navigation with Indonesian-English code-switched instructions. The dataset-generation pipeline produces multi-target navigation tasks, validates them in Habitat-Sim over HM3D scenes, segments successful rollouts, augments segments with visual tags, and generates bilingual step-level instructions.

## Intended Use

- Research on long-horizon VLN data generation.
- Evaluation of agents under Indonesian-English code-switched instructions.
- Analysis of bilingual instruction quality, switching intensity, and prompt-compliance behavior.
- Debugging synthetic dataset pipelines through traceable task, rollout, segment, and language artifacts.

This dataset-generation pipeline is not intended to claim full human-authored linguistic naturalness without additional human validation.

## Source Data and External Dependencies

- Indoor scenes and navigation meshes come from Habitat-HM3D.
- Rollout execution uses Habitat-Sim.
- Segment-level scene context uses RAM/RAM++ visual tags.
- Instruction generation is configured for Qwen2.5 or another OpenAI-compatible instruction model.
- The data-generation workflow adapts LH-VLN/NavGen and should be attributed accordingly.

Large assets such as HM3D scenes, model checkpoints, generated logs, and generated tasks are not stored in this repository.

## Language Profile

The target language setting is Indonesian-English code-switching. Prompts ask the LLM to generate instructions with Indonesian-dominant composition and embedded English action or connector phrases. The evaluation layer measures whether the realized output follows this target distribution.

Recommended reported metrics:

- Indonesian token ratio.
- English token ratio.
- Code-Mixing Index (CMI).
- I-index.
- Switch-point count.
- Span entropy, burstiness, memory, and T-index where available.

## Generation Process

1. Scene context is sampled from HM3D metadata, including regions and object categories.
2. A high-level long-horizon task is generated with a Qwen2.5-compatible LLM endpoint.
3. Habitat-Sim validates generated targets through rollout execution.
4. Successful rollouts are segmented into shorter action spans.
5. RAM/RAM++ attaches visual context tags to segment observations.
6. A step-level instruction generator verbalizes each segment in Indonesian-English code-switching.
7. Code-switching metrics evaluate the retained instructions.

## Filtering and Validation

Generated tasks should be filtered by:

- Object and region validity.
- Reachability in Habitat-Sim.
- Successful rollout completion.
- Instruction length and symbolic-task consistency.
- Indonesian-English code-switching metrics and prompt-compliance checks.

The pipeline separates raw generation yield from post-validation navigation reliability. Near-perfect validation scores should not be interpreted as the probability that every raw LLM generation is immediately usable.

## Known Limitations

- The instructions are synthetic and require human evaluation before claims about naturalness.
- Prompt-only bilingual ratio control may be weak and should be measured explicitly.
- Object affordance mismatch can reduce raw generation yield.
- Results depend on the chosen LLM endpoint, decoding settings, scene split, and RAM/RAM++ checkpoint.
- The repository does not include HM3D assets, model weights, or generated task corpora.

## Attribution

IDEN-VLN adapts LH-VLN/NavGen by HCPLab-SYSU and extends it with Qwen2.5-oriented generation plus Indonesian-English code-switching evaluation. Cite both the original LH-VLN/NavGen work and this adaptation when using generated data or code.

