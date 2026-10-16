# Source provenance and release decisions

Checked 16 October 2026 against IDEN-VLN base `1e45cb295352d23a0fcba472a99161ffd7e46962` and LH-VLN upstream `3492080382aba031e9d82a4f630c2b1cdf57df81`.

## Existing generation repository

Of 260 existing tracked files, 238 are byte-identical to matching upstream files; two CSVs differ at the byte level but have identical normalized text; 12 other matching files differ; eight paths have no counterpart in the compared snapshot. Counts are dominated by scene metadata and RAM. They are not percentages of scientific originality.

- `dataset.py` and `dataset_gen.py`: unchanged NavGen sources.
- `task_gen.py` and `split_task.py`: adapted imports/model-helper calls.
- `habitat_base/`: inherited implementation; two files change output paths.
- `recognize_anything/`: 48 byte-identical RAM files, including license and third-party notices.
- `scene/`: 181 text files, two CSVs, and one Python file inherited from the upstream snapshot.
- `llm.py`, `main.py`, and three prompts: adaptations for model configuration and Indonesian-English instructions.
- Language evaluator and wordlists: added paths relative to that upstream snapshot. Wordlist origin remains unresolved. Absence from upstream does not establish authorship.

## Additional paper implementations

The current paper also uses local/server implementations outside that generation snapshot: compact export, textual grounding, navigation metrics, goal resolvers, demonstration construction, IL training, recovery experiments, and STOP-counter diagnostics. Eighteen sampled implementation files had no identical blob in either compared repository snapshot. This is evidence of additional implementations, not a full copyright or derivation audit.

`source_inventory.json` records source locations and hashes for the selected publication files. Third-party frameworks and pretrained models remain dependencies: Habitat-Sim, PyTorch, Transformers, CLIP, XLM-R, scikit-learn, RAM, and the source NavGen pipeline. No new license for those components is granted.

## Local/server differences

At the root of the experiment directory, 114 of 117 server `.py`/`.sh` files matched local SHA256; three same-name sources differed. Local sources are preserved under `research/navigation/`, with server variants under `research/source_variants/server/`:

1. `audit_original_instruction_contract.py`: local adds metadata export and a `--text-only` path.
2. `freeze_paired_original_bank.py`: local writes explicit UTF-8 bytes rather than text-mode output; newline behavior can matter across platforms.
3. `validate_level2_il_candidates.py`: local uses snapped-center distance in one eligibility check; server uses raw-center distance. This is a meaningful protocol difference. Do not substitute one for the other without versioning the experiment.

No variants were merged and no historical result was recomputed. Per-run frozen sources take precedence over generic helpers when reconstructing a specific experiment. A server trainer variant is also retained rather than assumed identical to its local historical predecessor.

## Attribution and redistribution boundary

The upstream snapshot has no top-level license file found by this audit. Existing project attribution and RAM license/notices are preserved. This publication does not resolve the upstream licensing question, assign a permissive license, or imply rights to redistribute HM3D assets or pretrained weights. Obtain external assets from their official providers. The pipeline figure is the manuscript's illustrative schematic; its component artwork provenance has not been independently audited.
