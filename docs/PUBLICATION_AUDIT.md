# Paper companion packaging audit — 16 October 2026

## Charter

Goal: publish a paper-aligned companion package in `aimldlnlp/iden-vln`, using the manuscript, local research workspace, and server experiment artifacts. Scope: dataset/software publication. Object: the paper companion repository. Subjects: selected sources, task/split records, summaries, and construction schematic. This descriptive audit does not create new scientific results.

## Canonical sources

1. Originating project records: current manuscript and manuscript-to-artifact map, 15 October 2026.
2. Originating raw artifacts: compact JSONL release, frozen run summaries, contracts, and source snapshots.
3. Direct inspection: SHA256 comparisons between local and server copies; Git blobs of the IDEN-VLN and LH-VLN snapshots listed in `PROVENANCE.md`.
4. Existing Graphify graph: discovery aid only, stale for some recent additions; actual files supersede it.

## Findings ledger

- F1: Compact JSONLs contain 2,500 unique tasks and 3,291 unique segments across 155 scenes. Integrity and parent mappings pass `tools/verify_release.py`.
- F2: Both compact JSONLs match server SHA256: tasks `30484f5727b566b53ca103ea354ec326d3b3306b9bd79f8342e8dd5041028bcd`; segments `8e90ef4709312c25248c853217c7a0d721e57fb25831b335ba3901cff0b456ca`.
- F3: Both split files cover every task exactly once. Scene-held-out partitions have no scene overlap. XLM-R's actual loader reads random counts 2,000/250/250 and scene-held-out counts 2,019/247/234.
- F4: Additional paper implementation exists outside the prior GitHub generation snapshot. `source_inventory.json` records selected sources and both source/package hashes; `PAPER_CODE_MAP.md` maps paper components to implementations.
- F5: Three local/server same-name sources differ. Both variants are retained, including the raw-center versus snapped-center eligibility distinction. They were not merged or used to rerun results.
- F6: The pipeline PNG is rendered from the manuscript's original figure PDF and visually inspected. README local links resolve. Its panels remain labeled illustrative in the caption.
- F7: Python source parsing passes. The XLM-R `--help` command succeeds. Credential-pattern scanning finds no configured token/private-key pattern. A deliberately corrupted dataset copy is rejected by the integrity verifier.

## Scope filter

Revise the earlier GitHub-only interpretation: generation-repository file similarity does not measure originality of the complete paper. Runtime research code and compact data exist outside that snapshot.

## Object filter

A package of selected sources and summaries is not a fully restored simulator environment. No claim of independent authorship follows solely from a missing upstream blob. No training, simulator replay, or protocol change was performed for publication.

## Conclusions

F1-F3 support direct compact-data access and CPU integrity verification. F4-F6 support a broader paper companion with an explicit source map and construction overview. F5 and F7 support preserving variants and reporting verification limits. Historical numerical results remain unchanged.

## Next steps and risks

Remaining work for complete clean-machine navigation reproduction includes external assets, policy weights, caches, source trajectories, dynamic path dependencies, and exact environment restoration. Dataset/upstream licensing and artwork provenance are unresolved; no blanket license is granted. The inherited optional vision helper is unfinished and is documented in `REPRODUCIBILITY.md`. Credential-pattern scanning cannot prove absence of every possible secret. No fresh scientific-performance claim is made.
