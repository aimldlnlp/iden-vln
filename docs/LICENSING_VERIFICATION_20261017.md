# Scoped licensing verification — 17 October 2026

## Decision

The project owner authorized reuse of contributions he controls. MIT is applied to those original code contributions in the explicit source paths. A whole-dataset CC BY grant was not applied after reviewing source-provider restrictions on derived information.

## Evidence and checks

- Base repository: `a2d5232adebcf0cab92a77c6201b9b94ff537c0f`.
- Current LH-VLN upstream HEAD still matches inspected snapshot `3492080382aba031e9d82a4f630c2b1cdf57df81`.
- `docs/licensing_manifest.json` inventories all 427 files tracked at the base, using SHA256 of the original Git blobs rather than Windows working-tree newline conversions.
- 106 code paths are identified for the MIT grant on original owner-controlled contributions only. This is not a claim of exclusive ownership of every line.
- Existing RAM license and notices, inherited generator code, wordlists, scene metadata, media, experiment results, frozen research sources, task JSONLs and split assignments were not modified.
- The data integrity verifier passes: 2,500 tasks, 3,291 segments, 155 scenes, checksums, split integrity and parent mapping.
- Current package source-inventory hashes match files, including the updated data-card metadata.
- Local links in active README, data cards, scope documentation and dataset notice resolve.
- Dataset metadata explicitly records `cc_by_4_0_applied: false`; excluded NavGen files do not carry a new MIT classification.

## Primary sources

- MIT standard text: https://opensource.org/license/mit
- CC BY 4.0 scope and licensor considerations: https://creativecommons.org/licenses/by/4.0/legalcode.en
- Source-provider academic agreement, sections 2.2 and 2.4: https://matterport.com/legal/matterport-end-user-license-agreement-academic-use-model-data
- HM3D-Semantics availability: https://aihabitat.org/datasets/hm3d-semantics/

## Limits

This change grants only original rights controlled by the owner. It does not establish complete authorship provenance, settle upstream permission, determine whether substantial-source-data distribution provisions apply, or grant commercial use of HM3D-derived data/media. No external permission request was sent. No repository visibility, policy score, model checkpoint, original record, split, frozen goal bank, or evaluation rule was changed. The original export manifest retains its historical license-TBD entry; the new dataset notice and metadata explain the current deferral.

Manuscript license wording is updated separately in the local LaTeX project; its compiled PDF is not included in this repository commit.
