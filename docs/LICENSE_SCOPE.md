# License scope and source review

Updated 17 October 2026, based on repository snapshot `a2d5232adebcf0cab92a77c6201b9b94ff537c0f` and the owner's instruction to permit reuse of his contributions.

## Applied grant

- **MIT:** original IDEN-VLN code contributions controlled by Zinadine Zidan Alsyahana in `research/**/*.py` and `tools/verify_release.py`.
- Mixed-provenance content inside those files remains outside that grant to the extent owned by others. The source inventory and exclusions remain necessary; importing a package does not relicense the package.
- No source headers or frozen code bytes were changed. The path/hash inventory in `licensing_manifest.json` makes the scope explicit without rewriting experimental snapshots.

## Exclusions and outstanding rights

| Material | Treatment |
|---|---|
| Root NavGen generator, `habitat_base/`, inherited prompts and modifications | Upstream material excluded from the MIT grant. No clear top-level license was found in the inspected LH-VLN snapshot `3492080382aba031e9d82a4f630c2b1cdf57df81`; confirm permission before relying on redistribution rights. |
| `recognize_anything/` | Existing Apache-2.0 distribution text plus third-party notices retained. Its license does not cover all NavGen code. |
| `scene/`, source-derived dataset fields and split/curation package | Outside MIT; provider terms and derived-information questions remain. No whole-dataset CC BY grant. |
| English/Indonesian wordlists and language evaluator | Outside the new grant pending finer provenance review. Absence from upstream is not proof of ownership. |
| `assets/`, including navigation MP4 and construction schematic | No new license for scene imagery, robot/artwork components, or other third-party visual elements. Media availability is not rights clearance. |
| `results/`, trained weights/caches, original source trajectories | Not licensed by the code grant. Training on provider data does not erase provider conditions. Weights and raw trajectories are not included. |

## Why the proposed dataset license was deferred

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode.en) grants only the rights the licensor is authorized to grant, and licensors should identify excluded material. Its standard license is not modified here.

The [Matterport Academic Use Agreement](https://matterport.com/legal/matterport-end-user-license-agreement-academic-use-model-data), sections 2.2 and 2.4, covers academic/non-commercial use and derived information. The [HM3D-Semantics page](https://aihabitat.org/datasets/hm3d-semantics/) likewise describes academic non-commercial availability. These are concrete constraints on the earlier suggestion to put the whole compact dataset under CC BY. A copyright grant for original work cannot be represented as permission for every source-derived component or as unrestricted commercial dataset use.

The wording and split metadata have not been separated into a demonstrably independent, rights-cleared annotation release. No blanket CC BY or CC BY-NC label is applied in place of that missing review. No request was sent to upstream authors or providers.

## Authoritative references checked

- [MIT standard license, Open Source Initiative](https://opensource.org/license/mit).
- [Creative Commons BY 4.0 legal code](https://creativecommons.org/licenses/by/4.0/legalcode.en), scope and licensor considerations.
- [Matterport Academic Use Agreement](https://matterport.com/legal/matterport-end-user-license-agreement-academic-use-model-data), sections 2.2, 2.4(a)-(c).
- [HM3D-Semantics provider](https://aihabitat.org/datasets/hm3d-semantics/).
- [LH-VLN upstream](https://github.com/HCPLab-SYSU/LH-VLN), inspected Git snapshot and preserved source notices.

## Next action for a broader release

Resolve upstream code permission and the status of source-derived annotations/media. If clearance establishes separately licensable original data, apply the chosen dataset license in a new version, update metadata/checksums, and retain the exclusions. The current change does not make the repository public, establish full legal clearance, or alter scientific results.
