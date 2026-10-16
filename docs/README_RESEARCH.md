# README and repository organization — research and decision record

Reviewed 16 October 2026. This record supports a bounded documentation revision, not a systematic review of every research repository or a new scientific experiment.

## Charter

Goal: make the existing IDEN-VLN companion easy to understand, inspect and cite, and publish a playable navigation demonstration without changing scientific results. Scope: research software and dataset publication. Object: this repository's entry points and documentation. Subjects: README information order, data quickstart, source navigation, citation metadata and video attachments. Type: descriptive, because the project and evidence are understood and the task is to select an implementation structure. Knowledge depth: deep for the project contracts; working for GitHub publishing, verified against official documentation.

Questions: Q1 what must a new visitor find first? Q2 how should data, reproducibility and source history be separated? Q3 how can the approved MP4 become a GitHub player? Q4 which claims and reuse conditions must remain explicit?

## Canonical sources

| Rank | Source | Purpose |
|---|---|---|
| 1 | Frozen release manifest, checksums, result summaries and source map in this repository | Dataset counts, actual availability and scientific boundaries |
| 1 | [GitHub: About READMEs](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes) | Platform guidance on purpose, getting started, help and maintainers |
| 1 | [GitHub: About citation files](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files) | Machine-readable CITATION.cff and repository citation |
| 1 | [GitHub: Attaching files](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/attaching-files) and [CLI attachment guide](https://docs.github.com/en/github-cli/github-cli/attaching-files-with-github-cli) | Supported video formats, upload access and official upload route |
| 1 | [Gebru et al., Datasheets for Datasets, v8](https://arxiv.org/html/1803.09010v8) | Composition, collection, uses, distribution and maintenance documentation |
| 1 | [LH-VLN upstream README](https://github.com/HCPLab-SYSU/LH-VLN/blob/master/README.md) | Domain example and attribution, not a universal template |
| 1 | [HM3D provider](https://aihabitat.org/datasets/hm3d/) and [Habitat terms](https://aihabitat.org/terms-of-use/) | Separate third-party access conditions; no blanket project license inferred |
| 3 | Git checkout, authenticated repository metadata and final MP4 receipt | Current branch, access, file content and video identity |

Official pages and the datasheets full HTML were inspected. The upstream README was used only for the generation/setup pattern. No repository popularity ranking is treated as quality evidence.

## Findings ledger

| ID | Finding | Evidence | Answers |
|---|---|---|---|
| F1 | README should explain purpose, usefulness, getting started, support and maintenance | GitHub README guide | Q1 |
| F2 | The compact release contains 2,500 tasks, 3,291 segments and 155 scenes; it can be verified without ML dependencies | release manifest; tools/verify_release.py | Q1, Q2 |
| F3 | Dataset documentation needs composition, collection, intended use, distribution and maintenance context | Datasheets for Datasets, sections 3.1–3.7 | Q2, Q4 |
| F4 | Existing research snapshots retain absolute paths and missing external dependencies | docs/REPRODUCIBILITY.md; docs/PROVENANCE.md | Q2, Q4 |
| F5 | CITATION.cff enables GitHub's citation interface; repository citation need not invent a paper DOI | GitHub citation guide; existing CITATION.md | Q1, Q4 |
| F6 | MP4 is supported and H.264 is recommended; the approved video is 3,389,508 bytes and H.264 | GitHub attachment guide; video receipt and SHA256 | Q3 |
| F7 | Official CLI supports media attachment to issue/PR bodies and comments | CLI attachment guide; gh 2.102.0 help | Q3 |
| F8 | The target repository is private and existing scientific/source publication is ahead of the previous local checkout | Authenticated repository metadata; fast-forward from 1e45cb2 to 0f4a54a | Q2, Q3 |
| F9 | Teacher replay, training fit and different STOP contracts cannot replace historical learned-policy results | Included protocols and experiment recap | Q4 |
| F10 | Third-party assets and inherited code do not acquire a new license from a documentation change | Existing NOTICE.md and provider terms | Q4 |

## Scope filter

No universal best folder structure is asserted. A full source-tree reorganization would break historical path assumptions and provenance references without improving the requested reader journey. Existing code, datasets, numerical results and source variants remain in place. The documentation index supplies navigation instead.

GitHub Pages is an optional hosting mechanism, not a prerequisite for an in-README video. It was not selected because the user requested the repository and a player, and the repository is private. No visibility change or additional hosting deployment is inferred.

## Object filter

The prior statement that this was only a generation repository was superseded by commit 0f4a54a, which already included the compact release and research sources. The revision builds on that commit. It does not promise clean-machine simulator reproduction, rename the published dataset, grant an unverified license, or assert a venue/DOI.

Video completion is explicitly teacher/scorer completion in a selected example. The original instruction remains code mixed. The English interface and LM Roman typography are presentation choices, not dataset annotations or policy performance.

## Conclusions

1. Put dataset access and a small standard-library example before expensive installation; preserve deeper setup separately (F1–F4).
2. Keep a short README and a goal-oriented docs index, retaining frozen source/result paths (F1, F4, F9).
3. Add CITATION.cff consistent with the existing repository citation and preserve upstream attribution (F5, F10).
4. Use a repository media attachment for the approved H.264 video and retain an exact downloadable copy and provenance (F6–F9). Verify the actual rendered player after upload rather than promising it from file extension alone.
5. Preserve negative results, unresolved licensing and access conditions in reader-facing documentation (F8–F10).

## Next steps / Risks

Acceptance checks: compact integrity verifier passes; README example executes against the packaged schema; relative documentation/media links resolve; CFF matches the existing author/title; video bytes match the approved SHA256; remote README contains a controls-enabled video element; pushed commit matches remote main. Browser playback cannot be claimed from HTML inspection alone and must be reported separately if unavailable.

Existing runtime dependency pins, simulator path restoration, license clearance and comprehensive secret detection remain outside this documentation change. No new training or simulator benchmark is required. Private attachments require repository access. See the publication verification record for executed checks and any remaining platform limitations.
