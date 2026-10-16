# README and video publication checks

Date: 16 October 2026. Base: `0f4a54af900ac4b427abb091c97edc211942a544`. The clean generation checkout was fast-forwarded to the existing paper-companion publication before edits. Dataset records, split assignments, historical source snapshots and numerical results were not changed.

## Executed checks

- `python tools/verify_release.py`: passed for 2,500 tasks, 3,291 segments, 155 scenes, checksums, splits and parent mappings.
- The exact standard-library Python example in README executed successfully and returned scene-held-out task counts 2,019 / 247 / 234.
- Relative links in the revised README, documentation index, demo note and research decision record resolved locally.
- `CITATION.cff` passed the official Citation File Format 1.2.0 JSON Schema. An unsupported top-level `year` field was caught and removed before publication; no date, DOI or venue was invented.
- The downloadable MP4 matches the approved SHA256 `c00918afba53b2f104501c0111bdab91b35b3703cedd92db6cd9e36ed32a039f`.
- GitHub CLI 2.102.0 uploaded the MP4 through its official attachment flow. The portable official binary's checksum was verified; no GitHub token was printed or stored in project files.
- The completed [media record](https://github.com/aimldlnlp/iden-vln/issues/1) preserves the source caption and was closed as completed.
- GitHub's authenticated Markdown renderer produced exactly one `video` element with `controls` for the revised README.
- Downloading the rendered video's source returned 3,389,508 bytes with the exact approved SHA256. This verifies the uploaded media, not merely the local source.
- The approved local MP4 passed full-file FFmpeg decode and frame/layout inspection during video production.
- `git diff --check` passed before staging.

## Verification limits

The browser automation runtime could not start because its process security environment returned HRESULT 0x80070002. No manual or automated browser Play-button interaction is claimed. GitHub player markup and byte-for-byte media retrieval were verified through the platform API; browser-specific codec behavior remains a practical limit.

Repository metadata reported private visibility. This change does not change visibility, deploy Pages, grant new licenses or make promises about the lifetime of signed media links. The MP4 also remains in Git for a portable download independent of an attachment URL.

No simulator benchmark, policy training, clean-machine dependency installation, full license audit, or changes to the manuscript were performed. Documentation and machine-readable citation describe the existing repository and preserve its scientific limitations. The Git history identifies the published documentation commit; remote-branch and README checks are also recorded in the local publication receipts and delivery summary.
