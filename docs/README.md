# Documentation guide

Choose the path that matches your goal. The compact dataset can be inspected with Python's standard library; simulator experiments require additional assets and historical environment reconstruction.

| Goal | Start here | What to expect |
|---|---|---|
| Read or download the dataset | [Release data card](../dataset/DATA_CARD.md) | JSONL schema, validation status, splits and limitations |
| Obtain academic dataset access | [Private acceptance procedure](DATA_ACCESS.md) | Provider agreement, project terms and signed acceptance before delivery |
| Understand the dataset | [Dataset overview](../DATASET_CARD.md) | Motivation, intended uses and generation context |
| Verify downloaded records | [CPU integrity check](../tools/verify_release.py) | Checksums, identifiers, parent mappings and split integrity |
| Understand the paper implementation | [Paper-to-code map](PAPER_CODE_MAP.md) | Research components mapped to included source files |
| Reconstruct an experiment | [Reproducibility status](REPRODUCIBILITY.md) | Available evidence, dependencies and missing external assets |
| Interpret navigation numbers | [Experiment recap](experiment-records/experiment-results-recap-20261014.md) | Separate cohorts, STOP rules, denominators and negative results |
| Generate new instructions | [Generation guide](generation.md) | Qwen-compatible endpoint, Habitat/HM3D and RAM setup |
| Inspect the video | [Navigation demo](NAVIGATION_DEMO.md) | Controller provenance, task identity, encoding and scope |
| Trace source history | [Provenance](PROVENANCE.md) | Adapted components and preserved source variants |
| Cite or check reuse conditions | [Citation](../CITATION.md), [license scope](LICENSE_SCOPE.md), and [notices](../NOTICE.md) | Scoped MIT code grant, limited academic dataset permission and third-party conditions |

## Getting help

Use [GitHub Issues](https://github.com/aimldlnlp/iden-vln/issues) for dataset, documentation, and reproducibility questions. Include the repository commit, task ID or source filename, split, Python/environment details, and a minimal command when relevant. Remove credentials and private asset paths before posting. Raw HM3D scenes and model weights must be obtained from their providers.

## Working with historical sources

Keep frozen result directories and source variants intact. Use a new output directory when adapting experiment scripts. Equal filenames do not establish equal source versions, and equal update counts do not establish equal sample exposure. See each run's frozen protocol before comparing results.
