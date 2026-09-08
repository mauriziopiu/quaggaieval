# quaggAIeval

Detecting and measuring quagga mussel (_Dreissena bugensis_) coverage in lakebed
photographs as captured by BIS systems, using interactive segmentation based on a
re-trained variant of the [SAM2 image segmentation model by FAIR](https://github.com/facebookresearch/sam2).

## What's here

This repository consists of three independently installable subprojects, refer to [`docs/architecture.md`](docs/architecture.md)
for how they relate and why they're kept separate.

| Path                                           | What it is                                                                                                                     |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| [`scripts/`](scripts/)                         | Dataset preprocessing, SAM2 fine-tuning, and evaluation/comparison scripts. Headless, CLI-driven.                              |
| [`quaggaieval_desktop/`](quaggaieval_desktop/) | Field-use desktop app: SAM2-assisted segmentation, area calibration, and coverage evaluation, packaged as a standalone `.app`. |
| [`eval_viewer/`](eval_viewer/)                 | Dependency-free browser-based viewer for inspecting `scripts/` evaluation runs.                                                |
| [`examples/`](examples/)                       | Small sample dataset for trying the pipeline end to end. _(not yet populated)_                                                 |
| [`docs/`](docs/)                               | Architecture and setup documentation.                                                                                          |

## Getting started

Each subproject has its own environment and its own detailed README. Start there for actual
usage. Overview:

```bash
# Research pipeline (dataset prep, training, evaluation)
cd scripts
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
python -m quaggai_scripts.preprocessing.batch_preprocess --config configs/batch_config.yaml

# Desktop app
cd quaggaieval_desktop
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
python -m quaggaieval_desktop

# Evaluation viewer (no install needed)
cd eval_viewer
python3 -m http.server 8000
```

Both `scripts/` and `quaggaieval_desktop/` depend on a pinned SAM2 fork
(`mauriziopiu/sam2@quaggai-v1`) rather than upstream `main`. Please refer to
[`docs/architecture.md`](docs/architecture.md#sam2-dependency) for additional details on this choice.

## License and Usage Rights

All source code is provided "as-is" under the GNU AGPLv3 License.

### Citing QuaggAIeval

If you use the QuaggAIeval software or the underlying image segmentation model in your research, please cite this project in the following way:

- Title: QuaggAIeval - Towards an Automated Coverage Evaluation of the Invasive Quagga Mussel using Benthic Imaging Systems
- Authors: Maurizio Piu, Christoph Joachim Paulus, Thomas Müller, Alexandra Anh-Thu Weber, Piet Spaak
- Year: 2026
