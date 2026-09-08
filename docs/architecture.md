# Architecture

quaggAIeval is three independently installable subprojects that share a common goal —
measuring quagga mussel (_Dreissena bugensis_) coverage from lakebed photographs — but
have almost no code or dependency overlap with each other. Each has its own `pyproject.toml`
and its own virtual environment; none of them import from another.

```
                        ┌───────────────────────┐
                        │       scripts/          │
                        │  (research pipeline)     │
                        │                           │
                        │  preprocessing/  ──────┐  │
                        │  training/              │  │
                        │  evaluation/  ◄─────────┘  │
                        └────────────┬──────────────┘
                                     │
                        trained checkpoint (.pt)
                                     │
                                     ▼
                        ┌───────────────────────────┐
                        │    quaggaieval_desktop/     │
                        │   (field-use desktop app)    │
                        └───────────────────────────┘

           scripts/evaluation/*  ──writes──►  comparison_manifest.json
                                                        │
                                                        ▼
                                                 eval_viewer/
                                          (browser-based run inspector)
```

## The three subprojects

**`scripts/`** — dataset preprocessing, SAM2 fine-tuning, and evaluation/comparison, as an
installable package (`quaggai_scripts`) split into `preprocessing/`, `training/`, and
`evaluation/`. Headless, CLI-driven, meant to run on a machine with a GPU. Its evaluation
scripts write a `comparison_manifest.json` per run.

**`quaggaieval_desktop/`** — the field-use product: a wxPython desktop app that takes a
trained SAM2 checkpoint (produced by `scripts/training/`) and lets someone segment, calibrate,
and measure coverage in their own photos interactively. Packaged into a standalone `.app` via
PyInstaller for distribution to people who don't have a Python environment set up at all.

**`eval_viewer/`** — a small, dependency-free static HTML/JS page for browsing a
`comparison_manifest.json` produced by `scripts/evaluation/visualize_iterative_refinement.py`
(or `visualize_comparison.py`). No build step, no server-side code — point it at a manifest via
a `?data=` query parameter and it renders everything client-side.

## Why separate environments

The three subprojects have almost disjoint dependency footprints — a GUI toolkit (wxPython) vs.
headless training/inference (torch, hydra, pycocotools) vs. nothing at all (static JS). Sharing
one environment across all three would mean every contributor installs PyTorch just to touch the
viewer, or wxPython just to run a preprocessing script. Each subproject's own `pyproject.toml` and
venv keeps installs scoped to what that piece actually needs.

## SAM2 dependency

Both `scripts/` and `quaggaieval_desktop/` depend on Meta's [SAM2](https://github.com/facebookresearch/sam2).
Rather than depending on upstream directly, both pin to a specific commit on a fork
(`mauriziopiu/sam2`, tag `quaggai-v1`) via a `git+https` reference in `dependencies`. This
protects the build against upstream changes, force-pushes, or removal, since no active
development is expected against SAM2 itself — the fork exists purely to freeze a known-working
commit.
