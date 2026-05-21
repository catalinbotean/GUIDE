# guide

Uncertainty-guided diffusion priors for monocular dynamic Gaussian splatting.

Research codebase targeting CVPR 2027. See [PLAN.md](PLAN.md) for the phase-gated roadmap.

## Status

**Phase 0 — repository scaffolding.** No training code, no models, no data yet.

## Setup

Requires Python 3.11, an NVIDIA GPU with CUDA 12.4 drivers (for later phases),
and [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
make install      # uv sync --extra dev
make test         # run unit tests (currently smoke only)
make lint         # ruff check
make format       # ruff format + autofix
```

PyTorch is pulled from `https://download.pytorch.org/whl/cu124` via `[tool.uv.sources]`
in `pyproject.toml`. To target a different CUDA build (e.g. `cu121`, `cpu`) edit the
`tool.uv.index` URL there.

## Layout

```
.
├── PLAN.md              # phase-gated project plan
├── pyproject.toml       # uv-managed deps, ruff & pytest config
├── Makefile             # install / test / lint / format
├── configs/             # YAML configs (one per experiment)
├── scripts/             # entry points (download data, run training, eval)
├── notebooks/           # exploratory work
├── tests/               # pytest unit tests
└── src/guide/
    ├── data/            # dataset loaders, preprocessing
    ├── models/          # gaussian splatting, uncertainty, diffusion prior
    ├── training/        # training loops, losses, optimizers
    ├── eval/            # metrics, evaluation harness
    └── utils/           # shared helpers
```

## Choices made in Phase 0

PLAN.md left several details to my judgment. I picked:

- **Python**: 3.11 (best current CUDA wheel coverage; 3.12 still has gaps for some
  gsplat/diffusers builds).
- **PyTorch**: 2.5.x on CUDA 12.4.
- **License**: Apache-2.0 (matches Shape-of-Motion).
- **Package layout**: `src/guide/{data,models,training,eval,utils}/` so imports read
  `from guide.models.uncertainty import ...`. PLAN.md showed bare `src/` subdirs;
  this is the same intent with a proper package name.
- **Configs**: plain YAML + `pyyaml`. Skipped Hydra for now — too much ceremony for
  Phase 0–1. Trivial to add later if/when we need composition or sweeps.
- **Logging**: TensorBoard. Skipped `wandb` — defer until we actually need experiment
  tracking across machines.
- **Lint**: `ruff` (check + format). No `mypy` yet — research code moves too fast for
  strict typing to pull its weight at this stage.

## Open questions still on the table (from chat)

- Compute target — local 4090 vs. remote A100/H100 — likely affects Phase 3+.
- Shape-of-Motion integration strategy: submodule (per PLAN.md) vs. fork into the org
  so we can cleanly add hooks. To be decided at the start of Phase 1.
- Choice of first "real" DyCheck scene for Phase 2/4 evaluation (PLAN.md uses `apple`
  for plumbing only).

## License

Apache-2.0 — see [LICENSE](LICENSE).
