# Project Plan: Hindsight — Uncertainty-Guided Diffusion Priors for Monocular Dynamic Gaussian Splatting

## Project Goal

Build a research codebase to develop a method for monocular dynamic scene reconstruction that uses per-Gaussian uncertainty to target where a frozen pretrained video diffusion model (Stable Video Diffusion) provides SDS-style guidance. Target submission: CVPR 2027.

## Working Principles

- Small, verifiable increments. Each task ends with something runnable.
- Test as you build. Write minimal unit tests for non-trivial functions.
- Commit frequently with clear messages.
- Stop and ask the user before any decision that significantly changes the architecture or adds a major dependency.
- Use `uv` for Python dependency management; never use bare `pip install`.
- Default to PyTorch 2.x with CUDA 12.x. Target a single 4090 (24GB VRAM).

## Phase 0: Repository Setup

1. Initialize the repo with this structure:

   ```
   .
   ├── pyproject.toml
   ├── README.md
   ├── .gitignore
   ├── configs/
   ├── src/
   │   ├── data/
   │   ├── models/
   │   ├── training/
   │   ├── eval/
   │   └── utils/
   ├── scripts/
   ├── tests/
   └── notebooks/
   ```
2. Set up `pyproject.toml` with `uv`. Initial dependencies: torch, torchvision, numpy, einops, hydra-core (for configs), wandb, tqdm, pillow, opencv-python, plyfile.
3. Add a `.gitignore` covering: `data/`, `checkpoints/`, `outputs/`, `wandb/`, `__pycache__/`, `.venv/`, `*.ply`.
4. Write a README explaining the project goal, setup instructions, and current status.
5. Add a `Makefile` with common commands: `make install`, `make test`, `make lint`.

**Stop after Phase 0 and confirm with the user that the structure looks good before continuing.**

## Phase 1: Reproduce Shape-of-Motion Baseline

Goal: get the existing Shape-of-Motion method running on one DyCheck scene, match the reported numbers within a few percent.

1. Clone Shape-of-Motion into `third_party/shape-of-motion/` as a git submodule.
2. Read its README and CLAUDE.md (if any). Document its training pipeline in `docs/baselines/shape-of-motion.md`: what inputs it expects, what outputs it produces, what each loss term does.
3. Write a script `scripts/download_dycheck.py` that downloads the DyCheck iPhone dataset to `data/dycheck/`. Start with just the `apple` scene to validate the pipeline.
4. Write `scripts/run_shape_of_motion.py` that wraps the Shape-of-Motion training pipeline for one scene with our config system.
5. Test on the `apple` scene end-to-end. This will need a GPU — confirm with the user before they run it on RunPod, and estimate time/cost.
6. After training, run the official DyCheck evaluation. Log PSNR, SSIM, LPIPS, and masked-dynamic-region metrics to wandb.
7. Document results in `docs/baselines/results.md`.

**Stop and confirm baseline numbers with the user before moving to Phase 2.**

## Phase 2: Per-Gaussian Uncertainty Estimation

Goal: implement and visualize the per-Gaussian uncertainty signals we'll use to target diffusion guidance.

1. In `src/models/uncertainty.py`, implement three uncertainty signals per Gaussian:
   - **Observation count**: number of input frames where the Gaussian's projected location is inside the frame and not occluded by other Gaussians.
   - **View diversity**: variance of viewing angles across frames the Gaussian was observed in.
   - **Photometric residual**: average rendering error at the Gaussian's projected location across input frames.
2. Each signal returns a tensor of shape `[N_gaussians]` normalized to `[0, 1]`.
3. Write `scripts/visualize_uncertainty.py` that loads a trained Shape-of-Motion checkpoint and renders the scene with Gaussians colored by each uncertainty signal. Save outputs to `outputs/uncertainty_vis/`.
4. Add unit tests in `tests/test_uncertainty.py` for each function on a tiny synthetic scene (e.g., 100 Gaussians, 5 cameras).
5. **Sanity check we'll discuss with the user**: do the high-uncertainty Gaussians visually correspond to regions where the baseline produces artifacts? If not, the uncertainty signals are wrong and we need to rethink.

**Stop and review visualizations with the user before Phase 3.**

## Phase 3: Frozen SVD Integration

Goal: load Stable Video Diffusion frozen, render short clips from the Gaussian scene, get an SDS gradient flowing back to Gaussian parameters.

1. In `src/models/svd_prior.py`, wrap a frozen Stable Video Diffusion 1.1 model. Use the `diffusers` library. Load in fp16 to fit in 24GB.
2. Implement an SDS loss following the standard formulation: noise the rendered clip, predict noise with SVD, compute the gradient. Reference: DreamFusion paper for the math, ProlificDreamer for stability improvements.
3. Write `src/training/render_clip.py`: render a short clip (16 frames) from the Gaussian scene along a parameterized novel camera trajectory.
4. Write a minimal test: load a trained Shape-of-Motion checkpoint, render a clip, compute SDS loss, take one gradient step, verify the loss decreases and no NaNs appear.
5. **Critical compute check before running**: SVD inference on 16 frames at modest resolution should fit in ~16GB. Confirm memory usage before running long training.

**Stop here and discuss with the user. SDS is notoriously unstable. We may need to spend significant time on this step.**

## Phase 4: Uncertainty-Targeted SDS

Goal: combine Phase 2 and Phase 3 — apply SDS gradients masked by uncertainty.

1. In `src/training/targeted_sds.py`, implement the masking: for each Gaussian, scale its SDS gradient by its uncertainty score (so high-uncertainty Gaussians receive strong guidance, low-uncertainty Gaussians receive little or none).
2. Implement a novel-view sampling strategy that prefers viewpoints where many high-uncertainty Gaussians become visible (compute visibility via projection of the uncertainty-weighted Gaussian cloud).
3. Integrate this into a fine-tuning loop: start from a trained Shape-of-Motion checkpoint, fine-tune with targeted SDS for N additional iterations.
4. Run on one scene. Compare against (a) no SDS fine-tuning, (b) uniform SDS fine-tuning, (c) our targeted SDS.
5. Inspect qualitatively — do disoccluded regions look better?

**Stop. This is the make-or-break moment. If targeted SDS doesn't visibly help, we re-plan.**

## Phase 5: Full Benchmark Evaluation

(Plan this phase with the user after Phase 4 succeeds.)

## Phase 6: Ablations

(Plan this phase with the user after Phase 5.)

## What to Ask the User About

- Before downloading any large dataset
- Before any RunPod training run estimated to cost >$5
- Before adding any new major dependency
- Before changing the architecture in a non-trivial way
- After completing each phase, before starting the next

## What to Avoid

- Do not modify Shape-of-Motion source files in `third_party/`. Treat them as read-only; build adapters in `src/` instead.
- Do not download datasets to the repo (use `data/`, gitignored).
- Do not commit checkpoints, wandb logs, or rendered outputs.
- Do not run multi-hour training without user confirmation of compute cost.
- Do not silently swap libraries or model versions.
