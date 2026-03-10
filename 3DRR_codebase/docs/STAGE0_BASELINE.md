# Stage 0 Baseline Freeze

## Goal

Stage 0 freezes `3DRR_codebase` as the only baseline entrypoint for NTIRE 2026 preparation. The target is not to improve the model yet. The target is to make the official Blender scenes runnable, repeatable, and easy to compare before any LITA-GS idea is migrated.

## Scope

Stage 0 includes only three things:

1. Fix the official dataset paths in dedicated stage-0 configs.
2. Verify that every official scene matches the expected Blender layout.
3. Standardize how baseline runs are recorded.

Stage 0 does not include:

- low-light method changes
- depth priors
- structure priors
- denoiser migration
- any COLMAP-based replacement of official camera metadata

## Actual official layout in this repo

The repo contains two different official layouts, and stage 0 now freezes both as-is instead of forcing them into one artificial shape.

### Development scenes

The four development scenes are:

- `Chocolate`
- `Cupcake`
- `GearWorks`
- `Laboratory`

Their actual structure is:

```text
scene_root/
├─ train/
├─ transforms_train.json
└─ transforms_test.json
```

This matches the current `3DRR_codebase/train.py` behavior:

- training reads `train/` + `transforms_train.json`
- validation previews reuse the first views from `transforms_test.json`
- final render-only evaluation also uses `transforms_test.json`

### Validation reference scene

`BlueHawaii` in the validation package has the full three-split structure:

```text
scene_root/
├─ train/
├─ val/
├─ test/
├─ transforms_train.json
├─ transforms_val.json
└─ transforms_test.json
```

This scene is kept as a reference config in stage 0, but the main frozen baseline for migration work remains the four development scenes.

## Official stage-0 configs

| Scene | Track | Config | Data path |
| --- | --- | --- | --- |
| Chocolate | development | `config/stage0/chocolate.yaml` | `dataset/Development/lowlight_development/development/Chocolate` |
| Cupcake | development | `config/stage0/cupcake.yaml` | `dataset/Development/lowlight_development/development/Cupcake` |
| GearWorks | development | `config/stage0/gearworks.yaml` | `dataset/Development/lowlight_development/development/GearWorks` |
| Laboratory | development | `config/stage0/laboratory.yaml` | `dataset/Development/lowlight_development/development/Laboratory` |
| BlueHawaii | validation reference | `config/stage0/bluehawaii.yaml` | `dataset/Validation/lowlight_validation/validation/BlueHawaii` |

All stage-0 configs now use:

```yaml
CHECKPOINT_STEPS: [7000, 30000]
```

So baseline training only saves two checkpoints by default.

## Validation command

Run this first:

```powershell
python tools/stage0_verify.py
```

This checks:

- every stage-0 config exists
- `DATASET.NAME` matches the official scene name
- `DATASET.DATA_PATH` points to the official in-repo Blender dataset path
- development scenes have `train/ + transforms_train.json + transforms_test.json`
- `BlueHawaii` has `train/ + val/ + test/ + transforms_train/val/test.json`

## Training commands

Run baseline training from the `3DRR_codebase` root.

```powershell
python train.py -c config/stage0/chocolate.yaml
python train.py -c config/stage0/cupcake.yaml
python train.py -c config/stage0/gearworks.yaml
python train.py -c config/stage0/laboratory.yaml
```

Optional validation-reference run:

```powershell
python train.py -c config/stage0/bluehawaii.yaml
```

## Output contract

Each run keeps the default output structure:

```text
outputs/<experiment>/<timestamp>/
├─ config.yaml
├─ step_7000.pt
├─ step_30000.pt
├─ examples/
└─ test/
```

## Eval command

`eval.py` loads a specific checkpoint path and renders the test set again.

```powershell
python eval.py -w outputs/<experiment>/<timestamp>/step_30000.pt
```

For the stage-0 training schedule, use `step_30000.pt` as the final checkpoint.

## Acceptance criteria

Stage 0 is complete only when all of the following are true:

1. `python tools/stage0_verify.py` passes.
2. At least one development scene has been trained end-to-end with a stage-0 config.
3. Baseline results are recorded in `docs/STAGE0_BASELINE_RECORD.md`.
4. No run depends on `LITA-GS` code, COLMAP folders, or unofficial scene restructuring.
