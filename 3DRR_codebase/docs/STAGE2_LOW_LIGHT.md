# Stage 2 Low-Light Training

Stage 2 migrates the lowest-risk low-light ideas from LITA-GS into the official 3DRR training loop without introducing depth, structure priors, or denoiser branches.

## Output structure

Stage-2 runs are grouped as:

```text
outputs/stage2/<scene>/<timestamp>/
```

Examples:

- `outputs/stage2/BlueHawaii/<timestamp>/`
- `outputs/stage2/Chocolate/<timestamp>/`

## What changed

The training loop in [train.py](D:/github/3dgs-low-light/3DRR_codebase/train.py) is now split conceptually into:

- data loading
- low-light target preparation
- rendering
- loss composition
- checkpoint / validation scheduling

## New modules

- [core/libs/augment.py](D:/github/3dgs-low-light/3DRR_codebase/core/libs/augment.py)
- [core/libs/losses.py](D:/github/3dgs-low-light/3DRR_codebase/core/libs/losses.py)

## Stage-2 augmentation

The new augmentation path is config-driven. The default stage-2 configs use:

```yaml
AUGMENTATION:
  ENABLED: true
  MODE: hybrid
```

`hybrid` means:

1. apply gamma brightening
2. match the image mean brightness toward a configurable target exposure

This is closer to the exposure-target idea used in LITA-GS than the previous hard-coded `gamma_augment(image, gamma=0.5)` baseline.

## Stage-2 losses

Stage 2 keeps the main RGB reconstruction loss, and adds two lightweight terms:

- `LAMBDA_LOW_LIGHT`: preserves normalized luminance structure relative to the original low-light observation
- `LAMBDA_EXPOSURE`: encourages the rendered image to approach the configured target exposure

The total loss is now:

```text
rgb_loss + lambda_low_light * low_light_consistency_loss + lambda_exposure * exposure_control_loss
```

## Config layout

New stage-2 configs are under `config/stage2/`.

Example:

```powershell
python train.py -c config/stage2/chocolate.yaml
```

## Ablation path

You can compare:

- stage 0: `config/stage0/*.yaml`
- stage 2: `config/stage2/*.yaml`

This gives you a clean baseline-to-low-light comparison before adding any depth or structure priors.
