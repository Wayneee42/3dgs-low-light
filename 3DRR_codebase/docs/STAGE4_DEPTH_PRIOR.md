# Stage 4 Depth Prior Integration

Stage 4 migrates Marigold-based monocular depth priors into the official 3DRR training loop.

## What changed

Stage 4 adds three pieces:

1. depth extraction from official Blender-format scenes
2. optional depth loading through the official `auxiliaries/depth/` convention
3. a trainable `DepthPriorLoss` plugged into the stage-3 loss interface

## Marigold extraction

Use the new script:

```powershell
python tools/extract_marigold_depth.py <scene_root> --checkpoint <marigold_checkpoint_dir> --device cuda --skip-existing
```

Example:

```powershell
python tools/extract_marigold_depth.py dataset/Validation/lowlight_validation/validation/BlueHawaii --checkpoint /path/to/marigold-depth-lcm-v1-0 --device cuda --skip-existing
```

The script writes depth priors to:

```text
scene_root/auxiliaries/depth/<frame_key>.png
```

## Training integration

The depth prior is now a real loss module registered in [core/losses/modules.py](D:/github/3dgs-low-light/3DRR_codebase/core/losses/modules.py).

It uses:

- a global Pearson depth correlation term
- a local Pearson depth correlation term

This follows the spirit of the LITA-GS depth supervision while keeping the official Blender data layout unchanged.

## Stage-4 configs

Use the configs under `config/stage4/`.

Example:

```powershell
python train.py -c config/stage4/chocolate.yaml
```

These configs enable:

```yaml
PRIORS:
  DEPTH:
    ENABLED: true
```

and keep structure priors disabled for now.

## Important constraint

If `PRIORS.DEPTH.ENABLED` is true but the current sample has no depth prior under `auxiliaries/depth/`, training will fail with a clear error.
This is intentional, so stage-4 runs do not silently skip missing depth supervision.
