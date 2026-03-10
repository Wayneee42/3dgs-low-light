# Stage 1 Data Protocol

Stage 1 turns the previous ad-hoc Blender loader into an explicit data protocol for the whole migration.

## What changed

The loader in [core/data/blender.py](D:/github/3dgs-low-light/3DRR_codebase/core/data/blender.py) now treats the official Blender dataset as the only primary source of truth and exposes a stable sample schema.

Each sample now returns:

- `transforms`
- `images` when `load_images=True`
- `infos.frame_key`
- `infos.frame_name`
- `infos.frame_stem`
- `infos.split`
- `infos.relative_path`
- `low_light_image` as an optional field
- `depth` as an optional field
- `structure` as an optional field

The optional fields default to `None` and do not block training if they are absent.
`prior` is still exposed as a legacy alias for `structure` so older experiments do not break.

## Frame binding rule

All auxiliary assets must be bound by official frame key, never by directory order.

The frame key format is:

```text
<split>_<frame_stem>
```

Examples:

- `train_0001`
- `val_0017`
- `test_0036`

This key is generated from the official `file_path` inside `transforms_*.json`.

## Auxiliary directory convention

Optional modalities are resolved from:

```text
scene_root/
|- auxiliaries/
|  |- lowlight/
|  |  \- train_0001.png
|  |- depth/
|  |  \- train_0001.png
|  \- structure/
|     \- train_0001.png
```

The default root is `auxiliaries/`, configurable by `AUXILIARY_DIR` in the dataset config.
For backward compatibility, `auxiliaries/prior/` is also accepted as a legacy alias for `structure/`.

## Camera convention notes

The dataset metadata now records two explicit conventions:

- `camera_convention = blender_nerf_synthetic_c2w_opengl`
- `renderer_camera_convention = opencv_w2c_after_yz_flip`

This matches the conversion already implemented in [core/model/simple_3dgs.py](D:/github/3dgs-low-light/3DRR_codebase/core/model/simple_3dgs.py).

## Validator

Use the stage-1 validator before adding new auxiliary modalities:

```powershell
python tools/validate_blender_dataset.py dataset/Development/lowlight_development/development/Chocolate
```

It checks:

- required intrinsics in `transforms_*.json`
- required per-frame keys
- training image existence
- optional auxiliary alignment by `frame_key`

## Output naming fix

Render outputs no longer use names like `test_0031.JPG.png`.
They now use the normalized frame key:

- `test_0031.png`
- `test_0032.png`
