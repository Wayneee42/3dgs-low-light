# Stage 5 Explicit P_r Migration

Stage 5 is now split into two ablation-friendly sub-stages:

- `stage5a`: explicit `P_r` head only
- `stage5b`: joint `D_r + P_r`

Both stages keep the official Blender data layout and bind priors by `frame_key`.
They no longer use the deprecated "CIConv(rendered RGB)" structure-loss path.

## Offline structure prior extraction

Use:

```powershell
python tools/extract_structure_prior.py <scene_root> --device cuda --skip-existing
```

Outputs go to:

```text
scene_root/auxiliaries/structure/<frame_key>.png
```

## Stage selection

- For `P_r` only ablation, use `config/stage5a/*.yaml`
- For joint `D_r + P_r`, use `config/stage5b/*.yaml`

The old `config/stage5/` configs are retained only as historical reference and are no longer the recommended stage-5 path.
