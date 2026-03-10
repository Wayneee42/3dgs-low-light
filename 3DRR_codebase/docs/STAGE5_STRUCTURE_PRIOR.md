# Stage 5 Structure Prior Integration

Stage 5 migrates the structure-prior supervision idea from LITA-GS without inheriting its dataset-directory assumptions.

## What changed

Stage 5 adds three pieces:

1. a reusable CIConv-based structure extractor in [core/losses/structure_prior.py](D:/github/3dgs-low-light/3DRR_codebase/core/losses/structure_prior.py)
2. an offline extraction script that binds outputs by official `frame_key`
3. an optional `StructurePriorLoss` that supervises rendered RGB through the extracted structure map

## Offline extraction

Use the new script:

```powershell
python tools/extract_structure_prior.py <scene_root> --device cuda --skip-existing
```

Example:

```powershell
python tools/extract_structure_prior.py dataset/Development/lowlight_development/development/Chocolate --device cuda --skip-existing
```

The script writes structure priors to:

```text
scene_root/auxiliaries/structure/<frame_key>.png
```

The default extractor matches the LITA-GS recipe closely:

- invariant: `W`
- kernel size: `3`
- scale: `0.8`

## Training integration

The structure prior is now a real loss module registered in [core/losses/modules.py](D:/github/3dgs-low-light/3DRR_codebase/core/losses/modules.py).

This stage keeps the migration deliberately lightweight:

- no extra renderer branch is introduced
- the training loop reads only precomputed structure maps
- missing structure maps do not stop training; the structure loss is skipped for that sample

This follows the migration plan: keep the supervision idea, but decouple it from the original LITA-GS directory layout.

## Dataset convention

Preferred structure-prior location:

```text
scene_root/auxiliaries/structure/<frame_key>.png
```

Legacy compatibility:

```text
scene_root/auxiliaries/prior/<frame_key>.png
```

If both exist, `structure/` wins.

## Stage-5 configs

Use the configs under `config/stage5/`.

Example:

```powershell
python train.py -c config/stage5/chocolate.yaml
```

These configs keep depth priors enabled and additionally turn on:

```yaml
PRIORS:
  STRUCTURE:
    ENABLED: true
    WEIGHT: 0.05
    INVARIANT: 'W'
    KERNEL_SIZE: 3
    SCALE: 0.8
```

## Important constraint

Structure priors are treated as an ablation-friendly auxiliary signal.
If a scene or frame does not have a generated structure prior yet, training continues and logs `structure_prior=0` for that sample instead of failing.
