# Stage 3 Modular Loss Interface

Stage 3 converts the stage-2 hard-coded low-light losses into a pluggable interface so later priors can be added without rewriting the training loop.

## What changed

The training script [train.py](D:/github/3dgs-low-light/3DRR_codebase/train.py) no longer directly assembles individual losses inline.
It now uses a modular loss builder from [core/losses](D:/github/3dgs-low-light/3DRR_codebase/core/losses).

## New package

- [core/losses/__init__.py](D:/github/3dgs-low-light/3DRR_codebase/core/losses/__init__.py)
- [core/losses/modules.py](D:/github/3dgs-low-light/3DRR_codebase/core/losses/modules.py)
- [core/losses/builder.py](D:/github/3dgs-low-light/3DRR_codebase/core/losses/builder.py)

## Registered modules

Currently implemented and enabled on demand:

- `RGBReconstructionLoss`
- `LowLightConsistencyLoss`
- `ExposureControlLoss`

Registered as stage-3 interfaces but intentionally not implemented yet:

- `DepthPriorLoss`
- `StructurePriorLoss`

If one of the prior losses is enabled now, the code raises a clear `NotImplementedError` instead of silently doing the wrong thing.

## Config structure

Stage 3 introduces explicit `PRIORS` and `EXPERIMENT` sections.

```yaml
PRIORS:
  DEPTH:
    ENABLED: false
    WEIGHT: 0.0
  STRUCTURE:
    ENABLED: false
    WEIGHT: 0.0

EXPERIMENT:
  STAGE: stage3
```

This keeps the training loop stable while making stage-4 and stage-5 integration straightforward.

## Run example

```powershell
python train.py -c config/stage3/chocolate.yaml
```

Outputs will go to:

```text
outputs/stage3/Chocolate/
```
