# Stage 4 Explicit D_r Head

Stage 4 migrates the depth prior into an explicit auxiliary rendering head instead of supervising gsplat geometric expected depth.

## What changed

Stage 4 now consists of three pieces:

1. offline depth extraction under `auxiliaries/depth/`
2. a per-Gaussian scalar `depth_feat` latent
3. an explicit rendered depth prior head `D_r` supervised by the offline depth map

## Training behavior

The RGB branch remains the primary reconstruction path.
Depth supervision is applied only to `depth_aux` / `D_r`.
This stage does not enable any structure head.

## Config behavior

Use `config/stage4/*.yaml`.
These configs set:

```yaml
PRIORS:
  DEPTH:
    ENABLED: true
    WEIGHT: 0.02
    START_STEP: 5000
  STRUCTURE:
    ENABLED: false
```

## Important constraint

If `PRIORS.DEPTH.ENABLED` is true and a training frame has no depth prior, training fails with a clear frame-key error.
This is intentional so stage-4 runs stay auditable.



