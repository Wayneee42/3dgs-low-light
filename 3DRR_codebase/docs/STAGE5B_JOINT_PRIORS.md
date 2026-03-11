# Stage 5b Joint D_r + P_r

Stage 5b enables both explicit auxiliary heads while keeping staged activation to reduce early training interference.

## What changed

- `D_r` remains active from stage 4
- `P_r` is added as a second explicit auxiliary head
- the two heads are rendered independently from RGB and supervised independently

## Config behavior

Use `config/stage5b/*.yaml`.
These configs set:

```yaml
PRIORS:
  DEPTH:
    ENABLED: true
    WEIGHT: 0.02
    START_STEP: 5000
  STRUCTURE:
    ENABLED: true
    WEIGHT: 0.01
    START_STEP: 10000
```

## Validation goal

This stage exists only to test whether the joint auxiliary heads outperform `stage5a` and the stage-2 baseline.
If not, it should be treated as an ablation result, not as the default mainline.
