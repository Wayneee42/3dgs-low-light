# Stage 5a Explicit P_r Head

Stage 5a introduces an explicit rendered structure prior head `P_r` without enabling the depth prior head.

## What changed

- each Gaussian keeps a scalar `prior_feat`
- rasterization produces `prior_aux` / `P_r`
- `StructurePriorLoss` supervises `P_r` directly against the offline structure map
- RGB is not passed through CIConv inside the loss

## Config behavior

Use `config/stage5a/*.yaml`.
These configs set:

```yaml
PRIORS:
  DEPTH:
    ENABLED: false
  STRUCTURE:
    ENABLED: true
    WEIGHT: 0.01
    START_STEP: 5000
    INVARIANT: 'W'
    KERNEL_SIZE: 3
    SCALE: 0.8
```

## Missing prior policy

Missing structure priors do not stop training.
The structure loss is skipped for that sample and `structure_prior_available` logs as `0`.



