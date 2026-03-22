import torch

from .modules import (
    DepthPriorLoss,
    ExposureControlLoss,
    IlluminationRegularizationLoss,
    LowLightConsistencyLoss,
    MultiViewReprojectionLoss,
    ReconstructionLoss,
    RGBReconstructionLoss,
    StructurePriorLoss,
)



def _cfg_get(cfg, key, default):
    if cfg is None:
        return default
    try:
        return getattr(cfg, key)
    except AttributeError:
        return default



def build_loss_modules(meta_cfg, model_cfg):
    loss_cfg = _cfg_get(meta_cfg, "LOSS", None)
    priors_cfg = _cfg_get(meta_cfg, "PRIORS", None)
    depth_cfg = _cfg_get(priors_cfg, "DEPTH", None)
    structure_cfg = _cfg_get(priors_cfg, "STRUCTURE", None)
    multiview_cfg = _cfg_get(priors_cfg, "MULTIVIEW", None)

    lambda_ssim = float(_cfg_get(loss_cfg, "LAMBDA_SSIM", model_cfg.LAMBDA_SSIM))
    lambda_reconstruction = float(_cfg_get(loss_cfg, "LAMBDA_RECONSTRUCTION", 0.0))
    has_reconstruction = lambda_reconstruction > 0.0

    modules = [
        RGBReconstructionLoss(
            lambda_ssim=lambda_ssim,
            name="rgb_base" if has_reconstruction else "rgb",
            input_key="rgb_base_hwc" if has_reconstruction else "rendered",
            target_key="supervision_hwc",
        ),
        ReconstructionLoss(
            lambda_ssim=lambda_ssim,
            weight=lambda_reconstruction,
            start_step=int(_cfg_get(loss_cfg, "RECON_START_STEP", 0)),
            input_key="recon_hwc",
            target_key="proxy_target_hwc",
        ),
        IlluminationRegularizationLoss(weight=float(_cfg_get(loss_cfg, "LAMBDA_ILLUM_REG", 0.0))),
        LowLightConsistencyLoss(weight=float(_cfg_get(loss_cfg, "LAMBDA_LOW_LIGHT", 0.0))),
        ExposureControlLoss(weight=float(_cfg_get(loss_cfg, "LAMBDA_EXPOSURE", 0.0))),
    ]

    if bool(_cfg_get(depth_cfg, "ENABLED", False)):
        modules.append(
            DepthPriorLoss(
                weight=float(_cfg_get(depth_cfg, "WEIGHT", 0.0)),
                start_step=int(_cfg_get(depth_cfg, "START_STEP", 0)),
                end_step=_cfg_get(depth_cfg, "END_STEP", None),
                ramp_up_steps=int(_cfg_get(depth_cfg, "RAMP_UP_STEPS", 0)),
                ramp_down_steps=int(_cfg_get(depth_cfg, "RAMP_DOWN_STEPS", 0)),
                start_scale=float(_cfg_get(depth_cfg, "START_SCALE", 1.0)),
                end_scale=float(_cfg_get(depth_cfg, "END_SCALE", 0.0)),
                global_weight=float(_cfg_get(depth_cfg, "GLOBAL_WEIGHT", 1.0)),
                local_weight=float(_cfg_get(depth_cfg, "LOCAL_WEIGHT", 1.0)),
                box_size=int(_cfg_get(depth_cfg, "BOX_SIZE", 128)),
                sample_ratio=float(_cfg_get(depth_cfg, "SAMPLE_RATIO", 0.5)),
            )
        )
    if bool(_cfg_get(structure_cfg, "ENABLED", False)):
        modules.append(
            StructurePriorLoss(
                weight=float(_cfg_get(structure_cfg, "WEIGHT", 0.0)),
                start_step=int(_cfg_get(structure_cfg, "START_STEP", 0)),
            )
        )
    if bool(_cfg_get(multiview_cfg, "ENABLED", False)):
        modules.append(
            MultiViewReprojectionLoss(
                weight=float(_cfg_get(multiview_cfg, "WEIGHT", 0.0)),
                start_step=int(_cfg_get(multiview_cfg, "START_STEP", 0)),
                end_step=_cfg_get(multiview_cfg, "END_STEP", None),
                ramp_up_steps=int(_cfg_get(multiview_cfg, "RAMP_UP_STEPS", 0)),
                ramp_down_steps=int(_cfg_get(multiview_cfg, "RAMP_DOWN_STEPS", 0)),
                start_scale=float(_cfg_get(multiview_cfg, "START_SCALE", 1.0)),
                end_scale=float(_cfg_get(multiview_cfg, "END_SCALE", 0.0)),
                sample_stride=int(_cfg_get(multiview_cfg, "SAMPLE_STRIDE", 4)),
                min_alpha=float(_cfg_get(multiview_cfg, "MIN_ALPHA", 0.2)),
                relative_depth_thresh=float(_cfg_get(multiview_cfg, "RELATIVE_DEPTH_THRESH", 0.05)),
                absolute_depth_thresh=float(_cfg_get(multiview_cfg, "ABSOLUTE_DEPTH_THRESH", 0.02)),
                eps=float(_cfg_get(multiview_cfg, "EPS", 1.0e-4)),
            )
        )

    return [module for module in modules if module.enabled]



def compute_loss_modules(modules, context):
    total_loss = torch.zeros((), device=context["rendered"].device, dtype=context["rendered"].dtype)
    logs = {}

    for module in modules:
        effective_weight = float(module.current_weight(context))
        raw_loss, extra_logs = module.compute(context)
        weighted_loss = raw_loss * effective_weight
        total_loss = total_loss + weighted_loss
        logs[module.name] = float(weighted_loss.detach().item())
        logs[f"{module.name}_weight"] = effective_weight
        for key, value in extra_logs.items():
            logs[f"{module.name}_{key}"] = float(value)

    logs["total"] = float(total_loss.detach().item())
    return total_loss, logs



def required_aux_heads(loss_modules):
    heads = []
    if any(module.name == "depth_prior" for module in loss_modules):
        heads.append("depth")
    if any(module.name == "structure_prior" for module in loss_modules):
        heads.append("prior")
    if any(module.name in {"reconstruction", "illum_reg"} for module in loss_modules):
        heads.append("illum")
    return tuple(heads)



def requires_depth_render(loss_modules):
    return "depth" in required_aux_heads(loss_modules)


def requires_geom_depth_render(loss_modules):
    return any(module.name == "multiview_reproj" for module in loss_modules)
