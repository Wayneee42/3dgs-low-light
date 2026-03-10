import torch

from .modules import (
    DepthPriorLoss,
    ExposureControlLoss,
    LowLightConsistencyLoss,
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

    modules = [
        RGBReconstructionLoss(lambda_ssim=float(_cfg_get(loss_cfg, "LAMBDA_SSIM", model_cfg.LAMBDA_SSIM))),
        LowLightConsistencyLoss(weight=float(_cfg_get(loss_cfg, "LAMBDA_LOW_LIGHT", 0.0))),
        ExposureControlLoss(weight=float(_cfg_get(loss_cfg, "LAMBDA_EXPOSURE", 0.0))),
    ]

    if bool(_cfg_get(depth_cfg, "ENABLED", False)):
        modules.append(DepthPriorLoss(weight=float(_cfg_get(depth_cfg, "WEIGHT", 0.0))))
    if bool(_cfg_get(structure_cfg, "ENABLED", False)):
        modules.append(StructurePriorLoss(weight=float(_cfg_get(structure_cfg, "WEIGHT", 0.0))))

    return [module for module in modules if module.enabled]



def compute_loss_modules(modules, context):
    total_loss = torch.zeros((), device=context["rendered"].device, dtype=context["rendered"].dtype)
    logs = {}

    for module in modules:
        raw_loss, extra_logs = module.compute(context)
        weighted_loss = raw_loss * module.weight
        total_loss = total_loss + weighted_loss
        logs[module.name] = float(weighted_loss.detach().item())
        for key, value in extra_logs.items():
            logs[f"{module.name}_{key}"] = float(value)

    logs["total"] = float(total_loss.detach().item())
    return total_loss, logs
