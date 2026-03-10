import random

import torch



def _cfg_get(cfg, key, default):
    if cfg is None:
        return default
    try:
        return getattr(cfg, key)
    except AttributeError:
        return default



def gamma_augment(image, gamma):
    return torch.clamp(image, 0.0, 1.0).pow(gamma)



def exposure_match(image, target_mean, min_scale=1.0, max_scale=3.0, eps=1e-6):
    image = torch.clamp(image, 0.0, 1.0)
    current_mean = image.mean().clamp_min(eps)
    scale = torch.clamp(torch.tensor(target_mean, device=image.device) / current_mean, min_scale, max_scale)
    return torch.clamp(image * scale, 0.0, 1.0), float(scale.item())



def prepare_low_light_batch(image, aug_cfg=None, training=True):
    enabled = bool(_cfg_get(aug_cfg, "ENABLED", True))
    mode = str(_cfg_get(aug_cfg, "MODE", "gamma")).lower()

    base_target_mean = float(_cfg_get(aug_cfg, "TARGET_MEAN", 0.35))
    target_mean_jitter = float(_cfg_get(aug_cfg, "TARGET_MEAN_JITTER", 0.0))
    min_scale = float(_cfg_get(aug_cfg, "MIN_SCALE", 1.0))
    max_scale = float(_cfg_get(aug_cfg, "MAX_SCALE", 3.0))
    gamma_range = _cfg_get(aug_cfg, "GAMMA_RANGE", [0.5, 0.5])
    eval_gamma = float(_cfg_get(aug_cfg, "EVAL_GAMMA", 0.5))

    if not enabled:
        return {
            "supervision": image,
            "reference": image,
            "target_mean": float(image.mean().item()),
            "gamma": 1.0,
            "scale": 1.0,
            "mode": "identity",
        }

    if training:
        gamma = random.uniform(float(gamma_range[0]), float(gamma_range[1]))
        jitter = random.uniform(-target_mean_jitter, target_mean_jitter)
        target_mean = min(max(base_target_mean * (1.0 + jitter), 0.05), 0.95)
    else:
        gamma = eval_gamma
        target_mean = base_target_mean

    supervision = image
    scale = 1.0

    if mode in ("gamma", "hybrid"):
        supervision = gamma_augment(supervision, gamma)
    if mode in ("exposure_match", "hybrid"):
        supervision, scale = exposure_match(supervision, target_mean, min_scale=min_scale, max_scale=max_scale)

    return {
        "supervision": supervision,
        "reference": image,
        "target_mean": float(target_mean),
        "gamma": float(gamma),
        "scale": float(scale),
        "mode": mode,
    }
