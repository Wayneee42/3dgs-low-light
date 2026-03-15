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



def build_proxy_target(image, proxy_cfg=None, fallback_target_mean=0.35, fallback_min_scale=1.0, fallback_max_scale=3.0):
    if not bool(_cfg_get(proxy_cfg, "ENABLED", False)):
        proxy_target, proxy_gain = exposure_match(
            image,
            fallback_target_mean,
            min_scale=fallback_min_scale,
            max_scale=fallback_max_scale,
            eps=float(_cfg_get(proxy_cfg, "EPS", 1e-6)),
        )
        return proxy_target, float(proxy_gain)

    eps = float(_cfg_get(proxy_cfg, "EPS", 1e-6))
    target_mean = float(_cfg_get(proxy_cfg, "TARGET_MEAN", fallback_target_mean))
    min_gain = float(_cfg_get(proxy_cfg, "MIN_GAIN", 1.0))
    max_gain = float(_cfg_get(proxy_cfg, "MAX_GAIN", 32.0))

    image = torch.clamp(image, 0.0, 1.0)
    current_mean = image.mean().clamp_min(eps)
    gain = torch.clamp(torch.tensor(target_mean, device=image.device) / current_mean, min_gain, max_gain)
    proxy_target = torch.clamp(image * gain, 0.0, 1.0)
    return proxy_target, float(gain.item())



def prepare_low_light_batch(image, aug_cfg=None, training=True, proxy_cfg=None):
    enabled = bool(_cfg_get(aug_cfg, "ENABLED", True))
    mode = str(_cfg_get(aug_cfg, "MODE", "gamma")).lower()

    base_target_mean = float(_cfg_get(aug_cfg, "TARGET_MEAN", 0.35))
    target_mean_jitter = float(_cfg_get(aug_cfg, "TARGET_MEAN_JITTER", 0.0))
    min_scale = float(_cfg_get(aug_cfg, "MIN_SCALE", 1.0))
    max_scale = float(_cfg_get(aug_cfg, "MAX_SCALE", 3.0))
    gamma_range = _cfg_get(aug_cfg, "GAMMA_RANGE", [0.5, 0.5])
    eval_gamma = float(_cfg_get(aug_cfg, "EVAL_GAMMA", 0.5))

    if training:
        gamma = random.uniform(float(gamma_range[0]), float(gamma_range[1]))
        jitter = random.uniform(-target_mean_jitter, target_mean_jitter)
        target_mean = min(max(base_target_mean * (1.0 + jitter), 0.05), 0.95)
    else:
        gamma = eval_gamma
        target_mean = base_target_mean

    proxy_target, proxy_scale = build_proxy_target(
        image,
        proxy_cfg=proxy_cfg,
        fallback_target_mean=target_mean,
        fallback_min_scale=min_scale,
        fallback_max_scale=max_scale,
    )

    if not enabled:
        return {
            "supervision": image,
            "reference": image,
            "proxy_target": proxy_target,
            "target_mean": float(target_mean),
            "gamma": 1.0,
            "scale": 1.0,
            "proxy_scale": float(proxy_scale),
            "proxy_mean": float(proxy_target.mean().item()),
            "low_mean": float(image.mean().item()),
            "mode": "identity",
        }

    supervision = image
    scale = 1.0

    if mode in ("gamma", "hybrid"):
        supervision = gamma_augment(supervision, gamma)
    if mode in ("exposure_match", "hybrid"):
        supervision, scale = exposure_match(supervision, target_mean, min_scale=min_scale, max_scale=max_scale)

    return {
        "supervision": supervision,
        "reference": image,
        "proxy_target": proxy_target,
        "target_mean": float(target_mean),
        "gamma": float(gamma),
        "scale": float(scale),
        "proxy_scale": float(proxy_scale),
        "proxy_mean": float(proxy_target.mean().item()),
        "low_mean": float(image.mean().item()),
        "mode": mode,
    }
