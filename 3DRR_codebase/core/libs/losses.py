import torch

from .utils import ssim



def rgb_reconstruction_loss(rendered, target_hwc, lambda_ssim):
    l1_loss = torch.abs(rendered - target_hwc).mean()
    ssim_value = ssim(rendered, target_hwc)
    rgb_loss = (1.0 - lambda_ssim) * l1_loss + lambda_ssim * (1.0 - ssim_value)
    return {
        "total": rgb_loss,
        "l1": l1_loss,
        "ssim": ssim_value,
    }



def luminance(image_hwc):
    return 0.299 * image_hwc[..., 0] + 0.587 * image_hwc[..., 1] + 0.114 * image_hwc[..., 2]



def low_light_consistency_loss(rendered, reference_hwc, eps=1e-6):
    rendered_luma = luminance(rendered)
    reference_luma = luminance(reference_hwc)
    rendered_norm = rendered_luma / rendered_luma.mean().clamp_min(eps)
    reference_norm = reference_luma / reference_luma.mean().clamp_min(eps)
    return torch.abs(rendered_norm - reference_norm).mean()



def exposure_control_loss(rendered, target_mean):
    rendered_luma = luminance(rendered)
    target = torch.tensor(float(target_mean), device=rendered.device, dtype=rendered.dtype)
    return torch.abs(rendered_luma.mean() - target)


def robust_exposure_control_loss(rendered, target_median, target_p75, mask_low=0.05, mask_high=0.95):
    rendered_luma = luminance(rendered)
    valid_mask = (rendered_luma > float(mask_low)) & (rendered_luma < float(mask_high))
    valid_values = rendered_luma[valid_mask]
    if valid_values.numel() == 0:
        valid_values = rendered_luma.reshape(-1)

    median_value = torch.median(valid_values)
    p75_value = torch.quantile(valid_values, 0.75)
    target_median = torch.tensor(float(target_median), device=rendered.device, dtype=rendered.dtype)
    target_p75 = torch.tensor(float(target_p75), device=rendered.device, dtype=rendered.dtype)
    return torch.abs(median_value - target_median) + 0.5 * torch.abs(p75_value - target_p75)
