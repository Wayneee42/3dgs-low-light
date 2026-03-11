from core.libs.losses import exposure_control_loss, low_light_consistency_loss, rgb_reconstruction_loss
import torch


class BaseLossModule:
    def __init__(self, name, weight=1.0, enabled=True, start_step=0):
        self.name = name
        self.weight = float(weight)
        self.enabled = bool(enabled)
        self.start_step = int(start_step)

    def compute(self, context):
        raise NotImplementedError

    def is_active(self, context):
        return int(context.get("step", 0)) >= self.start_step


class RGBReconstructionLoss(BaseLossModule):
    def __init__(self, lambda_ssim):
        super().__init__(name="rgb", weight=1.0, enabled=True, start_step=0)
        self.lambda_ssim = float(lambda_ssim)

    def compute(self, context):
        result = rgb_reconstruction_loss(
            context["rendered"],
            context["supervision_hwc"],
            lambda_ssim=self.lambda_ssim,
        )
        return result["total"], {
            "l1": float(result["l1"].detach().item()),
            "ssim": float(result["ssim"].detach().item()),
        }


class LowLightConsistencyLoss(BaseLossModule):
    def __init__(self, weight):
        super().__init__(name="low_light", weight=weight, enabled=weight > 0.0, start_step=0)

    def compute(self, context):
        loss = low_light_consistency_loss(context["rendered"], context["reference_hwc"])
        return loss, {}


class ExposureControlLoss(BaseLossModule):
    def __init__(self, weight):
        super().__init__(name="exposure", weight=weight, enabled=weight > 0.0, start_step=0)

    def compute(self, context):
        loss = exposure_control_loss(context["rendered"], context["target_mean"])
        return loss, {}



def zero_scalar_like(context):
    return torch.zeros((), device=context["rendered"].device, dtype=context["rendered"].dtype)



def pearson_depth_loss(depth_src, depth_target):
    src = depth_src - depth_src.mean()
    target = depth_target - depth_target.mean()
    src = src / (src.std() + 1e-6)
    target = target / (target.std() + 1e-6)
    return 1.0 - (src * target).mean()



def local_pearson_loss(depth_src, depth_target, box_size, sample_ratio):
    box_size = int(max(4, min(box_size, depth_src.shape[0], depth_src.shape[1])))
    num_box_h = max(1, depth_src.shape[0] // box_size)
    num_box_w = max(1, depth_src.shape[1] // box_size)
    n_corr = max(1, int(sample_ratio * num_box_h * num_box_w))
    max_h = max(1, depth_src.shape[0] - box_size + 1)
    max_w = max(1, depth_src.shape[1] - box_size + 1)

    x_0 = torch.randint(0, max_h, (n_corr,), device=depth_src.device)
    y_0 = torch.randint(0, max_w, (n_corr,), device=depth_src.device)

    total = torch.zeros((), device=depth_src.device, dtype=depth_src.dtype)
    for x0, y0 in zip(x_0.tolist(), y_0.tolist()):
        x1 = x0 + box_size
        y1 = y0 + box_size
        total = total + pearson_depth_loss(
            depth_src[x0:x1, y0:y1].reshape(-1),
            depth_target[x0:x1, y0:y1].reshape(-1),
        )
    return total / n_corr



def squeeze_single_channel(image_tensor, label):
    if image_tensor is None:
        return None
    if image_tensor.dim() == 3 and image_tensor.shape[0] == 1:
        return image_tensor.squeeze(0)
    if image_tensor.dim() == 2:
        return image_tensor
    raise RuntimeError(f"Unsupported {label} tensor shape: {tuple(image_tensor.shape)}")



def standardize_map(image_tensor):
    return (image_tensor - image_tensor.mean()) / image_tensor.std().clamp_min(1e-6)



def minmax_normalize_map(image_tensor):
    min_value = image_tensor.min()
    max_value = image_tensor.max()
    return (image_tensor - min_value) / (max_value - min_value).clamp_min(1e-6)


class DepthPriorLoss(BaseLossModule):
    def __init__(self, weight, start_step=0, global_weight=1.0, local_weight=1.0, box_size=128, sample_ratio=0.5):
        super().__init__(name="depth_prior", weight=weight, enabled=weight > 0.0, start_step=start_step)
        self.global_weight = float(global_weight)
        self.local_weight = float(local_weight)
        self.box_size = int(box_size)
        self.sample_ratio = float(sample_ratio)

    def compute(self, context):
        depth_target = context.get("depth")
        frame_key = context.get("data", {}).get("infos", {}).get("frame_key", "unknown")
        if depth_target is None:
            raise RuntimeError(f"DepthPriorLoss is enabled, but frame '{frame_key}' has no depth prior.")

        if not self.is_active(context):
            zero = zero_scalar_like(context)
            return zero, {"global": 0.0, "local": 0.0}

        rendered_depth = context.get("depth_aux")
        if rendered_depth is None:
            raise RuntimeError("DepthPriorLoss requires depth_aux, but the model did not render the D_r head.")

        rendered_depth = squeeze_single_channel(rendered_depth, "rendered_depth")
        depth_target = squeeze_single_channel(depth_target.to(rendered_depth.device, dtype=rendered_depth.dtype), "depth")

        rendered_depth = standardize_map(rendered_depth)
        depth_target = standardize_map(depth_target)
        global_loss = pearson_depth_loss(rendered_depth.reshape(-1), depth_target.reshape(-1))
        local_loss = local_pearson_loss(rendered_depth, depth_target, self.box_size, self.sample_ratio)
        depth_loss = self.global_weight * global_loss + self.local_weight * local_loss
        return depth_loss, {
            "global": float(global_loss.detach().item()),
            "local": float(local_loss.detach().item()),
        }


class StructurePriorLoss(BaseLossModule):
    def __init__(self, weight, start_step=0):
        super().__init__(name="structure_prior", weight=weight, enabled=weight > 0.0, start_step=start_step)

    def compute(self, context):
        structure_target = context.get("structure")
        if structure_target is None:
            zero = zero_scalar_like(context)
            return zero, {"available": 0.0}

        if not self.is_active(context):
            zero = zero_scalar_like(context)
            return zero, {"available": 1.0}

        rendered_prior = context.get("prior_aux")
        if rendered_prior is None:
            raise RuntimeError("StructurePriorLoss requires prior_aux, but the model did not render the P_r head.")

        rendered_prior = squeeze_single_channel(rendered_prior, "rendered_prior")
        structure_target = squeeze_single_channel(structure_target.to(rendered_prior.device, dtype=rendered_prior.dtype), "structure")
        predicted = minmax_normalize_map(rendered_prior)
        target = minmax_normalize_map(structure_target)
        loss = torch.abs(predicted - target).mean()
        return loss, {"available": 1.0}
