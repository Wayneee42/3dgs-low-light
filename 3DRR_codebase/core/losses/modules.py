from core.libs.losses import exposure_control_loss, low_light_consistency_loss, rgb_reconstruction_loss
import torch

from .structure_prior import build_structure_extractor


class BaseLossModule:
    def __init__(self, name, weight=1.0, enabled=True):
        self.name = name
        self.weight = float(weight)
        self.enabled = bool(enabled)

    def compute(self, context):
        raise NotImplementedError


class RGBReconstructionLoss(BaseLossModule):
    def __init__(self, lambda_ssim):
        super().__init__(name="rgb", weight=1.0, enabled=True)
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
        super().__init__(name="low_light", weight=weight, enabled=weight > 0.0)

    def compute(self, context):
        loss = low_light_consistency_loss(context["rendered"], context["reference_hwc"])
        return loss, {}


class ExposureControlLoss(BaseLossModule):
    def __init__(self, weight):
        super().__init__(name="exposure", weight=weight, enabled=weight > 0.0)

    def compute(self, context):
        loss = exposure_control_loss(context["rendered"], context["target_mean"])
        return loss, {}



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


class DepthPriorLoss(BaseLossModule):
    def __init__(self, weight, global_weight=1.0, local_weight=1.0, box_size=128, sample_ratio=0.5):
        super().__init__(name="depth_prior", weight=weight, enabled=weight > 0.0)
        self.global_weight = float(global_weight)
        self.local_weight = float(local_weight)
        self.box_size = int(box_size)
        self.sample_ratio = float(sample_ratio)

    def compute(self, context):
        rendered_depth = context.get("rendered_depth")
        depth_target = context.get("depth")
        if rendered_depth is None:
            raise RuntimeError("DepthPriorLoss requires rendered_depth, but the model did not render depth.")
        if depth_target is None:
            raise RuntimeError("DepthPriorLoss is enabled, but the dataset sample has no depth prior.")

        rendered_depth = rendered_depth.squeeze(-1)
        depth_target = depth_target.to(rendered_depth.device)
        if depth_target.dim() == 3:
            depth_target = depth_target.squeeze(0)
        elif depth_target.dim() != 2:
            raise RuntimeError(f"Unsupported depth tensor shape: {tuple(depth_target.shape)}")

        global_loss = pearson_depth_loss(rendered_depth.reshape(-1), depth_target.reshape(-1))
        local_loss = local_pearson_loss(rendered_depth, depth_target, self.box_size, self.sample_ratio)
        depth_loss = self.global_weight * global_loss + self.local_weight * local_loss
        return depth_loss, {
            "global": float(global_loss.detach().item()),
            "local": float(local_loss.detach().item()),
        }


class StructurePriorLoss(BaseLossModule):
    def __init__(self, weight, invariant="W", kernel_size=3, scale=0.8):
        super().__init__(name="structure_prior", weight=weight, enabled=weight > 0.0)
        self.extractor = build_structure_extractor(
            invariant=str(invariant),
            kernel_size=int(kernel_size),
            scale=float(scale),
        )

    def compute(self, context):
        structure_target = context.get("structure")
        if structure_target is None:
            zero = torch.zeros((), device=context["rendered"].device, dtype=context["rendered"].dtype)
            return zero, {"available": 0.0}

        rendered = context["rendered"].permute(2, 0, 1).unsqueeze(0)
        structure_target = structure_target.to(device=rendered.device, dtype=rendered.dtype)
        if structure_target.dim() == 2:
            structure_target = structure_target.unsqueeze(0).unsqueeze(0)
        elif structure_target.dim() == 3 and structure_target.shape[0] == 1:
            structure_target = structure_target.unsqueeze(0)
        else:
            raise RuntimeError(f"Unsupported structure tensor shape: {tuple(structure_target.shape)}")

        self.extractor = self.extractor.to(device=rendered.device, dtype=rendered.dtype)
        predicted_structure = self.extractor(rendered).clamp(0.0, 1.0)
        target_structure = structure_target.clamp(0.0, 1.0)
        loss = torch.abs(predicted_structure - target_structure).mean()
        return loss, {"available": 1.0}
