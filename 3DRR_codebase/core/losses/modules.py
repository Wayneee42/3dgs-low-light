from core.libs.losses import (
    exposure_control_loss,
    low_light_consistency_loss,
    rgb_reconstruction_loss,
    robust_exposure_control_loss,
)
import torch
import torch.nn.functional as F


def _rgb_to_lab_hwc(rgb_hwc):
    rgb = torch.clamp(rgb_hwc, 0.0, 1.0)
    linear = torch.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )
    x = 0.4124564 * linear[..., 0] + 0.3575761 * linear[..., 1] + 0.1804375 * linear[..., 2]
    y = 0.2126729 * linear[..., 0] + 0.7151522 * linear[..., 1] + 0.0721750 * linear[..., 2]
    z = 0.0193339 * linear[..., 0] + 0.1191920 * linear[..., 1] + 0.9503041 * linear[..., 2]

    x = x / 0.95047
    z = z / 1.08883

    delta = 6.0 / 29.0
    delta_cube = delta ** 3
    delta_square = delta ** 2

    def _f(component):
        return torch.where(
            component > delta_cube,
            torch.pow(component.clamp_min(1.0e-8), 1.0 / 3.0),
            component / (3.0 * delta_square) + 4.0 / 29.0,
        )

    fx = _f(x)
    fy = _f(y)
    fz = _f(z)
    l = (116.0 * fy - 16.0) / 100.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return l, a, b


def _masked_mean(values, mask):
    if not bool(mask.any().item()):
        return values.new_zeros(())
    return values[mask].mean()


class BaseLossModule:
    def __init__(
        self,
        name,
        weight=1.0,
        enabled=True,
        start_step=0,
        end_step=None,
        ramp_up_steps=0,
        ramp_down_steps=0,
        start_scale=1.0,
        end_scale=0.0,
    ):
        self.name = name
        self.weight = float(weight)
        self.enabled = bool(enabled)
        self.start_step = int(start_step)
        self.end_step = None if end_step is None else int(end_step)
        self.ramp_up_steps = int(max(0, ramp_up_steps))
        self.ramp_down_steps = int(max(0, ramp_down_steps))
        self.start_scale = float(start_scale)
        self.end_scale = float(end_scale)

    def compute(self, context):
        raise NotImplementedError

    def is_active(self, context):
        return self.schedule_scale(context) > 0.0

    def schedule_scale(self, context):
        step = int(context.get("step", 0))
        if step < self.start_step:
            return 0.0
        if self.end_step is not None and step > self.end_step:
            return 0.0

        scale = 1.0
        if self.ramp_up_steps > 0:
            ramp_up_progress = (step - self.start_step) / float(self.ramp_up_steps)
            ramp_up_progress = min(max(ramp_up_progress, 0.0), 1.0)
            scale *= self.start_scale + (1.0 - self.start_scale) * ramp_up_progress

        if self.end_step is not None and self.ramp_down_steps > 0:
            ramp_down_start = self.end_step - self.ramp_down_steps
            if step > ramp_down_start:
                ramp_down_progress = (self.end_step - step) / float(self.ramp_down_steps)
                ramp_down_progress = min(max(ramp_down_progress, 0.0), 1.0)
                scale *= self.end_scale + (1.0 - self.end_scale) * ramp_down_progress

        return scale

    def current_weight(self, context):
        return self.weight * self.schedule_scale(context)


class RGBReconstructionLoss(BaseLossModule):
    def __init__(self, lambda_ssim, name="rgb", input_key="rendered", target_key="supervision_hwc"):
        super().__init__(name=name, weight=1.0, enabled=True, start_step=0)
        self.lambda_ssim = float(lambda_ssim)
        self.input_key = str(input_key)
        self.target_key = str(target_key)

    def compute(self, context):
        result = rgb_reconstruction_loss(
            context[self.input_key],
            context[self.target_key],
            lambda_ssim=self.lambda_ssim,
        )
        return result["total"], {
            "l1": float(result["l1"].detach().item()),
            "ssim": float(result["ssim"].detach().item()),
        }


class ReconstructionLoss(BaseLossModule):
    def __init__(self, lambda_ssim, weight, start_step=0, input_key="recon_hwc", target_key="proxy_target_hwc"):
        super().__init__(name="reconstruction", weight=weight, enabled=weight > 0.0, start_step=start_step)
        self.lambda_ssim = float(lambda_ssim)
        self.input_key = str(input_key)
        self.target_key = str(target_key)

    def compute(self, context):
        target = context.get(self.target_key)
        if target is None:
            raise RuntimeError("ReconstructionLoss requires proxy_target_hwc, but it is missing.")
        if not self.is_active(context):
            zero = zero_scalar_like(context)
            return zero, {"active": 0.0}
        result = rgb_reconstruction_loss(
            context[self.input_key],
            target,
            lambda_ssim=self.lambda_ssim,
        )
        return result["total"], {
            "l1": float(result["l1"].detach().item()),
            "ssim": float(result["ssim"].detach().item()),
            "active": 1.0,
        }


class IlluminationRegularizationLoss(BaseLossModule):
    def __init__(self, weight):
        super().__init__(name="illum_reg", weight=weight, enabled=weight > 0.0, start_step=0)

    def compute(self, context):
        illum_aux = context.get("illum_aux")
        if illum_aux is None:
            raise RuntimeError("IlluminationRegularizationLoss requires illum_aux, but the model did not render the illumination head.")
        illum_factor = 2.0 * torch.sigmoid(illum_aux)
        loss = torch.abs(illum_factor - 1.0).mean()
        return loss, {"factor_mean": float(illum_factor.detach().mean().item())}


class LowLightConsistencyLoss(BaseLossModule):
    def __init__(self, weight):
        super().__init__(name="low_light", weight=weight, enabled=weight > 0.0, start_step=0)

    def compute(self, context):
        loss = low_light_consistency_loss(context["rendered"], context["reference_hwc"])
        return loss, {}


class ExposureControlLoss(BaseLossModule):
    def __init__(self, weight, input_key="rendered", target_mean_key="target_mean", name="exposure"):
        super().__init__(name=name, weight=weight, enabled=weight > 0.0, start_step=0)
        self.input_key = str(input_key)
        self.target_mean_key = str(target_mean_key)

    def compute(self, context):
        target_mean = context.get(self.target_mean_key)
        if target_mean is None:
            raise RuntimeError(f"ExposureControlLoss requires '{self.target_mean_key}' in context.")
        loss = exposure_control_loss(context[self.input_key], target_mean)
        return loss, {}


class CanonicalObservationLoss(BaseLossModule):
    def __init__(self, lambda_ssim, weight=1.0):
        super().__init__(name="obs_photo", weight=weight, enabled=weight > 0.0, start_step=0)
        self.lambda_ssim = float(lambda_ssim)

    def compute(self, context):
        result = rgb_reconstruction_loss(
            context["view_calibrated_hwc"],
            context["reference_hwc"],
            lambda_ssim=self.lambda_ssim,
        )
        return result["total"], {
            "l1": float(result["l1"].detach().item()),
            "ssim": float(result["ssim"].detach().item()),
        }


class ViewCalibrationIdentityLoss(BaseLossModule):
    def __init__(self, weight, color_identity_rho=8.0):
        super().__init__(name="view_id", weight=weight, enabled=weight > 0.0, start_step=0)
        self.color_identity_rho = float(color_identity_rho)

    def compute(self, context):
        decoded = context.get("view_params_decoded")
        if decoded is None:
            raise RuntimeError("ViewCalibrationIdentityLoss requires 'view_params_decoded' in context.")
        a = decoded["a"]
        b = decoded["b"]
        u = decoded["u"]
        v = decoded["v"]
        loss = ((a - 1.0) ** 2 + b ** 2 + self.color_identity_rho * (u ** 2 + v ** 2)).mean()
        uv_mean = torch.sqrt(u ** 2 + v ** 2).mean()
        return loss, {
            "a_mean": float(a.detach().mean().item()),
            "b_mean": float(b.detach().mean().item()),
            "uv_mean": float(uv_mean.detach().item()),
        }


class ViewCalibrationPriorLoss(BaseLossModule):
    def __init__(
        self,
        weight,
        color_prior_rho=8.0,
        start_step=0,
        end_step=None,
        ramp_up_steps=0,
        ramp_down_steps=0,
        start_scale=1.0,
        end_scale=0.0,
    ):
        super().__init__(
            name="view_prior",
            weight=weight,
            enabled=weight > 0.0,
            start_step=start_step,
            end_step=end_step,
            ramp_up_steps=ramp_up_steps,
            ramp_down_steps=ramp_down_steps,
            start_scale=start_scale,
            end_scale=end_scale,
        )
        self.color_prior_rho = float(color_prior_rho)

    def compute(self, context):
        decoded = context.get("view_params_decoded")
        prior_d0 = context.get("view_prior_d0")
        if decoded is None:
            raise RuntimeError("ViewCalibrationPriorLoss requires 'view_params_decoded' in context.")
        if prior_d0 is None:
            raise RuntimeError("ViewCalibrationPriorLoss requires 'view_prior_d0' in context.")
        d = decoded["d"]
        s = decoded["s"]
        u = decoded["u"]
        v = decoded["v"]
        prior_d0 = torch.as_tensor(prior_d0, device=d.device, dtype=d.dtype).reshape_as(d)
        loss = ((d - prior_d0) ** 2 + self.color_prior_rho * ((1.0 - s) ** 2 + u ** 2 + v ** 2)).mean()
        uv_mean = torch.sqrt(u ** 2 + v ** 2).mean()
        return loss, {
            "d_mean": float(d.detach().mean().item()),
            "s_mean": float(s.detach().mean().item()),
            "uv_mean": float(uv_mean.detach().item()),
        }


class CanonicalExposureAnchorLoss(BaseLossModule):
    def __init__(self, weight, mask_low=0.05, mask_high=0.95):
        super().__init__(name="canon_exp", weight=weight, enabled=weight > 0.0, start_step=0)
        self.mask_low = float(mask_low)
        self.mask_high = float(mask_high)

    def compute(self, context):
        target_median = context.get("canonical_target_median")
        target_p75 = context.get("canonical_target_p75")
        if target_median is None:
            raise RuntimeError("CanonicalExposureAnchorLoss requires 'canonical_target_median' in context.")
        if target_p75 is None:
            raise RuntimeError("CanonicalExposureAnchorLoss requires 'canonical_target_p75' in context.")
        loss = robust_exposure_control_loss(
            context["rgb_base_hwc"],
            target_median=target_median,
            target_p75=target_p75,
            mask_low=self.mask_low,
            mask_high=self.mask_high,
        )
        return loss, {}


class TeacherChromaConsistencyLoss(BaseLossModule):
    def __init__(self, weight):
        super().__init__(name="teacher_chroma", weight=weight, enabled=weight > 0.0, start_step=0)

    def compute(self, context):
        rendered = context["rgb_base_hwc"]
        reference = context.get("teacher_rgb_hwc")
        if reference is None:
            raise RuntimeError("TeacherChromaConsistencyLoss requires 'teacher_rgb_hwc' in context.")
        rendered_cb = -0.168736 * rendered[..., 0] - 0.331264 * rendered[..., 1] + 0.5 * rendered[..., 2]
        rendered_cr = 0.5 * rendered[..., 0] - 0.418688 * rendered[..., 1] - 0.081312 * rendered[..., 2]
        reference_cb = -0.168736 * reference[..., 0] - 0.331264 * reference[..., 1] + 0.5 * reference[..., 2]
        reference_cr = 0.5 * reference[..., 0] - 0.418688 * reference[..., 1] - 0.081312 * reference[..., 2]
        loss = torch.abs(rendered_cb - reference_cb).mean() + torch.abs(rendered_cr - reference_cr).mean()
        return loss, {}


class TeacherColorAnchorLoss(BaseLossModule):
    def __init__(
        self,
        lambda_l,
        lambda_ab,
        lambda_c,
        mask_l_low=0.08,
        mask_l_high=0.95,
        mask_chroma_min=0.02,
        alpha_min=0.2,
        eps=1.0e-6,
    ):
        total_weight = float(lambda_l) + float(lambda_ab) + float(lambda_c)
        super().__init__(name="teacher_color", weight=1.0, enabled=total_weight > 0.0, start_step=0)
        self.lambda_l = float(lambda_l)
        self.lambda_ab = float(lambda_ab)
        self.lambda_c = float(lambda_c)
        self.mask_l_low = float(mask_l_low)
        self.mask_l_high = float(mask_l_high)
        self.mask_chroma_min = float(mask_chroma_min)
        self.alpha_min = float(alpha_min)
        self.eps = float(eps)

    def compute(self, context):
        student = context["rgb_base_hwc"]
        teacher = context.get("teacher_rgb_hwc")
        if teacher is None:
            raise RuntimeError("TeacherColorAnchorLoss requires 'teacher_rgb_hwc' in context.")

        l_s, a_s, b_s = _rgb_to_lab_hwc(student)
        l_t, a_t, b_t = _rgb_to_lab_hwc(teacher)
        c_s = torch.sqrt(a_s ** 2 + b_s ** 2 + self.eps)
        c_t = torch.sqrt(a_t ** 2 + b_t ** 2 + self.eps)

        luma_mask = (l_t > self.mask_l_low) & (l_t < self.mask_l_high)

        alpha_s = context.get("alphas")
        if alpha_s is not None:
            if alpha_s.dim() == 3 and alpha_s.shape[-1] == 1:
                alpha_s = alpha_s.squeeze(-1)
            luma_mask = luma_mask & (alpha_s > self.alpha_min)

        alpha_t = context.get("teacher_alphas")
        if alpha_t is not None:
            if alpha_t.dim() == 3 and alpha_t.shape[-1] == 1:
                alpha_t = alpha_t.squeeze(-1)
            luma_mask = luma_mask & (alpha_t > self.alpha_min)

        chroma_mask = luma_mask & (c_t > self.mask_chroma_min)

        if not bool(luma_mask.any().item()):
            zero = student.new_zeros(())
            return zero, {"l": 0.0, "ab": 0.0, "c": 0.0}

        l_loss = _masked_mean(torch.abs(l_s - l_t), luma_mask)
        if bool(chroma_mask.any().item()):
            ab_loss = _masked_mean(torch.abs(a_s - a_t) + torch.abs(b_s - b_t), chroma_mask)
            c_loss = _masked_mean(torch.abs(c_s - c_t), chroma_mask)
        else:
            ab_loss = student.new_zeros(())
            c_loss = student.new_zeros(())
        total = self.lambda_l * l_loss + self.lambda_ab * ab_loss + self.lambda_c * c_loss
        return total, {
            "l": float(l_loss.detach().item()),
            "ab": float(ab_loss.detach().item()),
            "c": float(c_loss.detach().item()),
        }


class TeacherLuminanceFloorLoss(BaseLossModule):
    def __init__(
        self,
        lambda_abs,
        lambda_floor,
        lambda_quantile,
        mask_l_low=0.18,
        mask_l_high=0.9,
        alpha_min=0.2,
        eta=0.9,
        eta50=0.9,
        eta75=0.9,
    ):
        total_weight = float(lambda_abs) + float(lambda_floor) + float(lambda_quantile)
        super().__init__(name="teacher_luma", weight=1.0, enabled=total_weight > 0.0, start_step=0)
        self.lambda_abs = float(lambda_abs)
        self.lambda_floor = float(lambda_floor)
        self.lambda_quantile = float(lambda_quantile)
        self.mask_l_low = float(mask_l_low)
        self.mask_l_high = float(mask_l_high)
        self.alpha_min = float(alpha_min)
        self.eta = float(eta)
        self.eta50 = float(eta50)
        self.eta75 = float(eta75)

    def compute(self, context):
        student = context["rgb_base_hwc"]
        teacher = context.get("teacher_rgb_hwc")
        if teacher is None:
            raise RuntimeError("TeacherLuminanceFloorLoss requires 'teacher_rgb_hwc' in context.")

        y_s = 0.299 * student[..., 0] + 0.587 * student[..., 1] + 0.114 * student[..., 2]
        y_t = 0.299 * teacher[..., 0] + 0.587 * teacher[..., 1] + 0.114 * teacher[..., 2]

        mask = (y_t > self.mask_l_low) & (y_t < self.mask_l_high)

        alpha_s = context.get("alphas")
        if alpha_s is not None:
            if alpha_s.dim() == 3 and alpha_s.shape[-1] == 1:
                alpha_s = alpha_s.squeeze(-1)
            mask = mask & (alpha_s > self.alpha_min)

        alpha_t = context.get("teacher_alphas")
        if alpha_t is not None:
            if alpha_t.dim() == 3 and alpha_t.shape[-1] == 1:
                alpha_t = alpha_t.squeeze(-1)
            mask = mask & (alpha_t > self.alpha_min)

        if not bool(mask.any().item()):
            zero = student.new_zeros(())
            return zero, {
                "abs": 0.0,
                "floor": 0.0,
                "quantile": 0.0,
                "q50_s": 0.0,
                "q50_t": 0.0,
                "q75_s": 0.0,
                "q75_t": 0.0,
                "valid_ratio": 0.0,
            }

        y_s_masked = y_s[mask]
        y_t_masked = y_t[mask]

        abs_loss = torch.abs(y_s_masked - y_t_masked).mean()
        floor_loss = torch.relu(self.eta * y_t_masked - y_s_masked).mean()

        q50_s = torch.quantile(y_s_masked, 0.5)
        q50_t = torch.quantile(y_t_masked, 0.5)
        q75_s = torch.quantile(y_s_masked, 0.75)
        q75_t = torch.quantile(y_t_masked, 0.75)
        quantile_loss = torch.relu(self.eta50 * q50_t - q50_s) + 0.5 * torch.relu(self.eta75 * q75_t - q75_s)

        total = self.lambda_abs * abs_loss + self.lambda_floor * floor_loss + self.lambda_quantile * quantile_loss
        valid_ratio = float(mask.detach().float().mean().item())
        return total, {
            "abs": float(abs_loss.detach().item()),
            "floor": float(floor_loss.detach().item()),
            "quantile": float(quantile_loss.detach().item()),
            "q50_s": float(q50_s.detach().item()),
            "q50_t": float(q50_t.detach().item()),
            "q75_s": float(q75_s.detach().item()),
            "q75_t": float(q75_t.detach().item()),
            "valid_ratio": valid_ratio,
        }



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
    if image_tensor.dim() == 2:
        return image_tensor
    if image_tensor.dim() == 3 and image_tensor.shape[0] == 1:
        return image_tensor.squeeze(0)
    if image_tensor.dim() == 3 and image_tensor.shape[-1] == 1:
        return image_tensor.squeeze(-1)
    raise RuntimeError(f"Unsupported {label} tensor shape: {tuple(image_tensor.shape)}")



def standardize_map(image_tensor):
    return (image_tensor - image_tensor.mean()) / image_tensor.std().clamp_min(1e-6)



def minmax_normalize_map(image_tensor):
    min_value = image_tensor.min()
    max_value = image_tensor.max()
    return (image_tensor - min_value) / (max_value - min_value).clamp_min(1e-6)


def build_intrinsics_tensor(intrinsics, device, dtype):
    if torch.is_tensor(intrinsics):
        return intrinsics.to(device=device, dtype=dtype)
    return torch.tensor(intrinsics, device=device, dtype=dtype)


def build_c2w_4x4(camtoworld, device, dtype):
    c2w = torch.eye(4, device=device, dtype=dtype)
    c2w[:3, :] = camtoworld.to(device=device, dtype=dtype)
    return c2w


def rgb_to_gray_map(image_tensor):
    if image_tensor is None:
        return None
    if image_tensor.dim() == 2:
        return image_tensor
    if image_tensor.dim() == 3 and image_tensor.shape[-1] == 3:
        return 0.299 * image_tensor[..., 0] + 0.587 * image_tensor[..., 1] + 0.114 * image_tensor[..., 2]
    if image_tensor.dim() == 3 and image_tensor.shape[0] == 3:
        return 0.299 * image_tensor[0] + 0.587 * image_tensor[1] + 0.114 * image_tensor[2]
    raise RuntimeError(f"Unsupported RGB tensor shape for grayscale conversion: {tuple(image_tensor.shape)}")


def sample_image_grid(height, width, stride, device, dtype):
    ys = torch.arange(0, height, stride, device=device, dtype=dtype)
    xs = torch.arange(0, width, stride, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return grid_x.reshape(-1), grid_y.reshape(-1)


def normalize_coords(u, v, width, height):
    x_norm = 2.0 * (u / max(width - 1, 1)) - 1.0
    y_norm = 2.0 * (v / max(height - 1, 1)) - 1.0
    return torch.stack([x_norm, y_norm], dim=-1)


def project_world_to_target(world_points, target_c2w, intrinsics, eps):
    flip = torch.diag(torch.tensor([1.0, -1.0, -1.0], device=world_points.device, dtype=world_points.dtype))
    rotation = target_c2w[:3, :3]
    translation = target_c2w[:3, 3]
    cam_gl = (world_points - translation) @ rotation
    cam_cv = cam_gl @ flip.T
    z = cam_cv[:, 2]
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    u = fx * (cam_cv[:, 0] / z.clamp_min(eps)) + cx
    v = fy * (cam_cv[:, 1] / z.clamp_min(eps)) + cy
    return u, v, z


def backproject_to_world(depth_map, camtoworld, intrinsics, stride, eps, min_alpha, alpha_map):
    height, width = depth_map.shape
    device = depth_map.device
    dtype = depth_map.dtype
    grid_x, grid_y = sample_image_grid(height, width, stride, device, dtype)
    pixel_x = grid_x.long()
    pixel_y = grid_y.long()
    depth_values = depth_map[pixel_y, pixel_x]
    alpha_values = alpha_map[pixel_y, pixel_x]

    valid = torch.isfinite(depth_values) & (depth_values > eps) & (alpha_values > min_alpha)
    if not valid.any():
        empty = torch.empty((0,), device=device, dtype=dtype)
        return empty, empty, empty

    grid_x = grid_x[valid]
    grid_y = grid_y[valid]
    depth_values = depth_values[valid]

    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    x = (grid_x - cx) / fx * depth_values
    y = (grid_y - cy) / fy * depth_values
    cam_points_cv = torch.stack([x, y, depth_values], dim=-1)

    flip = torch.diag(torch.tensor([1.0, -1.0, -1.0], device=device, dtype=dtype))
    c2w = build_c2w_4x4(camtoworld, device, dtype)
    world_points = (cam_points_cv @ flip.T) @ c2w[:3, :3].T + c2w[:3, 3]
    return world_points, grid_x, grid_y


def backproject_pixels_to_world(depth_values, pixel_u, pixel_v, camtoworld, intrinsics):
    device = depth_values.device
    dtype = depth_values.dtype
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    x = (pixel_u - cx) / fx * depth_values
    y = (pixel_v - cy) / fy * depth_values
    cam_points_cv = torch.stack([x, y, depth_values], dim=-1)
    flip = torch.diag(torch.tensor([1.0, -1.0, -1.0], device=device, dtype=dtype))
    c2w = build_c2w_4x4(camtoworld, device, dtype)
    return (cam_points_cv @ flip.T) @ c2w[:3, :3].T + c2w[:3, 3]


def sample_target_map(target_map, coords_norm):
    sampled = F.grid_sample(
        target_map[None, None],
        coords_norm.view(1, 1, -1, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled.view(-1)


def sample_target_patch_map(target_map, coords_norm):
    sampled = F.grid_sample(
        target_map[None, None],
        coords_norm.view(1, coords_norm.shape[0], coords_norm.shape[1], 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled.view(coords_norm.shape[0], coords_norm.shape[1])


def compute_gray_gradient_confidence(gray_map):
    grad_x = torch.zeros_like(gray_map)
    grad_y = torch.zeros_like(gray_map)
    grad_x[:, 1:-1] = 0.5 * torch.abs(gray_map[:, 2:] - gray_map[:, :-2])
    grad_y[1:-1, :] = 0.5 * torch.abs(gray_map[2:, :] - gray_map[:-2, :])
    grad = torch.maximum(grad_x, grad_y)
    grad = grad / grad.max().clamp_min(1.0e-6)
    grad[[0, -1], :] = 1.0
    grad[:, [0, -1]] = 1.0
    return grad.clamp(0.0, 1.0)


def build_image_weight_map(gray_map, low_texture_thresh, highlight_thresh, dark_thresh, dark_grad_thresh):
    if gray_map is None:
        return None
    grad_conf = compute_gray_gradient_confidence(gray_map)
    img_weight = grad_conf.square()
    low_texture_mask = grad_conf < low_texture_thresh
    highlight_mask = gray_map > highlight_thresh
    dark_reflective_mask = (gray_map < dark_thresh) & (grad_conf < dark_grad_thresh)
    img_weight[low_texture_mask | highlight_mask | dark_reflective_mask] = 0.0
    return img_weight


def weighted_masked_mean(values, weights, eps=1.0e-6):
    if values.numel() == 0 or weights.numel() == 0:
        return torch.zeros((), device=values.device if values.numel() > 0 else weights.device, dtype=values.dtype if values.numel() > 0 else weights.dtype)
    denom = weights.sum()
    if float(denom.detach().item()) <= eps:
        return torch.zeros((), device=values.device, dtype=values.dtype)
    return (values * weights).sum() / denom.clamp_min(eps)


def lncc_loss(ref_patches, target_patches):
    batch_size, total_patch_size = target_patches.shape
    patch_size = int(round(total_patch_size ** 0.5))
    if patch_size * patch_size != total_patch_size:
        raise RuntimeError(f"Invalid patch vector length for LNCC: {total_patch_size}")

    ref = ref_patches.view(batch_size, 1, patch_size, patch_size)
    target = target_patches.view(batch_size, 1, patch_size, patch_size)
    ref_target = ref * target
    ref_sq = ref.pow(2)
    target_sq = target.pow(2)
    filters = torch.ones(1, 1, patch_size, patch_size, device=ref.device, dtype=ref.dtype)
    padding = patch_size // 2

    ref_sum = F.conv2d(ref, filters, stride=1, padding=padding)[:, :, padding, padding]
    target_sum = F.conv2d(target, filters, stride=1, padding=padding)[:, :, padding, padding]
    ref_sq_sum = F.conv2d(ref_sq, filters, stride=1, padding=padding)[:, :, padding, padding]
    target_sq_sum = F.conv2d(target_sq, filters, stride=1, padding=padding)[:, :, padding, padding]
    ref_target_sum = F.conv2d(ref_target, filters, stride=1, padding=padding)[:, :, padding, padding]

    ref_avg = ref_sum / total_patch_size
    target_avg = target_sum / total_patch_size
    cross = ref_target_sum - target_avg * ref_sum
    ref_var = ref_sq_sum - ref_avg * ref_sum
    target_var = target_sq_sum - target_avg * target_sum
    cc = cross * cross / (ref_var * target_var + 1.0e-8)
    ncc = 1.0 - cc
    ncc = torch.clamp(ncc, 0.0, 2.0).mean(dim=1)
    valid = ncc < 0.9
    return ncc, valid


def compute_patch_lncc_loss(
    source_gray,
    target_gray,
    source_depth,
    source_u,
    source_v,
    weights,
    source_camtoworld,
    target_camtoworld,
    intrinsics,
    patch_size,
    patch_samples,
    eps,
):
    if source_gray is None or target_gray is None or source_u.numel() == 0:
        zero = torch.zeros((), device=source_depth.device, dtype=source_depth.dtype)
        return zero, {"lncc_count": 0.0}

    valid_indices = torch.arange(source_u.shape[0], device=source_u.device)
    if valid_indices.numel() > patch_samples:
        choice = torch.randperm(valid_indices.numel(), device=valid_indices.device)[:patch_samples]
        valid_indices = valid_indices[choice]

    source_u = source_u[valid_indices]
    source_v = source_v[valid_indices]
    weights = weights[valid_indices]
    patch_radius = int(max(1, patch_size))
    offset_values = torch.arange(-patch_radius, patch_radius + 1, device=source_u.device, dtype=source_u.dtype)
    offset_y, offset_x = torch.meshgrid(offset_values, offset_values, indexing="ij")
    offset_x = offset_x.reshape(1, -1)
    offset_y = offset_y.reshape(1, -1)

    source_patch_u = source_u[:, None] + offset_x
    source_patch_v = source_v[:, None] + offset_y
    height = int(source_gray.shape[0])
    width = int(source_gray.shape[1])
    source_inside = (source_patch_u >= 0.0) & (source_patch_u <= float(width - 1))
    source_inside &= (source_patch_v >= 0.0) & (source_patch_v <= float(height - 1))

    source_patch_coords = normalize_coords(source_patch_u.reshape(-1), source_patch_v.reshape(-1), width, height)
    source_patch_coords = source_patch_coords.view(source_patch_u.shape[0], source_patch_u.shape[1], 2)
    source_patch_gray = sample_target_patch_map(source_gray, source_patch_coords)
    source_patch_depth = sample_target_patch_map(source_depth, source_patch_coords)
    patch_depth_valid = torch.isfinite(source_patch_depth) & (source_patch_depth > eps) & source_inside
    if not patch_depth_valid.any():
        zero = torch.zeros((), device=source_depth.device, dtype=source_depth.dtype)
        return zero, {"lncc_count": 0.0}

    world_points = backproject_pixels_to_world(
        source_patch_depth.reshape(-1),
        source_patch_u.reshape(-1),
        source_patch_v.reshape(-1),
        source_camtoworld,
        intrinsics,
    ).view(source_patch_depth.shape[0], source_patch_depth.shape[1], 3)
    target_c2w = build_c2w_4x4(target_camtoworld, source_depth.device, source_depth.dtype)
    projected_u, projected_v, projected_z = project_world_to_target(
        world_points.reshape(-1, 3),
        target_c2w,
        intrinsics,
        eps,
    )
    projected_u = projected_u.view(source_patch_depth.shape)
    projected_v = projected_v.view(source_patch_depth.shape)
    projected_z = projected_z.view(source_patch_depth.shape)
    target_inside = torch.isfinite(projected_z) & (projected_z > eps)
    target_inside &= (projected_u >= 0.0) & (projected_u <= float(width - 1))
    target_inside &= (projected_v >= 0.0) & (projected_v <= float(height - 1))
    patch_valid = patch_depth_valid & target_inside
    patch_center_valid = patch_valid.all(dim=1)
    if not patch_center_valid.any():
        zero = torch.zeros((), device=source_depth.device, dtype=source_depth.dtype)
        return zero, {"lncc_count": 0.0}

    target_patch_coords = normalize_coords(projected_u.reshape(-1), projected_v.reshape(-1), width, height)
    target_patch_coords = target_patch_coords.view(projected_u.shape[0], projected_u.shape[1], 2)
    target_patch_gray = sample_target_patch_map(target_gray, target_patch_coords)

    source_patch_gray = source_patch_gray[patch_center_valid]
    target_patch_gray = target_patch_gray[patch_center_valid]
    weights = weights[patch_center_valid]

    ncc_values, ncc_valid = lncc_loss(source_patch_gray, target_patch_gray)
    if not ncc_valid.any():
        zero = torch.zeros((), device=source_depth.device, dtype=source_depth.dtype)
        return zero, {"lncc_count": 0.0}

    ncc_values = ncc_values[ncc_valid]
    weights = weights[ncc_valid]
    return weighted_masked_mean(ncc_values, weights), {"lncc_count": float(ncc_valid.sum().detach().item())}


def compute_reprojection_directional_loss(
    source_depth,
    source_alpha,
    source_camtoworld,
    target_depth,
    target_alpha,
    target_camtoworld,
    intrinsics,
    stride,
    min_alpha,
    rel_thresh,
    abs_thresh,
    eps,
    pixel_noise_thresh=0.0,
    source_image_weight_map=None,
    target_image_weight_map=None,
    source_gray=None,
    target_gray=None,
    lncc_enabled=False,
    patch_size=3,
    patch_samples=256,
):
    world_points, source_u, source_v = backproject_to_world(
        depth_map=source_depth,
        camtoworld=source_camtoworld,
        intrinsics=intrinsics,
        stride=stride,
        eps=eps,
        min_alpha=min_alpha,
        alpha_map=source_alpha,
    )
    total_samples = max(1, ((source_depth.shape[0] + stride - 1) // stride) * ((source_depth.shape[1] + stride - 1) // stride))
    if world_points.numel() == 0:
        zero = torch.zeros((), device=source_depth.device, dtype=source_depth.dtype)
        return zero, zero, {
            "valid_ratio": 0.0,
            "geom_valid_ratio": 0.0,
            "occ_valid_ratio": 0.0,
            "valid_count": 0.0,
            "sample_count": float(total_samples),
            "pixel_noise_mean": 0.0,
            "lncc": 0.0,
        }

    target_c2w = build_c2w_4x4(target_camtoworld, source_depth.device, source_depth.dtype)
    u, v, z_proj = project_world_to_target(world_points, target_c2w, intrinsics, eps)

    width = int(target_depth.shape[1])
    height = int(target_depth.shape[0])
    valid = torch.isfinite(z_proj) & (z_proj > eps)
    valid &= (u >= 0.0) & (u <= float(width - 1))
    valid &= (v >= 0.0) & (v <= float(height - 1))
    if not valid.any():
        zero = torch.zeros((), device=source_depth.device, dtype=source_depth.dtype)
        return zero, zero, {
            "valid_ratio": 0.0,
            "geom_valid_ratio": 0.0,
            "occ_valid_ratio": 0.0,
            "valid_count": 0.0,
            "sample_count": float(total_samples),
            "pixel_noise_mean": 0.0,
            "lncc": 0.0,
        }

    u = u[valid]
    v = v[valid]
    z_proj = z_proj[valid]
    source_u = source_u[valid]
    source_v = source_v[valid]

    coords_norm = normalize_coords(u, v, width, height)

    sampled_target_depth = sample_target_map(target_depth, coords_norm)
    sampled_target_alpha = sample_target_map(target_alpha, coords_norm)

    geom_valid = torch.isfinite(sampled_target_depth) & (sampled_target_depth > eps)
    geom_valid &= sampled_target_alpha > min_alpha
    depth_delta = torch.abs(z_proj - sampled_target_depth)
    geom_valid &= depth_delta <= (abs_thresh + rel_thresh * sampled_target_depth.abs().clamp_min(eps))
    if not geom_valid.any():
        zero = torch.zeros((), device=source_depth.device, dtype=source_depth.dtype)
        return zero, zero, {
            "valid_ratio": 0.0,
            "geom_valid_ratio": 0.0,
            "occ_valid_ratio": 0.0,
            "valid_count": 0.0,
            "sample_count": float(total_samples),
            "pixel_noise_mean": 0.0,
            "lncc": 0.0,
        }

    z_proj = z_proj[geom_valid]
    sampled_target_depth = sampled_target_depth[geom_valid]
    u = u[geom_valid]
    v = v[geom_valid]
    source_u = source_u[geom_valid]
    source_v = source_v[geom_valid]

    depth_error = torch.abs(z_proj - sampled_target_depth) / sampled_target_depth.abs().clamp_min(eps)
    depth_loss = depth_error.mean()
    valid_ratio = float(geom_valid.sum().detach().item() / float(total_samples))
    zero = torch.zeros((), device=source_depth.device, dtype=source_depth.dtype)
    return depth_loss, zero, {
        "valid_ratio": valid_ratio,
        "geom_valid_ratio": valid_ratio,
        "occ_valid_ratio": valid_ratio,
        "valid_count": float(geom_valid.sum().detach().item()),
        "sample_count": float(total_samples),
        "pixel_noise_mean": 0.0,
        "lncc": 0.0,
    }


class DepthPriorLoss(BaseLossModule):
    def __init__(
        self,
        weight,
        start_step=0,
        end_step=None,
        ramp_up_steps=0,
        ramp_down_steps=0,
        start_scale=1.0,
        end_scale=0.0,
        global_weight=1.0,
        local_weight=1.0,
        box_size=128,
        sample_ratio=0.5,
    ):
        super().__init__(
            name="depth_prior",
            weight=weight,
            enabled=weight > 0.0,
            start_step=start_step,
            end_step=end_step,
            ramp_up_steps=ramp_up_steps,
            ramp_down_steps=ramp_down_steps,
            start_scale=start_scale,
            end_scale=end_scale,
        )
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


class MultiViewReprojectionLoss(BaseLossModule):
    def __init__(
        self,
        weight,
        start_step=0,
        end_step=None,
        ramp_up_steps=0,
        ramp_down_steps=0,
        start_scale=1.0,
        end_scale=0.0,
        sample_stride=4,
        min_alpha=0.2,
        relative_depth_thresh=0.05,
        absolute_depth_thresh=0.02,
        eps=1.0e-4,
        pixel_noise_thresh=0.0,
        low_texture_thresh=-1.0,
        highlight_thresh=2.0,
        dark_thresh=-1.0,
        dark_grad_thresh=-1.0,
        lncc_enabled=False,
        lncc_weight=0.15,
        patch_size=3,
        patch_samples=256,
    ):
        super().__init__(
            name="multiview_reproj",
            weight=weight,
            enabled=weight > 0.0,
            start_step=start_step,
            end_step=end_step,
            ramp_up_steps=ramp_up_steps,
            ramp_down_steps=ramp_down_steps,
            start_scale=start_scale,
            end_scale=end_scale,
        )
        self.sample_stride = int(max(1, sample_stride))
        self.min_alpha = float(min_alpha)
        self.relative_depth_thresh = float(relative_depth_thresh)
        self.absolute_depth_thresh = float(absolute_depth_thresh)
        self.eps = float(eps)
        self.pixel_noise_thresh = float(pixel_noise_thresh)
        self.low_texture_thresh = float(low_texture_thresh)
        self.highlight_thresh = float(highlight_thresh)
        self.dark_thresh = float(dark_thresh)
        self.dark_grad_thresh = float(dark_grad_thresh)
        self.lncc_enabled = bool(lncc_enabled)
        self.lncc_weight = float(lncc_weight)
        self.patch_size = int(max(1, patch_size))
        self.patch_samples = int(max(1, patch_samples))

    def compute(self, context):
        if not self.is_active(context):
            zero = zero_scalar_like(context)
            return zero, {"active": 0.0, "valid_ratio": 0.0, "lncc": 0.0}

        source_depth = context.get("geom_depth")
        target_depth = context.get("neighbor_geom_depth")
        source_alpha = context.get("alphas")
        target_alpha = context.get("neighbor_alphas")
        source_camtoworld = context.get("camtoworld")
        target_camtoworld = context.get("neighbor_camtoworld")
        intrinsics = context.get("intrinsics")
        if source_depth is None or target_depth is None or source_alpha is None or target_alpha is None:
            zero = zero_scalar_like(context)
            return zero, {"active": 0.0, "valid_ratio": 0.0, "lncc": 0.0}
        if source_camtoworld is None or target_camtoworld is None or intrinsics is None:
            zero = zero_scalar_like(context)
            return zero, {"active": 0.0, "valid_ratio": 0.0, "lncc": 0.0}

        source_depth = squeeze_single_channel(source_depth, "geom_depth")
        target_depth = squeeze_single_channel(target_depth, "neighbor_geom_depth")
        source_alpha = squeeze_single_channel(source_alpha, "alphas")
        target_alpha = squeeze_single_channel(target_alpha, "neighbor_alphas")
        intrinsics = build_intrinsics_tensor(intrinsics, source_depth.device, source_depth.dtype)
        src_to_tgt_loss, src_to_tgt_lncc, src_logs = compute_reprojection_directional_loss(
            source_depth=source_depth,
            source_alpha=source_alpha,
            source_camtoworld=source_camtoworld,
            target_depth=target_depth,
            target_alpha=target_alpha,
            target_camtoworld=target_camtoworld,
            intrinsics=intrinsics,
            stride=self.sample_stride,
            min_alpha=self.min_alpha,
            rel_thresh=self.relative_depth_thresh,
            abs_thresh=self.absolute_depth_thresh,
            eps=self.eps,
        )
        tgt_to_src_loss, tgt_to_src_lncc, tgt_logs = compute_reprojection_directional_loss(
            source_depth=target_depth,
            source_alpha=target_alpha,
            source_camtoworld=target_camtoworld,
            target_depth=source_depth,
            target_alpha=source_alpha,
            target_camtoworld=source_camtoworld,
            intrinsics=intrinsics,
            stride=self.sample_stride,
            min_alpha=self.min_alpha,
            rel_thresh=self.relative_depth_thresh,
            abs_thresh=self.absolute_depth_thresh,
            eps=self.eps,
        )
        depth_loss = 0.5 * (src_to_tgt_loss + tgt_to_src_loss)
        loss = depth_loss
        valid_ratio = 0.5 * (src_logs["valid_ratio"] + tgt_logs["valid_ratio"])
        return loss, {
            "active": 1.0,
            "valid_ratio": valid_ratio,
            "src_valid": src_logs["valid_count"],
            "tgt_valid": tgt_logs["valid_count"],
        }

