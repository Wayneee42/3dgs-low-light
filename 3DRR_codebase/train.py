#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xchu.contact@gmail.com)

import argparse
import math
import os
import random
import warnings
from pathlib import Path

import gsplat
import numpy as np
import torch
import yaml
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from core.data import Blender
from core.libs import ConfigDict
from core.libs.augment import prepare_low_light_batch
from core.losses import (
    build_loss_modules,
    compute_loss_modules,
    required_aux_heads,
)
from core.model import SceneCalibration, Simple3DGS
from core.model import ViewCalibrationTable



def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)



def _cfg_get(cfg, key, default):
    if cfg is None:
        return default
    try:
        return getattr(cfg, key)
    except AttributeError:
        return default



def infer_stage_name(config_path, meta_cfg):
    experiment_cfg = _cfg_get(meta_cfg, "EXPERIMENT", None)
    explicit_stage = _cfg_get(experiment_cfg, "STAGE", None)
    if explicit_stage is not None:
        return str(explicit_stage)

    config_parts = Path(config_path).parts
    if "config" in config_parts:
        config_index = config_parts.index("config")
        if config_index + 1 < len(config_parts) - 1:
            return config_parts[config_index + 1]
    return "manual"



def build_output_dir(config_path, meta_cfg):
    stage_name = infer_stage_name(config_path, meta_cfg)
    scene_name = str(meta_cfg.DATASET.NAME)
    return os.path.join("outputs", stage_name, scene_name)


def rgb_chw_to_gray(image_tensor):
    if image_tensor is None:
        return None
    return 0.299 * image_tensor[0] + 0.587 * image_tensor[1] + 0.114 * image_tensor[2]


def luminance_chw(image_tensor):
    return 0.299 * image_tensor[0] + 0.587 * image_tensor[1] + 0.114 * image_tensor[2]


def compute_view_prior_d0_map(train_dataset, canonical_calib_cfg):
    canonical_mode = str(_cfg_get(canonical_calib_cfg, "VIEW_CALIB_MODE", "free_affine"))
    if canonical_mode != "degradation_only":
        return {}

    eps = 1.0e-4
    d_max = float(_cfg_get(canonical_calib_cfg, "D_MAX", 4.0))
    target_median = float(
        _cfg_get(
            canonical_calib_cfg,
            "CANONICAL_TARGET_MEDIAN",
            _cfg_get(canonical_calib_cfg, "CANONICAL_TARGET_MEAN", 0.38),
        )
    )
    target_median = float(np.clip(target_median, eps, 1.0 - eps))
    target_logit = math.log(target_median / (1.0 - target_median))

    prior_map = {}
    for frame_key in train_dataset._records_keys:
        image_tensor = train_dataset._records[frame_key]["img_tensor"]
        luma = luminance_chw(image_tensor).reshape(-1)
        median_value = float(torch.median(luma).item())
        median_value = float(np.clip(median_value, eps, 1.0 - eps))
        median_logit = math.log(median_value / (1.0 - median_value))
        prior_map[frame_key] = float(np.clip(target_logit - median_logit, 0.0, d_max))
    return prior_map


def should_step_optimizer(optimizer_name, canonical_calib_cfg, current_step):
    canonical_mode = str(_cfg_get(canonical_calib_cfg, "VIEW_CALIB_MODE", "free_affine"))
    if canonical_mode != "degradation_only":
        return True
    view_only_steps = int(_cfg_get(canonical_calib_cfg, "VIEW_ONLY_STEPS", 0))
    if optimizer_name == "view_calibration":
        return True
    if optimizer_name == "scene_calibration":
        return True
    if optimizer_name in {"sh0", "shN"}:
        return current_step > view_only_steps
    return False


def is_multiview_active(loss_modules, step):
    context = {"step": int(step)}
    return any(module.name == "multiview_reproj" and module.is_active(context) for module in loss_modules)



def build_step_dir(output_dir, step):
    return os.path.join(output_dir, f"step_{int(step)}")



def save_config(path, meta_cfg):
    with open(path, "w") as handle:
        yaml.dump(dict(meta_cfg), handle, default_flow_style=False)



def resolve_checkpoint_steps(cfg):
    checkpoint_steps = _cfg_get(cfg, "CHECKPOINT_STEPS", None)
    if checkpoint_steps is None:
        checkpoint_steps = [7000, cfg.TRAIN_TOTAL_STEP]
    resolved = sorted({int(step) for step in checkpoint_steps if 0 < int(step) <= int(cfg.TRAIN_TOTAL_STEP)})
    if int(cfg.TRAIN_TOTAL_STEP) not in resolved:
        resolved.append(int(cfg.TRAIN_TOTAL_STEP))
    return resolved



def save_checkpoint(model, output_dir, step, meta_cfg):
    step_dir = build_step_dir(output_dir, step)
    os.makedirs(step_dir, exist_ok=True)
    save_config(os.path.join(step_dir, "config.yaml"), meta_cfg)
    checkpoint_path = os.path.join(step_dir, f"step_{int(step)}.pt")
    torch.save(model.splats.state_dict(), checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")
    return checkpoint_path


def save_view_calibration(output_dir, step, view_calibration, frame_keys):
    if view_calibration is None:
        return None
    step_dir = build_step_dir(output_dir, step)
    os.makedirs(step_dir, exist_ok=True)
    sidecar_path = os.path.join(step_dir, "view_calib.pt")
    torch.save(
        {
            "state_dict": view_calibration.state_dict(),
            "frame_keys": list(frame_keys),
        },
        sidecar_path,
    )
    return sidecar_path


def load_view_calibration(view_calibration, checkpoint_path, frame_keys, device):
    if view_calibration is None or checkpoint_path is None:
        return False
    sidecar_path = os.path.join(os.path.dirname(os.path.expanduser(str(checkpoint_path))), "view_calib.pt")
    if not os.path.exists(sidecar_path):
        return False
    payload = torch.load(sidecar_path, map_location=device)
    saved_keys = payload.get("frame_keys", [])
    state_dict = payload.get("state_dict", {})
    if list(saved_keys) == list(frame_keys):
        try:
            view_calibration.load_state_dict(state_dict, strict=True)
            print(f"[WarmStart] loaded view calibration from {sidecar_path}")
            return True
        except RuntimeError:
            pass
    saved_weight = state_dict.get("embedding.weight")
    if saved_weight is None:
        return False
    current_weight = view_calibration.embedding.weight.data
    if saved_weight.shape[1] != current_weight.shape[1]:
        print(
            f"[WarmStart] skipped incompatible view calibration sidecar {sidecar_path} "
            f"(saved_dim={saved_weight.shape[1]}, current_dim={current_weight.shape[1]})"
        )
        return False
    saved_index = {key: idx for idx, key in enumerate(saved_keys)}
    copied = 0
    for idx, key in enumerate(frame_keys):
        src_idx = saved_index.get(key)
        if src_idx is None:
            continue
        current_weight[idx].copy_(saved_weight[src_idx].to(device=device, dtype=current_weight.dtype))
        copied += 1
    print(f"[WarmStart] partially loaded view calibration from {sidecar_path}, copied={copied}")
    return copied > 0


def save_scene_calibration(output_dir, step, scene_calibration):
    if scene_calibration is None:
        return None
    step_dir = build_step_dir(output_dir, step)
    os.makedirs(step_dir, exist_ok=True)
    sidecar_path = os.path.join(step_dir, "scene_calib.pt")
    torch.save({"state_dict": scene_calibration.state_dict()}, sidecar_path)
    return sidecar_path


def load_scene_calibration(scene_calibration, checkpoint_path, device):
    if scene_calibration is None or checkpoint_path is None:
        return False
    sidecar_path = os.path.join(os.path.dirname(os.path.expanduser(str(checkpoint_path))), "scene_calib.pt")
    if not os.path.exists(sidecar_path):
        return False
    payload = torch.load(sidecar_path, map_location=device)
    state_dict = payload.get("state_dict", {})
    if not state_dict:
        return False
    scene_calibration.load_state_dict(state_dict, strict=True)
    print(f"[WarmStart] loaded scene calibration from {sidecar_path}")
    return True




def load_warmstart_checkpoint(model, checkpoint_path, device):
    checkpoint_path = os.path.expanduser(str(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or "means" not in checkpoint:
        raise RuntimeError(f"Warm-start checkpoint must be a splat state_dict with 'means': {checkpoint_path}")

    expected_keys = list(model.splats.keys())
    means = checkpoint["means"]
    if not torch.is_tensor(means) or means.ndim != 2 or means.shape[1] != 3:
        raise RuntimeError(f"Warm-start checkpoint has invalid means tensor: {checkpoint_path}")

    num_points = int(means.shape[0])
    normalized = {}
    missing_keys = []
    converted_illum = False
    for key in expected_keys:
        tensor = checkpoint.get(key)
        if tensor is None:
            if key in {"depth_feat", "prior_feat", "illum_feat"}:
                tensor = torch.zeros(num_points, 1, dtype=means.dtype)
                missing_keys.append(key)
            else:
                raise RuntimeError(f"Warm-start checkpoint is missing required key '{key}': {checkpoint_path}")
        if not torch.is_tensor(tensor):
            raise RuntimeError(f"Warm-start checkpoint key '{key}' is not a tensor: {checkpoint_path}")

        tensor = tensor.detach().to(device)
        if tensor.shape[0] != num_points:
            raise RuntimeError(
                f"Warm-start checkpoint key '{key}' has inconsistent gaussian count {tensor.shape[0]} != {num_points}: {checkpoint_path}"
            )

        if key in {"depth_feat", "prior_feat", "illum_feat"}:
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(-1)
            if key == "illum_feat" and tensor.ndim == 2 and tensor.shape[1] != 1:
                tensor = tensor.mean(dim=1, keepdim=True)
                converted_illum = True
            elif tensor.ndim != 2 or tensor.shape[1] != 1:
                tensor = tensor[:, :1]

        normalized[key] = torch.nn.Parameter(tensor.contiguous())

    model.splats = torch.nn.ParameterDict(normalized)
    model.sh_degree = model.sh_degree_max
    missing_str = ",".join(missing_keys) if missing_keys else "none"
    print(
        f"[WarmStart] checkpoint={checkpoint_path}, gaussians={num_points}, missing={missing_str}, "
        f"illum_converted={int(converted_illum)}, sh_degree={model.sh_degree}"
    )


def build_teacher_model(cfg, data_info, init_context, checkpoint_path, device):
    if checkpoint_path is None:
        return None
    teacher_model = Simple3DGS(cfg, data_info, init_context=init_context).to(device)
    load_warmstart_checkpoint(teacher_model, checkpoint_path, device)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad_(False)
    return teacher_model


def log_teacher_init_diff(student_model, teacher_model):
    if teacher_model is None:
        return
    keys = ["means", "sh0", "shN", "opacities", "scales", "quats"]
    stats = []
    with torch.no_grad():
        for key in keys:
            if key not in student_model.splats or key not in teacher_model.splats:
                continue
            diff = torch.abs(student_model.splats[key] - teacher_model.splats[key]).mean().item()
            stats.append(f"{key}:{diff:.3e}")
    if stats:
        print("[TeacherInit] mean_abs_diff " + ", ".join(stats))


def apply_scene_calibration_if_enabled(rgb_hwc, scene_calibration):
    if scene_calibration is None:
        return rgb_hwc, None
    return scene_calibration.apply(rgb_hwc)



def save_render_outputs(render_outputs, frame_key, root_dir, save_canonical=False, scene_calibration=None):
    canonical_image, _ = apply_scene_calibration_if_enabled(render_outputs["rgb"], scene_calibration)
    final_image = canonical_image if save_canonical and canonical_image is not None else render_outputs["recon_rgb"]
    save_image(final_image.permute(2, 0, 1).clamp(0, 1), os.path.join(root_dir, f"{frame_key}.png"))

    illum_aux = render_outputs.get("illum_aux")
    if illum_aux is None or save_canonical:
        return final_image

    base_dir = os.path.join(root_dir, "base")
    illum_dir = os.path.join(root_dir, "illum")
    recon_dir = os.path.join(root_dir, "recon")
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(illum_dir, exist_ok=True)
    os.makedirs(recon_dir, exist_ok=True)

    base_image = canonical_image if canonical_image is not None else render_outputs["rgb"]
    illum_image = torch.clamp(2.0 * torch.sigmoid(illum_aux), 0.0, 2.0) / 2.0
    save_image(base_image.permute(2, 0, 1).clamp(0, 1), os.path.join(base_dir, f"{frame_key}.png"))
    save_image(illum_image.permute(2, 0, 1).clamp(0, 1), os.path.join(illum_dir, f"{frame_key}.png"))
    save_image(final_image.permute(2, 0, 1).clamp(0, 1), os.path.join(recon_dir, f"{frame_key}.png"))
    return final_image



def train(config_path, device="cuda"):
    meta_cfg = ConfigDict(config_path=config_path)
    seed = int(_cfg_get(meta_cfg, "SEED", 3407))
    set_seed(seed)
    print(f"[Seed] {seed}")
    print(meta_cfg)
    cfg = meta_cfg.MODEL
    augmentation_cfg = _cfg_get(meta_cfg, "AUGMENTATION", None)
    proxy_cfg = _cfg_get(meta_cfg, "PROXY_TARGET", None)
    checkpoint_steps = set(resolve_checkpoint_steps(cfg))
    loss_modules = build_loss_modules(meta_cfg, cfg)
    aux_heads = required_aux_heads(loss_modules)
    has_reconstruction = "illum" in aux_heads
    canonical_calib_cfg = _cfg_get(meta_cfg, "CANONICAL_CALIB", None)
    canonical_calib_enabled = bool(_cfg_get(canonical_calib_cfg, "ENABLED", False))
    canonical_view_mode = str(_cfg_get(canonical_calib_cfg, "VIEW_CALIB_MODE", "free_affine"))
    scene_calib_cfg = _cfg_get(canonical_calib_cfg, "SCENE_CALIB", None)
    scene_calib_enabled = canonical_calib_enabled and bool(_cfg_get(scene_calib_cfg, "ENABLED", False))
    teacher_color_cfg = _cfg_get(canonical_calib_cfg, "COLOR_TEACHER", None)
    teacher_chroma_enabled = canonical_calib_enabled and float(_cfg_get(_cfg_get(meta_cfg, "LOSS", None), "LAMBDA_TEACHER_CHROMA", 0.0)) > 0.0
    teacher_color_enabled = canonical_calib_enabled and bool(_cfg_get(teacher_color_cfg, "ENABLED", False))
    teacher_enabled = teacher_chroma_enabled or teacher_color_enabled

    output_dir = build_output_dir(config_path, meta_cfg)
    os.makedirs(output_dir, exist_ok=True)
    save_config(os.path.join(output_dir, "config.yaml"), meta_cfg)

    multiview_cfg = _cfg_get(_cfg_get(meta_cfg, "PRIORS", None), "MULTIVIEW", None)
    train_dataset = Blender(meta_cfg.DATASET, split="train")
    val_dataset = Blender(meta_cfg.DATASET, split="val", load_images=False)
    num_train = len(train_dataset._records_keys)

    init_records = []
    for key in train_dataset._records_keys:
        rec = train_dataset._records[key]
        init_records.append(
            {
                "frame_key": rec["frame_key"],
                "transform_matrix": rec["transform_matrix"],
                "file_path": rec["file_path"],
            }
        )
    init_context = {
        "scene_root": str(meta_cfg.DATASET.DATA_PATH),
        "records": init_records,
    }

    train_frame_keys = list(train_dataset._records_keys)
    calib_index_by_frame = {frame_key: idx for idx, frame_key in enumerate(train_frame_keys)}
    view_prior_d0_map = compute_view_prior_d0_map(train_dataset, canonical_calib_cfg)

    model = Simple3DGS(cfg, train_dataset._data_info, init_context=init_context).to(device)
    teacher_model = None
    view_calibration = None
    scene_calibration = None
    if canonical_calib_enabled:
        if scene_calib_enabled:
            scene_calibration = SceneCalibration(
                chroma_scale=float(_cfg_get(scene_calib_cfg, "CHROMA_SCALE", 0.05)),
                max_lift=float(_cfg_get(scene_calib_cfg, "MAX_LIFT", 0.2)),
                max_gain=float(_cfg_get(scene_calib_cfg, "MAX_GAIN", 2.0)),
            ).to(device)
        view_calibration = ViewCalibrationTable(
            num_views=len(train_frame_keys),
            chroma_scale=float(_cfg_get(canonical_calib_cfg, "CHROMA_SCALE", 0.05)),
            mode=canonical_view_mode,
        ).to(device)
    warmstart_checkpoint = _cfg_get(cfg, "WARMSTART_CHECKPOINT", None)
    if warmstart_checkpoint:
        load_warmstart_checkpoint(model, warmstart_checkpoint, device)
        load_scene_calibration(scene_calibration, warmstart_checkpoint, device)
        load_view_calibration(view_calibration, warmstart_checkpoint, train_frame_keys, device)
        if teacher_enabled:
            teacher_model = build_teacher_model(cfg, train_dataset._data_info, init_context, warmstart_checkpoint, device)
            log_teacher_init_diff(model, teacher_model)
    print(f"Initialized {model.num_gaussians} Gaussians")

    lr_map = {
        "means": cfg.LR_MEANS,
        "quats": cfg.LR_QUATS,
        "scales": cfg.LR_SCALES,
        "opacities": cfg.LR_OPACITIES,
        "sh0": cfg.LR_SH0,
        "shN": cfg.LR_SHN,
        "depth_feat": cfg.LR_SHN,
        "prior_feat": cfg.LR_SHN,
        "illum_feat": float(_cfg_get(cfg, "LR_ILLUM", cfg.LR_SHN)),
    }
    optimizers = {}
    for name, param in model.splats.items():
        optimizers[name] = torch.optim.Adam([param], lr=lr_map[name], eps=1e-15)
    if view_calibration is not None:
        optimizers["view_calibration"] = torch.optim.Adam(
            view_calibration.parameters(),
            lr=float(_cfg_get(canonical_calib_cfg, "LR_VIEW_CALIB", 1.0e-2)),
            eps=1e-15,
        )
    if scene_calibration is not None:
        optimizers["scene_calibration"] = torch.optim.Adam(
            scene_calibration.parameters(),
            lr=float(_cfg_get(scene_calib_cfg, "LR", 5.0e-3)),
            eps=1e-15,
        )

    total_steps = int(cfg.TRAIN_TOTAL_STEP)
    lr_final_factor = cfg.LR_MEANS_FINAL / cfg.LR_MEANS
    schedulers = {
        "means": torch.optim.lr_scheduler.ExponentialLR(
            optimizers["means"], gamma=lr_final_factor ** (1.0 / total_steps)
        )
    }

    strategy = gsplat.DefaultStrategy(
        verbose=False,
        refine_start_iter=cfg.DENSIFY_START_STEP,
        refine_stop_iter=cfg.DENSIFY_STOP_STEP,
        refine_every=cfg.DENSIFY_INTERVAL,
        grow_grad2d=cfg.DENSIFY_GRAD_THRESH,
        reset_every=cfg.OPACITY_RESET_INTERVAL,
    )
    strategy_state = strategy.initialize_state(scene_scale=cfg.SCENE_SCALE)
    intrinsics = torch.tensor(
        [
            [float(train_dataset._data_info["fl_x"]), 0.0, float(train_dataset._data_info["cx"])],
            [0.0, float(train_dataset._data_info["fl_y"]), float(train_dataset._data_info["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    )

    train_aug_images = []
    train_proxy_images = []
    train_proxy_global_images = []
    train_proxy_shadow_images = []
    train_proxy_weight_images = []
    pbar = tqdm(range(total_steps))
    for step in pbar:
        current_step = step + 1

        if step > 0 and step % cfg.SH_UPGRADE_INTERVAL == 0:
            model.sh_degree = min(model.sh_degree + 1, model.sh_degree_max)

        data = train_dataset[random.randint(0, num_train - 1)]
        input_image = data["images"].to(device)
        train_batch = prepare_low_light_batch(input_image, augmentation_cfg, training=True, proxy_cfg=proxy_cfg)
        supervision_image = train_batch["supervision"]
        reference_image = train_batch["reference"]
        proxy_target_image = train_batch["proxy_target"]

        camtoworld = data["transforms"].to(device)
        H, W = supervision_image.shape[1], supervision_image.shape[2]
        multiview_active = is_multiview_active(loss_modules, current_step)
        render_outputs = model(camtoworld, H, W, render_heads=aux_heads, render_geom_depth=multiview_active)
        rendered = render_outputs["recon_rgb"]
        scene_calibrated_hwc = None
        scene_params_decoded = None
        view_calibrated_hwc = None
        view_params_raw = None
        view_params_decoded = None
        canonical_target_mean = None
        canonical_target_median = None
        canonical_target_p75 = None
        view_prior_d0 = None
        teacher_rgb_hwc = None
        teacher_alphas = None
        if canonical_calib_enabled:
            scene_calibrated_hwc, scene_params_decoded = apply_scene_calibration_if_enabled(render_outputs["rgb"], scene_calibration)
            if scene_calibrated_hwc is None:
                scene_calibrated_hwc = render_outputs["rgb"]
            frame_key = data["infos"]["frame_key"]
            calib_index = torch.tensor([calib_index_by_frame[frame_key]], dtype=torch.long, device=device)
            view_calibration_input = scene_calibrated_hwc.detach() if scene_calibration is not None else scene_calibrated_hwc
            view_calibrated_hwc, view_params_decoded = view_calibration.apply(view_calibration_input, calib_index)
            view_params_raw = view_params_decoded["raw"]
            rendered = view_calibrated_hwc
            canonical_target_mean = float(_cfg_get(canonical_calib_cfg, "CANONICAL_TARGET_MEAN", 0.38))
            canonical_target_median = float(
                _cfg_get(canonical_calib_cfg, "CANONICAL_TARGET_MEDIAN", canonical_target_mean)
            )
            canonical_target_p75 = float(_cfg_get(canonical_calib_cfg, "CANONICAL_TARGET_P75", 0.55))
            view_prior_d0 = float(view_prior_d0_map.get(frame_key, 0.0))
            if teacher_model is not None:
                with torch.no_grad():
                    teacher_outputs = teacher_model(camtoworld, H, W, render_heads=(), render_geom_depth=False)
                teacher_rgb_hwc = teacher_outputs["rgb"]
                teacher_alphas = teacher_outputs["alphas"]

        neighbor_record = train_dataset.get_pose_neighbor(data["infos"]["frame_key"]) if multiview_active else None
        neighbor_outputs = None
        neighbor_camtoworld = None
        neighbor_distance = 0.0
        if neighbor_record is not None:
            neighbor_camtoworld = neighbor_record["transform_matrix"].to(device)
            neighbor_outputs = model(neighbor_camtoworld, H, W, render_heads=(), render_geom_depth=True, render_rgb=False)
            neighbor_distance = float(
                torch.norm(
                    data["transforms"][:3, 3].to(dtype=torch.float32) - neighbor_record["transform_matrix"][:3, 3].to(dtype=torch.float32)
                ).item()
            )

        context = {
            "step": current_step,
            "rendered": rendered,
            "rgb_base_hwc": scene_calibrated_hwc if scene_calibrated_hwc is not None else render_outputs["rgb"],
            "rgb_model_hwc": render_outputs["rgb"],
            "recon_hwc": render_outputs["recon_rgb"],
            "depth_aux": render_outputs["depth_aux"],
            "geom_depth": render_outputs["geom_depth"],
            "alphas": render_outputs["alphas"],
            "prior_aux": render_outputs["prior_aux"],
            "illum_aux": render_outputs["illum_aux"],
            "view_calibrated_hwc": view_calibrated_hwc,
            "scene_params_decoded": scene_params_decoded,
            "view_params_raw": view_params_raw,
            "view_params_decoded": view_params_decoded,
            "canonical_target_mean": canonical_target_mean,
            "canonical_target_median": canonical_target_median,
            "canonical_target_p75": canonical_target_p75,
            "view_prior_d0": view_prior_d0,
            "teacher_rgb_hwc": teacher_rgb_hwc,
            "teacher_alphas": teacher_alphas,
            "supervision_hwc": supervision_image.permute(1, 2, 0),
            "proxy_shadow_weight_hwc": train_batch["proxy_shadow_weight"],
            "reference_hwc": reference_image.permute(1, 2, 0),
            "proxy_target_hwc": proxy_target_image.permute(1, 2, 0),
            "target_mean": train_batch["target_mean"],
            "data": data,
            "batch": train_batch,
            "camtoworld": camtoworld,
            "neighbor_camtoworld": neighbor_camtoworld,
            "neighbor_geom_depth": None if neighbor_outputs is None else neighbor_outputs["geom_depth"],
            "neighbor_alphas": None if neighbor_outputs is None else neighbor_outputs["alphas"],
            "intrinsics": intrinsics,
            "depth": data["depth"].to(device) if data["depth"] is not None else None,
            "structure": data["structure"].to(device) if data["structure"] is not None else None,
        }
        loss, loss_logs = compute_loss_modules(loss_modules, context)
        loss_logs["illumination_available"] = float(render_outputs["illum_aux"] is not None)
        loss_logs["proxy_mean"] = float(train_batch["proxy_mean"])
        loss_logs["proxy_stat_mean"] = float(train_batch["proxy_stat_mean"])
        loss_logs["proxy_gain"] = float(train_batch["proxy_scale"])
        loss_logs["proxy_form"] = str(train_batch["proxy_form"])
        loss_logs["proxy_global_mean"] = float(train_batch["proxy_global_mean"])
        loss_logs["proxy_shadow_mean"] = float(train_batch["proxy_shadow_mean"])
        loss_logs["proxy_blend_mean"] = float(train_batch["proxy_blend_mean"])
        loss_logs["proxy_shadow_weight_mean"] = float(train_batch["proxy_shadow_weight_mean"])
        loss_logs["low_mean"] = float(train_batch["low_mean"])
        loss_logs["neighbor_distance"] = float(neighbor_distance)

        strategy.step_pre_backward(model.splats, optimizers, strategy_state, step, render_outputs["info"])
        loss.backward()
        strategy.step_post_backward(model.splats, optimizers, strategy_state, step, render_outputs["info"], packed=False)

        for optimizer_name, optimizer in optimizers.items():
            if not canonical_calib_enabled or should_step_optimizer(optimizer_name, canonical_calib_cfg, current_step):
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        for scheduler in schedulers.values():
            scheduler.step()
        if step % cfg.LOG_INTERVAL_STEP == 0:
            with torch.no_grad():
                if canonical_calib_enabled:
                    base_target = context["reference_hwc"]
                    base_render = context["view_calibrated_hwc"]
                else:
                    base_target = context["supervision_hwc"]
                    base_render = context["rgb_base_hwc"] if has_reconstruction else context["rendered"]
                mse_base = ((base_render - base_target) ** 2).mean()
                psnr_base = -10.0 * math.log10(mse_base.clamp_min(1e-10).item())
                if has_reconstruction:
                    recon_target = context["proxy_target_hwc"]
                    mse_recon = ((context["recon_hwc"] - recon_target) ** 2).mean()
                    psnr_recon = -10.0 * math.log10(mse_recon.clamp_min(1e-10).item())
                else:
                    psnr_recon = psnr_base
            postfix = {
                # "loss": f"{loss_logs['total']:.4f}",
                "n_gs": model.num_gaussians,
            }
            if canonical_calib_enabled:
                postfix["obs"] = f"{loss_logs.get('obs_photo', 0.0):.4f}"
                postfix["psnr"] = f"{psnr_base:.2f}"
                if loss_logs.get("view_prior_weight", 0.0) > 0.0:
                    postfix["vpr"] = f"{loss_logs.get('view_prior', 0.0):.4f}"
                    postfix["d"] = f"{loss_logs.get('view_prior_d_mean', 0.0):.2f}"
                    postfix["cs"] = f"{loss_logs.get('view_prior_s_mean', 1.0):.2f}"
                    postfix["uv"] = f"{loss_logs.get('view_prior_uv_mean', 0.0):.3f}"
                elif loss_logs.get("view_id_weight", 0.0) > 0.0:
                    postfix["vid"] = f"{loss_logs.get('view_id', 0.0):.4f}"
                    postfix["a"] = f"{loss_logs.get('view_id_a_mean', 1.0):.2f}"
                    postfix["b"] = f"{loss_logs.get('view_id_b_mean', 0.0):.2f}"
                    postfix["uv"] = f"{loss_logs.get('view_id_uv_mean', 0.0):.3f}"
                if loss_logs.get("canon_exp_weight", 0.0) > 0.0:
                    postfix["cex"] = f"{loss_logs.get('canon_exp', 0.0):.4f}"
                if loss_logs.get("teacher_chroma_weight", 0.0) > 0.0:
                    postfix["tchr"] = f"{loss_logs.get('teacher_chroma', 0.0):.4f}"
                if loss_logs.get("teacher_color_weight", 0.0) > 0.0:
                    postfix["tclr"] = f"{loss_logs.get('teacher_color', 0.0):.4f}"
                if loss_logs.get("teacher_luma_weight", 0.0) > 0.0:
                    postfix["tlum"] = f"{loss_logs.get('teacher_luma', 0.0):.4f}"
                if scene_params_decoded is not None:
                    postfix["sg"] = f"{float(scene_params_decoded['gain'].detach().item()):.3f}"
                    postfix["sc"] = f"{float(scene_params_decoded['contrast'].detach().item()):.3f}"
                    postfix["sl"] = f"{float(scene_params_decoded['lift'].detach().item()):.3f}"
            elif has_reconstruction:
                postfix["rgb_b"] = f"{loss_logs.get('rgb_base', 0.0):.4f}"
                postfix["rec"] = f"{loss_logs.get('reconstruction', 0.0):.4f}"
                postfix["psnr_b"] = f"{psnr_base:.2f}"
                postfix["psnr_r"] = f"{psnr_recon:.2f}"
            else:
                postfix["rgb"] = f"{loss_logs.get('rgb', 0.0):.4f}"
                # postfix["psnr"] = f"{psnr_base:.2f}"

            if loss_logs.get("depth_prior_weight", 0.0) > 0.0:
                postfix["dep"] = f"{loss_logs.get('depth_prior', 0.0):.4f}"
                postfix["dep_w"] = f"{loss_logs.get('depth_prior_weight', 0.0):.3f}"
            if loss_logs.get("multiview_reproj_weight", 0.0) > 0.0:
                postfix["mv"] = f"{loss_logs.get('multiview_reproj', 0.0):.4f}"
                postfix["mv_w"] = f"{loss_logs.get('multiview_reproj_weight', 0.0):.3f}"
                postfix["mv_v"] = f"{loss_logs.get('multiview_reproj_valid_ratio', 0.0):.2f}"
            if loss_logs.get("structure_prior", 0.0) > 0.0:
                postfix["st"] = f"{loss_logs.get('structure_prior', 0.0):.4f}"
            pbar.set_postfix(**postfix)

        if train_aug_images is not None:
            train_aug_images.append(supervision_image.clamp(0, 1))
            train_proxy_images.append(proxy_target_image.clamp(0, 1))
            train_proxy_global_images.append(train_batch["proxy_global"].clamp(0, 1))
            train_proxy_shadow_images.append(train_batch["proxy_shadow"].clamp(0, 1))
            proxy_weight_image = train_batch["proxy_shadow_weight"].unsqueeze(0).repeat(3, 1, 1).clamp(0, 1)
            train_proxy_weight_images.append(proxy_weight_image)
            if len(train_aug_images) >= 4:
                step_dir = build_step_dir(output_dir, current_step)
                examples_dir = os.path.join(step_dir, "examples")
                os.makedirs(examples_dir, exist_ok=True)
                aug_grid = make_grid(train_aug_images[:4], nrow=2)
                proxy_grid = make_grid(train_proxy_images[:4], nrow=2)
                proxy_global_grid = make_grid(train_proxy_global_images[:4], nrow=2)
                proxy_shadow_grid = make_grid(train_proxy_shadow_images[:4], nrow=2)
                proxy_weight_grid = make_grid(train_proxy_weight_images[:4], nrow=2)
                save_image(aug_grid, os.path.join(examples_dir, "train_aug.jpg"))
                save_image(proxy_grid, os.path.join(examples_dir, "proxy_target.jpg"))
                save_image(proxy_global_grid, os.path.join(examples_dir, "proxy_global.jpg"))
                save_image(proxy_shadow_grid, os.path.join(examples_dir, "proxy_shadow.jpg"))
                save_image(proxy_weight_grid, os.path.join(examples_dir, "proxy_shadow_weight.jpg"))
                train_aug_images = None
                train_proxy_images = None
                train_proxy_global_images = None
                train_proxy_shadow_images = None
                train_proxy_weight_images = None

        if current_step % cfg.VAL_INTERVAL_STEP == 0:
            validate(model, val_dataset, current_step, device, output_dir, aux_heads, canonical_calib_enabled, scene_calibration)

        if current_step in checkpoint_steps:
            save_checkpoint(model, output_dir, current_step, meta_cfg)
            save_scene_calibration(output_dir, current_step, scene_calibration)
            save_view_calibration(output_dir, current_step, view_calibration, train_frame_keys)

    test_dataset = Blender(meta_cfg.DATASET, split="test", load_images=False)
    evaluate(model, test_dataset, device, output_dir, total_steps, aux_heads, canonical_calib_enabled, scene_calibration)


@torch.no_grad()
def validate(model, val_dataset, step, device, output_dir, render_heads, save_canonical=False, scene_calibration=None):
    model.eval()
    step_dir = build_step_dir(output_dir, step)
    examples_dir = os.path.join(step_dir, "examples")
    os.makedirs(examples_dir, exist_ok=True)

    H, W = val_dataset._data_info["img_h"], val_dataset._data_info["img_w"]
    num_val = len(val_dataset._records_keys)
    recon_images = []
    base_images = []
    illum_images = []
    for index in range(num_val):
        data = val_dataset[index]
        camtoworld = data["transforms"].to(device)
        render_outputs = model(camtoworld, H, W, render_heads=render_heads)
        canonical_image, _ = apply_scene_calibration_if_enabled(render_outputs["rgb"], scene_calibration)
        output_image = canonical_image if save_canonical and canonical_image is not None else render_outputs["recon_rgb"]
        recon_images.append(output_image.permute(2, 0, 1).clamp(0, 1))
        if render_outputs["illum_aux"] is not None and not save_canonical:
            base_images.append((canonical_image if canonical_image is not None else render_outputs["rgb"]).permute(2, 0, 1).clamp(0, 1))
            illum_images.append((torch.clamp(2.0 * torch.sigmoid(render_outputs["illum_aux"]), 0.0, 2.0) / 2.0).permute(2, 0, 1))
        if len(recon_images) >= 4:
            break
    if recon_images:
        recon_grid = make_grid(recon_images, nrow=2)
        save_image(recon_grid, os.path.join(examples_dir, f"val_step{step}.jpg"))
        save_image(recon_grid, os.path.join(examples_dir, f"val_recon_step{step}.jpg"))
    if base_images:
        save_image(make_grid(base_images, nrow=2), os.path.join(examples_dir, f"val_base_step{step}.jpg"))
    if illum_images:
        save_image(make_grid(illum_images, nrow=2), os.path.join(examples_dir, f"val_illum_step{step}.jpg"))
    print(f"\n[Step {step}] {model.num_gaussians} Gaussians")
    model.train()


@torch.no_grad()
def evaluate(model, test_dataset, device, output_dir, step, render_heads, save_canonical=False, scene_calibration=None):
    model.eval()
    step_dir = build_step_dir(output_dir, step)
    test_dir = os.path.join(step_dir, "test")
    os.makedirs(test_dir, exist_ok=True)

    H, W = test_dataset._data_info["img_h"], test_dataset._data_info["img_w"]
    num_test = len(test_dataset._records_keys)
    for index in range(num_test):
        data = test_dataset[index]
        camtoworld = data["transforms"].to(device)
        render_outputs = model(camtoworld, H, W, render_heads=render_heads)
        frame_key = data["infos"]["frame_key"]
        save_render_outputs(render_outputs, frame_key, test_dir, save_canonical=save_canonical, scene_calibration=scene_calibration)
    print(f"Test renders saved to {test_dir}/")
    model.train()


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message=".*You are using the default legacy behaviour of the.*")
    warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", "-c", required=True, type=str)
    args = parser.parse_args()
    print("Command Line Args: {}".format(args))

    train(args.config_path)






