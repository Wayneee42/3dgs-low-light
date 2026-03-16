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
from core.losses import build_loss_modules, compute_loss_modules, required_aux_heads
from core.model import Simple3DGS



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



def save_render_outputs(render_outputs, frame_key, root_dir):
    final_image = render_outputs["recon_rgb"]
    save_image(final_image.permute(2, 0, 1).clamp(0, 1), os.path.join(root_dir, f"{frame_key}.png"))

    illum_aux = render_outputs.get("illum_aux")
    if illum_aux is None:
        return final_image

    base_dir = os.path.join(root_dir, "base")
    illum_dir = os.path.join(root_dir, "illum")
    recon_dir = os.path.join(root_dir, "recon")
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(illum_dir, exist_ok=True)
    os.makedirs(recon_dir, exist_ok=True)

    base_image = render_outputs["rgb"]
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

    output_dir = build_output_dir(config_path, meta_cfg)
    os.makedirs(output_dir, exist_ok=True)
    save_config(os.path.join(output_dir, "config.yaml"), meta_cfg)

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

    model = Simple3DGS(cfg, train_dataset._data_info, init_context=init_context).to(device)
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

    train_aug_images = []
    train_proxy_images = []
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
        render_outputs = model(camtoworld, H, W, render_heads=aux_heads)
        rendered = render_outputs["recon_rgb"]

        context = {
            "step": current_step,
            "rendered": rendered,
            "rgb_base_hwc": render_outputs["rgb"],
            "recon_hwc": render_outputs["recon_rgb"],
            "depth_aux": render_outputs["depth_aux"],
            "prior_aux": render_outputs["prior_aux"],
            "illum_aux": render_outputs["illum_aux"],
            "supervision_hwc": supervision_image.permute(1, 2, 0),
            "reference_hwc": reference_image.permute(1, 2, 0),
            "proxy_target_hwc": proxy_target_image.permute(1, 2, 0),
            "target_mean": train_batch["target_mean"],
            "data": data,
            "batch": train_batch,
            "depth": data["depth"].to(device) if data["depth"] is not None else None,
            "structure": data["structure"].to(device) if data["structure"] is not None else None,
        }
        loss, loss_logs = compute_loss_modules(loss_modules, context)
        loss_logs["illumination_available"] = float(render_outputs["illum_aux"] is not None)
        loss_logs["proxy_mean"] = float(train_batch["proxy_mean"])
        loss_logs["proxy_gain"] = float(train_batch["proxy_scale"])
        loss_logs["low_mean"] = float(train_batch["low_mean"])

        strategy.step_pre_backward(model.splats, optimizers, strategy_state, step, render_outputs["info"])
        loss.backward()
        strategy.step_post_backward(model.splats, optimizers, strategy_state, step, render_outputs["info"], packed=False)

        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        for scheduler in schedulers.values():
            scheduler.step()

        if step % cfg.LOG_INTERVAL_STEP == 0:
            with torch.no_grad():
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
                "loss": f"{loss_logs['total']:.4f}",
                "rgb": f"{loss_logs.get('rgb', 0.0):.4f}",
                "rgb_b": f"{loss_logs.get('rgb_base', 0.0):.4f}",
                "rec": f"{loss_logs.get('reconstruction', 0.0):.4f}",
                "illum_r": f"{loss_logs.get('illum_reg', 0.0):.4f}",
                "illum_f": f"{loss_logs.get('illum_reg_factor_mean', 1.0):.3f}",
                "exp": f"{loss_logs.get('exposure', 0.0):.4f}",
                "dep": f"{loss_logs.get('depth_prior', 0.0):.4f}",
                "st": f"{loss_logs.get('structure_prior', 0.0):.4f}",
                "illum_a": f"{loss_logs.get('illumination_available', 0.0):.0f}",
                "proxy_m": f"{loss_logs.get('proxy_mean', 0.0):.3f}",
                "proxy_g": f"{loss_logs.get('proxy_gain', 0.0):.2f}",
                "low_m": f"{loss_logs.get('low_mean', 0.0):.3f}",
                "n_gs": model.num_gaussians,
                "psnr_b": f"{psnr_base:.2f}",
            }
            if has_reconstruction:
                postfix["psnr_r"] = f"{psnr_recon:.2f}"
            else:
                postfix["psnr"] = f"{psnr_base:.2f}"
            pbar.set_postfix(**postfix)

        if train_aug_images is not None:
            train_aug_images.append(supervision_image.clamp(0, 1))
            train_proxy_images.append(proxy_target_image.clamp(0, 1))
            if len(train_aug_images) >= 4:
                step_dir = build_step_dir(output_dir, current_step)
                examples_dir = os.path.join(step_dir, "examples")
                os.makedirs(examples_dir, exist_ok=True)
                aug_grid = make_grid(train_aug_images[:4], nrow=2)
                proxy_grid = make_grid(train_proxy_images[:4], nrow=2)
                save_image(aug_grid, os.path.join(examples_dir, "train_aug.jpg"))
                save_image(proxy_grid, os.path.join(examples_dir, "proxy_target.jpg"))
                train_aug_images = None
                train_proxy_images = None

        if current_step % cfg.VAL_INTERVAL_STEP == 0:
            validate(model, val_dataset, current_step, device, output_dir, aux_heads)

        if current_step in checkpoint_steps:
            save_checkpoint(model, output_dir, current_step, meta_cfg)

    test_dataset = Blender(meta_cfg.DATASET, split="test", load_images=False)
    evaluate(model, test_dataset, device, output_dir, total_steps, aux_heads)


@torch.no_grad()
def validate(model, val_dataset, step, device, output_dir, render_heads):
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
        recon_images.append(render_outputs["recon_rgb"].permute(2, 0, 1).clamp(0, 1))
        if render_outputs["illum_aux"] is not None:
            base_images.append(render_outputs["rgb"].permute(2, 0, 1).clamp(0, 1))
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
def evaluate(model, test_dataset, device, output_dir, step, render_heads):
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
        save_render_outputs(render_outputs, frame_key, test_dir)
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

