#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xchu.contact@gmail.com)

import argparse
import json
import math
import os
import warnings

import torch
import yaml
from torchvision.utils import save_image
from tqdm import tqdm

from core.data import Blender
from core.libs import ConfigDict, ssim
from core.model import Simple3DGS



def psnr(rendered, target, eps=1e-10):
    mse = ((rendered - target) ** 2).mean().clamp_min(eps)
    return float((-10.0 * torch.log10(mse)).item())



def try_build_lpips(device):
    try:
        import lpips
    except ImportError:
        return None
    model = lpips.LPIPS(net="vgg").to(device)
    model.eval()
    return model



def lpips_score(lpips_model, rendered, target):
    if lpips_model is None:
        return None
    rendered_nchw = rendered.permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0
    target_nchw = target.permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0
    return float(lpips_model(rendered_nchw, target_nchw).mean().item())



def can_compute_metrics(dataset):
    if len(dataset._records_keys) == 0:
        return False
    return all(os.path.exists(dataset._records[key]["file_path"]) for key in dataset._records_keys)



def write_metric_outputs(ckpt_dir, summary, per_view):
    summary_path = os.path.join(ckpt_dir, "results.json")
    per_view_path = os.path.join(ckpt_dir, "per_view.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with open(per_view_path, "w", encoding="utf-8") as handle:
        json.dump(per_view, handle, indent=2)
    print(f"Metric summary written to {summary_path}")
    print(f"Per-view metrics written to {per_view_path}")



@torch.no_grad()
def evaluate(checkpoint_path, device="cuda"):
    ckpt_dir = os.path.dirname(checkpoint_path)
    config_path = os.path.join(ckpt_dir, "config.yaml")
    with open(config_path) as f:
        config_dict = yaml.load(f, Loader=yaml.Loader)
    config_dict["EXP_STR"] = ""
    config_dict["TIME_STR"] = ""
    meta_cfg = ConfigDict(config_path=config_dict)
    cfg = meta_cfg.MODEL

    test_dataset = Blender(meta_cfg.DATASET, split="test", load_images=False)
    metric_dataset = Blender(meta_cfg.DATASET, split="test", load_images=True) if can_compute_metrics(test_dataset) else None
    H, W = test_dataset._data_info["img_h"], test_dataset._data_info["img_w"]

    model = Simple3DGS(cfg, test_dataset._data_info).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    for k, v in ckpt.items():
        model.splats[k] = torch.nn.Parameter(v)
    model.sh_degree = model.sh_degree_max
    model.eval()

    output_dir = os.path.join(ckpt_dir, "test")
    os.makedirs(output_dir, exist_ok=True)
    num_test = len(test_dataset._records_keys)

    lpips_model = try_build_lpips(device)
    metric_values = {"PSNR": [], "SSIM": []}
    per_view = {"PSNR": {}, "SSIM": {}}
    if lpips_model is not None:
        metric_values["LPIPS"] = []
        per_view["LPIPS"] = {}

    for i in tqdm(range(num_test), desc="Rendering"):
        data = test_dataset[i]
        camtoworld = data["transforms"].to(device)
        rendered, _, _ = model(camtoworld, H, W)
        frame_key = data["infos"]["frame_key"]
        save_image(
            rendered.permute(2, 0, 1).clamp(0, 1),
            os.path.join(output_dir, f"{frame_key}.png"),
        )

        if metric_dataset is not None:
            gt_data = metric_dataset[i]
            gt_hwc = gt_data["images"].to(device).permute(1, 2, 0)
            psnr_value = psnr(rendered, gt_hwc)
            ssim_value = float(ssim(rendered, gt_hwc).item())
            metric_values["PSNR"].append(psnr_value)
            metric_values["SSIM"].append(ssim_value)
            per_view["PSNR"][frame_key] = psnr_value
            per_view["SSIM"][frame_key] = ssim_value

            if lpips_model is not None:
                lpips_value = lpips_score(lpips_model, rendered, gt_hwc)
                metric_values["LPIPS"].append(lpips_value)
                per_view["LPIPS"][frame_key] = lpips_value

    print(f"Rendered {num_test} images to {output_dir}/ | {model.num_gaussians} Gaussians")

    if metric_dataset is not None:
        summary = {metric_name: float(sum(values) / len(values)) for metric_name, values in metric_values.items() if values}
        write_metric_outputs(ckpt_dir, summary, per_view)
        for metric_name, value in summary.items():
            print(f"{metric_name}: {value:.6f}")
    else:
        print("Ground-truth test images not found; skipped metric computation.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message=".*You are using the default legacy behaviour of the.*")
    warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", "-w", required=True, type=str)
    args = parser.parse_args()
    evaluate(args.checkpoint)
