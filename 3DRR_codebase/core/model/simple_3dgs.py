# Copyright (c) Xuangeng Chu (xchu.contact@gmail.com)

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from gsplat import rasterization


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".JPG", ".JPEG")
NPY_EXTS = (".npy",)


class Simple3DGS(nn.Module):
    def __init__(self, model_cfg, data_info, init_context=None):
        super().__init__()
        self.fl_x = data_info["fl_x"]
        self.fl_y = data_info["fl_y"]
        self.cx = data_info["cx"]
        self.cy = data_info["cy"]
        self.bg_color = data_info["bg_color"]
        self.sh_degree_max = model_cfg.SH_DEGREE
        self.sh_degree = 0

        num_points = model_cfg.NUM_INIT_POINTS
        num_sh_bases = (self.sh_degree_max + 1) ** 2

        init_mode = str(getattr(model_cfg, "INIT_MODE", "random")).lower()
        if init_mode == "depth_backproject" and init_context is not None:
            means = self._init_means_from_depth_backproject(model_cfg, data_info, init_context)
        elif init_mode == "depth_backproject" and init_context is None:
            print("[Init] depth_backproject requested but no init_context provided; fallback to random means.")
            means = (torch.rand(num_points, 3) - 0.5) * 10.0
        else:
            means = (torch.rand(num_points, 3) - 0.5) * 10.0

        quats = torch.zeros(num_points, 4)
        quats[:, 0] = 1.0
        scales = torch.log(torch.full((num_points, 3), 0.005))
        opacities = torch.logit(torch.full((num_points,), 0.1))
        sh0 = torch.zeros(num_points, 1, 3)
        shN = torch.zeros(num_points, num_sh_bases - 1, 3)
        depth_feat = torch.zeros(num_points, 1)
        prior_feat = torch.zeros(num_points, 1)

        self.splats = nn.ParameterDict(
            {
                "means": nn.Parameter(means),
                "quats": nn.Parameter(quats),
                "scales": nn.Parameter(scales),
                "opacities": nn.Parameter(opacities),
                "sh0": nn.Parameter(sh0),
                "shN": nn.Parameter(shN),
                "depth_feat": nn.Parameter(depth_feat),
                "prior_feat": nn.Parameter(prior_feat),
            }
        )

    @property
    def num_gaussians(self):
        return self.splats["means"].shape[0]

    def _resolve_depth_path(self, depth_root, frame_key):
        for ext in NPY_EXTS + IMAGE_EXTS:
            candidate = depth_root / f"{frame_key}{ext}"
            if candidate.exists():
                return candidate
        return None

    def _load_depth(self, depth_path, width, height):
        if depth_path.suffix.lower() == ".npy":
            depth = np.load(depth_path).astype(np.float32)
        else:
            depth = np.asarray(Image.open(depth_path).convert("L"), dtype=np.float32) / 255.0
        if depth.shape != (height, width):
            depth = np.asarray(
                Image.fromarray(np.asarray(np.clip(depth, 0.0, 1.0) * 255.0, dtype=np.uint8), mode="L").resize(
                    (width, height), resample=Image.BILINEAR
                ),
                dtype=np.float32,
            ) / 255.0
        return depth

    def _normalize_depth_for_init(self, depth, model_cfg, depth_eps):
        mode = str(getattr(model_cfg, "INIT_DEPTH_NORMALIZATION", "none")).lower()
        if mode == "none":
            return depth
        if mode != "per_frame_robust":
            raise RuntimeError(f"Unsupported INIT_DEPTH_NORMALIZATION mode: {mode}")

        valid = np.isfinite(depth) & (depth > depth_eps)
        if not np.any(valid):
            return depth

        low_q = float(getattr(model_cfg, "INIT_DEPTH_NORM_LOW_Q", 1.0))
        high_q = float(getattr(model_cfg, "INIT_DEPTH_NORM_HIGH_Q", 99.0))
        low = float(np.percentile(depth[valid], low_q))
        high = float(np.percentile(depth[valid], high_q))
        if not np.isfinite(low) or not np.isfinite(high) or high <= low + 1e-8:
            return depth

        normalized = np.zeros_like(depth, dtype=np.float32)
        normalized[valid] = np.clip((depth[valid] - low) / (high - low), 0.0, 1.0)
        return normalized

    def _voxel_downsample_points(self, points, voxel_size):
        if voxel_size <= 0.0 or points.shape[0] == 0:
            return points

        voxel_indices = np.floor(points / voxel_size).astype(np.int64)
        _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)
        unique_indices = np.sort(unique_indices)
        return points[unique_indices]

    def _init_means_from_depth_backproject(self, model_cfg, data_info, init_context):
        records = list(init_context.get("records", []))
        scene_root = init_context.get("scene_root", None)
        if scene_root is None:
            raise RuntimeError("INIT_MODE=depth_backproject requires init_context['scene_root']")
        if not records:
            raise RuntimeError("INIT_MODE=depth_backproject requires non-empty train records")

        num_points = int(model_cfg.NUM_INIT_POINTS)
        depth_rel_dir = str(getattr(model_cfg, "INIT_DEPTH_DIR", "auxiliaries/depth"))
        depth_root = Path(depth_rel_dir) if os.path.isabs(depth_rel_dir) else Path(scene_root) / depth_rel_dir
        if not depth_root.exists():
            raise RuntimeError(f"Depth init directory does not exist: {depth_root}")

        near = float(getattr(model_cfg, "INIT_BACKPROJECT_NEAR", 0.2))
        far = float(getattr(model_cfg, "INIT_BACKPROJECT_FAR", 6.0))
        stride = int(max(1, int(getattr(model_cfg, "INIT_BACKPROJECT_SAMPLE_STRIDE", 4))))
        depth_eps = float(getattr(model_cfg, "INIT_BACKPROJECT_DEPTH_EPS", 1e-3))
        min_valid_points = int(getattr(model_cfg, "INIT_BACKPROJECT_MIN_VALID_POINTS", num_points))
        invert_depth = bool(getattr(model_cfg, "INIT_BACKPROJECT_DEPTH_INVERT", False))
        voxel_enabled = bool(getattr(model_cfg, "INIT_VOXEL_DOWNSAMPLE", True))
        voxel_size = float(getattr(model_cfg, "INIT_VOXEL_SIZE", 0.01))

        width = int(data_info["img_w"])
        height = int(data_info["img_h"])
        fx = float(self.fl_x)
        fy = float(self.fl_y)
        cx = float(self.cx)
        cy = float(self.cy)

        xs = np.arange(0, width, stride, dtype=np.float32)
        ys = np.arange(0, height, stride, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(xs, ys)

        points_world_all = []
        used_frames = 0
        dropped_frames = 0

        for rec in records:
            frame_key = rec.get("frame_key", None)
            transform_matrix = rec.get("transform_matrix", None)
            if frame_key is None or transform_matrix is None:
                raise RuntimeError("Each init record must include frame_key and transform_matrix")

            depth_path = self._resolve_depth_path(depth_root, frame_key)
            if depth_path is None:
                raise RuntimeError(f"Missing init depth for train frame '{frame_key}' in {depth_root}")

            depth = self._load_depth(depth_path, width, height)
            depth = self._normalize_depth_for_init(depth, model_cfg, depth_eps)
            sampled = depth[::stride, ::stride]
            if sampled.shape != grid_x.shape:
                sampled = depth[np.ix_(ys.astype(np.int32), xs.astype(np.int32))]
            valid = np.isfinite(sampled) & (sampled > depth_eps)
            if not np.any(valid):
                dropped_frames += 1
                continue

            d = sampled[valid]
            if invert_depth:
                d = 1.0 - d
            z = near + d * (far - near)

            u = grid_x[valid]
            v = grid_y[valid]
            x_cam = (u - cx) / fx * z
            y_cam = -(v - cy) / fy * z
            z_cam = -z

            points_cam = np.stack([x_cam, y_cam, z_cam, np.ones_like(x_cam)], axis=1)

            c2w = np.eye(4, dtype=np.float32)
            if torch.is_tensor(transform_matrix):
                c2w[:3, :] = transform_matrix.detach().cpu().numpy().astype(np.float32)
            else:
                c2w[:3, :] = np.asarray(transform_matrix, dtype=np.float32)

            points_world = (c2w @ points_cam.T).T[:, :3]
            points_world_all.append(points_world.astype(np.float32))
            used_frames += 1

        if not points_world_all:
            raise RuntimeError("Depth back-projection produced no valid 3D points.")

        points_world_all = np.concatenate(points_world_all, axis=0)
        raw_total_points = int(points_world_all.shape[0])

        if voxel_enabled:
            points_world_all = self._voxel_downsample_points(points_world_all, voxel_size=voxel_size)

        total_points = int(points_world_all.shape[0])
        if total_points < min_valid_points:
            raise RuntimeError(
                f"Depth back-projection points after voxel downsample ({total_points}) below "
                f"INIT_BACKPROJECT_MIN_VALID_POINTS ({min_valid_points})."
            )
        if total_points < num_points:
            raise RuntimeError(
                f"Depth back-projection points after voxel downsample ({total_points}) below "
                f"NUM_INIT_POINTS ({num_points}). Reduce INIT_VOXEL_SIZE or NUM_INIT_POINTS."
            )

        choice = np.random.choice(total_points, size=num_points, replace=False)
        sampled_points = points_world_all[choice]

        print(
            "[Init] depth_backproject: "
            f"depth_dir={depth_root}, used_frames={used_frames}, dropped_frames={dropped_frames}, "
            f"raw_points={raw_total_points}, voxel_points={total_points}, sampled_points={num_points}, "
            f"normalization={getattr(model_cfg, 'INIT_DEPTH_NORMALIZATION', 'none')}, "
            f"voxel_enabled={voxel_enabled}, voxel_size={voxel_size}"
        )

        return torch.from_numpy(sampled_points).float()

    def _build_camera(self, camtoworld):
        device = self.splats["means"].device
        c2w = torch.eye(4, device=device, dtype=torch.float32)
        c2w[:3, :] = camtoworld.to(device)
        viewmat = torch.linalg.inv(c2w)
        viewmat[1, :] *= -1
        viewmat[2, :] *= -1
        viewmat = viewmat[None]

        intrinsics = torch.tensor(
            [
                [self.fl_x, 0.0, self.cx],
                [0.0, self.fl_y, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )[None]
        return viewmat, intrinsics

    def _rasterize(self, colors, viewmats, intrinsics, img_h, img_w, backgrounds, sh_degree):
        return rasterization(
            means=self.splats["means"],
            quats=self.splats["quats"],
            scales=torch.exp(self.splats["scales"]),
            opacities=torch.sigmoid(self.splats["opacities"]),
            colors=colors,
            viewmats=viewmats,
            Ks=intrinsics,
            width=img_w,
            height=img_h,
            sh_degree=sh_degree,
            backgrounds=backgrounds,
            render_mode="RGB",
            packed=False,
        )

    def render_rgb(self, camtoworld, img_h, img_w):
        device = self.splats["means"].device
        viewmats, intrinsics = self._build_camera(camtoworld)
        colors = torch.cat([self.splats["sh0"], self.splats["shN"]], dim=1)
        backgrounds = torch.full((1, 3), self.bg_color, dtype=torch.float32, device=device)
        renders, alphas, info = self._rasterize(
            colors=colors,
            viewmats=viewmats,
            intrinsics=intrinsics,
            img_h=img_h,
            img_w=img_w,
            backgrounds=backgrounds,
            sh_degree=self.sh_degree,
        )
        return renders[0], alphas[0], info

    def render_aux_heads(self, camtoworld, img_h, img_w, heads):
        device = self.splats["means"].device
        viewmats, intrinsics = self._build_camera(camtoworld)
        backgrounds = torch.zeros((1, 3), dtype=torch.float32, device=device)
        outputs = {}
        for head in heads:
            feature_name = f"{head}_feat"
            if feature_name not in self.splats:
                outputs[head] = None
                continue
            scalar_feature = self.splats[feature_name]
            colors = scalar_feature.repeat(1, 3)
            renders, _, _ = self._rasterize(
                colors=colors,
                viewmats=viewmats,
                intrinsics=intrinsics,
                img_h=img_h,
                img_w=img_w,
                backgrounds=backgrounds,
                sh_degree=None,
            )
            outputs[head] = renders[0].mean(dim=-1, keepdim=True)
        return outputs

    def forward(self, camtoworld, img_h, img_w, render_heads=()):
        rgb, alphas, info = self.render_rgb(camtoworld, img_h, img_w)
        head_outputs = self.render_aux_heads(camtoworld, img_h, img_w, render_heads) if render_heads else {}
        return {
            "rgb": rgb,
            "depth_aux": head_outputs.get("depth"),
            "prior_aux": head_outputs.get("prior"),
            "alphas": alphas,
            "info": info,
        }
