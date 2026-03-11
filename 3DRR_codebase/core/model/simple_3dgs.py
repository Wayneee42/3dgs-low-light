# Copyright (c) Xuangeng Chu (xchu.contact@gmail.com)

import torch
import torch.nn as nn
from gsplat import rasterization


class Simple3DGS(nn.Module):
    def __init__(self, model_cfg, data_info):
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
