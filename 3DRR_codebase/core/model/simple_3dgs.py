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

        self.splats = nn.ParameterDict(
            {
                "means": nn.Parameter(means),
                "quats": nn.Parameter(quats),
                "scales": nn.Parameter(scales),
                "opacities": nn.Parameter(opacities),
                "sh0": nn.Parameter(sh0),
                "shN": nn.Parameter(shN),
            }
        )

    @property
    def num_gaussians(self):
        return self.splats["means"].shape[0]

    def forward(self, camtoworld, img_h, img_w, return_depth=False, depth_render_mode="RGB+ED"):
        device = self.splats["means"].device

        c2w = torch.eye(4, device=device, dtype=torch.float32)
        c2w[:3, :] = camtoworld.to(device)
        viewmat = torch.linalg.inv(c2w)
        viewmat[1, :] *= -1
        viewmat[2, :] *= -1
        viewmat = viewmat[None]

        K = torch.tensor(
            [
                [self.fl_x, 0.0, self.cx],
                [0.0, self.fl_y, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )[None]

        colors = torch.cat([self.splats["sh0"], self.splats["shN"]], dim=1)
        bg = torch.full((1, 3), self.bg_color, dtype=torch.float32, device=device)
        render_mode = depth_render_mode if return_depth else "RGB"

        renders, alphas, info = rasterization(
            means=self.splats["means"],
            quats=self.splats["quats"],
            scales=torch.exp(self.splats["scales"]),
            opacities=torch.sigmoid(self.splats["opacities"]),
            colors=colors,
            viewmats=viewmat,
            Ks=K,
            width=img_w,
            height=img_h,
            sh_degree=self.sh_degree,
            backgrounds=bg,
            render_mode=render_mode,
            packed=False,
        )

        if return_depth:
            rendered_rgb = renders[0][..., :3]
            rendered_depth = renders[0][..., 3:4]
            return rendered_rgb, rendered_depth, alphas[0], info
        return renders[0], None, alphas[0], info
