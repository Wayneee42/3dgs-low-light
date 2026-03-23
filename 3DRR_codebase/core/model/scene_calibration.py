import math

import torch
import torch.nn as nn

INIT_NEG_BIAS = -6.0


def _rgb_to_ycbcr(rgb_hwc):
    r = rgb_hwc[..., 0]
    g = rgb_hwc[..., 1]
    b = rgb_hwc[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b
    return y, cb, cr


def _ycbcr_to_rgb(y, cb, cr):
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    return torch.stack([r, g, b], dim=-1)


class SceneCalibration(nn.Module):
    def __init__(self, chroma_scale=0.05, eps=1.0e-4, max_lift=0.2, max_gain=2.0):
        super().__init__()
        self.chroma_scale = float(chroma_scale)
        self.eps = float(eps)
        self.max_lift = float(max_lift)
        self.max_gain = float(max_gain)
        init = torch.tensor([INIT_NEG_BIAS, 0.0, INIT_NEG_BIAS, 0.0, 0.0], dtype=torch.float32)
        self.params = nn.Parameter(init)
        self._sigmoid_init = float(1.0 / (1.0 + math.exp(-INIT_NEG_BIAS)))
        self._sigmoid_scale = float(1.0 / (1.0 - self._sigmoid_init))

    def _positive_zero_aligned(self, raw_value, max_value):
        value = (torch.sigmoid(raw_value) - self._sigmoid_init) * self._sigmoid_scale
        return max_value * torch.clamp(value, 0.0, 1.0)

    def decode(self):
        raw_gain, raw_contrast, raw_lift, raw_u, raw_v = self.params
        gain = self._positive_zero_aligned(raw_gain, self.max_gain)
        contrast = torch.exp(torch.clamp(0.35 * raw_contrast, -0.5, 0.5))
        lift = self._positive_zero_aligned(raw_lift, self.max_lift)
        u = self.chroma_scale * torch.tanh(raw_u)
        v = self.chroma_scale * torch.tanh(raw_v)
        return {
            'raw': self.params,
            'gain': gain,
            'contrast': contrast,
            'lift': lift,
            'u': u,
            'v': v,
        }

    def apply(self, rgb_hwc):
        decoded = self.decode()
        y, cb, cr = _rgb_to_ycbcr(torch.clamp(rgb_hwc, 0.0, 1.0))
        y = torch.clamp(y, self.eps, 1.0 - self.eps)
        y_logit = torch.logit(y, eps=self.eps)
        y_scene = torch.sigmoid(decoded['contrast'] * y_logit + decoded['gain'])
        y_scene = y_scene + decoded['lift'] * (1.0 - y_scene)
        y_scene = torch.clamp(y_scene, self.eps, 1.0 - self.eps)
        cb_scene = cb + decoded['u']
        cr_scene = cr + decoded['v']
        calibrated = torch.clamp(_ycbcr_to_rgb(y_scene, cb_scene, cr_scene), 0.0, 1.0)
        return calibrated, decoded

