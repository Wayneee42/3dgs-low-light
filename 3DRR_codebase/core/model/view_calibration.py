import torch
import torch.nn as nn

SOFTPLUS_ZERO_OFFSET = 0.6931471805599453


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


class ViewCalibrationTable(nn.Module):
    def __init__(self, num_views, chroma_scale=0.05, eps=1.0e-4, mode="free_affine"):
        super().__init__()
        self.num_views = int(num_views)
        self.chroma_scale = float(chroma_scale)
        self.eps = float(eps)
        self.mode = str(mode)
        if self.mode == "degradation_only":
            embedding_dim = 4
        else:
            embedding_dim = 4
        self.embedding = nn.Embedding(self.num_views, embedding_dim)
        nn.init.zeros_(self.embedding.weight)

    def decode(self, indices):
        raw = self.embedding(indices)
        if self.mode == "degradation_only":
            raw_d = raw[..., 0]
            raw_s = raw[..., 1]
            raw_u = raw[..., 2]
            raw_v = raw[..., 3]
            d = torch.clamp_min(torch.nn.functional.softplus(raw_d) - SOFTPLUS_ZERO_OFFSET, 0.0)
            chroma_atten = torch.exp(-torch.clamp_min(torch.nn.functional.softplus(raw_s) - SOFTPLUS_ZERO_OFFSET, 0.0))
            u = self.chroma_scale * torch.tanh(raw_u)
            v = self.chroma_scale * torch.tanh(raw_v)
            return {
                "raw": raw,
                "d": d,
                "s": chroma_atten,
                "u": u,
                "v": v,
            }

        log_a = raw[..., 0]
        raw_b = raw[..., 1]
        raw_u = raw[..., 2]
        raw_v = raw[..., 3]
        a = torch.exp(torch.clamp(log_a, -2.0, 2.0))
        b = 0.5 * torch.tanh(raw_b)
        u = self.chroma_scale * torch.tanh(raw_u)
        v = self.chroma_scale * torch.tanh(raw_v)
        return {
            "raw": raw,
            "a": a,
            "b": b,
            "u": u,
            "v": v,
        }

    def apply(self, rgb_hwc, indices):
        decoded = self.decode(indices)
        y, cb, cr = _rgb_to_ycbcr(torch.clamp(rgb_hwc, 0.0, 1.0))
        y = torch.clamp(y, self.eps, 1.0 - self.eps)
        y_logit = torch.logit(y, eps=self.eps)
        if self.mode == "degradation_only":
            y_obs = torch.sigmoid(y_logit - decoded["d"])
            cb_obs = decoded["s"] * cb + decoded["u"]
            cr_obs = decoded["s"] * cr + decoded["v"]
        else:
            y_obs = torch.sigmoid(decoded["a"] * y_logit + decoded["b"])
            cb_obs = cb + decoded["u"]
            cr_obs = cr + decoded["v"]
        calibrated = torch.clamp(_ycbcr_to_rgb(y_obs, cb_obs, cr_obs), 0.0, 1.0)
        return calibrated, decoded
