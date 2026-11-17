"""Test stimuli as sRGB float frames [F, H, W, 3] (so low contrasts are not quantised)."""
import math
import numpy as np
import torch


def lin_to_srgb(x):
    x = x.clamp(0, 1)
    return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)


def grating(n_frames, fps, width_deg, sf=1.0, tf=4.0, contrast=0.5, ori=0.0, radius=None, mean=0.5, nx=128, ny=96,
            onset=0.0, offset=None, counterphase=False, mask=0.0, surround=0.0, surround_ori=None, center=(0.0, 0.0)):
    """Drifting (or counterphase) grating in a soft-edged patch; optional orthogonal mask and surround annulus."""
    ppd = nx / width_deg
    x = (torch.arange(nx) - (nx - 1) / 2) / ppd - center[0]; y = ((ny - 1) / 2 - torch.arange(ny)) / ppd - center[1]
    Y, X = torch.meshgrid(y, x, indexing='ij'); t = torch.arange(n_frames).view(-1, 1, 1) / fps
    th = math.radians(ori); u = -X * math.sin(th) + Y * math.cos(th); v = X * math.cos(th) + Y * math.sin(th)
    def carrier(u_):
        ph = 2 * math.pi * sf * u_[None]
        return torch.cos(ph) * torch.cos(2 * math.pi * tf * t) if counterphase else torch.cos(ph - 2 * math.pi * tf * t)
    m = contrast * carrier(u) + mask * carrier(v)
    if radius is not None:
        r = torch.hypot(X, Y); e = max(0.08 * radius, 1.5 / ppd)
        wc = 1 - ((r - (radius - e)) / (2 * e)).clamp(0, 1); m = m * wc[None]
        if surround:
            ths = math.radians(ori if surround_ori is None else surround_ori)
            ws = ((r - (radius * 1.15 - e)) / (2 * e)).clamp(0, 1)
            m = m + surround * ws[None] * carrier(-X * math.sin(ths) + Y * math.cos(ths))
    on = (t >= onset) & ((t < offset) if offset is not None else True)
    lum = mean * (1 + m.clamp(-1, 1) * on)
    return lin_to_srgb(lum)[..., None].expand(-1, -1, -1, 3).contiguous()


def gameplay(n_frames, fps, nx=160, ny=120, seed=0):
    """A stand-in for game footage: textured scene, moving and looming objects, flashes, slow camera pan."""
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(torch.linspace(-0.75, 0.75, ny), torch.linspace(-1, 1, nx), indexing='ij')
    def noise(scale):
        n = torch.randn(1, 1, max(2, int(ny / scale)), max(2, int(nx / scale)), generator=g)
        return torch.nn.functional.interpolate(n, size=(ny * 2, nx * 2), mode='bicubic', align_corners=False)[0, 0]
    bg = 0.5 * noise(24) + 0.3 * noise(8) + 0.2 * noise(3); bg = (bg - bg.mean()) / bg.std()
    K = 7; pos = torch.rand(K, 2, generator=g) * 2 - 1; vel = (torch.rand(K, 2, generator=g) - 0.5) * 0.9
    size = 0.05 + 0.2 * torch.rand(K, generator=g); col = torch.rand(K, 3, generator=g); kind = torch.randint(0, 3, (K,), generator=g)
    frames = torch.empty(n_frames, ny, nx, 3)
    for f in range(n_frames):
        t = f / fps
        ox = int((0.5 + 0.5 * math.sin(0.23 * t)) * nx); oy = int((0.5 + 0.5 * math.sin(0.17 * t + 1)) * ny)
        b = bg[oy:oy + ny, ox:ox + nx]
        img = (0.32 + 0.13 * b)[..., None].repeat(1, 1, 3) * (0.8 + 0.4 * (0.5 + 0.5 * math.sin(0.31 * t)))
        for k in range(K):
            p = pos[k] + vel[k] * t; p = ((p + 1) % 2) - 1
            s = float(size[k]) * (1 + 0.6 * math.sin(0.9 * t + k))
            d = torch.hypot(xx - p[0], (yy - p[1] * 0.75))
            if kind[k] == 0: m = (d < s).float()
            elif kind[k] == 1: m = (d < s).float() * (0.5 + 0.5 * torch.sign(torch.sin(40 * (xx - p[0]) + 6 * t)))
            else: m = torch.exp(-(d / s) ** 2) * (0.5 + 0.5 * torch.cos(60 * (yy - p[1] * 0.75) - 9 * t))
            img = img * (1 - m[..., None]) + m[..., None] * col[k]
        if int(t * 0.7) % 4 == 3 and (t * 0.7) % 1 < 0.08: img = img * 0.3 + 0.7
        frames[f] = img
    return lin_to_srgb(frames)
