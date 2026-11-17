"""V1: quadrature Gabor bank on the LGN signal, energy, divisive normalisation with a delayed pool.

  v1_drive   [T, S, 8] at probes   filter amplitude sqrt(e^2 + o^2), LGN spikes/s
  v1_rate    [T, S, 8] at probes   complex-cell firing rate, spikes/s
Feature maps handed to the voxel read-out (per band, unweighted by eccentricity):
  E  orientation-summed energy, (spikes/s)^2      N  the same energy, pooled over space and low-passed in time
"""
import math
import torch
import torch.nn.functional as F
from .frontend import FIR

GCOMP = [1.0, 1.268, 1.344, 1.364, 1.37]      # undoes the pyramid low-pass at each band's centre frequency


def gabor_bank(device):
    """8 orientations x quadrature pair, period 4 px, ~1.4 octave bandwidth, ~20 deg orientation half-width."""
    R, f, sx, sy = 5, 0.25, 1.66, 2.0
    d = torch.arange(-R, R + 1, dtype=torch.float64)
    dy, dx = torch.meshgrid(d, d, indexing='ij'); dy = -dy            # image rows run downward; keep y up
    ev, od = [], []
    for k in range(8):
        th = k * math.pi / 8
        x = -dx * math.sin(th) + dy * math.cos(th); y = dx * math.cos(th) + dy * math.sin(th)
        env = torch.exp(-x * x / (2 * sx * sx) - y * y / (2 * sy * sy)) * ((dx * dx + dy * dy) <= 30.3)
        e = env * torch.cos(2 * math.pi * f * x); o = env * torch.sin(2 * math.pi * f * x)
        e = e - env * e.sum() / env.sum()
        ev.append(e / (e * torch.cos(2 * math.pi * f * x)).sum()); od.append(o / (o * torch.sin(2 * math.pi * f * x)).sum())
    K = torch.stack(ev + od)

    def pooled(ff):                                # summed energy over orientations for a unit grating
        acc = 0.0
        for th0 in (0.0, math.pi / 16):
            for ph in range(8):
                g = torch.cos(2 * math.pi * ff * (-dx * math.sin(th0) + dy * math.cos(th0)) + ph * math.pi / 4)
                r = (K * g).sum((1, 2)); acc += float((r[:8] ** 2 + r[8:] ** 2).sum())
        return acc / 16
    return K.float().unsqueeze(1).to(device), pooled(0.25) + 0.5 * pooled(0.125)


def gauss2d(sigma, device):
    r = int(math.ceil(3 * sigma)); d = torch.arange(-r, r + 1, dtype=torch.float32, device=device)
    g = torch.exp(-d * d / (2 * sigma * sigma)); g = g / g.sum()
    return g, r


def blur(x, g, r):
    """Separable Gaussian on [N, C, h, w] with replicate borders."""
    N, C, h, w = x.shape; x = x.reshape(N * C, 1, h, w)
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode='replicate'), g.view(1, 1, 1, -1))
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode='replicate'), g.view(1, 1, -1, 1))
    return x.reshape(N, C, h, w)


class V1:
    def __init__(self, grid, p, dt):
        self.g, self.p, self.dt, dev = grid, p, dt, grid.device
        self.K, self.kn = gabor_bank(dev)
        self.S = p.n_bands
        self.sf = [grid.ppd * 0.25 / 2 ** s for s in range(self.S)]           # band centre SF, cycles/deg
        self.down = (torch.tensor([1., 3., 3., 1.], device=dev) / 8)
        self.gl, self.rl = gauss2d(1.5, dev); self.gs, self.rs = gauss2d(5.0, dev)
        ecc = grid.ecc[None, None]; self.popw = []
        for s in range(self.S):
            e = F.avg_pool2d(ecc, 2 ** s) if s else ecc
            fp = 1 / (p.sf_pref[0] * e + p.sf_pref[1])
            self.popw.append(torch.exp(-(torch.log2(self.sf[s] / fp) / p.sf_pop_sigma) ** 2))   # squared weight, for energies
        a = 1 - math.exp(-dt / p.v1_tau_norm); L = int(math.ceil(6 * p.v1_tau_norm / dt))
        self.hn = a * (1 - a) ** torch.arange(L, dtype=torch.float32); self.hn = self.hn / self.hn.sum()
        self.reset()

    def reset(self):
        self.fpool = [FIR(self.hn) for _ in range(self.S)]; self.fsur = [FIR(self.hn) for _ in range(self.S)]
        self.ffeat = [FIR(self.hn) for _ in range(self.S)]; self.adapt = [None] * self.S

    def pyramid(self, x):
        out = [x]
        for _ in range(1, self.S):
            y = F.pad(out[-1], (1, 1, 1, 1), mode='replicate')
            y = F.conv2d(y, self.down.view(1, 1, 1, 4), stride=(1, 2)); y = F.conv2d(y, self.down.view(1, 1, 4, 1), stride=(2, 1))
            out.append(y)
        return out

    def run(self, lgn, adapt_on=True):
        """lgn [T, 8, H, W] spikes/s at the V1 time step -> per-band tensors."""
        p = self.p
        d = lgn[:, 0::2] - lgn[:, 1::2]                                  # ON minus OFF: P-L, P-M, magno, konio
        lum = ((2 * d[:, 0] + d[:, 1]) / 3 + p.v1_magno_weight * d[:, 2])[:, None]
        T = lum.shape[0]; E, Ew, loc = [], [], []
        for s, x in enumerate(self.pyramid(lum)):
            r = F.conv2d(F.pad(x, (5, 5, 5, 5), mode='replicate'), self.K) * GCOMP[s]
            e = r[:, :8] ** 2 + r[:, 8:] ** 2                            # [T, 8, h, w]
            E.append(e); Ew.append(e * self.popw[s])
            loc.append(blur(Ew[s].sum(1, keepdim=True), self.gl, self.rl) / self.kn)
        out = dict(E=[], N=[], rate=[], drive=[])
        for s in range(self.S):
            lx = loc[s]
            if s > 0: lx = lx + 0.5 * F.interpolate(loc[s - 1], size=lx.shape[-2:], mode='area')
            if s < self.S - 1: lx = lx + 0.5 * F.interpolate(loc[s + 1], size=lx.shape[-2:], mode='bilinear', align_corners=False)
            sur = blur(Ew[s], self.gs, self.rs); sun = sur.sum(1, keepdim=True) / self.kn
            pool = p.v1_w_local * lx + p.v1_w_surround * (p.v1_surround_tuning * sur + (1 - p.v1_surround_tuning) * sun)
            pool = (1 - p.v1_delay_frac) * pool + p.v1_delay_frac * self.fpool[s](pool)   # delayed normalisation: most of the pool lags the drive
            if self.adapt[s] is None: self.adapt[s] = torch.zeros_like(Ew[s][0])
            sig = p.v1_sigma * (1 + (p.v1_k_adapt * self.adapt[s] if adapt_on else 0))
            h = p.v1_n / 2
            rate = p.v1_rmax * Ew[s].clamp(min=1e-12) ** h / (sig ** p.v1_n + pool.clamp(min=1e-12) ** h)
            if adapt_on:
                self.adapt[s] += (1 - math.exp(-T * self.dt / p.v1_tau_adapt)) * (rate.mean(0) / p.v1_rmax - self.adapt[s])
            es = E[s].sum(1, keepdim=True)
            ns = blur(es, self.gs, self.rs) / self.kn
            out['E'].append(es[:, 0]); out['N'].append(((1 - p.v1_delay_frac) * ns + p.v1_delay_frac * self.ffeat[s](ns))[:, 0])
            out['rate'].append(rate); out['drive'].append(torch.sqrt(Ew[s]))
        return out
