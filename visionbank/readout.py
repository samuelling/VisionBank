"""Voxel read-out: turn the model's V1 feature maps into predicted BOLD, and fit it to data.

Per voxel (all fitted):
  x0, y0, prf_sigma     population receptive field in retinal coordinates (deg)
  sf_peak, sf_bw        log-Gaussian weighting over the SF bands (the pSFT), cycles/deg and octaves
  c50, n                normalisation constant and exponent of the contrast response (the pCRF), LGN spikes/s
  hrf_scale             stretches the double-gamma HRF in time
  gain, baseline        linear, solved in closed form
Stages kept for inspection:  drive -> neural (normalised population rate, 0-1) -> bold (% signal change at each TR)
"""
import math
import torch
import torch.nn.functional as F

NAMES = ['x0', 'y0', 'prf_sigma', 'sf_peak', 'sf_bw', 'c50', 'n', 'hrf_scale']


def hrf_kernel(scale, dt, p, length=32.0):
    """Double-gamma HRF sampled at dt, one row per voxel, unit area of the positive lobe."""
    t = (torch.arange(int(length / dt), device=scale.device) * dt)[None] / scale[:, None] + 1e-6
    def gam(a): return torch.exp((a - 1) * torch.log(t) - t - math.lgamma(a))
    h = gam(p['peak']) - p['ratio'] * gam(p['under'])
    return h / h.clamp(min=0).sum(1, keepdim=True)


class FeatureBank:
    """V1 feature maps pre-blurred at a ladder of pRF sizes, so a Gaussian pRF of any size and position is one lookup."""

    def __init__(self, run, device='cpu'):
        E, N = run.E.to(device), run.N.to(device)                  # [Tb, S, Hf, Wf]
        self.T, self.S, Hf, Wf = E.shape; self.dt = run.dt_feat; self.sf = torch.tensor(run.sf, device=device)
        self.w, self.h = run.width_deg, run.height_deg; px = self.w / Wf
        x = torch.stack([E, N], 2).reshape(self.T * self.S * 2, 1, Hf, Wf)
        self.sig = [0.5 * px]; self.levels = [x.reshape(1, -1, Hf, Wf)]
        s = 1.0
        while s * px < self.w / 3:
            r = int(math.ceil(3 * s)); d = torch.arange(-r, r + 1, dtype=torch.float32, device=device)
            g = torch.exp(-d * d / (2 * s * s)); g = g / g.sum()
            y = F.conv2d(F.pad(x, (r, r, 0, 0), mode='replicate'), g.view(1, 1, 1, -1)); y = F.conv2d(F.pad(y, (0, 0, r, r), mode='replicate'), g.view(1, 1, -1, 1))
            f = 2 ** int(math.floor(math.log2(max(1.0, s / 1.5))))
            if f > 1: y = F.avg_pool2d(y, f, ceil_mode=True)
            self.sig.append(math.sqrt(s * s + 0.25) * px); self.levels.append(y.reshape(1, -1, *y.shape[-2:])); s *= 2 ** 0.5
        self.sig = torch.tensor(self.sig, device=device)

    def sample(self, x0, y0, sigma):
        """-> E, N each [V, T, S]: the feature maps averaged under each voxel's Gaussian pRF."""
        V = x0.numel(); grid = torch.stack([x0 / (self.w / 2), -y0 / (self.h / 2)], -1).view(1, V, 1, 2)
        pos = torch.log(sigma.clamp(float(self.sig[0]), float(self.sig[-1])))
        ls = torch.log(self.sig); out = 0
        for k in range(len(self.sig)):                             # linear interpolation in log sigma between ladder levels
            lo = ls[k - 1] if k > 0 else None; hi = ls[k + 1] if k < len(ls) - 1 else None
            wk = torch.ones_like(pos)
            if lo is not None: wk = torch.where(pos < ls[k], ((pos - lo) / (ls[k] - lo)).clamp(0, 1), wk)
            if hi is not None: wk = torch.where(pos >= ls[k], ((hi - pos) / (hi - ls[k])).clamp(0, 1), wk)
            if float(wk.detach().max()) == 0: continue
            v = F.grid_sample(self.levels[k], grid, mode='bilinear', padding_mode='border', align_corners=False)[0, :, :, 0]
            out = out + v * wk[None]
        out = out.reshape(self.T, self.S, 2, V).permute(3, 0, 1, 2)
        return out[..., 0], out[..., 1]


class Voxels:
    """Forward model for a set of voxels. Parameters are stored unconstrained; .params() returns them in natural units."""

    def __init__(self, bank, n_voxels, hrf, tr, n_tr, fit_n=False):
        self.bank, self.V, self.hrf, self.tr, self.n_tr, self.fit_n = bank, n_voxels, hrf, tr, n_tr, fit_n
        z = lambda v: torch.full((n_voxels,), float(v), device=bank.sf.device)
        self.raw = dict(x0=z(0), y0=z(0), prf_sigma=z(math.log(1.0)), sf_peak=z(math.log2(float(bank.sf[1]))), sf_bw=z(math.log(1.5)),
                        c50=z(math.log(6.0)), n=z(0.0), hrf_scale=z(0.0))

    def set(self, **kw):
        tf = dict(prf_sigma=torch.log, sf_peak=torch.log2, sf_bw=torch.log, c50=torch.log, hrf_scale=torch.log, n=lambda v: torch.logit(((v - 1) / 3).clamp(1e-4, 1 - 1e-4)))
        for k, v in kw.items():
            v = torch.as_tensor(v, dtype=torch.float32, device=self.bank.sf.device).expand(self.V).clone()
            self.raw[k] = tf[k](v) if k in tf else v
        return self

    def params(self):
        r = self.raw
        return dict(x0=r['x0'], y0=r['y0'], prf_sigma=torch.exp(r['prf_sigma']), sf_peak=2 ** r['sf_peak'], sf_bw=torch.exp(r['sf_bw']),
                    c50=torch.exp(r['c50']), n=1 + 3 * torch.sigmoid(r['n']), hrf_scale=torch.exp(r['hrf_scale']))

    def neural(self):
        """-> dict with drive, pool, neural, each [V, T] at the feature time step."""
        q = self.params(); E, N = self.bank.sample(q['x0'], q['y0'], q['prf_sigma'])
        a = torch.exp(-0.5 * ((torch.log2(self.bank.sf)[None] - torch.log2(q['sf_peak'])[:, None]) / q['sf_bw'][:, None]) ** 2)
        a = a / a.sum(1, keepdim=True)
        D = (E * a[:, None]).sum(2).clamp(min=1e-9); P = (N * a[:, None]).sum(2).clamp(min=1e-9); h = q['n'][:, None] / 2
        return dict(drive=torch.sqrt(D), pool=torch.sqrt(P), neural=D ** h / (q['c50'][:, None] ** (2 * h) + P ** h))

    def bold_basis(self, neural):
        """Neural time course convolved with each voxel's HRF and sampled at the TRs -> [V, n_tr]."""
        k = hrf_kernel(self.params()['hrf_scale'], self.bank.dt, self.hrf); L = k.shape[1]
        y = F.conv1d(F.pad(neural[None], (L - 1, 0)), k.flip(1)[:, None], groups=self.V)[0]
        idx = (torch.arange(self.n_tr, device=y.device) * self.tr / self.bank.dt).round().long().clamp(max=y.shape[1] - 1)
        return y[:, idx]

    def predict(self, data=None, gain=None, baseline=None):
        """Predicted BOLD [V, n_tr]. With data, gain and baseline come from least squares; otherwise pass them in."""
        st = self.neural(); b = self.bold_basis(st['neural'])
        if data is not None:
            bm, dm = b.mean(1, keepdim=True), data.mean(1, keepdim=True)
            gain = ((b - bm) * (data - dm)).sum(1) / ((b - bm) ** 2).sum(1).clamp(min=1e-12); baseline = dm[:, 0] - gain * bm[:, 0]
        st.update(bold=gain[:, None] * b + baseline[:, None], gain=gain, baseline=baseline)
        return st


def r2(pred, data):
    return 1 - ((pred - data) ** 2).sum(1) / ((data - data.mean(1, keepdim=True)) ** 2).sum(1)


def fit(bank, data, tr, hrf, n_iter=300, lr=0.05, fit_n=False, verbose=False):
    """Fit every voxel in `data` [V, n_tr] (percent signal change). Coarse grid search, then gradient descent on all voxels at once."""
    V, n_tr = data.shape; dev = data.device
    # coarse search: pRF centre x size x SF peak, other parameters at defaults
    xs = torch.linspace(-bank.w / 2, bank.w / 2, 9, device=dev)[1:-1]; ys = torch.linspace(-bank.h / 2, bank.h / 2, 7, device=dev)[1:-1]
    sg = torch.tensor([0.7, 1.5, 3.0], device=dev) * (bank.w / 20); sp = bank.sf
    gx, gy, gs, gp = [t.flatten() for t in torch.meshgrid(xs, ys, sg, sp, indexing='ij')]
    cand = Voxels(bank, gx.numel(), hrf, tr, n_tr).set(x0=gx, y0=gy, prf_sigma=gs, sf_peak=gp)
    with torch.no_grad():
        b = cand.bold_basis(cand.neural()['neural'])
        bz = (b - b.mean(1, keepdim=True)); bz = bz / bz.norm(dim=1, keepdim=True).clamp(min=1e-9)
        dz = (data - data.mean(1, keepdim=True)); dz = dz / dz.norm(dim=1, keepdim=True).clamp(min=1e-9)
        best = (dz @ bz.t()).argmax(1)
    vox = Voxels(bank, V, hrf, tr, n_tr, fit_n).set(x0=gx[best], y0=gy[best], prf_sigma=gs[best], sf_peak=gp[best])
    keys = [k for k in NAMES if fit_n or k != 'n']
    for k in keys: vox.raw[k].requires_grad_(True)
    opt = torch.optim.Adam([vox.raw[k] for k in keys], lr=lr); var = data.var(1).clamp(min=1e-12)
    for i in range(n_iter):
        opt.zero_grad(); out = vox.predict(data)
        loss = (((out['bold'] - data) ** 2).mean(1) / var).sum()
        loss.backward(); opt.step()
        with torch.no_grad():                                      # keep parameters in a sane range
            vox.raw['x0'].clamp_(-bank.w / 2, bank.w / 2); vox.raw['y0'].clamp_(-bank.h / 2, bank.h / 2)
            vox.raw['sf_bw'].clamp_(math.log(0.5), math.log(4.0)); vox.raw['hrf_scale'].clamp_(math.log(0.6), math.log(1.6))
            vox.raw['sf_peak'].clamp_(math.log2(float(bank.sf[-1])) - 1, math.log2(float(bank.sf[0])) + 1)
        if verbose and (i % 50 == 0 or i == n_iter - 1): print('  iter %3d  mean R2 %.3f' % (i, float(1 - loss / V)))
    for k in keys: vox.raw[k].requires_grad_(False)
    with torch.no_grad(): out = vox.predict(data)
    return vox, out
