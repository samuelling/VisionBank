"""Front end: display -> retinal image -> cones -> retinal ganglion cells -> LGN.

Every stage keeps physical units:
  rstar          isomerisations / cone / s          [T, 3 (L, M, S), H, W]
  cone_current   photocurrent, pA                   [T, 3, H, W]
  rgc_generator  spikes/s before rectification      [T, 4 (P L-centre, P M-centre, parasol, bistratified), H, W]
  rgc            firing rate, spikes/s              [T, 8, H, W]  (see CELLS)
  lgn            firing rate, spikes/s              [T, 8, H, W]
"""
import math
import torch
import torch.nn.functional as F

CELLS = ['P L-on', 'P L-off', 'P M-on', 'P M-off', 'M on', 'M off', 'K on', 'K off']
RGB2LMS = torch.tensor([[0.31392, 0.63956, 0.04653], [0.15529, 0.75795, 0.08672], [0.01772, 0.10945, 0.87275]])


class Grid:
    """Retinotopic sampling grid centred on the fovea (x right, y up, degrees)."""

    def __init__(self, width_deg, nx, ny=None, device='cpu'):
        self.nx, self.ny = nx, ny or int(round(nx * 0.75))
        self.width_deg, self.ppd, self.device = width_deg, nx / width_deg, device
        xs = (torch.arange(self.nx, device=device) - (self.nx - 1) / 2) / self.ppd
        ys = ((self.ny - 1) / 2 - torch.arange(self.ny, device=device)) / self.ppd
        self.y, self.x = torch.meshgrid(ys, xs, indexing='ij')
        self.ecc = torch.hypot(self.x, self.y)

    def index(self, x_deg, y_deg):
        col = int(round(x_deg * self.ppd + (self.nx - 1) / 2)); row = int(round((self.ny - 1) / 2 - y_deg * self.ppd))
        return min(max(row, 0), self.ny - 1), min(max(col, 0), self.nx - 1)


def srgb_to_linear(c):
    c = c.float() / 255.0 if c.dtype == torch.uint8 else c.float()
    return torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def retinal_rstar(frames, gaze, screen_width_deg, grid, p):
    """Screen frames + gaze -> isomerisation rates on the retinal grid.

    frames [F, Hs, Ws, 3] sRGB (uint8 or 0-1 float); gaze [F, 2] degrees from screen centre (x right, y up).
    The image is re-centred on the fovea frame by frame, then passed through the eye's optics.
    """
    Fn, Hs, Ws, _ = frames.shape
    lin = srgb_to_linear(frames.to(grid.device)).permute(0, 3, 1, 2)
    r = grid.ppd / (Ws / screen_width_deg)
    if r < 0.8:                                    # anti-alias when the video is finer than the model grid
        lin = F.interpolate(lin, size=(max(2, round(Hs * r)), max(2, round(Ws * r))), mode='area')
    hw, hh = screen_width_deg / 2, screen_width_deg * Hs / Ws / 2
    gaze = torch.as_tensor(gaze, dtype=torch.float32, device=grid.device).reshape(Fn, 2)
    gx = (grid.x[None] + gaze[:, 0, None, None]) / hw
    gy = -(grid.y[None] + gaze[:, 1, None, None]) / hh
    samp = F.grid_sample(lin, torch.stack([gx, gy], -1), mode='bilinear', padding_mode='zeros', align_corners=False)
    on = ((gx.abs() <= 1) & (gy.abs() <= 1)).float()[:, None]
    lin = on * samp + (1 - on) * p.offscreen
    lms = torch.einsum('ij,fjhw->fihw', RGB2LMS.to(grid.device), lin)
    td = p.peak_luminance * math.pi * (p.pupil_mm / 2) ** 2
    w = torch.tensor(p.cone_weights, device=grid.device).view(1, 3, 1, 1)
    return optics(lms * w * td * p.rstar_per_td, grid, p)


def optics(img, grid, p):
    """Mean human MTF for a given pupil (Watson 2013): (1 + (u/u1)^2)^-0.62 * sqrt(diffraction MTF at 555 nm)."""
    d = p.pupil_mm
    fy = torch.fft.fftfreq(grid.ny, 1 / grid.ppd, device=img.device)
    fx = torch.fft.rfftfreq(grid.nx, 1 / grid.ppd, device=img.device)
    u = torch.hypot(fy[:, None], fx[None])
    u1 = 21.95 - 5.512 * d + 0.3922 * d * d
    uh = (u / (d * math.pi * 1e6 / (555 * 180))).clamp(max=1)
    diff = (2 / math.pi) * (torch.acos(uh) - uh * torch.sqrt(1 - uh * uh))
    mtf = (1 + (u / u1) ** 2) ** -0.62 * torch.sqrt(diff)
    return torch.fft.irfft2(torch.fft.rfft2(img) * mtf, s=(grid.ny, grid.nx)).clamp(min=0)


# --------------------------------------------------------------------------------------------- cones
def cone_consts(c):
    q = 2 * c['beta'] * c['cdark'] / (c['k'] * c['gdark'] ** c['h'])
    smax = c['eta'] / c['phi'] * c['gdark'] * (1 + (c['cdark'] / c['kGc']) ** c['n'])
    return q, smax


def cone_update(s, R, dt, c, q, smax, freeze_slow=False):
    """One semi-implicit step of the phototransduction cascade (same equations as ISETBio osAdaptTemporal)."""
    s['opsin'] = (s['opsin'] + dt * c['OpsinGain'] * R) / (1 + dt * c['sigma'])
    s['pde'] = (s['pde'] + dt * (s['opsin'] + c['eta'])) / (1 + dt * c['phi'])
    cur = c['k'] * s['cgmp'] ** c['h'] / (1 + s['ca_slow'] / c['cdark'])
    s['ca'] = (s['ca'] + dt * q * cur) / (1 + dt * c['beta'])
    if not freeze_slow:
        s['ca_slow'] = (s['ca_slow'] + dt * c['betaSlow'] * s['ca']) / (1 + dt * c['betaSlow'])
    st = smax / (1 + (s['ca'] / c['kGc']) ** c['n'])
    s['cgmp'] = (s['cgmp'] + dt * st) / (1 + dt * s['pde'])
    return -c['k'] * s['cgmp'] ** c['h'] / (1 + s['ca_slow'] / c['cdark'])


def cone_dark_state(c, like):
    one = torch.ones_like(like)
    return dict(opsin=0 * one, pde=one * c['eta'] / c['phi'], cgmp=one * c['gdark'], ca=one * c['cdark'], ca_slow=one * c['cdark'])


def cone_settle(R, c):
    q, smax = cone_consts(c); s = cone_dark_state(c, R)
    for _ in range(400):
        cur = cone_update(s, R, 0.05, c, q, smax)
    for _ in range(300):
        cur = cone_update(s, R, 0.001, c, q, smax)
    return s, cur


def cone_transfer(c, background, dt, nsub, n=512):
    """Discrete-time small-signal response of the fast cascade: pA per unit Weber contrast, sampled every dt.
    The slow calcium pool is held fixed so the equaliser built from this stays short."""
    R0 = torch.tensor([float(background)], dtype=torch.float64)
    c = {k: float(v) for k, v in c.items()}
    s, i0 = cone_settle(R0, c); q, smax = cone_consts(c)
    eps, out = 0.01, []
    for t in range(n):
        R = R0 * (1 + eps) if t == 0 else R0
        for _ in range(nsub):
            cur = cone_update(s, R, dt / nsub, c, q, smax, freeze_slow=True)
        out.append((cur - i0) / eps)
    return torch.cat(out)


# ------------------------------------------------------------------------------ receptive-field sizes
def midget_spacing(ecc, p):
    """Single-polarity midget RF spacing in degrees (Watson 2014), averaged over the four meridians."""
    d = 0
    for a, r2, re in p.watson_meridians:
        d = d + (a * (1 + ecc / r2) ** -2 + (1 - a) * torch.exp(-ecc / re))
    d = d / 4 * 2 * p.watson_dc0 / p.watson_f0
    dm = 0.5 * d * p.watson_f0 / (1 + ecc / p.watson_rm)
    return torch.sqrt(2 / (math.sqrt(3) * dm))


def rf_radii(ecc, p):
    """Characteristic (1/e) radii in degrees for each cell class."""
    rcP = p.midget_rc_per_spacing * midget_spacing(ecc, p) * p.rf_scale
    rcM = p.parasol_rc_ratio * rcP
    rsM = torch.maximum((p.m_rs_fit[0] * ecc.clamp(min=1e-3) ** p.m_rs_fit[1]).clamp(min=0.1) * p.rf_scale, 3 * rcM)
    rcK = p.k_rc_ratio * rcM
    return dict(Pc=rcP, Ps=p.p_rs_over_rc * rcP, Mc=rcM, Ms=rsM, Kc=rcK, Ks=1.3 * rcK)


class SpaceVariantBlur:
    """Gaussian blur whose width varies across the image: exact Gaussians on a 1/3-octave ladder, interpolated per pixel."""

    def __init__(self, ny, nx, sig_max, device, sig_min=0.3, per_octave=3):
        self.per = per_octave; self.ny, self.nx = ny, nx
        K = max(2, int(math.ceil(math.log2(max(sig_max, sig_min * 2) / sig_min) * per_octave)) + 1)
        self.sig = sig_min * 2 ** (torch.arange(K, device=device) / per_octave)
        self.pad = int(min(max(8, math.ceil(2.5 * float(self.sig[-1]))), 48))
        Hp, Wp = ny + 2 * self.pad, nx + 2 * self.pad
        f2 = torch.fft.fftfreq(Hp, device=device)[:, None] ** 2 + torch.fft.rfftfreq(Wp, device=device)[None] ** 2
        self.G = torch.exp(-2 * math.pi ** 2 * self.sig[:, None, None] ** 2 * f2)
        self.shape = (Hp, Wp)

    def weights(self, sigma_px):
        pos = (torch.log2(sigma_px.clamp(float(self.sig[0]), float(self.sig[-1])) / self.sig[0]) * self.per).clamp(0, len(self.sig) - 1 - 1e-4)
        i0 = pos.floor().long(); fr = pos - i0
        W = torch.zeros(len(self.sig), *sigma_px.shape, device=sigma_px.device)
        W.scatter_(0, i0[None], (1 - fr)[None]); W.scatter_add_(0, (i0 + 1)[None], fr[None])
        return W

    def __call__(self, x, wlist):
        """x [N, H, W] -> one blurred [N, H, W] per weight map in wlist."""
        xp = F.pad(x[:, None], (self.pad,) * 4, mode='replicate')[:, 0]
        X = torch.fft.rfft2(xp); outs = [torch.zeros_like(x) for _ in wlist]
        use = torch.stack([w.flatten(1).amax(1) > 0 for w in wlist]).any(0)
        for k in range(len(self.sig)):
            if not use[k]:
                continue
            b = torch.fft.irfft2(X * self.G[k], s=self.shape)[:, self.pad:self.pad + self.ny, self.pad:self.pad + self.nx]
            for o, w in zip(outs, wlist):
                o += w[k] * b
        return outs


# ----------------------------------------------------------------------------------- temporal filters
def bk_response(f, A, D, Hs, tauS, tauL, NL):
    """Benardete & Kaplan linear filter: delay, one subtractive high-pass stage, NL low-pass stages."""
    w = 2j * math.pi * f
    hp = 1 - Hs / (1 + w * tauS) if Hs else 1
    return A * torch.exp(-w * D) * hp * (1 + w * tauL) ** (-NL)


class FIR:
    """Causal temporal filter applied along the first axis, with state carried between chunks."""

    def __init__(self, h):
        self.k = h.flip(0).reshape(1, 1, -1).float(); self.L = h.numel(); self.tail = None

    def __call__(self, x):
        T = x.shape[0]; z = x.reshape(T, -1).t().unsqueeze(1)
        if self.tail is None:
            self.tail = z[:, :, :1].expand(-1, -1, self.L - 1).clone()
        z = torch.cat([self.tail, z], 2); self.tail = z[:, :, -(self.L - 1):].clone()
        return F.conv1d(z, self.k.to(x.device)).squeeze(1).t().reshape(x.shape)


def design_fir(Hfun, dt, seconds, Href=None, n=4096):
    """Impulse response of a discrete-time filter with frequency response Hfun(f) / Href(f)."""
    f = torch.fft.rfftfreq(n, dt).double()
    H = Hfun(f)
    if Href is not None:
        H = H * Href.conj() / (Href.abs() ** 2 + (1e-3 * Href.abs().max()) ** 2)
    h = torch.fft.irfft(H, n)
    L = int(round(seconds / dt))
    return h[:L].float(), float(h[L:].abs().sum() / h.abs().sum())


class FrontEnd:
    def __init__(self, grid, p, dt):
        self.g, self.p, self.dt, dev = grid, p, dt, grid.device
        self.nsub = max(1, int(math.ceil(dt / p.cone_dt)))
        fov = (grid.ecc <= p.cone_fovea_cutoff_deg)[None]
        self.cone = {k: (torch.where(fov, torch.tensor(float(p.cone_fovea[k])), torch.tensor(float(p.cone_periphery[k]))).to(dev)
                         if p.cone_fovea[k] != p.cone_periphery[k] else float(p.cone_fovea[k])) for k in p.cone_fovea}
        self.q, self.smax = cone_consts(self.cone)
        # receptive fields
        ecc = grid.ecc; r = rf_radii(ecc, p); self.radii = r
        sig = {k: v / math.sqrt(2) * grid.ppd for k, v in r.items()}          # Gaussian sigma in pixels
        sig['Mf'] = sig['Mc'] / 3
        sig['pool'] = sig['Mc'] * p.m_contrast_pool
        sig['sf'] = sig['Ms'] * p.lgn_sf_radius
        self.blur = SpaceVariantBlur(grid.ny, grid.nx, float(max(v.max() for v in sig.values())), dev)
        self.w = {k: self.blur.weights(v) for k, v in sig.items()}
        self.purity = torch.exp(-ecc / p.cone_purity_ecc)
        self.scP = p.p_sc_ratio[0] + p.p_sc_ratio[1] * ecc
        self.sfac = torch.clamp((ecc + 1.5 / grid.ppd - 0.1) / 0.3, 0, 1)      # no S cones in the foveola
        # temporal kernels: published retina-to-spikes filters divided by the cone stage, so the cascade
        # reproduces them exactly for small signals at the luminance of the original recordings
        ref = p.bk_reference_td * p.rstar_per_td
        hc = cone_transfer(p.cone_fovea, ref, dt, self.nsub)
        n = 4096; f = torch.fft.rfftfreq(n, dt).double()
        self.Hcone = (hc[None] * torch.exp(-2j * math.pi * f[:, None] * dt * torch.arange(hc.numel()))).sum(1)
        mk = p.m_kernel
        self.kPc, e1 = design_fir(lambda f: bk_response(f, 1.0, **p.p_center), dt, 0.6, self.Hcone)
        self.kPs, e2 = design_fir(lambda f: bk_response(f, 1.0, **p.p_surround), dt, 0.6, self.Hcone)
        self.kM, e3 = design_fir(lambda f: bk_response(f, 1.0, mk['D'], 0, 1, mk['tauL'], mk['NL']), dt, 0.6, self.Hcone)
        self.fir_truncation = max(e1, e2, e3)
        self.reset()

    def reset(self):
        self.state = None; self.iref = None
        self.f = {k: FIR(h) for k, h in [('PLc', self.kPc), ('PMc', self.kPc), ('Ps', self.kPs), ('M', self.kM), ('Mf', self.kM), ('K', self.kPc)]}
        self.m = None; self.xdc = None

    def run(self, rstar):
        """rstar [T, 3, H, W] at the model time step -> dict of stage tensors."""
        p, dt, T = self.p, self.dt, rstar.shape[0]
        if self.state is None:
            self.state, _ = cone_settle(rstar[0], self.cone)
            self.iref = cone_settle(rstar[0].mean((1, 2), keepdim=True).expand_as(rstar[0]), self.cone)[1]
        cur = torch.empty_like(rstar)
        for t in range(T):
            for _ in range(self.nsub):
                c = cone_update(self.state, rstar[t], dt / self.nsub, self.cone, self.q, self.smax)
            cur[t] = c
        y = cur - self.iref                                   # pA relative to the adapted scene mean
        lum = (2 * y[:, 0] + y[:, 1]) / 3
        w = self.w
        Lc, = self.blur(y[:, 0], [w['Pc']]); Mc, = self.blur(y[:, 1], [w['Pc']]); Sc, = self.blur(y[:, 2], [w['Kc']])
        lPc, lPs, lMc, lMs, lMf, lKs = self.blur(lum, [w['Pc'], w['Ps'], w['Mc'], w['Ms'], w['Mf'], w['Ks']])
        pur = self.purity
        sur = self.f['Ps'](lPs) * self.scP
        gPL = p.p_gain * (self.f['PLc'](pur * Lc + (1 - pur) * lPc) - sur)
        gPM = p.p_gain * (self.f['PMc'](pur * Mc + (1 - pur) * lPc) - sur)
        gK = p.k_gain * self.f['K'](self.sfac * (Sc - lKs))
        xM = self.f['M'](lMc - p.m_sc_ratio * lMs)            # contrast units, before the adaptive high-pass stage
        xf = self.f['Mf'](lMf - lMc)                          # fine structure inside the centre (bipolar subunits)
        xac = xM - self._slow(xM)                              # temporal contrast only: static luminance steps do not drive gain control
        q, qsf = self.blur(xac * xac, [w['pool'], w['sf']])
        uf, = self.blur(xf.abs(), [w['Mc']])
        yM, sub, cM, cSF = self._magno(xM, q, qsf, uf)
        gM = p.m_gain * yM; gS = p.m_subunit_weight * p.m_gain * sub
        R0 = p.maintained
        rgc = torch.stack([R0['P'] + gPL, R0['P'] - gPL, R0['P'] + gPM, R0['P'] - gPM,
                           R0['M'] + gM + gS, R0['M'] - gM + gS, R0['K'] + gK, R0['K'] - 0.6 * gK], 1).clamp(min=0)
        base = torch.tensor([R0['P']] * 4 + [R0['M']] * 2 + [R0['K']] * 2, device=rgc.device).view(1, 8, 1, 1)
        s = cSF / (cSF + p.lgn_sf_c50)
        supp = torch.stack([1 / (1 + 0.5 * p.lgn_sf_weight * s)] * 4 + [1 / (1 + p.lgn_sf_weight * s)] * 2 + [1 / (1 + 0.5 * p.lgn_sf_weight * s)] * 2, 1)
        lgn = (p.lgn_transfer * p.relay_gain * (rgc - p.lgn_threshold * base)).clamp(min=0) * supp
        return dict(rstar=rstar, cone_current=cur, rgc_generator=torch.stack([gPL, gPM, gM, gK], 1), m_contrast=cM[:, None], rgc=rgc, lgn=lgn)

    def _slow(self, x, tau=0.5):
        if self.xdc is None: self.xdc = x[0].clone()
        a = 1 - math.exp(-self.dt / tau); out = torch.empty_like(x)
        for t in range(x.shape[0]):
            self.xdc += a * (x[t] - self.xdc); out[t] = self.xdc
        return out

    def _magno(self, x, q, qsf, uf):
        """Parasol contrast gain control (Benardete & Kaplan 1999): the high-pass time constant shortens with local contrast."""
        p, dt, mk = self.p, self.dt, self.p.m_kernel
        if self.m is None:
            self.m = dict(c2=q[0].clone(), c2sf=qsf[0].clone(), s=x[0].clone(), us=uf[0].clone(), xp=x[0].clone())
        m = self.m; ac = 1 - math.exp(-dt / p.m_contrast_tau); au = 1 - math.exp(-dt / 0.1)
        y = torch.empty_like(x); sub = torch.empty_like(x); cM = torch.empty_like(x); cS = torch.empty_like(x)
        for t in range(x.shape[0]):
            m['c2'] += ac * (q[t] - m['c2']); m['c2sf'] += ac * (qsf[t] - m['c2sf'])
            c = torch.sqrt(2 * m['c2'].clamp(min=0))
            tau = mk['T0'] / (1 + (c.clamp(max=p.m_contrast_max) / mk['C_half']) ** 2)
            m['s'] += dt / (2 * tau + dt) * (x[t] + m['xp'] - 2 * m['s'])      # trapezoidal: exact low-frequency gain even when tau < dt
            m['xp'] = x[t]
            y[t] = x[t] - mk['Hs'] * m['s']
            m['us'] += au * (uf[t] - m['us']); sub[t] = uf[t] - m['us']
            cM[t] = c; cS[t] = torch.sqrt(2 * m['c2sf'].clamp(min=0))
        return y, sub, cM, cS
