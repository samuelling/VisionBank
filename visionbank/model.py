"""Run the whole cascade on a video and keep every stage inspectable."""
import math
import numpy as np
import torch
import torch.nn.functional as F
from .params import Params
from .frontend import Grid, FrontEnd, retinal_rstar, CELLS
from .v1 import V1

UNITS = dict(rstar='isomerisations/cone/s', cone_current='pA', rgc_generator='spikes/s', m_contrast='contrast',
             rgc='spikes/s', lgn='spikes/s', v1_drive='LGN spikes/s', v1_rate='spikes/s')
CHANNELS = dict(rstar=['L', 'M', 'S'], cone_current=['L', 'M', 'S'], rgc_generator=['P L-centre', 'P M-centre', 'parasol', 'bistratified'],
                m_contrast=['parasol local contrast'], rgc=CELLS, lgn=CELLS)


class Run:
    """Result of Model.run: probe traces for every stage, optional map snapshots, and the feature bank for voxel fitting."""

    def __init__(self):
        self.traces, self.maps, self.units, self.channels = {}, {}, UNITS, CHANNELS

    def __repr__(self):
        s = ['Run: %.2f s, front-end dt %.2f ms, V1 dt %.2f ms, features every %.0f ms' % (self.duration, 1e3 * self.dt, 1e3 * self.dt_v1, 1e3 * self.dt_feat)]
        for k, v in self.traces.items():
            s.append('  %-14s %-18s [%s]' % (k, tuple(v.shape), UNITS[k]))
        return '\n'.join(s)


class Model:
    def __init__(self, width_deg, nx=96, ny=None, fps=30, dt=1 / 240, v1_stride=2, dt_feat=0.05, params=None, device='cpu'):
        self.p = params or Params()
        ny = ny or int(round(nx * 0.75)); m = 2 ** (self.p.n_bands - 1)
        assert nx % m == 0 and ny % m == 0, 'grid size must be a multiple of %d' % m
        self.grid = Grid(width_deg, nx, ny, device); self.fps = fps
        self.k = max(1, round(1 / (fps * dt))); self.dt = 1 / (fps * self.k)          # model steps per video frame
        self.stride = v1_stride; self.dt_v1 = self.dt * v1_stride
        self.bin = max(1, round(dt_feat / self.dt_v1)); self.dt_feat = self.dt_v1 * self.bin
        need = self.stride * self.bin; f0 = need // math.gcd(need, self.k)
        self.chunk = f0 * max(1, round(0.5 * fps / f0))                              # frames per chunk (about 0.5 s)
        self.fe = FrontEnd(self.grid, self.p, self.dt); self.v1 = V1(self.grid, self.p, self.dt_v1)

    def run(self, frames, gaze=None, screen_width_deg=None, probes=((0.0, 0.0),), maps_at=(), adapt=True, features=True, progress=None):
        """frames [F, H, W, 3] sRGB video at self.fps (array, tensor, or an iterable of such chunks).
        gaze [F, 2] in degrees from screen centre; probes: (x, y) in degrees from the fovea; maps_at: times (s) to snapshot full maps."""
        g, dev = self.grid, self.grid.device
        self.fe.reset(); self.v1.reset()
        if not hasattr(frames, 'shape'):
            frames = torch.cat([torch.as_tensor(np.asarray(f)) for f in frames])
        frames = torch.as_tensor(np.asarray(frames)) if not torch.is_tensor(frames) else frames
        nF = frames.shape[0]; gaze = torch.zeros(nF, 2) if gaze is None else torch.as_tensor(np.asarray(gaze), dtype=torch.float32)
        sw = screen_width_deg or g.width_deg
        pix = [g.index(x, y) for x, y in probes]; rows = torch.tensor([r for r, c in pix], device=dev); cols = torch.tensor([c for r, c in pix], device=dev)
        tr = {k: [] for k in UNITS}; fe, fn = [], []; run = Run(); snaps = sorted(maps_at)
        for a in range(0, nF, self.chunk):
            fr, gz = frames[a:a + self.chunk], gaze[a:a + self.chunk]; n = fr.shape[0]
            if n < self.chunk:                                                        # pad the last chunk, trim after
                fr = torch.cat([fr, fr[-1:].expand(self.chunk - n, -1, -1, -1)]); gz = torch.cat([gz, gz[-1:].expand(self.chunk - n, -1)])
            r = retinal_rstar(fr, gz, sw, g, self.p).repeat_interleave(self.k, 0)
            st = self.fe.run(r)
            T = st['lgn'].shape[0]; lg = st['lgn'].reshape(T // self.stride, self.stride, *st['lgn'].shape[1:]).mean(1)
            v = self.v1.run(lg, adapt_on=adapt)
            keep = n * self.k; kv = keep // self.stride
            for k in ('rstar', 'cone_current', 'rgc_generator', 'm_contrast', 'rgc', 'lgn'):
                tr[k].append(st[k][:keep, :, rows, cols].permute(0, 2, 1).cpu())
            for k, src in (('v1_drive', 'drive'), ('v1_rate', 'rate')):
                tr[k].append(torch.stack([v[src][s][:kv, :, rows >> s, cols >> s].permute(0, 2, 1) for s in range(self.v1.S)], 2).cpu())
            while snaps and snaps[0] < (a + n) / self.fps:
                t = snaps.pop(0); i = min(T - 1, max(0, int(round((t - a / self.fps) / self.dt)))); j = min(i // self.stride, T // self.stride - 1)
                run.maps[t] = dict({k: st[k][i].cpu() for k in ('rstar', 'cone_current', 'rgc', 'lgn')}, v1_rate=[x[j].cpu() for x in v['rate']])
            if features:
                hf, wf = g.ny // 2, g.nx // 2; kb = kv // self.bin
                for name, dst in (('E', fe), ('N', fn)):
                    bands = []
                    for s in range(self.v1.S):
                        x = v[name][s][:kb * self.bin]; x = x.reshape(kb, self.bin, *x.shape[1:]).mean(1)[:, None]
                        x = F.interpolate(x, size=(hf, wf), mode='area') if s == 0 else (x if s == 1 else F.interpolate(x, size=(hf, wf), mode='bilinear', align_corners=False))
                        bands.append(x[:, 0])
                    dst.append(torch.stack(bands, 1).cpu())
            if progress: progress(min(nF, a + self.chunk) / nF)
        run.traces = {k: torch.cat(v2).numpy() for k, v2 in tr.items()}
        run.dt, run.dt_v1, run.dt_feat, run.duration = self.dt, self.dt_v1, self.dt_feat, nF / self.fps
        run.t = np.arange(run.traces['lgn'].shape[0]) * self.dt; run.t_v1 = np.arange(run.traces['v1_rate'].shape[0]) * self.dt_v1
        run.probes, run.sf, run.width_deg, run.height_deg = list(probes), self.v1.sf, g.width_deg, g.ny / g.ppd
        if features:
            run.E, run.N = torch.cat(fe), torch.cat(fn)
        return run
