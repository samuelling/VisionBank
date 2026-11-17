"""Parameter recovery: simulate voxels with known parameters from game-like video with eye movements,
add noise, fit on one run, test on another.  Run: python examples/recover.py"""
import sys, os, math, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np, torch
from visionbank import Model
from visionbank.stimuli import gameplay
from visionbank.readout import FeatureBank, Voxels, fit, r2

torch.manual_seed(1); torch.set_num_threads(2)
FPS, SECS, TR, V, NOISE = 30, 90, 1.0, 60, 0.6
SCREEN, FIELD = 20.0, 24.0                         # screen width, and width of the model's retinal grid (deg)


def gaze_trace(n, seed):
    """Fixations of 0.3-0.7 s joined by saccades, within +/-3 deg of screen centre."""
    g = torch.Generator().manual_seed(seed); out = torch.zeros(n, 2); pos = torch.zeros(2); i = 0
    while i < n:
        d = int((0.3 + 0.4 * torch.rand(1, generator=g)) * FPS); pos = (pos + 1.5 * torch.randn(2, generator=g)).clamp(-3, 3)
        out[i:i + d] = pos; i += d
    return out


def make_run(model, seed):
    frames = gameplay(SECS * FPS, FPS, seed=seed); gaze = gaze_trace(SECS * FPS, seed)
    t0 = time.time(); run = model.run(frames, gaze=gaze, screen_width_deg=SCREEN, probes=((0.0, 0.0),))
    print('  model run: %d s of video in %.0f s' % (SECS, time.time() - t0))
    return FeatureBank(run)


def main():
    model = Model(FIELD, nx=64, ny=48, fps=FPS, dt=1 / 120)
    print('Retinal grid %d x %d over %.0f deg, bands at %s c/deg' % (model.grid.nx, model.grid.ny, FIELD, ', '.join('%.2f' % f for f in model.v1.sf)))
    A, B = make_run(model, 1), make_run(model, 2)
    n_tr = int(SECS / TR); u = lambda lo, hi: lo + (hi - lo) * torch.rand(V)
    true = dict(x0=u(-7, 7), y0=u(-5, 5), prf_sigma=u(0.8, 3.0), sf_peak=2 ** u(math.log2(0.15), math.log2(0.7)), sf_bw=u(1.0, 2.0),
                c50=torch.exp(u(math.log(4), math.log(25))), hrf_scale=u(0.85, 1.2))
    gain, base = torch.full((V,), 3.0), torch.zeros(V)
    sim = lambda bank: Voxels(bank, V, model.p.hrf, TR, n_tr).set(**true).predict(gain=gain, baseline=base)['bold']
    cleanA, cleanB = sim(A), sim(B)
    sd = NOISE * cleanA.std(1, keepdim=True)
    dataA, dataB = cleanA + sd * torch.randn_like(cleanA), cleanB + sd * torch.randn_like(cleanB)
    print('Simulated %d voxels, TR %.1f s, noise SD = %.1f x signal SD (ceiling R2 = %.2f)' % (V, TR, NOISE, 1 / (1 + NOISE ** 2)))
    t0 = time.time(); vox, out = fit(A, dataA, TR, model.p.hrf, n_iter=300, verbose=True)
    print('  fit: %.0f s' % (time.time() - t0))
    est = vox.params()
    held = Voxels(B, V, model.p.hrf, TR, n_tr); held.raw = vox.raw
    predB = held.predict(gain=out['gain'], baseline=out['baseline'])['bold']
    print('\nR2 on the fitted run   median %.2f' % float(r2(out['bold'], dataA).median()))
    print('R2 on the held-out run median %.2f   (true parameters would give %.2f)' % (float(r2(predB, dataB).median()), float(r2(cleanB, dataB).median())))
    print('\nparameter      r(true, fitted)   median |error|')
    res = {}
    for k in ('x0', 'y0', 'prf_sigma', 'sf_peak', 'c50', 'hrf_scale'):
        a, b = true[k], est[k]
        if k in ('sf_peak', 'c50', 'prf_sigma'): r = float(torch.corrcoef(torch.stack([a.log(), b.log()]))[0, 1]); e = float((b / a).log2().abs().median()); unit = 'octaves'
        else: r = float(torch.corrcoef(torch.stack([a, b]))[0, 1]); e = float((b - a).abs().median()); unit = 'deg' if k != 'hrf_scale' else ''
        res[k] = (r, e); print('  %-11s  %6.2f           %.2f %s' % (k, r, e, unit))
    return true, est, res


if __name__ == '__main__':
    main()
