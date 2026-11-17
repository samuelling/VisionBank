"""Front-end checks against the published kernels. Run: python tests/check_frontend.py"""
import sys, math, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np, torch
from visionbank import Model, Params
from visionbank.frontend import bk_response
from visionbank.stimuli import grating

torch.set_num_threads(2)
FPS = 240


def f1(x, f, dt, t0):
    n = np.arange(len(x)); k = n * dt >= t0
    return 2 * np.mean(x[k] * np.exp(-2j * np.pi * f * n[k] * dt))


def flicker(m, f, c, secs=2.0, probe=(0.5, 0.0)):
    fr = grating(int(secs * FPS), FPS, m.grid.width_deg, sf=0.0, tf=f, contrast=c, nx=16, ny=16)
    return m.run(fr, probes=(probe,), features=False)


def main():
    p = Params(); p.pupil_mm = 4.0
    p.peak_luminance = p.bk_reference_td / (math.pi * 2 ** 2 * 0.5)          # mean grey = 1123 td, as in the recordings
    m = Model(2.0, nx=16, ny=16, fps=FPS, dt=1 / FPS, params=p)
    E = float(m.grid.ecc[m.grid.index(0.5, 0.0)]); sc = p.p_sc_ratio[0] + p.p_sc_ratio[1] * E
    print('FIR truncation error: %.3f' % m.fe.fir_truncation)
    print('\nP cell (L-centre) generator, uniform flicker, contrast 0.05: model vs Benardete-Kaplan kernels')
    print('  Hz   model amp  target amp   ratio   model phase  target phase (deg)')
    worst = 0
    for f in (1, 2, 4, 8, 16, 32):
        r = flicker(m, f, 0.05); z = f1(r.traces['rgc_generator'][:, 0, 0], f, m.dt, 1.0) / 0.05
        ff = torch.tensor([float(f)], dtype=torch.float64)
        k = p.p_gain * (bk_response(ff, 1.0, **p.p_center) - sc * bk_response(ff, 1.0, **p.p_surround))[0]
        k = complex(k); worst = max(worst, abs(abs(z) / abs(k) - 1))
        print('  %2d   %8.2f   %8.2f   %6.3f   %8.1f    %8.1f' % (f, abs(z), abs(k), abs(z) / abs(k), np.degrees(np.angle(z)), np.degrees(np.angle(k))))
    print('  worst amplitude error: %.1f%%' % (100 * worst))
    print('\nParasol generator, uniform flicker: gain (spikes/s per unit contrast) and phase by contrast')
    print('  (above ~0.1 contrast the model holds the time constant, so it is deliberately not compared with the published fit there)')
    mk = p.m_kernel; dog = 1 - p.m_sc_ratio
    print('  Hz  ' + '   '.join('c=%.2f: model / B&K' % c for c in (0.01, 0.05, 0.1)))
    for f in (1, 2, 4, 8, 16, 32):
        row = []
        for c in (0.01, 0.05, 0.1):
            r = flicker(m, f, c); z = f1(r.traces['rgc_generator'][:, 0, 2], f, m.dt, 1.0) / c
            ce = c * dog; ff = torch.tensor([float(f)], dtype=torch.float64)
            k = complex(p.m_gain * dog * bk_response(ff, 1.0, mk['D'], mk['Hs'], mk['T0'] / (1 + (ce / mk['C_half']) ** 2), mk['tauL'], mk['NL'])[0])
            row.append('%6.1f /%6.1f  (%4.0f /%4.0f deg) c^=%.3f' % (abs(z), abs(k), np.degrees(np.angle(z)), np.degrees(np.angle(k)), r.traces['m_contrast'][240:, 0, 0].mean()))
        print('  %2d  ' % f + '  '.join(row))
    print('\nCone adaptation: photocurrent modulation for 10% contrast, 4 Hz, across backgrounds')
    for lum in (1, 10, 100, 1000):
        q = Params(); q.peak_luminance = lum
        mm = Model(2.0, nx=16, ny=16, fps=FPS, dt=1 / FPS, params=q)
        r = flicker(mm, 4, 0.1); x = r.traces['cone_current'][:, 0, 0]; rs = r.traces['rstar'][:, 0, 0].mean()
        print('  white %5d cd/m2  mean %8.0f R*/s   current %6.1f pA   modulation %.3f pA per unit contrast' % (lum, rs, x[240:].mean(), abs(f1(x, 4, mm.dt, 1.0)) / 0.1))


if __name__ == '__main__':
    main()
