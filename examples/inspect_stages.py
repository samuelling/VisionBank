"""Every stage of the cascade at one retinal location, in physical units.  Run: python examples/inspect_stages.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from visionbank import Model
from visionbank.stimuli import grating

torch.set_num_threads(2)
BLUE, ORANGE, AQUA, VIOLET, INK, MUTE = '#2a78d6', '#eb6834', '#1baf7a', '#6250d6', '#0b0b0b', '#52514e'
m = Model(8.0, nx=64, ny=48, fps=120, dt=1 / 240)
frames = grating(300, 120, 8.0, sf=2.0, tf=4.0, contrast=0.5, nx=128, ny=96, onset=0.5, offset=1.75)
run = m.run(frames, probes=((2.0, 0.0),), features=False)
print(run)
t, tv, T = run.t, run.t_v1, run.traces
fig, ax = plt.subplots(6, 1, figsize=(7.2, 10.5), sharex=True)
def panel(a, title, unit, series):
    for x, y, lab, col, ls in series: a.plot(x, y, color=col, lw=1.6, ls=ls, label=lab)
    a.set_title(title, loc='left', fontsize=10.5, color=INK, fontweight='bold'); a.set_ylabel(unit, fontsize=9, color=MUTE)
    a.legend(fontsize=8, frameon=False, ncol=4, loc='lower right', bbox_to_anchor=(1.0, 0.98), handlelength=1.6, borderaxespad=0)
    a.grid(axis='y', color='#e6e5e0', lw=0.7); a.tick_params(labelsize=8, colors=MUTE)
    for s in ('top', 'right'): a.spines[s].set_visible(False)
    for s in ('left', 'bottom'): a.spines[s].set_color('#c3c2b7')
    a.axvspan(0.5, 1.75, color='#f1f0ea', lw=0, zorder=0)
panel(ax[0], 'Retinal image at the probe', 'R*/cone/s', [(t, T['rstar'][:, 0, 0], 'L', ORANGE, '-'), (t, T['rstar'][:, 0, 1], 'M', AQUA, '-'), (t, T['rstar'][:, 0, 2], 'S', BLUE, '-')])
panel(ax[1], 'Cone photocurrent', 'pA', [(t, T['cone_current'][:, 0, 0], 'L', ORANGE, '-'), (t, T['cone_current'][:, 0, 1], 'M', AQUA, '-'), (t, T['cone_current'][:, 0, 2], 'S', BLUE, '-')])
panel(ax[2], 'Retinal ganglion cells', 'spikes/s', [(t, T['rgc'][:, 0, 0], 'midget ON', BLUE, '-'), (t, T['rgc'][:, 0, 1], 'midget OFF', BLUE, '--'), (t, T['rgc'][:, 0, 4], 'parasol ON', ORANGE, '-'), (t, T['rgc'][:, 0, 5], 'parasol OFF', ORANGE, '--')])
panel(ax[3], 'LGN relay cells', 'spikes/s', [(t, T['lgn'][:, 0, 0], 'parvo ON', BLUE, '-'), (t, T['lgn'][:, 0, 1], 'parvo OFF', BLUE, '--'), (t, T['lgn'][:, 0, 4], 'magno ON', ORANGE, '-'), (t, T['lgn'][:, 0, 5], 'magno OFF', ORANGE, '--')])
panel(ax[4], 'V1 filter drive (2 c/deg band)', 'LGN spikes/s', [(tv, T['v1_drive'][:, 0, 0, 0], 'preferred orientation', AQUA, '-'), (tv, T['v1_drive'][:, 0, 0, 4], 'orthogonal', VIOLET, '-')])
panel(ax[5], 'V1 complex cell after normalisation', 'spikes/s', [(tv, T['v1_rate'][:, 0, 0, 0], 'preferred orientation', AQUA, '-'), (tv, T['v1_rate'][:, 0, 0, 4], 'orthogonal', VIOLET, '-')])
ax[5].set_xlabel('time (s)', fontsize=9, color=MUTE)
fig.suptitle('One probe, 2 deg eccentricity: 50% contrast grating, 4 Hz drift, on from 0.5 to 1.75 s', x=0.02, ha='left', fontsize=11, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.975)); fig.savefig(os.path.join(os.path.dirname(__file__), '..', 'docs', 'stages.png'), dpi=140)
