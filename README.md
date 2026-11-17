# VisionBank

An image-computable model of the early human visual pathway: retina → LGN → V1, with a voxel read-out that turns model responses into predicted BOLD and can be fitted to data.

Two parts share the repo:

- **`visionbank/` (Python, PyTorch)**: the reference model. Offline, differentiable at the read-out, every stage in physical units.
- **`index.html`**: a live browser dashboard (webcam, video or test stimuli). It uses faster, approximate filters and is a viewer, not the reference.

## Python model

```python
from visionbank import Model
from visionbank.video import read_video
from visionbank.readout import FeatureBank, fit

frames, fps = read_video('gameplay.mp4')            # [F, H, W, 3]
model = Model(width_deg=24, nx=96, ny=72, fps=round(fps))
run = model.run(frames, gaze=gaze_deg, screen_width_deg=20, probes=[(0, 0), (4, 2)])
print(run)                                          # every stage, with shapes and units
run.traces['lgn']                                   # [time, probe, cell class] in spikes/s

bank = FeatureBank(run)
voxels, out = fit(bank, bold, tr=1.0, hrf=model.p.hrf)   # bold: [voxels, TRs]
voxels.params()                                     # pRF, SF tuning, c50, HRF per voxel
out['neural'], out['bold']                          # population rate and predicted BOLD
```

### Stages and units

| Stage | Trace | Units |
|---|---|---|
| Retinal image, after gaze re-centring and optics | `rstar` | isomerisations/cone/s |
| Cones (phototransduction cascade) | `cone_current` | pA |
| Retinal ganglion cells: midget L/M ON/OFF, parasol ON/OFF, bistratified | `rgc_generator`, `rgc` | spikes/s |
| LGN relay cells | `lgn` | spikes/s |
| V1 filter drive and complex-cell rate, per SF band and orientation | `v1_drive`, `v1_rate` | LGN spikes/s, spikes/s |
| Voxel: pooled drive → normalised population rate → BOLD | `drive`, `neural`, `bold` | –, 0–1, % signal |

`maps_at=[t, ...]` in `Model.run` also stores full maps of every stage at those times.

### What each stage is built from

- **Optics**: Watson (2013) mean human MTF for the chosen pupil.
- **Cones**: the ISETBio / Rieke-lab phototransduction cascade, foveal and peripheral parameter sets.
- **RGC space**: centre sizes from Watson's (2014) midget density formula; surround size and weight from Croner & Kaplan (1995).
- **RGC time**: Benardete & Kaplan kernels (P centre and surround; M with contrast gain control). The kernels are divided by the cone stage's own transfer function, so cones plus post-receptoral filter reproduce the published kernel at the luminance of the recordings.
- **LGN**: relay gain, threshold, divisive suppressive field.
- **V1**: quadrature Gabor bank (8 orientations × 4 octave bands), energy, divisive normalisation with a local pool and a tuned surround, a delayed pool (onset transients, sub-additive temporal summation), slow contrast adaptation.
- **Read-out**: Gaussian pRF, log-Gaussian SF weighting, voxel-level normalisation (c50, exponent), double-gamma HRF.

`visionbank/params.py` lists every constant with its source, tagged **[V]** (checked against the source while building) or **[A]** (approximate; a starting point).

### Checks

- `python tests/check_frontend.py`: midget generator matches the Benardete–Kaplan kernels within 6% in amplitude and 5° in phase from 1 to 32 Hz; parasol matches within about 10% from 2 Hz up (20% at 1 Hz) for contrasts to 10%, including the contrast-dependent phase advance.
- `python examples/recover.py`: 60 simulated voxels driven by game-like video with eye movements, noise added, fitted on one run and tested on another. Held-out R² 0.72 against a ceiling of 0.74; pRF position recovered at r = 0.98, c50 0.79, HRF timing 0.82, pRF size 0.71, SF peak 0.46.
- `python examples/inspect_stages.py`: writes `docs/stages.png`, every stage at one probe.

### Known limits

- LGN, V1 normalisation timing and the troland-to-isomerisation conversion are **[A]** values.
- The cone model saturates above roughly 5 × 10⁴ isomerisations/s (no pigment bleaching).
- Spatial tuning has not yet been compared against Croner & Kaplan's spatial-frequency curves.
- Foveal midget cells are finer than any practical grid; near the fovea the grid, not the cell, sets resolution.
- SF peak is weakly identified with four octave bands and short runs.
- Monocular, deterministic, no corticogeniculate feedback, no rods.

## Dashboard

Open `index.html` over HTTPS (GitHub Pages works) or from disk. Sources live in `src/`; `node build.mjs` bundles them.
