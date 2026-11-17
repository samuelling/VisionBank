# VisionBank

An image-computable model of the early human visual pathway (retina → LGN → V1) that runs in the browser on a webcam, a video or photo, or built-in test stimuli. Everything is one self-contained page: `index.html`.

Open `index.html` over HTTPS (GitHub Pages works) or from disk and press **Webcam**.

## What is modelled

| Stage | Contents |
|---|---|
| Optics, cones | sRGB → L, M, S; eccentricity-dependent blur; photocurrent low-pass; Weber gain control per cone class with a slow local adaptation pool |
| Retinal ganglion cells | Midget (L- and M-centre, mixed surround), parasol (transient, divisive contrast gain control, rectified subunits), small bistratified (S − (L+M)); receptive fields scale with eccentricity; ON and OFF rectified separately |
| LGN | Parvo, magno and konio relays: spike threshold, divisive suppressive field, state-dependent relay gain |
| V1 | Quadrature Gabor pairs (8 orientations × 4 octave SF bands), energy model, divisive normalization (local cross-orientation and cross-SF pool plus an orientation-tuned surround), orientation-specific contrast adaptation, direction from phase advance, single- and double-opponent colour, eccentricity-dependent SF preference, cortical flat map |

The **stimulus bench** runs contrast-response, spatial-frequency, orientation and size sweeps at the probe location, so the nonlinearities can be measured rather than assumed.

## Layout

- `src/shaders.js`: one fragment shader per model stage
- `src/engine.js`: WebGL2 plumbing and the per-frame pipeline
- `src/ui.js`, `src/body.html`, `src/style.css`: the dashboard
- `build.mjs`: bundles `src/` into `index.html` (`node build.mjs`)

## Limits

The camera sets the ceiling: 30 frames/s undersamples the magno pathway in time, and foveal midget cells are finer than the pixels unless the field of view is reduced. Monocular, deterministic (no noise or spikes), no corticogeniculate feedback, no rods. Parameters follow macaque physiology and human fMRI where available and are all adjustable in the page.
