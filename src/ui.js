'use strict';
/* ---------- Dashboard: views, probe read-outs, stimulus bench, parameters ---------- */
const INK = '#e9ebf1', INK2 = '#aab1c0', INK3 = '#747c8d', GRID = '#262c38';
const C_BLUE = '#3987e5', C_ORANGE = '#d95926', C_AQUA = '#199e70', C_YELLOW = '#c98500', C_MAGENTA = '#d55181', C_VIOLET = '#9085e9';
const BAND_COL = ['#cde2fb', '#86b6ef', '#3987e5', '#1c5cab'];   // one hue, light = fine, dark = coarse

const PARAMS = [
  { g: 'Geometry', id: 'fov', label: 'Image width in the visual field', unit: '°', min: 4, max: 100, v: 30, log: 1, d: 0 },
  { g: 'Optics and cones', id: 'optics', label: 'Optical blur', unit: '×', min: 0, max: 6, v: 1, d: 2 },
  { g: 'Optics and cones', id: 'tauA', label: 'Adaptation time constant', unit: ' s', min: 0.03, max: 3, v: 0.3, log: 1, d: 2 },
  { g: 'Optics and cones', id: 'sigA', label: 'Adaptation pool σ at the fovea', unit: '°', min: 0.1, max: 4, v: 0.6, log: 1, d: 2 },
  { g: 'Optics and cones', id: 'gamma', label: 'Local vs scene-wide adaptation', unit: '', min: 0, max: 1, v: 0.75, d: 2 },
  { g: 'Optics and cones', id: 'Id', label: 'Dark light', unit: '', min: 0.001, max: 0.3, v: 0.02, log: 1, d: 3 },
  { g: 'Retina', id: 'rf', label: 'Receptive-field size', unit: '×', min: 0.5, max: 4, v: 1, log: 1, d: 2 },
  { g: 'Retina', id: 'ksP', label: 'Midget surround weight', unit: '', min: 0, max: 1, v: 0.6, d: 2 },
  { g: 'Retina', id: 'ksM', label: 'Parasol surround weight', unit: '', min: 0, max: 1, v: 0.6, d: 2 },
  { g: 'Retina', id: 'wM', label: 'Parasol transience', unit: '', min: 0, max: 1, v: 0.85, d: 2 },
  { g: 'Retina', id: 'c50M', label: 'Parasol contrast-gain constant', unit: '', min: 0.005, max: 0.5, v: 0.06, log: 1, d: 3 },
  { g: 'Retina', id: 'wSub', label: 'Parasol nonlinear subunits', unit: '', min: 0, max: 3, v: 0.6, d: 2 },
  { g: 'LGN', id: 'sfL', label: 'Suppressive field strength', unit: '', min: 0, max: 3, v: 1, d: 2 },
  { g: 'LGN', id: 'relay', label: 'Relay gain (state)', unit: '×', min: 0.2, max: 2, v: 1, d: 2 },
  { g: 'LGN', id: 'thr', label: 'Spike threshold (× maintained rate)', unit: '', min: 0, max: 1.5, v: 0.6, d: 2 },
  { g: 'V1', id: 'wMV1', label: 'Magno weight into V1', unit: '', min: 0, max: 1.5, v: 0.3, d: 2 },
  { g: 'V1', id: 'sig', label: 'Semi-saturation σ', unit: '', min: 0.02, max: 1.5, v: 0.35, log: 1, d: 3 },
  { g: 'V1', id: 'n', label: 'Exponent n', unit: '', min: 1, max: 4, v: 2, d: 2 },
  { g: 'V1', id: 'wLoc', label: 'Local pool (cross-orientation, cross-SF)', unit: '', min: 0, max: 3, v: 1, d: 2 },
  { g: 'V1', id: 'wSur', label: 'Surround pool', unit: '', min: 0, max: 4, v: 1, d: 2 },
  { g: 'V1', id: 'tun', label: 'Surround orientation tuning', unit: '', min: 0, max: 1, v: 0.7, d: 2 },
  { g: 'V1', id: 'kAd', label: 'Contrast adaptation strength', unit: '', min: 0, max: 5, v: 2, d: 2 },
  { g: 'V1', id: 'tauAd', label: 'Adaptation time constant', unit: ' s', min: 0.5, max: 60, v: 6, log: 1, d: 1 },
  { g: 'V1', id: 'sw', label: 'SF population bandwidth σ', unit: ' oct', min: 0.4, max: 3, v: 1, d: 2 }
];
const STIM = [
  { id: 'c', label: 'Contrast', unit: '', min: 0.005, max: 1, log: 1, d: 3 },
  { id: 'f', label: 'Spatial frequency', unit: ' c/deg', min: 0.05, max: 12, log: 1, d: 2 },
  { id: 'ori', label: 'Orientation', unit: '°', min: 0, max: 180, d: 0 },
  { id: 'tf', label: 'Drift', unit: ' Hz', min: -8, max: 8, d: 2 },
  { id: 'R', label: 'Patch radius', unit: '°', min: 0.2, max: 60, log: 1, d: 2 },
  { id: 'mask', label: 'Orthogonal mask contrast', unit: '', min: 0, max: 0.5, d: 2 },
  { id: 'sur', label: 'Surround contrast', unit: '', min: 0, max: 1, d: 2 }
];
PARAMS.forEach((p) => { P[p.id] = p.v; });
P.gM = 30; P.gP = 2.8;

const VIEWS = [
  { id: 'input', grp: 'Eye', name: 'Retinal image', u: () => ({ u_mode: 0, u_t0: M.lms.t }) },
  { id: 'cones', grp: 'Eye', name: 'Cones, adapted', u: () => ({ u_mode: 1, u_t0: M.cone.t, u_gain: 2.2 }) },
  { id: 'rgc-rg', grp: 'RGC', name: 'Midget L − M', u: () => ({ u_mode: 2, u_t0: M.rgcA.t, u_w: [1, -1, -1, 1], u_cm: 1, u_gain: 11 }) },
  { id: 'rgc-p', grp: 'RGC', name: 'Midget ON − OFF', u: () => ({ u_mode: 2, u_t0: M.rgcA.t, u_w: [2 / 3, -2 / 3, 1 / 3, -1 / 3], u_cm: 0, u_gain: 3.2 }) },
  { id: 'rgc-m', grp: 'RGC', name: 'Parasol ON − OFF', u: () => ({ u_mode: 2, u_t0: M.rgcB.t, u_w: [1, -1, 0, 0], u_cm: 0, u_gain: 1.5 }) },
  { id: 'rgc-k', grp: 'RGC', name: 'Bistratified S−(L+M)', u: () => ({ u_mode: 2, u_t0: M.rgcB.t, u_w: [0, 0, 1, -1], u_cm: 2, u_gain: 4 }) },
  { id: 'lgn-rg', grp: 'LGN', name: 'Parvo L − M', u: () => ({ u_mode: 2, u_t0: M.lgnA.t, u_w: [1, -1, -1, 1], u_cm: 1, u_gain: 11 }) },
  { id: 'lgn-p', grp: 'LGN', name: 'Parvo ON − OFF', u: () => ({ u_mode: 2, u_t0: M.lgnA.t, u_w: [2 / 3, -2 / 3, 1 / 3, -1 / 3], u_cm: 0, u_gain: 4.5 }) },
  { id: 'lgn-m', grp: 'LGN', name: 'Magno ON − OFF', u: () => ({ u_mode: 2, u_t0: M.lgnB.t, u_w: [1, -1, 0, 0], u_cm: 0, u_gain: 2.4 }) },
  { id: 'lgn-k', grp: 'LGN', name: 'Konio S−(L+M)', u: () => ({ u_mode: 2, u_t0: M.lgnB.t, u_w: [0, 0, 1, -1], u_cm: 2, u_gain: 4.5 }) },
  { id: 'v1', grp: 'V1', name: 'Orientation, all bands', u: () => ({ u_mode: 3, u_t0: M.sumA.t, u_gain: 1.6 }) },
  { id: 'v1-0', grp: 'V1', band: 0, u: () => ({ u_mode: 4, u_t0: M.sc[0].RA.t, u_t1: M.sc[0].RB.t, u_gain: 2.4 }) },
  { id: 'v1-1', grp: 'V1', band: 1, u: () => ({ u_mode: 4, u_t0: M.sc[1].RA.t, u_t1: M.sc[1].RB.t, u_gain: 2.4 }) },
  { id: 'v1-2', grp: 'V1', band: 2, u: () => ({ u_mode: 4, u_t0: M.sc[2].RA.t, u_t1: M.sc[2].RB.t, u_gain: 2.4 }) },
  { id: 'v1-3', grp: 'V1', band: 3, u: () => ({ u_mode: 4, u_t0: M.sc[3].RA.t, u_t1: M.sc[3].RB.t, u_gain: 2.4 }) },
  { id: 'motion', grp: 'V1', name: 'Direction of motion', u: () => ({ u_mode: 5, u_t1: M.sumB.t, u_gain: 2.5 }) },
  { id: 'colour', grp: 'V1', name: 'Colour opponency', u: () => ({ u_mode: 6, u_t0: M.col.t, u_gain: 1.0 }) },
  { id: 'adapt', grp: 'V1', name: 'Adaptation state', u: () => ({ u_mode: 4, u_t0: M.sc[1].ad[M.cur][0].t, u_t1: M.sc[1].ad[M.cur][1].t, u_gain: 4 }) },
  { id: 'cortex', grp: 'V1', name: 'Flat map: LH | RH', u: () => ({ u_mode: 7, u_t0: M.sumA.t, u_gain: 1.6 }) }
];
const viewName = (v) => v.name || ('Band ' + (v.band + 1) + ': ' + fmtSF(bandSF(v.band)) + ' c/deg');
const fmtSF = (f) => f >= 10 ? f.toFixed(0) : (f >= 1 ? f.toFixed(1) : f.toFixed(2));

const UI = { view: 'all', tap: 'probe', layout: null, fps: 0, hist: [], lastProbe: null };
const BENCH = { running: false, result: null, hover: -1, geom: null };
const raf = () => new Promise((r) => requestAnimationFrame(r));

/* ----- controls ----- */
function slider(host, spec, obj, onChange) {
  const el = document.createElement('div'); el.className = 'ctl';
  const id = 'c-' + spec.id + (obj === S ? '-s' : '');
  el.innerHTML = '<label for="' + id + '">' + spec.label + '</label><output id="' + id + '-o"></output><input type="range" id="' + id + '" min="0" max="1000" step="1">';
  host.appendChild(el);
  const inp = el.querySelector('input'), out = el.querySelector('output');
  const toT = (v) => spec.log ? Math.log(v / spec.min) / Math.log(spec.max / spec.min) : (v - spec.min) / (spec.max - spec.min);
  const toV = (t) => spec.log ? spec.min * Math.pow(spec.max / spec.min, t) : spec.min + t * (spec.max - spec.min);
  const show = () => { out.textContent = (+obj[spec.id]).toFixed(spec.d) + spec.unit; };
  spec.sync = () => { inp.value = Math.round(1000 * clamp(toT(obj[spec.id]), 0, 1)); show(); };
  inp.addEventListener('input', () => { obj[spec.id] = +toV(inp.value / 1000).toFixed(4); show(); if (onChange) onChange(spec.id); });
  spec.sync();
}
function buildControls() {
  const host = $('params'); let grp = null, box = null;
  PARAMS.forEach((p) => {
    if (p.g !== grp) {
      grp = p.g; const d = document.createElement('details'); d.open = (grp === 'Geometry' || grp === 'V1');
      d.innerHTML = '<summary>' + grp + '</summary><div class="ctls"></div>'; host.appendChild(d); box = d.querySelector('.ctls');
    }
    slider(box, p, P, (id) => { if (id === 'fov') layout(); });
  });
  STIM.forEach((s) => slider($('stim-ctls'), s, S));
  $('reset').addEventListener('click', () => { PARAMS.forEach((p) => { P[p.id] = p.v; p.sync(); }); layout(); });
  $('s-surrel').addEventListener('change', (e) => { S.surRel = +e.target.value; });
  $('s-col').addEventListener('change', (e) => { S.col = +e.target.value; });
}

/* ----- sources ----- */
function setSource(kind) {
  if (ST.src === 'cam' && kind !== 'cam') stopCam();
  ST.src = kind; M.reset = 1;
  ['demo', 'stim', 'cam'].forEach((k) => $('src-' + k).setAttribute('aria-pressed', k === kind ? 'true' : 'false'));
  $('src-file-l').style.borderColor = kind === 'file' ? 'var(--accent)' : '';
}
function say(text) { const m = $('msg'); m.hidden = !text; m.textContent = text || ''; }
function stopCam() { const v = $('vid'); if (v.srcObject) { v.srcObject.getTracks().forEach((t) => t.stop()); v.srcObject = null; } }
async function startCam() {
  say('');
  try {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) throw new Error('no camera API');
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' }, audio: false });
    const v = $('vid'); v.removeAttribute('src'); v.srcObject = stream; await v.play();
    SRC.el = v; SRC.ready = false; setSource('cam');
  } catch (e) {
    say('The camera is blocked on this page. Claude-hosted pages are sandboxed, so open this same file from GitHub Pages or from disk to use the webcam. Here you can load a video or photo instead (on a phone, the picker can record one).');
  }
}
function loadFile(file) {
  if (!file) return; say('');
  const url = URL.createObjectURL(file);
  if (file.type.indexOf('video') === 0) {
    stopCam(); const v = $('vid'); v.srcObject = null; v.src = url; v.loop = true; v.muted = true;
    v.play().then(() => { SRC.el = v; SRC.ready = false; setSource('file'); }).catch(() => say('That video could not be played in this browser.'));
  } else {
    const img = new Image();
    img.onload = () => { SRC.el = img; SRC.ready = false; setSource('file'); };
    img.onerror = () => say('That file could not be read as an image.');
    img.src = url;
  }
}

/* ----- viewer layout and drawing ----- */
function layout() {
  const stage = $('stage'), cssW = stage.clientWidth || 320, dpr = Math.min(window.devicePixelRatio || 1, 2), gap = 3, tiles = [];
  let cssH;
  if (UI.view === 'all') {
    const cols = cssW < 560 ? 2 : (cssW < 860 ? 3 : 4), tw = (cssW - gap * (cols - 1)) / cols, th = tw * 0.75;
    VIEWS.forEach((v, i) => tiles.push({ v, x: (i % cols) * (tw + gap), y: Math.floor(i / cols) * (th + gap), w: tw, h: th }));
    cssH = Math.ceil(VIEWS.length / cols) * (th + gap) - gap;
  } else {
    const v = VIEWS.find((q) => q.id === UI.view), maxH = Math.max(240, window.innerHeight * 0.72);
    const w = Math.min(cssW, maxH * 4 / 3), h = w * 0.75;
    tiles.push({ v, x: (cssW - w) / 2, y: 0, w, h }); cssH = h;
  }
  canvas.style.height = cssH + 'px';
  const pw = Math.round(cssW * dpr), ph = Math.round(cssH * dpr);
  if (canvas.width !== pw || canvas.height !== ph) { canvas.width = pw; canvas.height = ph; }
  UI.layout = { tiles, dpr, cssW, cssH };
  const lab = $('labels'); lab.innerHTML = '';
  tiles.forEach((t) => {
    const s = document.createElement('span');
    s.style.left = (100 * t.x / cssW) + '%'; s.style.top = (100 * t.y / cssH) + '%';
    s.innerHTML = '<i>' + t.v.grp + '</i>' + viewName(t.v); lab.appendChild(s);
  });
  document.querySelectorAll('#chips button').forEach((b) => b.setAttribute('aria-pressed', b.dataset.v === UI.view ? 'true' : 'false'));
  $('viewhint').textContent = UI.view === 'all' ? 'Tap a view to open it.' : (UI.view === 'cortex' ? 'Rings mark 1, 2, 5, 10 and 20° eccentricity. Upper visual field maps to the lower half.' : 'Tap the image to move the ' + (UI.tap === 'probe' ? 'probe' : 'fixation point') + '.');
  document.querySelectorAll('#chips button').forEach((b) => { const v = VIEWS.find((q) => q.id === b.dataset.v); if (v) b.textContent = v.grp + ' · ' + viewName(v); });
}
function draw() {
  const L = UI.layout; if (!L) return;
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, canvas.width, canvas.height); gl.clearColor(0.043, 0.051, 0.071, 1); gl.clear(gl.COLOR_BUFFER_BIT);
  const C = common(), em = emax();
  L.tiles.forEach((t) => {
    const x = Math.round(t.x * L.dpr), w = Math.round(t.w * L.dpr), h = Math.round(t.h * L.dpr), y = canvas.height - Math.round(t.y * L.dpr) - h;
    const u = Object.assign({ u_t0: M.sumA.t, u_t1: M.sumB.t, u_cm: 0, u_gain: 1, u_w: [0, 0, 0, 0], u_marks: 1, u_emax: em, u_probe: ST.probe, u_tile: [w, h], u_lms2rgb: LMS2RGB }, C, t.v.u());
    run(PROG.view, null, u, [x, y, w, h]);
  });
}
function onTap(ev) {
  const L = UI.layout, r = canvas.getBoundingClientRect(), x = ev.clientX - r.left, y = ev.clientY - r.top;
  const t = L.tiles.find((q) => x >= q.x && x <= q.x + q.w && y >= q.y && y <= q.y + q.h); if (!t) return;
  if (UI.view === 'all') { UI.view = t.v.id; layout(); return; }
  if (t.v.id === 'cortex') return;
  const uv = [clamp((x - t.x) / t.w, 0.01, 0.99), clamp(1 - (y - t.y) / t.h, 0.01, 0.99)];
  if (UI.tap === 'probe') ST.probe = uv; else ST.fix = uv;
}
function buildChips() {
  const host = $('chips'), mk = (id, text) => { const b = document.createElement('button'); b.dataset.v = id; b.textContent = text; b.addEventListener('click', () => { UI.view = id; layout(); }); host.appendChild(b); };
  mk('all', 'All stages'); VIEWS.forEach((v) => mk(v.id, v.id));
}
function drawOriKey() {
  const cv = $('orikey'), g = prep(cv), c = g.c;
  for (let k = 0; k < 8; k++) {
    const th = k * Math.PI / 8, h = th / Math.PI, cx = 8 + k * 16.5, cy = g.h / 2;
    const col = [0, 0.33, 0.67].map((o) => Math.round(255 * (0.5 + 0.5 * Math.cos(2 * Math.PI * (h + o)))));
    c.strokeStyle = 'rgb(' + col.join(',') + ')'; c.lineWidth = 3; c.lineCap = 'round';
    c.beginPath(); c.moveTo(cx - 6 * Math.cos(th), cy + 6 * Math.sin(th)); c.lineTo(cx + 6 * Math.cos(th), cy - 6 * Math.sin(th)); c.stroke();
  }
}

/* ----- small chart kit (canvas 2D) ----- */
function prep(cv) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2), w = cv.clientWidth, h = cv.clientHeight;
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
  const c = cv.getContext('2d'); c.setTransform(dpr, 0, 0, dpr, 0, 0); c.clearRect(0, 0, w, h);
  return { c, w, h };
}
function rrect(c, x, y, w, h) { c.beginPath(); if (c.roundRect) c.roundRect(x, y, w, h, [3, 3, 0, 0]); else c.rect(x, y, w, h); c.fill(); }
function niceMax(v) { if (!(v > 0)) return 1; const e = Math.pow(10, Math.floor(Math.log10(v))), m = v / e; return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 5 ? 5 : 10) * e; }
function axes(cv, o) {
  const g = prep(cv), c = g.c, m = { l: 38, r: 10, t: 8, b: o.xlab ? 34 : 22 }, pw = g.w - m.l - m.r, ph = g.h - m.t - m.b;
  const fx = o.xlog ? (v) => Math.log(v / o.xmin) / Math.log(o.xmax / o.xmin) : (v) => (v - o.xmin) / (o.xmax - o.xmin);
  const sx = (v) => m.l + pw * fx(v), sy = (v) => m.t + ph * (1 - (v - o.ymin) / (o.ymax - o.ymin));
  c.font = '11px "IBM Plex Mono", ui-monospace, monospace'; c.lineWidth = 1;
  const mant = Math.round(Math.max(Math.abs(o.ymax), Math.abs(o.ymin)) / Math.pow(10, Math.floor(Math.log10(Math.max(Math.abs(o.ymax), Math.abs(o.ymin)) * 1.0001))));
  const ny = (mant === 5 && o.ymin === 0) ? 5 : 4;
  for (let i = 0; i <= ny; i++) {
    const v = o.ymin + (o.ymax - o.ymin) * i / ny, y = Math.round(sy(v)) + 0.5;
    c.strokeStyle = GRID; c.beginPath(); c.moveTo(m.l, y); c.lineTo(m.l + pw, y); c.stroke();
    c.fillStyle = INK3; c.textAlign = 'right'; c.textBaseline = 'middle'; c.fillText(+v.toPrecision(3) + '', m.l - 6, y);
  }
  c.textAlign = 'center'; c.textBaseline = 'top';
  (o.xticks || []).forEach((t) => { const x = sx(t.v); if (x < m.l - 1 || x > m.l + pw + 1) return; c.fillStyle = INK3; c.fillText(t.label, x, m.t + ph + 5); c.strokeStyle = GRID; c.beginPath(); c.moveTo(Math.round(x) + 0.5, m.t + ph); c.lineTo(Math.round(x) + 0.5, m.t + ph + 3); c.stroke(); });
  if (o.xlab) { c.fillStyle = INK2; c.fillText(o.xlab, m.l + pw / 2, m.t + ph + 19); }
  return { c, sx, sy, m, pw, ph, w: g.w, h: g.h };
}
function lines(a, series) {
  const c = a.c;
  series.forEach((s) => {
    c.strokeStyle = s.color; c.lineWidth = s.width || 2; c.setLineDash(s.dash ? [5, 4] : []); c.lineJoin = 'round'; c.beginPath();
    let pen = false;
    s.x.forEach((x, i) => { const y = s.y[i]; if (y == null || isNaN(y)) { pen = false; return; } const X = a.sx(x), Y = a.sy(y); if (pen) c.lineTo(X, Y); else c.moveTo(X, Y); pen = true; });
    c.stroke(); c.setLineDash([]);
    if (s.dots) s.x.forEach((x, i) => { const y = s.y[i]; if (y == null || isNaN(y)) return; c.fillStyle = s.color; c.beginPath(); c.arc(a.sx(x), a.sy(y), 2.6, 0, 6.3); c.fill(); });
  });
}
function legend(id, items) {
  $(id).innerHTML = items.map((s) => '<span class="k"><span class="ln' + (s.dash ? ' dash' : '') + '" style="border-color:' + s.color + '"></span>' + s.name + '</span>').join('');
}

/* ----- probe read-outs ----- */
function buildStats() {
  const defs = [['ecc', 'eccentricity'], ['psf', 'V1 preferred SF here'], ['cmf', 'cortical magnification'], ['mid', 'midget centre σ'], ['par', 'parasol centre σ'], ['nyq', 'image limit (Nyquist)']];
  $('stats').innerHTML = defs.map((d) => '<div class="stat"><div class="v" id="st-' + d[0] + '">–</div><div class="l">' + d[1] + '</div></div>').join('');
  legend('lg-strip', [{ name: 'LGN magno', color: C_YELLOW }, { name: 'LGN parvo', color: C_MAGENTA }, { name: 'V1 (peak cell)', color: C_BLUE }]);
  legend('lg-cells', [{ name: 'retinal ganglion cell', color: C_VIOLET }, { name: 'LGN relay cell', color: C_YELLOW }]);
  legend('lg-sf', [{ name: 'response per band', color: C_BLUE }, { name: 'population weight at this eccentricity', color: INK2, dash: 1 }]);
}
function updateProbe(r) {
  const E = eccAt(ST.probe), fpk = 1 / (0.12 * E + 0.35), set = (k, v) => { $('st-' + k).textContent = v; };
  set('ecc', E.toFixed(1) + '°'); set('psf', fpk.toFixed(2) + ' c/deg'); set('cmf', (17.3 / (E + 0.75)).toFixed(1) + ' mm/°');
  set('mid', ((0.010 + 0.0025 * E) * P.rf * 60).toFixed(1) + '′'); set('par', ((0.040 + 0.0050 * E) * P.rf * 60).toFixed(1) + '′'); set('nyq', (ppd() / 2).toFixed(1) + ' c/deg');
  // orientation tuning per band
  let ymax = 0.1; r.R.forEach((b) => b.forEach((v) => { ymax = Math.max(ymax, v); }));
  const xs = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5, 180];
  let a = axes($('ch-ori'), { xmin: 0, xmax: 180, ymin: 0, ymax: niceMax(ymax), xticks: [0, 45, 90, 135, 180].map((v) => ({ v, label: v + '°' })) });
  lines(a, r.R.map((b, s) => ({ x: xs, y: b.concat([b[0]]), color: BAND_COL[s], dots: 1 })));
  legend('lg-ori', r.R.map((b, s) => ({ name: fmtSF(bandSF(s)) + ' c/deg', color: BAND_COL[s] })));
  // population SF tuning
  const fs = [0, 1, 2, 3].map(bandSF), tot = r.R.map((b) => b.reduce((p, q) => p + q, 0)), tmax = niceMax(Math.max(0.2, ...tot));
  const x0 = fs[3] / 1.8, x1 = fs[0] * 1.8;
  a = axes($('ch-sf'), { xlog: 1, xmin: x0, xmax: x1, ymin: 0, ymax: tmax, xticks: fs.map((f) => ({ v: f, label: fmtSF(f) })), xlab: 'cycles/deg' });
  const bw = 0.11 * a.pw;
  fs.forEach((f, s) => { const X = a.sx(f), Y = a.sy(tot[s]), y0 = a.sy(0); a.c.fillStyle = C_BLUE; rrect(a.c, X - bw / 2, Y, bw, Math.max(0, y0 - Y)); });
  const wx = [], wy = []; for (let i = 0; i <= 40; i++) { const f = x0 * Math.pow(x1 / x0, i / 40), lo = Math.log2(f / fpk) / P.sw; wx.push(f); wy.push(tmax * Math.exp(-0.5 * lo * lo)); }
  lines(a, [{ x: wx, y: wy, color: INK2, dash: 1, width: 1.5 }]);
  // cell classes
  const names = ['L+', 'L−', 'M+', 'M−', 'Par+', 'Par−', 'S+', 'S−'], rg = r.rgcA.concat(r.rgcB), lg = r.lgnA.concat(r.lgnB);
  a = axes($('ch-cells'), { xmin: 0, xmax: 8, ymin: 0, ymax: niceMax(Math.max(0.2, ...rg, ...lg)), xticks: names.map((n, i) => ({ v: i + 0.5, label: n })) });
  const b2 = a.pw / 8 * 0.3;
  names.forEach((n, i) => {
    [[rg[i], C_VIOLET, -1], [lg[i], C_YELLOW, 1]].forEach(([v, col, side]) => {
      const X = a.sx(i + 0.5) + (side < 0 ? -b2 - 1 : 1), Y = a.sy(v), y0 = a.sy(0);
      a.c.fillStyle = col; rrect(a.c, X, Y, b2, Math.max(0, y0 - Y));
    });
  });
  // strip chart
  UI.hist.push({ t: performance.now() / 1000, m: r.lgnB[0] - r.lgnB[1], p: (2 * (r.lgnA[0] - r.lgnA[1]) + (r.lgnA[2] - r.lgnA[3])) / 3, v: r.sumB[2] });
  const tn = performance.now() / 1000; while (UI.hist.length && UI.hist[0].t < tn - 8) UI.hist.shift();
  let lo = -0.2, hi = 0.2; UI.hist.forEach((h) => { lo = Math.min(lo, h.m, h.p); hi = Math.max(hi, h.m, h.p, h.v); });
  const lim = niceMax(Math.max(hi, -lo));
  a = axes($('ch-strip'), { xmin: -8, xmax: 0, ymin: -lim, ymax: lim, xticks: [-8, -6, -4, -2, 0].map((v) => ({ v, label: v + ' s' })) });
  const hx = UI.hist.map((h) => h.t - tn);
  lines(a, [{ x: hx, y: UI.hist.map((h) => h.m), color: C_YELLOW, width: 1.5 }, { x: hx, y: UI.hist.map((h) => h.p), color: C_MAGENTA, width: 1.5 }, { x: hx, y: UI.hist.map((h) => h.v), color: C_BLUE, width: 2 }]);
}

/* ----- stimulus bench ----- */
async function measure() {
  const WARM = 24, MEAS = 16, R = [0, 1, 2, 3].map(() => new Array(8).fill(0));
  let mc = 0, ms = 0, pc = 0, ps = 0;
  for (let i = 0; i < WARM + MEAS; i++) {
    step(1 / 30);
    if (i >= WARM) {
      const r = readProbe(), ph = 2 * Math.PI * S.phase, cs = 2 * Math.cos(ph) / MEAS, sn = 2 * Math.sin(ph) / MEAS;
      for (let s = 0; s < 4; s++) for (let k = 0; k < 8; k++) R[s][k] += r.R[s][k] / MEAS;
      mc += r.lgnB[0] * cs; ms += r.lgnB[0] * sn; pc += r.lgnA[0] * cs; ps += r.lgnA[0] * sn;
    }
    if (i % 5 === 4) { draw(); await raf(); }
  }
  return { R, lgnM: Math.hypot(mc, ms), lgnP: Math.hypot(pc, ps) };
}
const logspace = (a, b, n) => Array.from({ length: n }, (_, i) => a * Math.pow(b / a, i / (n - 1)));
async function runSweep(kind) {
  if (BENCH.running) return;
  BENCH.running = true;
  const keep = { S: Object.assign({}, S), src: ST.src, ad: ST.adOn }, prog = $('prog');
  document.querySelectorAll('[id^="sw-"]').forEach((b) => { b.disabled = true; });
  prog.hidden = false; prog.value = 0;
  ST.src = 'stim'; ST.adOn = 0; M.reset = 1;
  Object.assign(S, { tf: 3.75, col: 0, mask: 0, sur: 0, surRel: 0, phase: 0 });
  try {
    const E = eccAt(ST.probe), fpk = 1 / (0.12 * E + 0.35);
    let sb = 0, best = 1e9; for (let s = 0; s < 4; s++) { const d = Math.abs(Math.log2(bandSF(s) / fpk)); if (d < best) { best = d; sb = s; } }
    const f0 = bandSF(sb), ok = Math.round(((keep.S.ori % 180) + 180) % 180 / 22.5) % 8, ori0 = ok * 22.5;
    let res, done = 0, total = 1;
    const tick = () => { prog.value = ++done / total; };
    const unit = 'V1 complex cell, ' + fmtSF(f0) + ' c/deg, ' + ori0 + '°';
    if (kind === 'crf') {
      const cs = logspace(0.01, 1, 9); total = 27;
      Object.assign(S, { f: f0, ori: ori0, R: 1.25 / f0 });
      const y = { a: [], m: [], s: [], M: [], Pv: [] };
      for (const c of cs) { S.c = c; const q = await measure(); y.a.push(q.R[sb][ok]); y.M.push(q.lgnM); y.Pv.push(q.lgnP); tick(); }
      S.mask = 0.3; for (const c of cs) { if (c + 0.3 > 1.001) { y.m.push(null); tick(); continue; } S.c = c; y.m.push((await measure()).R[sb][ok]); tick(); }
      S.mask = 0; S.sur = 0.5; for (const c of cs) { S.c = c; y.s.push((await measure()).R[sb][ok]); tick(); }
      res = { title: 'Contrast response at ' + E.toFixed(1) + '° eccentricity', xlab: 'grating contrast', xlog: 1, x: cs, xfmt: (v) => v.toFixed(3),
        series: [{ name: unit, color: C_BLUE, y: y.a }, { name: '+ orthogonal mask, 30%', color: C_ORANGE, y: y.m }, { name: '+ iso-oriented surround, 50%', color: C_AQUA, y: y.s },
          { name: 'LGN magno ON cell (F1)', color: C_YELLOW, y: y.M, dash: 1 }, { name: 'LGN parvo ON cell (F1)', color: C_MAGENTA, y: y.Pv, dash: 1 }],
        note: 'Patch radius ' + (1.25 / f0).toFixed(2) + '°, 3.75 Hz drift. The mask and the surround shift the V1 curve along the contrast axis, which is the signature of divisive normalization; magno saturates early and parvo stays nearly linear.' };
    } else if (kind === 'sf') {
      const fsw = logspace(bandSF(3) / 2.2, ppd() / 2 * 0.95, 14); total = fsw.length;
      Object.assign(S, { ori: ori0, R: 200, c: keep.S.c });
      const b = [[], [], [], []], pop = [], lm = [], lp = [];
      for (const f of fsw) { S.f = f; const q = await measure(); for (let s = 0; s < 4; s++) b[s].push(q.R[s][ok]); pop.push(q.R.reduce((p, r) => p + r[ok], 0)); lm.push(q.lgnM); lp.push(q.lgnP); tick(); }
      res = { title: 'Spatial frequency tuning at ' + E.toFixed(1) + '° eccentricity', xlab: 'cycles/deg', xlog: 1, x: fsw, xfmt: (v) => v.toFixed(2),
        series: [{ name: 'V1 population (sum of bands)', color: C_AQUA, y: pop, width: 3 }].concat(b.map((y, s) => ({ name: 'band ' + fmtSF(bandSF(s)), color: BAND_COL[s], y })))
          .concat([{ name: 'LGN magno (F1)', color: C_YELLOW, y: lm, dash: 1 }, { name: 'LGN parvo (F1)', color: C_MAGENTA, y: lp, dash: 1 }]),
        note: 'Full-field grating, contrast ' + S.c.toFixed(2) + ', ' + ori0 + '°. The population curve is the model’s pSFT at this location; the image sampling limit is ' + (ppd() / 2).toFixed(1) + ' c/deg.' };
    } else if (kind === 'ori') {
      const os = Array.from({ length: 13 }, (_, i) => -90 + i * 15), chi = Math.max(keep.S.c, 0.2), clo = chi / 4; total = 26;
      Object.assign(S, { f: f0, R: 200 });
      const hi = [], lo = [];
      for (const d of os) { S.ori = ori0 + d; S.c = chi; hi.push((await measure()).R[sb][ok]); tick(); }
      for (const d of os) { S.ori = ori0 + d; S.c = clo; lo.push((await measure()).R[sb][ok]); tick(); }
      const mh = Math.max(...hi), ml = Math.max(...lo);
      res = { title: 'Orientation tuning at ' + E.toFixed(1) + '° eccentricity', xlab: 'grating orientation relative to preferred', x: os, xfmt: (v) => v + '°', xt: [-90, -45, 0, 45, 90],
        series: [{ name: unit + ', contrast ' + chi.toFixed(2), color: C_BLUE, y: hi }, { name: 'contrast ' + clo.toFixed(2), color: C_ORANGE, y: lo },
          { name: 'low contrast, rescaled to the same peak', color: C_ORANGE, y: lo.map((v) => v * mh / (ml || 1)), dash: 1 }],
        note: 'If the rescaled low-contrast curve lies on the high-contrast one, tuning width is contrast invariant.' };
    } else {
      const Rs = logspace(0.3 / f0, Math.min(14 / f0, P.fov * 0.6), 10); total = Rs.length;
      Object.assign(S, { f: f0, ori: ori0, c: Math.max(keep.S.c, 0.2) });
      const v = [], lm = [], lp = [];
      for (const R of Rs) { S.R = R; const q = await measure(); v.push(q.R[sb][ok]); lm.push(q.lgnM); lp.push(q.lgnP); tick(); }
      res = { title: 'Size tuning at ' + E.toFixed(1) + '° eccentricity', xlab: 'patch radius (deg)', xlog: 1, x: Rs, xfmt: (v2) => v2.toFixed(2),
        series: [{ name: unit, color: C_BLUE, y: v }, { name: 'LGN magno (F1)', color: C_YELLOW, y: lm, dash: 1 }, { name: 'LGN parvo (F1)', color: C_MAGENTA, y: lp, dash: 1 }],
        note: 'Contrast ' + S.c.toFixed(2) + '. V1 peaks at its summation field and is then suppressed by the surround pool; the LGN suppressive field gives magno cells a milder version.' };
    }
    BENCH.result = res; BENCH.hover = -1; renderBench();
  } finally {
    Object.assign(S, keep.S); ST.src = keep.src; ST.adOn = keep.ad; M.reset = 1; BENCH.running = false; prog.hidden = true;
    STIM.forEach((s) => s.sync());
    document.querySelectorAll('[id^="sw-"]').forEach((b) => { b.disabled = false; });
  }
}
function renderBench() {
  const r = BENCH.result, cv = $('bench-chart');
  if (!r) { axes(cv, { xmin: 0, xmax: 1, ymin: 0, ymax: 1, xticks: [] }); return; }
  $('bench-title').textContent = r.title; $('bench-note').textContent = r.note;
  legend('bench-legend', r.series);
  let ymax = 0; r.series.forEach((s) => s.y.forEach((v) => { if (v != null) ymax = Math.max(ymax, v); }));
  const x0 = r.x[0], x1 = r.x[r.x.length - 1];
  let xt;
  if (r.xlog) { xt = []; for (const b of [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]) if (b >= x0 * 0.999 && b <= x1 * 1.001) xt.push({ v: b, label: b + '' }); }
  else xt = (r.xt || r.x).map((v) => ({ v, label: r.xfmt(v) }));
  const a = axes(cv, { xlog: r.xlog, xmin: x0, xmax: x1, ymin: 0, ymax: niceMax(ymax), xticks: xt, xlab: r.xlab });
  if (BENCH.hover >= 0) { const X = Math.round(a.sx(r.x[BENCH.hover])) + 0.5; a.c.strokeStyle = INK3; a.c.lineWidth = 1; a.c.beginPath(); a.c.moveTo(X, a.m.t); a.c.lineTo(X, a.m.t + a.ph); a.c.stroke(); }
  lines(a, r.series.map((s) => ({ x: r.x, y: s.y, color: s.color, dash: s.dash, width: s.width, dots: 1 })));
  BENCH.geom = a;
  const tb = $('bench-table');
  tb.innerHTML = '<tr><th>' + r.xlab + '</th>' + r.series.map((s) => '<th>' + s.name + '</th>').join('') + '</tr>' +
    r.x.map((x, i) => '<tr><td>' + r.xfmt(x) + '</td>' + r.series.map((s) => '<td>' + (s.y[i] == null ? '' : s.y[i].toFixed(4)) + '</td>').join('') + '</tr>').join('');
  $('bench-copy').disabled = false;
}
function benchHover(ev) {
  const r = BENCH.result, a = BENCH.geom, tip = $('bench-tip'); if (!r || !a) return;
  const rect = $('bench-chart').getBoundingClientRect(), x = ev.clientX - rect.left;
  let bi = 0, bd = 1e9; r.x.forEach((v, i) => { const d = Math.abs(a.sx(v) - x); if (d < bd) { bd = d; bi = i; } });
  BENCH.hover = bi; renderBench();
  tip.innerHTML = '<div><b>' + r.xfmt(r.x[bi]) + '</b>&nbsp;' + r.xlab + '</div>' + r.series.filter((s) => s.y[bi] != null).map((s) => '<div><span class="ln" style="border-top:2px ' + (s.dash ? 'dashed ' : 'solid ') + s.color + ';width:12px;display:inline-block"></span>' + s.y[bi].toFixed(3) + '&nbsp; ' + s.name + '</div>').join('');
  tip.hidden = false;
  const tx = a.sx(r.x[bi]); tip.style.top = '6px';
  if (tx > rect.width / 2) { tip.style.left = 'auto'; tip.style.right = (rect.width - tx + 10) + 'px'; } else { tip.style.right = 'auto'; tip.style.left = (tx + 10) + 'px'; }
}
function copyBench() {
  const r = BENCH.result; if (!r) return;
  const tsv = [[r.xlab].concat(r.series.map((s) => s.name)).join('\t')].concat(r.x.map((x, i) => [x].concat(r.series.map((s) => s.y[i] == null ? '' : s.y[i])).join('\t'))).join('\n');
  const b = $('bench-copy');
  navigator.clipboard.writeText(tsv).then(() => { b.textContent = 'Copied'; setTimeout(() => { b.textContent = 'Copy as TSV'; }, 1500); }).catch(() => { b.textContent = 'Copy blocked here; use the table'; });
}

/* ----- main loop ----- */
let lastT = 0, lastRead = 0;
function frame(now) {
  requestAnimationFrame(frame);
  const dt = clamp((now - lastT) / 1000, 1 / 240, 0.1); lastT = now;
  if (BENCH.running) return;
  if (!ST.paused) { uploadSource(); step(dt); UI.fps = UI.fps ? UI.fps * 0.92 + 0.08 / dt : 1 / dt; }
  draw();
  if (now - lastRead > 90) {
    lastRead = now; const r = readProbe(); UI.lastProbe = r; updateProbe(r);
    $('status').innerHTML = '<b>' + UI.fps.toFixed(0) + '</b> frames/s · ' + M.W + ' × ' + M.H + ' · ' + ppd().toFixed(1) + ' px/° · ' + (use32 ? 'float32' : 'float16');
  }
}
function boot() {
  try { initGL(); alloc(256); }
  catch (e) {
    document.querySelector('.wrap').innerHTML = '<div class="fatal"><h1>The model could not start</h1><p style="margin-top:8px">' + e.message + ' Try a current version of Chrome, Safari or Firefox with hardware acceleration on.</p></div>';
    return;
  }
  buildControls(); buildChips(); buildStats(); layout(); drawOriKey(); renderBench();
  $('src-demo').addEventListener('click', () => { say(''); setSource('demo'); });
  $('src-stim').addEventListener('click', () => { say(''); setSource('stim'); });
  $('src-cam').addEventListener('click', startCam);
  $('src-file').addEventListener('change', (e) => { loadFile(e.target.files[0]); e.target.value = ''; });
  $('pause').addEventListener('click', (e) => { ST.paused = !ST.paused; e.target.setAttribute('aria-pressed', ST.paused ? 'true' : 'false'); e.target.textContent = ST.paused ? 'Resume' : 'Pause'; });
  $('mirror').addEventListener('change', (e) => { ST.mirror = e.target.checked; });
  $('res').addEventListener('change', (e) => { if (BENCH.running) return; alloc(+e.target.value); UI.hist = []; layout(); });
  const tapSet = (k) => { UI.tap = k; $('tap-probe').setAttribute('aria-pressed', k === 'probe' ? 'true' : 'false'); $('tap-fix').setAttribute('aria-pressed', k === 'fix' ? 'true' : 'false'); layout(); };
  $('tap-probe').addEventListener('click', () => tapSet('probe')); $('tap-fix').addEventListener('click', () => tapSet('fix'));
  canvas.addEventListener('click', onTap);
  ['crf', 'sf', 'ori', 'size'].forEach((k) => $('sw-' + k).addEventListener('click', () => runSweep(k)));
  $('bench-chart').addEventListener('pointermove', benchHover);
  $('bench-chart').addEventListener('pointerdown', benchHover);
  $('bench-chart').addEventListener('pointerleave', () => { $('bench-tip').hidden = true; BENCH.hover = -1; renderBench(); });
  $('bench-copy').addEventListener('click', copyBench);
  new ResizeObserver(() => { layout(); renderBench(); }).observe($('stage'));
  window.addEventListener('resize', () => { layout(); renderBench(); });
  canvas.addEventListener('webglcontextlost', (e) => { e.preventDefault(); say('The graphics context was lost. Reload the page to restart the model.'); });
  requestAnimationFrame((t) => { lastT = t; requestAnimationFrame(frame); });
}
window.VB = { P, S, ST, M, runSweep, BENCH, UI, setSource, layout };
boot();
