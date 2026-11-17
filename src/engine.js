'use strict';
/* ---------- WebGL2 plumbing + the model pipeline ---------- */
const $ = (id) => document.getElementById(id);
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));

let gl = null, canvas = null, use32 = false, FMT = 0, PROBE_FMT = 0;
const PROG = {};
const GLOBJ = { tex: [], fbo: [] };

const P = {};                                  // model parameters (filled from PARAMS)
const S = { c: 0.5, f: 1, ori: 0, tf: 3.75, R: 4, mask: 0, sur: 0, surRel: 0, col: 0, phase: 0 };
const ST = { src: 'demo', mirror: true, paused: false, probe: [0.64, 0.5], fix: [0.5, 0.5], adOn: 1 };
const M = { W: 0, H: 0, ping: 0, cur: 1, reset: 1, time: 0, NS: 4 };
const SRC = { tex: null, el: null, aspect: 4 / 3, ready: false };
const GCOMP = [1, 1.268, 1.344, 1.364];        // undoes the pyramid low-pass at each band's centre SF
let KERN = null, KN = 1.5, LMS2RGB = null, PROBE = null;

function initGL() {
  canvas = $('gl');
  gl = canvas.getContext('webgl2', { antialias: false, alpha: false, depth: false, stencil: false, powerPreference: 'high-performance' });
  if (!gl) throw new Error('WebGL2 is not available in this browser.');
  const eF = gl.getExtension('EXT_color_buffer_float'), eH = gl.getExtension('EXT_color_buffer_half_float');
  const eL = gl.getExtension('OES_texture_float_linear');
  if (!eF && !eH) throw new Error('This GPU cannot render to floating-point textures, which the model needs.');
  use32 = !!(eF && eL) && location.hash !== '#half';
  FMT = use32 ? gl.RGBA32F : gl.RGBA16F;
  PROBE_FMT = eF ? gl.RGBA32F : gl.RGBA16F;
  gl.bindVertexArray(gl.createVertexArray());
  for (const name in FS) PROG[name] = mkProg(name, FS[name]);
  makeKernels();
  LMS2RGB = inv3([0.31392, 0.63956, 0.04653, 0.15529, 0.75795, 0.08672, 0.01772, 0.10945, 0.87275]);
  SRC.tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, SRC.tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([128, 128, 128, 255]));
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.generateMipmap(gl.TEXTURE_2D);
  const pt = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, pt);
  gl.texStorage2D(gl.TEXTURE_2D, 1, PROBE_FMT, 8, 8);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  const pf = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, pf);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, pt, 0);
  if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) throw new Error('Floating-point framebuffer is incomplete on this GPU.');
  PROBE = { f: pf, w: 8, h: 8, buf: new Float32Array(256), half: null };
}

function inv3(m) {                              // row-major in, column-major Float32Array out (for GLSL)
  const [a, b, c, d, e, f, g, h, i] = m;
  const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g, det = a * A + b * B + c * C;
  const r = [A / det, -(b * i - c * h) / det, (b * f - c * e) / det, B / det, (a * i - c * g) / det, -(a * f - c * d) / det,
    C / det, -(a * h - b * g) / det, (a * e - b * d) / det];
  return new Float32Array([r[0], r[3], r[6], r[1], r[4], r[7], r[2], r[5], r[8]]);
}

function mkProg(name, fsSrc) {
  const sh = (type, src) => {
    const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error('Shader "' + name + '": ' + gl.getShaderInfoLog(s));
    return s;
  };
  const p = gl.createProgram();
  gl.attachShader(p, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(p, sh(gl.FRAGMENT_SHADER, fsSrc)); gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error('Program "' + name + '": ' + gl.getProgramInfoLog(p));
  const uni = {}, n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
  for (let i = 0; i < n; i++) { const a = gl.getActiveUniform(p, i); uni[a.name] = { loc: gl.getUniformLocation(p, a.name), type: a.type }; }
  return { p, uni, name };
}

function tex(w, h, levels) {
  const t = gl.createTexture(); levels = levels || 1;
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texStorage2D(gl.TEXTURE_2D, levels, FMT, w, h);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, levels > 1 ? gl.LINEAR_MIPMAP_LINEAR : gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  const o = { t, w, h, levels }; GLOBJ.tex.push(t); return o;
}
function fbo(atts) {
  const f = gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, f);
  const bufs = [];
  atts.forEach(([tx, lv], i) => {
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, gl.TEXTURE_2D, tx.t, lv);
    bufs.push(gl.COLOR_ATTACHMENT0 + i);
  });
  gl.drawBuffers(bufs);
  if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) throw new Error('Framebuffer incomplete (float render targets unsupported?)');
  GLOBJ.fbo.push(f);
  const [t0, l0] = atts[0];
  return { f, w: Math.max(1, t0.w >> l0), h: Math.max(1, t0.h >> l0) };
}
const nLevels = (w, h) => Math.floor(Math.log2(Math.max(w, h))) + 1;
function mkPyr(w, h) {
  const L = nLevels(w, h), tx = tex(w, h, L), f = [];
  for (let k = 0; k < L; k++) f.push(fbo([[tx, k]]));
  return { t: tx.t, tx, L, f };
}
function buildPyr(py) {
  for (let k = 1; k < py.L; k++) {
    gl.bindTexture(gl.TEXTURE_2D, py.t);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_BASE_LEVEL, k - 1);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAX_LEVEL, k - 1);
    run(PROG.down, py.f[k], { u_src: py.t });
  }
  gl.bindTexture(gl.TEXTURE_2D, py.t);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_BASE_LEVEL, 0);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAX_LEVEL, py.L - 1);
}

function run(prog, target, u, vp) {
  gl.bindFramebuffer(gl.FRAMEBUFFER, target ? target.f : null);
  if (vp) gl.viewport(vp[0], vp[1], vp[2], vp[3]); else gl.viewport(0, 0, target.w, target.h);
  gl.useProgram(prog.p);
  let unit = 0;
  for (const name in prog.uni) {
    const U = prog.uni[name], v = u[name];
    if (v === undefined) { if (!prog['_w' + name]) { prog['_w' + name] = 1; console.warn('uniform not set', prog.name, name); } continue; }
    switch (U.type) {
      case gl.SAMPLER_2D: gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, v); gl.uniform1i(U.loc, unit++); break;
      case gl.FLOAT: gl.uniform1f(U.loc, v); break;
      case gl.INT: gl.uniform1i(U.loc, v); break;
      case gl.FLOAT_VEC2: gl.uniform2fv(U.loc, v); break;
      case gl.FLOAT_VEC4: gl.uniform4fv(U.loc, v); break;
      case gl.FLOAT_MAT3: gl.uniformMatrix3fv(U.loc, false, v); break;
    }
  }
  gl.drawArrays(gl.TRIANGLES, 0, 3);
}

/* Quadrature Gabor kernels: 8 orientations, period 4 px, 1.4-octave SF bandwidth, ~20 deg orientation half-width. */
function makeKernels() {
  const R = 5, N = 11, f = 0.25, sx = 1.66, sy = 2.0, data = new Float32Array(121 * 4 * 4), K = [];
  for (let k = 0; k < 8; k++) {
    const th = k * Math.PI / 8, nx = -Math.sin(th), ny = Math.cos(th), bx = Math.cos(th), by = Math.sin(th);
    const ev = new Float64Array(121), od = new Float64Array(121), xp = new Float64Array(121);
    let se = 0, sn = 0, ge = 0, go = 0; const en = new Float64Array(121);
    for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
      const dx = i - R, dy = j - R, t = j * N + i, x = dx * nx + dy * ny, y = dx * bx + dy * by;
      const e = (dx * dx + dy * dy > 30.3) ? 0 : Math.exp(-x * x / (2 * sx * sx) - y * y / (2 * sy * sy));
      xp[t] = x; en[t] = e; ev[t] = e * Math.cos(2 * Math.PI * f * x); od[t] = e * Math.sin(2 * Math.PI * f * x); se += ev[t]; sn += e;
    }
    for (let t = 0; t < 121; t++) { ev[t] -= en[t] * se / sn; ge += ev[t] * Math.cos(2 * Math.PI * f * xp[t]); go += od[t] * Math.sin(2 * Math.PI * f * xp[t]); }
    for (let t = 0; t < 121; t++) { ev[t] /= ge; od[t] /= go; }
    K.push({ ev, od });
    const row = k >> 1, off = (k & 1) * 2;
    for (let t = 0; t < 121; t++) { data[(row * 121 + t) * 4 + off] = ev[t]; data[(row * 121 + t) * 4 + off + 1] = od[t]; }
  }
  // Pool normaliser: summed energy across orientations (and the next-finer band) for a unit grating.
  const pooled = (ff) => {
    let acc = 0, cnt = 0;
    for (const th0 of [0, Math.PI / 16]) for (let ph = 0; ph < 8; ph++) {
      let tot = 0;
      for (let k = 0; k < 8; k++) {
        let e = 0, o = 0;
        for (let j = 0; j < 11; j++) for (let i = 0; i < 11; i++) {
          const dx = i - 5, dy = j - 5, g = Math.cos(2 * Math.PI * ff * (-dx * Math.sin(th0) + dy * Math.cos(th0)) + ph * Math.PI / 4);
          e += K[k].ev[j * 11 + i] * g; o += K[k].od[j * 11 + i] * g;
        }
        tot += e * e + o * o;
      }
      acc += tot; cnt++;
    }
    return acc / cnt;
  };
  KN = pooled(0.25) + 0.5 * pooled(0.125);
  KERN = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, KERN);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, 121, 4, 0, gl.RGBA, gl.FLOAT, data);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
}

function alloc(W) {
  GLOBJ.tex.forEach((t) => gl.deleteTexture(t)); GLOBJ.fbo.forEach((f) => gl.deleteFramebuffer(f));
  GLOBJ.tex = []; GLOBJ.fbo = [];
  const H = Math.round(W * 0.75); M.W = W; M.H = H;
  const T = () => tex(W, H);
  M.lms = mkPyr(W, H); M.cone = mkPyr(W, H); M.sub = mkPyr(W, H); M.v1in = mkPyr(W, H);
  M.coneA = [T(), T()]; M.coneI = [T(), T()];
  M.fbCone = [0, 1].map((i) => fbo([[M.coneA[i], 0], [M.coneI[i], 0], [M.cone.tx, 0]]));
  M.gen = T(); M.genS = [T(), T()];
  M.fbGen = [0, 1].map((i) => fbo([[M.gen, 0], [M.genS[i], 0], [M.sub.tx, 0]]));
  M.rgcA = T(); M.rgcB = T(); M.lgnA = T(); M.lgnB = T();
  M.fbRL = fbo([[M.rgcA, 0], [M.rgcB, 0], [M.lgnA, 0], [M.lgnB, 0]]);
  M.sc = [];
  for (let s = 0; s < M.NS; s++) {
    const w = W >> s, h = H >> s, t = () => tex(w, h), L = nLevels(w, h), o = { w, h };
    o.lin = [[t(), t(), t(), t()], [t(), t(), t(), t()]];
    o.fbLin = o.lin.map((a) => fbo(a.map((x) => [x, 0])));
    o.EA = tex(w, h, L); o.EB = tex(w, h, L); o.DA = t(); o.DB = t();
    o.fbE = fbo([[o.EA, 0], [o.EB, 0], [o.DA, 0], [o.DB, 0]]);
    o.RA = t(); o.RB = t(); o.ad = [[t(), t()], [t(), t()]];
    o.fbN = [0, 1].map((i) => fbo([[o.RA, 0], [o.RB, 0], [o.ad[i][0], 0], [o.ad[i][1], 0]]));
    M.sc.push(o);
  }
  M.sumA = T(); M.sumB = T(); M.fbSum = fbo([[M.sumA, 0], [M.sumB, 0]]);
  M.col = T(); M.fbCol = fbo([[M.col, 0]]);
  M.reset = 1; M.ping = 0; M.cur = 1;
}

const ppd = () => M.W / P.fov;
const bandSF = (s) => ppd() * 0.25 / (1 << s);        // centre SF of band s, cycles/deg
const common = () => ({ u_res: [M.W, M.H], u_fix: ST.fix, u_ppd: ppd(), u_rf: P.rf });
function eccAt(uv) { return Math.hypot((uv[0] - ST.fix[0]) * M.W, (uv[1] - ST.fix[1]) * M.H) / ppd(); }
function emax() {
  let m = 0; for (const c of [[0, 0], [1, 0], [0, 1], [1, 1]]) m = Math.max(m, eccAt(c)); return m;
}

function uploadSource() {
  const el = SRC.el; if (!el) return;
  const isVid = el.tagName === 'VIDEO';
  if (isVid && el.readyState < 2) return;
  if (!isVid && SRC.ready) return;
  gl.bindTexture(gl.TEXTURE_2D, SRC.tex);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
  try { gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, el); gl.generateMipmap(gl.TEXTURE_2D); SRC.ready = true; } catch (e) { /* frame not ready */ }
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
  SRC.aspect = isVid ? (el.videoWidth / el.videoHeight || 4 / 3) : (el.naturalWidth / el.naturalHeight || 4 / 3);
}

/* One model time step. */
function step(dt) {
  const C = common(), n = M.ping, p = 1 - n, am = M.W / M.H;
  const mode = ST.src === 'stim' ? 2 : (ST.src === 'demo' ? 1 : 0);
  if (mode === 2) S.phase = (S.phase + S.tf * dt) % 1;
  const sScale = SRC.aspect > am ? [am / SRC.aspect, 1] : [1, SRC.aspect / am];
  const d2r = Math.PI / 180;
  run(PROG.ingest, M.lms.f[0], Object.assign({
    u_src: SRC.tex, u_mode: mode, u_srcScale: sScale, u_mirror: (ST.src === 'cam' && ST.mirror) ? 1 : 0, u_time: M.time, u_probe: ST.probe,
    u_sC: S.c, u_sF: S.f, u_sOri: S.ori * d2r, u_sR: S.R, u_sMask: S.mask, u_sSur: S.sur, u_sSurOri: (S.ori + S.surRel) * d2r, u_sPhase: S.phase, u_sCol: S.col
  }, C));
  buildPyr(M.lms);
  run(PROG.cone, M.fbCone[n], Object.assign({
    u_lms: M.lms.t, u_pA: M.coneA[p].t, u_pI: M.coneI[p].t, u_dt: dt, u_tauC: 0.025, u_tauA: P.tauA, u_sigA: P.sigA,
    u_gamma: P.gamma, u_Id: P.Id, u_optics: P.optics, u_reset: M.reset
  }, C));
  buildPyr(M.cone);
  run(PROG.gen, M.fbGen[n], Object.assign({
    u_cone: M.cone.t, u_pS: M.genS[p].t, u_dt: dt, u_reset: M.reset, u_ksP: P.ksP, u_ksM: P.ksM, u_srr: 6, u_purE: 20,
    u_wP: 0.2, u_wM: P.wM, u_tauP: 0.15, u_tauM: 0.06, u_tauK: 0.08
  }, C));
  buildPyr(M.sub);
  run(PROG.rl, M.fbRL, Object.assign({
    u_gen: M.gen.t, u_sub: M.sub.t, u_wSub: P.wSub, u_gM: P.gM, u_c50M: P.c50M, u_gP: P.gP, u_c50P: 1.0, u_gK: 3.0,
    u_relay: P.relay, u_thr: P.thr, u_sfL: P.sfL
  }, C));
  run(PROG.v1in, M.v1in.f[0], Object.assign({ u_lA: M.lgnA.t, u_lB: M.lgnB.t, u_wMV1: P.wMV1 }, C));
  buildPyr(M.v1in);
  for (let s = 0; s < M.NS; s++) {
    const o = M.sc[s], c = o.lin[n], q = o.lin[p];
    run(PROG.gabor, o.fbLin[n], { u_in: M.v1in.t, u_kern: KERN, u_lod: s, u_gain: GCOMP[s] });
    run(PROG.energy, o.fbE, Object.assign({
      u_c0: c[0].t, u_c1: c[1].t, u_c2: c[2].t, u_c3: c[3].t, u_p0: q[0].t, u_p1: q[1].t, u_p2: q[2].t, u_p3: q[3].t,
      u_fs: bandSF(s), u_sw: P.sw
    }, C));
    gl.bindTexture(gl.TEXTURE_2D, o.EA.t); gl.generateMipmap(gl.TEXTURE_2D);
    gl.bindTexture(gl.TEXTURE_2D, o.EB.t); gl.generateMipmap(gl.TEXTURE_2D);
  }
  for (let s = 0; s < M.NS; s++) {
    const o = M.sc[s], f = M.sc[s - 1] || o, c = M.sc[s + 1] || o;
    run(PROG.norm, o.fbN[n], Object.assign({
      u_EA: o.EA.t, u_EB: o.EB.t, u_EAf: f.EA.t, u_EBf: f.EB.t, u_EAc: c.EA.t, u_EBc: c.EB.t, u_pA: o.ad[p][0].t, u_pB: o.ad[p][1].t,
      u_hasF: s > 0 ? 1 : 0, u_hasC: s < M.NS - 1 ? 1 : 0, u_sig: P.sig, u_n: P.n, u_wLoc: P.wLoc, u_wSur: P.wSur, u_tun: P.tun,
      u_kAd: ST.adOn ? P.kAd : 0, u_tauAd: P.tauAd, u_dt: dt, u_adOn: ST.adOn, u_lodL: 1.5, u_lodS: 3.3, u_kn: KN
    }, C));
  }
  const U = Object.assign({}, C);
  for (let s = 0; s < 4; s++) { const o = M.sc[s]; U['u_RA' + s] = o.RA.t; U['u_RB' + s] = o.RB.t; U['u_DA' + s] = o.DA.t; U['u_DB' + s] = o.DB.t; }
  run(PROG.summary, M.fbSum, U);
  run(PROG.colour, M.fbCol, Object.assign({ u_in: M.v1in.t }, C));
  M.cur = n; M.ping = p; M.reset = 0; M.time += dt;
}

/* Read every stage at the probe location: 36 texels gathered into one small float buffer. */
function probeTextures() {
  const a = [M.cone.t, M.gen.t, M.rgcA.t, M.rgcB.t, M.lgnA.t, M.lgnB.t, M.v1in.t, M.col.t, M.sumA.t, M.sumB.t];
  for (let s = 0; s < 4; s++) { const o = M.sc[s]; a.push(o.RA.t, o.RB.t, o.DA.t, o.DB.t, o.ad[M.cur][0].t, o.ad[M.cur][1].t); }
  a.push(M.lms.t, M.coneA[M.cur].t);
  return a;
}
function halfToFloat(h) {
  const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 31, f = h & 1023;
  return e === 0 ? s * Math.pow(2, -14) * (f / 1024) : (e === 31 ? (f ? NaN : s * Infinity) : s * Math.pow(2, e - 15) * (1 + f / 1024));
}
function readProbe() {
  const list = probeTextures();
  list.forEach((t, i) => run(PROG.copy, PROBE, { u_t: t, u_p: ST.probe }, [i % 8, i >> 3, 1, 1]));
  gl.bindFramebuffer(gl.FRAMEBUFFER, PROBE.f);
  if (PROBE.half === null) {
    gl.readPixels(0, 0, 8, 8, gl.RGBA, gl.FLOAT, PROBE.buf);
    PROBE.half = gl.getError() !== gl.NO_ERROR;
    if (PROBE.half) PROBE.hbuf = new Uint16Array(256);
  }
  if (PROBE.half) { gl.readPixels(0, 0, 8, 8, gl.RGBA, gl.HALF_FLOAT, PROBE.hbuf); for (let i = 0; i < 256; i++) PROBE.buf[i] = halfToFloat(PROBE.hbuf[i]); }
  else gl.readPixels(0, 0, 8, 8, gl.RGBA, gl.FLOAT, PROBE.buf);
  const b = PROBE.buf, px = (i) => [b[i * 4], b[i * 4 + 1], b[i * 4 + 2], b[i * 4 + 3]];
  const r = { cone: px(0), gen: px(1), rgcA: px(2), rgcB: px(3), lgnA: px(4), lgnB: px(5), v1in: px(6), col: px(7), sumA: px(8), sumB: px(9), R: [], D: [], A: [], lms: px(34) };
  for (let s = 0; s < 4; s++) {
    const k = 10 + s * 6;
    r.R.push(px(k).concat(px(k + 1))); r.D.push(px(k + 2).concat(px(k + 3))); r.A.push(px(k + 4).concat(px(k + 5)));
  }
  return r;
}
