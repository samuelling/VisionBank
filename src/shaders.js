'use strict';
/* ---------- GLSL: every stage of the model is one fragment shader ---------- */
const VS = `#version 300 es
out vec2 uv;
void main(){ vec2 p = vec2(float((gl_VertexID<<1)&2), float(gl_VertexID&2)); uv = p; gl_Position = vec4(p*2.0-1.0,0.0,1.0); }`;

const HEAD = `#version 300 es
precision highp float; precision highp int; precision highp sampler2D;
in vec2 uv;
uniform vec2 u_res;   // model resolution, px
uniform vec2 u_fix;   // fixation point, uv
uniform float u_ppd;  // model px per degree of visual angle
uniform float u_rf;   // receptive-field size multiplier
const float TAU = 6.28318530718;
float eccDeg(vec2 p){ return length((p - u_fix)*u_res)/u_ppd; }
// Space-variant Gaussian blur: read a Gaussian pyramid at the level whose sigma matches.
vec4 pyr(sampler2D t, vec2 p, float sigPx){
  float lam = max(0.0, 0.5*log2(2.4*sigPx*sigPx + 0.6));
  return textureLod(t, p, lam);
}
// RF centre sigma (deg) vs eccentricity; after Croner & Kaplan (1995), human-scaled fovea.
float sigMidget(float E){ return (0.010 + 0.0025*E)*u_rf; }
float sigParasol(float E){ return (0.040 + 0.0050*E)*u_rf; }
float sigBistrat(float E){ return (0.060 + 0.0070*E)*u_rf; }
float sum4(vec4 v){ return v.x+v.y+v.z+v.w; }
`;

const FS = {};

/* Stage 0. Image -> linear cone excitations (L, M, S). */
FS.ingest = HEAD + `
uniform sampler2D u_src; uniform int u_mode; uniform vec2 u_srcScale; uniform float u_mirror, u_time;
uniform vec2 u_probe;
uniform float u_sC, u_sF, u_sOri, u_sR, u_sMask, u_sSur, u_sSurOri, u_sPhase; uniform int u_sCol;
out vec4 o;
const mat3 RGB2LMS = mat3(0.31392,0.15529,0.01772, 0.63956,0.75795,0.10945, 0.04653,0.08672,0.87275);
vec3 srgb2lin(vec3 c){ return mix(c/12.92, pow((c+0.055)/1.055, vec3(2.4)), step(0.04045, c)); }
float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1,311.7)))*43758.5453); }
float vnoise(vec2 p){ vec2 i=floor(p), f=fract(p); f=f*f*(3.0-2.0*f);
  return mix(mix(hash(i),hash(i+vec2(1,0)),f.x), mix(hash(i+vec2(0,1)),hash(i+vec2(1,1)),f.x), f.y); }
vec3 demo(vec2 p, float t){
  vec2 q = (p-0.5)*vec2(u_res.x/u_res.y,1.0);
  float n = 0.5*vnoise(q*6.0+t*0.05) + 0.3*vnoise(q*14.0) + 0.2*vnoise(q*31.0);
  vec3 col = vec3(0.30 + 0.44*(n-0.5));
  col *= 0.7 + 0.6*p.x;
  vec2 c1 = q - vec2(-0.36, 0.12); float r1 = length(c1); float a1 = atan(c1.y,c1.x) + t*0.6;
  col = mix(col, vec3(mix(0.05,0.9,step(0.0, sin(a1*4.0)))), smoothstep(0.2,0.19,r1));
  vec2 c2 = q - vec2(0.38,0.18); col += 0.3*exp(-dot(c2,c2)/0.012)*cos(84.0*(c2.x+c2.y) - t*6.0);
  vec2 c3 = q - vec2(0.36,-0.22); col += 0.3*exp(-dot(c3,c3)/0.02)*cos(45.0*c3.y - t*4.0);
  for(int k=0;k<4;k++){ float ang = t*0.35 + float(k)*1.5708; vec2 cc = q - vec2(-0.02,-0.08) - 0.2*vec2(cos(ang),sin(ang));
    vec3 dc = k==0?vec3(0.85,0.12,0.10): k==1?vec3(0.10,0.60,0.15): k==2?vec3(0.12,0.20,0.90): vec3(0.85,0.75,0.10);
    col = mix(col, dc, smoothstep(0.062,0.055,length(cc))); }
  float bx = fract(t*0.08)*1.6-0.8; col = mix(col, vec3(0.95), smoothstep(0.012,0.008,abs(q.x-bx))*step(q.y,-0.3));
  if(q.y>0.38){ vec2 ch = floor((q+vec2(t*0.02,0.0))*24.0); col = vec3(mod(ch.x+ch.y,2.0)<0.5?0.08:0.8); }
  return clamp(col,0.0,1.0);
}
vec3 stim(vec2 p){
  vec2 d = (p - u_probe)*u_res/u_ppd; float r = length(d);
  float ph = TAU*u_sPhase;
  vec2 n1 = vec2(-sin(u_sOri), cos(u_sOri)); vec2 n2 = vec2(-n1.y, n1.x);
  vec2 n3 = vec2(-sin(u_sSurOri), cos(u_sSurOri));
  float g1 = cos(TAU*u_sF*dot(d,n1) - ph), g2 = cos(TAU*u_sF*dot(d,n2) - ph), g3 = cos(TAU*u_sF*dot(d,n3) - ph);
  float e = max(0.08*u_sR, 1.5/u_ppd);
  float wc = 1.0 - smoothstep(u_sR - e, u_sR + e, r);
  float ws = smoothstep(u_sR*1.15 - e, u_sR*1.15 + e, r);
  float m = clamp(wc*(u_sC*g1 + u_sMask*g2) + ws*u_sSur*g3, -1.0, 1.0);
  vec3 dir = u_sCol==0 ? vec3(1.0) : (u_sCol==1 ? vec3(0.5,-1.0,0.0) : vec3(0.0,0.0,1.0));
  return 0.5*(1.0 + m*dir);
}
void main(){
  vec3 lms;
  if(u_mode==2){ lms = stim(uv); }
  else {
    vec3 rgb;
    if(u_mode==1) rgb = demo(uv, u_time);
    else { vec2 p = (uv-0.5)*u_srcScale + 0.5; if(u_mirror>0.5) p.x = 1.0-p.x; rgb = srgb2lin(texture(u_src, p).rgb); }
    lms = RGB2LMS*rgb;
  }
  o = vec4(lms, (2.0*lms.r+lms.g)/3.0);
}`;

/* Gaussian pyramid step: [1 3 3 1]/8 separable, decimate by 2 (keeps pixel centres aligned). */
FS.down = `#version 300 es
precision highp float; precision highp sampler2D;
uniform sampler2D u_src; out vec4 o;
void main(){
  ivec2 sz = textureSize(u_src,0); ivec2 c = ivec2(gl_FragCoord.xy)*2;
  const float w[4] = float[4](1.0,3.0,3.0,1.0);
  vec4 s = vec4(0.0);
  for(int j=0;j<4;j++) for(int i=0;i<4;i++){
    s += w[i]*w[j]*texelFetch(u_src, clamp(c+ivec2(i-1,j-1), ivec2(0), sz-1), 0); }
  o = s/64.0;
}`;

/* Stage 1. Optics + cones: blur, photocurrent low-pass, luminance gain control (per cone class). */
FS.cone = HEAD + `
uniform sampler2D u_lms, u_pA, u_pI;
uniform float u_dt, u_tauC, u_tauA, u_sigA, u_gamma, u_Id, u_optics, u_reset;
layout(location=0) out vec4 oA; layout(location=1) out vec4 oI; layout(location=2) out vec4 oY;
void main(){
  float E = eccDeg(uv); ivec2 q = ivec2(gl_FragCoord.xy);
  float so = (0.6 + 0.05*E)/60.0*u_optics;                      // optical PSF sigma, deg
  vec3 I; I.rg = pyr(u_lms, uv, so*u_ppd).rg;
  I.b = pyr(u_lms, uv, sqrt(so*so + 0.0016)*u_ppd).b;            // S cones: chromatic aberration, sparse mosaic
  vec3 pI = texelFetch(u_pI,q,0).rgb, pA = texelFetch(u_pA,q,0).rgb;
  vec3 In = mix(pI, I, 1.0-exp(-u_dt/u_tauC));
  vec3 loc = pyr(u_lms, uv, u_sigA*(1.0+E/4.0)*u_ppd).rgb;       // local adaptation pool
  vec3 glob = textureLod(u_lms, vec2(0.5), 30.0).rgb;            // scene mean
  vec3 pool = pow(max(loc,1e-4), vec3(u_gamma))*pow(max(glob,1e-4), vec3(1.0-u_gamma));
  vec3 An = mix(pA, pool, 1.0-exp(-u_dt/u_tauA));
  if(u_reset>0.5){ In = I; An = pool; }
  vec3 y = (In - An)/(In + An + u_Id);                            // Weber gain control, compressive
  oA = vec4(An,1.0); oI = vec4(In,1.0); oY = vec4(y, (2.0*y.r + y.g)/3.0);
}`;

/* Stage 2a. Retinal ganglion cell generator signals: midget (L, M centre), parasol, small bistratified. */
FS.gen = HEAD + `
uniform sampler2D u_cone, u_pS;
uniform float u_dt, u_reset, u_ksP, u_ksM, u_srr, u_purE, u_wP, u_wM, u_tauP, u_tauM, u_tauK;
layout(location=0) out vec4 oG; layout(location=1) out vec4 oS; layout(location=2) out vec4 oU;
void main(){
  float E = eccDeg(uv); ivec2 q = ivec2(gl_FragCoord.xy);
  float sp = sigMidget(E)*u_ppd, sm = sigParasol(E)*u_ppd, sk = sigBistrat(E)*u_ppd;
  vec4 cP = pyr(u_cone, uv, sp), sP = pyr(u_cone, uv, sp*u_srr);
  float pur = exp(-E/u_purE);                                     // single-cone centres near the fovea
  vec4 raw;
  raw.r = mix(cP.a,cP.r,pur) - u_ksP*sP.a;                        // L-centre midget
  raw.g = mix(cP.a,cP.g,pur) - u_ksP*sP.a;                        // M-centre midget
  raw.b = pyr(u_cone,uv,sm).a - u_ksM*pyr(u_cone,uv,sm*u_srr).a;  // parasol
  raw.a = smoothstep(0.1,0.4,E+3.0/u_ppd)*(pyr(u_cone,uv,sk).b - pyr(u_cone,uv,sk*1.5).a); // S-(L+M)
  vec4 k = 1.0-exp(-u_dt/vec4(u_tauP,u_tauP,u_tauM,u_tauK));
  vec4 st = mix(texelFetch(u_pS,q,0), raw, k);
  if(u_reset>0.5) st = raw;
  vec4 g; g.rg = raw.rg - u_wP*st.rg;                             // sustained
  g.b = raw.b - u_wM*st.b;                                        // transient (band-pass in time)
  g.a = st.a;                                                     // sluggish
  float gl = (2.0*g.r+g.g)/3.0;
  float b = ((2.0*raw.r+raw.g) - (2.0*st.r+st.g))/3.0;            // fine-scale transient (bipolar subunits)
  oG = g; oS = st; oU = vec4(max(b,0.0), max(-b,0.0), raw.b*raw.b, gl*gl);   // gain-control pools see contrast before the transient filter
}`;

/* Stage 2b + 3. RGC firing rates (contrast gain control, rectified subunits, rectification), then LGN relay. */
FS.rl = HEAD + `
uniform sampler2D u_gen, u_sub;
uniform float u_wSub, u_gM, u_c50M, u_gP, u_c50P, u_gK, u_relay, u_thr, u_sfL;
layout(location=0) out vec4 oRA; layout(location=1) out vec4 oRB; layout(location=2) out vec4 oLA; layout(location=3) out vec4 oLB;
void main(){
  float E = eccDeg(uv); ivec2 q = ivec2(gl_FragCoord.xy);
  float sp = sigMidget(E)*u_ppd, sm = sigParasol(E)*u_ppd;
  vec4 g = texelFetch(u_gen,q,0);
  vec4 uM = pyr(u_sub, uv, sm);
  float cM = sqrt(max(pyr(u_sub, uv, max(sm*2.5,1.0)).b, 0.0));   // local RMS contrast, magno band
  float cP = sqrt(max(pyr(u_sub, uv, max(sp*2.5,1.0)).a, 0.0));
  float gainM = u_gM/(1.0 + cM/u_c50M), gainP = u_gP/(1.0 + cP/u_c50P);
  const vec2 b0 = vec2(0.09, 0.05);                               // maintained rate: ON cells higher, OFF more rectified
  vec4 base = vec4(b0.x,b0.y,b0.x,b0.y);
  oRA = max(vec4(0.0), base + gainP*vec4(g.r,-g.r,g.g,-g.g));
  oRB = max(vec4(0.0), vec4(b0.x + gainM*( g.b + u_wSub*uM.r), b0.y + gainM*(-g.b + u_wSub*uM.g),
                            0.06 + u_gK*g.a, 0.03 - 0.6*u_gK*g.a));
  float sM = sqrt(max(pyr(u_sub, uv, max(sm*5.0,1.5)).b, 0.0));   // LGN suppressive field
  float sP = sqrt(max(pyr(u_sub, uv, max(sp*5.0,1.5)).a, 0.0));
  // divisive suppression driven by the saturating local-contrast signal (after Bonin, Mante & Carandini 2005)
  float lM = u_relay/(1.0 + u_sfL*0.6*sM/(sM + u_c50M)), lP = u_relay/(1.0 + u_sfL*0.6*sP/(sP + u_c50P));
  oLA = max(vec4(0.0), lP*(oRA - u_thr*base));
  oLB = max(vec4(0.0), vec4(lM,lM,lP,lP)*(oRB - u_thr*vec4(b0.x,b0.y,0.06,0.03)));
}`;

/* Stage 4a. Geniculate input to V1: luminance (P + M), red-green, blue-yellow. */
FS.v1in = HEAD + `
uniform sampler2D u_lA, u_lB; uniform float u_wMV1;
out vec4 o;
void main(){
  ivec2 q = ivec2(gl_FragCoord.xy); vec4 A = texelFetch(u_lA,q,0), B = texelFetch(u_lB,q,0);
  float l = A.r-A.g, m = A.b-A.a, mg = B.r-B.g;
  o = vec4((2.0*l+m)/3.0 + u_wMV1*mg, l-m, B.b-B.a, mg);
}`;

/* Stage 4b. V1 linear receptive fields: quadrature Gabor pairs, 8 orientations, one SF band. */
FS.gabor = `#version 300 es
precision highp float; precision highp int; precision highp sampler2D;
uniform sampler2D u_in, u_kern; uniform int u_lod; uniform float u_gain;
layout(location=0) out vec4 o0; layout(location=1) out vec4 o1; layout(location=2) out vec4 o2; layout(location=3) out vec4 o3;
void main(){
  ivec2 p = ivec2(gl_FragCoord.xy); ivec2 sz = textureSize(u_in, u_lod);
  vec4 a0=vec4(0.0), a1=vec4(0.0), a2=vec4(0.0), a3=vec4(0.0);
  for(int j=-5;j<=5;j++) for(int i=-5;i<=5;i++){
    float v = texelFetch(u_in, clamp(p+ivec2(i,j), ivec2(0), sz-1), u_lod).r;
    int t = (j+5)*11 + (i+5);
    a0 += v*texelFetch(u_kern, ivec2(t,0), 0); a1 += v*texelFetch(u_kern, ivec2(t,1), 0);
    a2 += v*texelFetch(u_kern, ivec2(t,2), 0); a3 += v*texelFetch(u_kern, ivec2(t,3), 0);
  }
  o0 = a0*u_gain; o1 = a1*u_gain; o2 = a2*u_gain; o3 = a3*u_gain;
}`;

/* Stage 4c. Complex-cell energy (e^2 + o^2), weighted by the local SF population, and direction from phase advance. */
FS.energy = HEAD + `
uniform sampler2D u_c0,u_c1,u_c2,u_c3,u_p0,u_p1,u_p2,u_p3;
uniform float u_fs, u_sw;
layout(location=0) out vec4 oEA; layout(location=1) out vec4 oEB; layout(location=2) out vec4 oDA; layout(location=3) out vec4 oDB;
vec2 ed(vec2 c, vec2 p, float w2){
  float e = dot(c,c), ep = dot(p,p);
  return vec2(e*w2, (c.y*p.x - c.x*p.y)/(sqrt(e*ep) + 1e-7));
}
void main(){
  ivec2 q = ivec2(gl_FragCoord.xy); float E = eccDeg(uv);
  float fp = 1.0/(0.12*E + 0.35);                                 // preferred SF vs eccentricity (Broderick et al. 2022)
  float lo = log2(u_fs/fp)/u_sw; float w2 = exp(-lo*lo);          // population weight, squared
  vec4 c0=texelFetch(u_c0,q,0), c1=texelFetch(u_c1,q,0), c2=texelFetch(u_c2,q,0), c3=texelFetch(u_c3,q,0);
  vec4 p0=texelFetch(u_p0,q,0), p1=texelFetch(u_p1,q,0), p2=texelFetch(u_p2,q,0), p3=texelFetch(u_p3,q,0);
  vec2 r0=ed(c0.xy,p0.xy,w2), r1=ed(c0.zw,p0.zw,w2), r2=ed(c1.xy,p1.xy,w2), r3=ed(c1.zw,p1.zw,w2);
  vec2 r4=ed(c2.xy,p2.xy,w2), r5=ed(c2.zw,p2.zw,w2), r6=ed(c3.xy,p3.xy,w2), r7=ed(c3.zw,p3.zw,w2);
  oEA = vec4(r0.x,r1.x,r2.x,r3.x); oEB = vec4(r4.x,r5.x,r6.x,r7.x);
  oDA = vec4(r0.y,r1.y,r2.y,r3.y); oDB = vec4(r4.y,r5.y,r6.y,r7.y);
}`;

/* Stage 4d. Divisive normalization (local cross-orientation / cross-SF pool + orientation-tuned surround) and contrast adaptation. */
FS.norm = HEAD + `
uniform sampler2D u_EA,u_EB,u_EAf,u_EBf,u_EAc,u_EBc,u_pA,u_pB;
uniform float u_hasF,u_hasC,u_sig,u_n,u_wLoc,u_wSur,u_tun,u_kAd,u_tauAd,u_dt,u_adOn,u_lodL,u_lodS,u_kn;
layout(location=0) out vec4 oRA; layout(location=1) out vec4 oRB; layout(location=2) out vec4 oAA; layout(location=3) out vec4 oAB;
void main(){
  ivec2 q = ivec2(gl_FragCoord.xy);
  vec4 EA = texelFetch(u_EA,q,0), EB = texelFetch(u_EB,q,0);
  float loc = sum4(textureLod(u_EA,uv,u_lodL)) + sum4(textureLod(u_EB,uv,u_lodL));
  if(u_hasF>0.5) loc += 0.5*(sum4(textureLod(u_EAf,uv,u_lodL+1.0)) + sum4(textureLod(u_EBf,uv,u_lodL+1.0)));
  if(u_hasC>0.5) loc += 0.5*(sum4(textureLod(u_EAc,uv,max(u_lodL-1.0,0.0))) + sum4(textureLod(u_EBc,uv,max(u_lodL-1.0,0.0))));
  loc /= u_kn;
  vec4 sA = textureLod(u_EA,uv,u_lodS), sB = textureLod(u_EB,uv,u_lodS);
  float sun = (sum4(sA)+sum4(sB))/u_kn;
  vec4 pA = texelFetch(u_pA,q,0), pB = texelFetch(u_pB,q,0);
  float h = 0.5*u_n;
  vec4 dA = pow(u_sig*(1.0+u_kAd*pA), vec4(u_n)) + pow(max(u_wLoc*loc + u_wSur*(u_tun*sA + (1.0-u_tun)*sun), 1e-12), vec4(h));
  vec4 dB = pow(u_sig*(1.0+u_kAd*pB), vec4(u_n)) + pow(max(u_wLoc*loc + u_wSur*(u_tun*sB + (1.0-u_tun)*sun), 1e-12), vec4(h));
  oRA = pow(max(EA,1e-12), vec4(h))/dA; oRB = pow(max(EB,1e-12), vec4(h))/dB;
  float k = u_adOn*(1.0-exp(-u_dt/u_tauAd));
  oAA = mix(pA, oRA, k); oAB = mix(pB, oRB, k);
}`;

/* V1 read-out across SF bands: orientation vector, total, SF centroid, motion vector. */
FS.summary = HEAD + `
uniform sampler2D u_RA0,u_RB0,u_DA0,u_DB0,u_RA1,u_RB1,u_DA1,u_DB1,u_RA2,u_RB2,u_DA2,u_DB2,u_RA3,u_RB3,u_DA3,u_DB3;
layout(location=0) out vec4 o0; layout(location=1) out vec4 o1;
void acc(vec4 R, vec4 D, int base, float s, inout vec2 ov, inout float tot, inout float sc, inout vec2 mv, inout float mx){
  for(int k=0;k<4;k++){ float th = float(base+k)*0.39269908; float r = R[k];
    ov += r*vec2(cos(2.0*th), sin(2.0*th)); tot += r; sc += r*s; mx = max(mx, r);
    mv += r*D[k]*vec2(-sin(th), cos(th)); }
}
void main(){
  vec2 ov = vec2(0.0), mv = vec2(0.0); float tot = 0.0, sc = 0.0, mx = 0.0;
  acc(texture(u_RA0,uv), texture(u_DA0,uv), 0, 0.0, ov,tot,sc,mv,mx); acc(texture(u_RB0,uv), texture(u_DB0,uv), 4, 0.0, ov,tot,sc,mv,mx);
  acc(texture(u_RA1,uv), texture(u_DA1,uv), 0, 1.0, ov,tot,sc,mv,mx); acc(texture(u_RB1,uv), texture(u_DB1,uv), 4, 1.0, ov,tot,sc,mv,mx);
  acc(texture(u_RA2,uv), texture(u_DA2,uv), 0, 2.0, ov,tot,sc,mv,mx); acc(texture(u_RB2,uv), texture(u_DB2,uv), 4, 2.0, ov,tot,sc,mv,mx);
  acc(texture(u_RA3,uv), texture(u_DA3,uv), 0, 3.0, ov,tot,sc,mv,mx); acc(texture(u_RB3,uv), texture(u_DB3,uv), 4, 3.0, ov,tot,sc,mv,mx);
  o0 = vec4(ov, tot, sc/(tot+1e-6)); o1 = vec4(mv, mx, 1.0);
}`;

/* V1 colour: single-opponent (blob-like, low-pass) and double-opponent (chromatic contrast) responses. */
FS.colour = HEAD + `
uniform sampler2D u_in; out vec4 o;
void main(){
  float E = eccDeg(uv); float sb = (0.12 + 0.03*E)*u_rf*u_ppd;
  vec4 f = pyr(u_in, uv, sb*0.5), c = pyr(u_in, uv, sb), w = pyr(u_in, uv, sb*3.0);
  vec2 so = c.gb; vec2 dop = abs(f.gb - w.gb);
  o = vec4(so/(0.15+abs(so)), dop/(0.1+dop));
}`;

/* Copy one texel (the probe). */
FS.copy = `#version 300 es
precision highp float; precision highp sampler2D;
uniform sampler2D u_t; uniform vec2 u_p; out vec4 o;
void main(){ o = textureLod(u_t, u_p, 0.0); }`;

/* Display. */
FS.view = HEAD + `
uniform sampler2D u_t0, u_t1; uniform int u_mode, u_cm; uniform float u_gain, u_marks, u_emax; uniform vec4 u_w;
uniform vec2 u_probe, u_tile; uniform mat3 u_lms2rgb;
out vec4 o;
const vec3 BG = vec3(0.045,0.05,0.065);
vec3 cyc(float h){ return 0.5+0.5*cos(TAU*(h+vec3(0.0,0.33,0.67))); }
vec3 lin2srgb(vec3 c){ c = clamp(c,0.0,1.0); return mix(c*12.92, 1.055*pow(c, vec3(1.0/2.4))-0.055, step(0.0031308, c)); }
vec3 dv(float x){
  vec3 pos = u_cm==0 ? vec3(1.0,0.74,0.32) : (u_cm==1 ? vec3(1.0,0.30,0.32) : vec3(0.45,0.52,1.0));
  vec3 neg = u_cm==0 ? vec3(0.30,0.62,1.0) : (u_cm==1 ? vec3(0.15,0.80,0.55) : vec3(0.95,0.85,0.25));
  return mix(BG, x>0.0?pos:neg, min(1.0, abs(x)));
}
vec3 oriCol(vec2 ov, float tot){
  float sel = length(ov)/(tot+1e-5); float v = 1.0-exp(-tot*u_gain);
  return BG + v*mix(vec3(0.75), cyc(atan(ov.y,ov.x)/TAU), clamp(sel*1.6,0.0,1.0));
}
vec2 oriVec(vec4 A, vec4 B, out float tot){
  vec2 ov = vec2(0.0); tot = 0.0;
  for(int k=0;k<4;k++){ float t0 = float(k)*0.39269908, t1 = float(k+4)*0.39269908;
    ov += A[k]*vec2(cos(2.0*t0),sin(2.0*t0)) + B[k]*vec2(cos(2.0*t1),sin(2.0*t1)); tot += A[k]+B[k]; }
  return ov;
}
void main(){
  vec3 col; vec2 p = uv;
  if(u_mode==0){ col = lin2srgb(u_lms2rgb*texture(u_t0,p).rgb); }
  else if(u_mode==1){ col = clamp(u_lms2rgb*(0.5+0.5*u_gain*texture(u_t0,p).rgb), 0.0, 1.0); }
  else if(u_mode==2){ col = dv(dot(texture(u_t0,p),u_w)*u_gain); }
  else if(u_mode==3){ vec4 s = texture(u_t0,p); col = oriCol(s.xy, s.z); }
  else if(u_mode==4){ float tot; vec2 ov = oriVec(texture(u_t0,p), texture(u_t1,p), tot); col = oriCol(ov, tot); }
  else if(u_mode==5){ vec2 mv = texture(u_t1,p).xy; col = BG + (1.0-exp(-length(mv)*u_gain))*cyc(atan(mv.y,mv.x)/TAU); }
  else if(u_mode==6){ vec4 c = texture(u_t0,p);
    col = vec3(0.16) + u_gain*(c.r*vec3(0.55,-0.35,-0.08) + c.g*vec3(-0.22,-0.16,0.6)) + 0.35*(c.b+c.a); col = clamp(col,0.0,1.0); }
  else {
    // V1 flat map: complex-log (monopole) map, M = 17.3/(E+0.75) mm/deg (Horton & Hoyt 1991)
    float a = 0.75, k = 17.3; bool lh = p.x < 0.5; float px = lh ? p.x*2.0 : (1.0-p.x)*2.0;
    float Wmm = k*log((u_emax+a)/a), Hmm = 2.0*k*atan(u_emax/a);
    float asp = 0.5*u_tile.x/u_tile.y;                       // one hemisphere's width / height
    float sc = max(Wmm/asp, Hmm)*1.04;                       // mm per tile height
    vec2 wmm = vec2((px-0.02)*asp*sc, (p.y-0.5)*sc);
    float ex = exp(wmm.x/k); vec2 z = a*vec2(ex*cos(wmm.y/k)-1.0, ex*sin(wmm.y/k));
    float E = length(z);
    if(wmm.x < 0.0 || z.x < 0.0 || E > u_emax || abs(wmm.y/k) > 1.5708){ col = vec3(0.075,0.085,0.11); }
    else {
      vec2 vf = vec2(lh ? z.x : -z.x, -z.y);                 // LH sees the right field; upper field maps ventrally (down)
      vec2 q = u_fix + vf*u_ppd/u_res;
      if(q.x<0.0||q.x>1.0||q.y<0.0||q.y>1.0) col = vec3(0.10,0.11,0.14);
      else { vec4 s = texture(u_t0,q); col = oriCol(s.xy, s.z); }
      float ring = 1.0;
      for(int i=0;i<5;i++){ float e0 = i==0?1.0: i==1?2.0: i==2?5.0: i==3?10.0: 20.0; ring = min(ring, abs(E-e0)/(E+a)); }
      col = mix(col, vec3(0.8), 0.35*(1.0-smoothstep(0.004,0.012,ring)));
    }
  }
  if(u_marks>0.5 && u_mode!=7){
    vec2 dp = (p-u_probe)*u_tile; float d = length(dp);
    float ring = 1.0-smoothstep(1.0,2.2,abs(d-7.0));
    col = mix(col, vec3(0.0), 0.6*(1.0-smoothstep(1.5,3.5,abs(d-7.0)))); col = mix(col, vec3(1.0,0.72,0.3), ring);
    vec2 df = abs((p-u_fix)*u_tile); float cr = (1.0-smoothstep(0.6,1.6,min(df.x,df.y)))*step(max(df.x,df.y),6.0);
    col = mix(col, vec3(1.0), cr);
    if(u_mode==0){ float E = eccDeg(p); float rr = 1.0;
      for(int i=0;i<5;i++){ float e0 = i==0?2.0: i==1?5.0: i==2?10.0: i==3?20.0: 40.0; rr = min(rr, abs(E-e0)); }
      col = mix(col, vec3(1.0), 0.28*(1.0-smoothstep(0.5,1.5, rr*u_ppd*u_tile.x/u_res.x))); }
  }
  o = vec4(col,1.0);
}`;
