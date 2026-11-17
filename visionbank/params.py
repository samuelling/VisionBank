"""All model constants in one place, each with its source.

Tags:  [V] value checked against the cited source (or code that reproduces it) while building this file
       [A] approximate: recalled or chosen to match a qualitative result; treat as a starting point
"""
from dataclasses import dataclass, field


@dataclass
class Params:
    # ---- display and eye -------------------------------------------------------------------
    peak_luminance: float = 100.0      # cd/m^2 of display white
    pupil_mm: float = 4.0              # pupil diameter
    rstar_per_td: float = 15.0         # [A] isomerisations / cone / s per troland (order of magnitude)
    cone_weights: tuple = (1.0, 0.86, 0.12)   # [A] relative L, M, S catch for display white
    offscreen: float = 0.02            # luminance outside the screen, as a fraction of white

    # ---- cones: phototransduction cascade (ISETBio osBioPhys; Angueyra & Rieke) ---------------
    # [V] equations and both parameter sets copied from isetbio/cones/outersegment/@osBioPhys
    cone_fovea: dict = field(default_factory=lambda: dict(
        sigma=10.0, phi=22.0, eta=700.0, gdark=20.5, k=0.02, h=3.0, cdark=1.0, beta=5.0, betaSlow=0.4, n=4.0, kGc=0.5, OpsinGain=12.0))
    cone_periphery: dict = field(default_factory=lambda: dict(
        sigma=22.0, phi=22.0, eta=2000.0, gdark=20.5, k=0.02, h=3.0, cdark=1.0, beta=9.0, betaSlow=0.4, n=4.0, kGc=0.5, OpsinGain=10.0))
    cone_fovea_cutoff_deg: float = 10.0   # [V] ISETBio switches parameter sets at 10 deg
    cone_dt: float = 0.001                # integration sub-step, s

    # ---- retinal ganglion cells: space ------------------------------------------------------
    # [V] Watson (2014) midget RF density formula (parameters from isetbio/calculators/@WatsonRGCModel)
    watson_meridians: tuple = ((0.9851, 1.058, 22.14), (0.9935, 1.035, 16.35), (0.9729, 1.084, 7.633), (0.996, 0.9932, 12.13))
    watson_dc0: float = 14804.6
    watson_f0: float = 1 / 1.12
    watson_rm: float = 41.03
    midget_rc_per_spacing: float = 1.0    # [A] centre radius (1/e) = this x single-polarity midget spacing;
    #                                         reproduces Croner & Kaplan's 0.03 deg (0-5 deg) and 0.05 deg (5-10 deg)
    parasol_rc_ratio: float = 2.2         # [V] "about twice" neighbouring P cells (Croner & Kaplan 1995 abstract)
    p_rs_over_rc: float = 6.7             # [V] Croner & Kaplan, as tabulated in ISETBio constants
    p_sc_ratio: tuple = (0.466, 0.007)    # [V] integrated surround/centre = 0.466 + 0.007*ecc (C&K Fig. 11)
    m_rs_fit: tuple = (0.203, 0.472)      # [V] surround radius = 0.203*ecc^0.472, floor 0.1 deg (C&K Fig. 4, M and P pooled)
    m_sc_ratio: float = 0.55              # [A] C&K mean integrated surround/centre
    cone_purity_ecc: float = 20.0         # [A] e-folding eccentricity of single-cone-type midget centres
    k_rc_ratio: float = 1.2               # [A] small-bistratified S-ON radius relative to parasol centre
    rf_scale: float = 1.0                 # global multiplier on all RF sizes

    # ---- retinal ganglion cells: time (Benardete & Kaplan) ---------------------------------
    # K(w) = A exp(-iwD) (1 - Hs/(1 + iw tauS)) (1 + iw tauL)^-NL
    # [V] P cells: population medians, Benardete & Kaplan 1999 J Physiol, Fig. 7 / Tables 2-3
    p_center: dict = field(default_factory=lambda: dict(NL=27.0, tauL=2.05e-3, Hs=0.68, tauS=21.25e-3, D=4.0e-3))
    p_surround: dict = field(default_factory=lambda: dict(NL=37.0, tauL=1.54e-3, Hs=0.61, tauS=24.41e-3, D=4.0e-3))
    p_gain: float = 100.0                 # [A] centre gain, spikes/s per unit contrast (C&K Table 1 order of magnitude)
    # [V] M cells: Benardete & Kaplan 1999 Vis Neurosci, as reproduced in J Vis 2025 (articleid 2818391)
    m_kernel: dict = field(default_factory=lambda: dict(NL=30.3, tauL=1.44e-3, Hs=1.0, T0=37.34e-3, C_half=0.051, D=2.0e-3))
    m_gain: float = 499.77                # [V] spikes/s per unit contrast (same source)
    m_contrast_max: float = 0.10          # [A] the published fit covers contrasts up to ~0.12; the time constant is held beyond this
    m_contrast_tau: float = 0.10          # [A] time constant of the contrast signal driving gain control
    m_contrast_pool: float = 2.0          # [A] pool radius for that signal, in units of centre radius
    m_subunit_weight: float = 0.5         # [A] rectified bipolar subunits (Y-like, frequency doubled)
    k_gain: float = 60.0                  # [A]
    maintained: dict = field(default_factory=lambda: dict(P=20.0, M=12.0, K=10.0))   # [A] spikes/s
    bk_reference_td: float = 1123.0       # [V] retinal illuminance of the P-cell recordings

    # ---- LGN ----------------------------------------------------------------------------------
    lgn_transfer: float = 0.6             # [A] fraction of retinal spikes relayed
    lgn_threshold: float = 0.5            # [A] x maintained rate
    lgn_sf_weight: float = 0.6            # [A] maximum divisive suppression from the suppressive field
    lgn_sf_c50: float = 0.15              # [A] contrast at half suppression
    lgn_sf_radius: float = 2.0            # [A] x parasol surround radius
    relay_gain: float = 1.0               # state (arousal) multiplier

    # ---- V1 -----------------------------------------------------------------------------------
    n_bands: int = 4
    v1_magno_weight: float = 0.3          # [A]
    v1_sigma: float = 10.0                # [A] semi-saturation, in LGN spikes/s of filter drive
    v1_n: float = 2.0
    v1_rmax: float = 60.0                 # [A] spikes/s
    v1_w_local: float = 1.0
    v1_w_surround: float = 1.0
    v1_surround_tuning: float = 0.7
    v1_tau_norm: float = 0.08             # [A] low-pass on the normalisation pool (delayed normalisation), s
    v1_delay_frac: float = 0.6            # [A] share of the pool that is delayed; sets the onset transient (about 2.5x sustained)
    v1_tau_adapt: float = 6.0             # [A] slow contrast adaptation, s
    v1_k_adapt: float = 2.0               # [A]
    sf_pref: tuple = (0.12, 0.35)         # [A] preferred period = 0.12*ecc + 0.35 deg (form verified; constants recalled)
    sf_pop_sigma: float = 1.0             # [A] octaves

    # ---- BOLD ---------------------------------------------------------------------------------
    hrf: dict = field(default_factory=lambda: dict(peak=6.0, under=16.0, ratio=1 / 6.0))   # double-gamma, s
