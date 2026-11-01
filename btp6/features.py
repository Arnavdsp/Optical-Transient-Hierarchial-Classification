"""
v6 light-curve features: rest-frame, extinction-corrected, phase-anchored.

What was wrong with the v4/v5 feature set, measured on the real v5 run:

* `peak_abs_mag` came from the Villar amplitude A, not from the peak of the fitted
  curve. With beta != 0 and a finite rise the curve never reaches A, so the number is
  ~7x noisier than the curve's own maximum (0.255 vs 0.035 mag scatter).
* `rise_half` / `fade_half` were measured from t0, the midpoint of the rising
  sigmoid, which sits days before the actual peak.
* Every timescale was in the observer frame. Time dilation stretches a z = 0.2 SLSN
  by 20 % and a z = 0.03 SN II by 3 %, so the classifier had to learn redshift and
  shape jointly from 30 objects per class.
* No colour was measured at a fixed phase. Colour at peak and its evolution over the
  first month is the single most informative photometric discriminant between SN Ia,
  stripped-envelope and hydrogen-rich SNe (Villar et al. 2019, Fig. 9; Hosseinzadeh
  et al. 2020; de Soto et al. 2024, Table 3), and v5 only had a ratio of fit amplitudes.
* Nothing captured the SN IIP plateau or the SN Ia secondary r-band maximum
  directly: the Villar model is smooth, so both show up only as residuals.

Everything below is computed from the same ALeRCE detection CSVs the v5 loop already
saved, with the same Villar fit (`fit_villar`, verbatim from v4), so it re-uses every
light curve on the user's Drive.

Physics choices, each stated:

* Absolute magnitude: M = m_peak - DM(z) - A_band + 2.5 log10(1 + z). The last term
  is the bandwidth part of the K-correction for an AB magnitude (Hogg et al. 2002,
  eq. 13 with a flat f_nu); the colour-dependent part is left to the classifier,
  which sees colour directly.
* Milky Way extinction: A_band = R_band * E(B-V), with E(B-V) from the OSC (which
  carries SFD values) when available, else BTS A_V / 3.1, else zero. R_g = 3.172,
  R_r = 2.271: Schlafly & Finkbeiner (2011) Table 6, R_V = 3.1, Pan-STARRS1 g and r,
  the closest tabulated match to the ZTF g and r passbands.
* Rest-frame phase: (t - t_peak) / (1 + z), with t_peak the r-band curve maximum
  (g if r is missing).
* Model quantities are reported only inside the window the data cover (plus a small
  margin). Extrapolating a parametric fit past the last detection is how v4 produced
  60-day declines for objects observed for 20 days.
"""

import numpy as np
import pandas as pd

try:                                    # in the notebook these already exist as globals
    from btp5.photometry import (ZP, mag_to_flux, clean_band, fit_villar, villar,
                                 distance_modulus)
except ImportError:                     # pragma: no cover - notebook path
    pass

R_BAND = {'g': 3.172, 'r': 2.271}
COLOR_PHASES = (-10, 0, 15, 30, 60)
DECLINE_PHASES = (15, 30, 60)
EXTRAPOLATION_MARGIN = 10.0             # observer-frame days past the data window

V6_FEATURES = (
    ['redshift', 'log1pz', 'ebv'] +
    [f'M_peak_{b}' for b in 'gr'] +
    [f'{q}_{b}' for b in 'gr' for q in ('rise_half', 'fall_half', 'rise_10', 'fall_20')] +
    [f'dm{d}_{b}' for b in 'gr' for d in DECLINE_PHASES] +
    [f'gr_{p:+d}'.replace('+', 'p').replace('-', 'm') for p in COLOR_PHASES] +  # gr_m10, gr_p0, ...
    ['gr_slope_0_30', 'dt_peak_gr',
     'slope_early_r', 'slope_late_r', 'slope_early_g', 'slope_late_g', 'plateau_ratio_r',
     'resid_bump_r', 'resid_bump_g', 'gr_obs_peak',
     'frac_prepeak', 'n_det_g', 'n_det_r', 'rest_span', 'log_chisq_r', 'log_chisq_g',
     'v_beta_r', 'v_log_gamma_rest_r', 'v_log_tau_rise_rest_r', 'v_log_tau_fall_rest_r',
     'v_log_tau_fall_rest_g', 'v_log_tau_rise_rest_g']
)


def _phase_name(p):
    return f'gr_{p:+d}'.replace('+', 'p').replace('-', 'm')


def _crossing(tt, m, level, i, before):
    """Observer-frame time between the peak (grid index i) and the nearest crossing
    of `level` before or after it."""
    if before:
        side = np.where(m[:i] < level)[0]
        return np.nan if not len(side) else tt[i] - tt[side[-1]]
    side = np.where(m[i:] < level)[0]
    return np.nan if not len(side) else tt[i + side[0]] - tt[i]


def _slope(t, mag, lo, hi):
    """Linear decline in mag per rest-frame day over [lo, hi] days after peak."""
    k = (t >= lo) & (t <= hi)
    if k.sum() < 3 or np.ptp(t[k]) < 5:
        return np.nan
    return float(np.polyfit(t[k], mag[k], 1)[0])


def fit_bands(df):
    """ALeRCE detections -> {'g'|'r': (t, f, fe, params, chisq)} using v4's own fit."""
    bands = {}
    for fid, b in [(1, 'g'), (2, 'r')]:
        s = df[df['fid'] == fid]
        if len(s) < 5:
            continue
        sig = s['sigmapsf'].values if 'sigmapsf' in s else np.full(len(s), 0.1)
        f, fe = mag_to_flux(s['magpsf'].values, sig)
        c = clean_band(s['mjd'].values, f, fe)
        if c is None:
            continue
        out = fit_villar(*c)
        if out is None:
            continue
        bands[b] = (c[0], c[1], c[2], out[0], out[1])
    return bands


def v6_features(df, redshift, ebv=np.nan, bands=None):
    """One dict of V6_FEATURES for one object. NaN where a quantity is undefined.

    `df` is the detection table (mjd, fid, magpsf, sigmapsf). `bands` may be passed
    to reuse fits already made by the quality gate.
    """
    out = {k: np.nan for k in V6_FEATURES}
    z = float(redshift) if redshift is not None and np.isfinite(redshift) and redshift > 0 else np.nan
    ebv = float(ebv) if ebv is not None and np.isfinite(ebv) else 0.0
    out['redshift'] = z
    out['log1pz'] = np.log10(1 + z) if np.isfinite(z) else np.nan
    out['ebv'] = ebv
    if bands is None:
        bands = fit_bands(df)
    if not bands:
        return out
    zz = 1.0 + (z if np.isfinite(z) else 0.0)
    dm = distance_modulus(z) if np.isfinite(z) else np.nan

    grids = {}
    for b, (t, f, fe, p, chi) in bands.items():
        tt = np.arange(t.min() - 60.0, t.max() + 120.0, 0.25)
        m = villar(tt, **p)
        i = int(np.argmax(m))
        grids[b] = (tt, m, i)
        fpk = float(m[i])
        out[f'log_chisq_{b}'] = float(np.log10(max(chi, 1e-8)))
        out[f'n_det_{b}'] = int(len(t))
        out[f'v_log_tau_fall_rest_{b}'] = np.log10(max(p['tau_fall'], 1e-3) / zz)
        out[f'v_log_tau_rise_rest_{b}'] = np.log10(max(p['tau_rise'], 1e-3) / zz)
        if b == 'r':
            out['v_beta_r'] = float(p['beta'])
            out['v_log_gamma_rest_r'] = np.log10(max(p['gamma'], 1e-3) / zz)
        if fpk <= 0:
            continue
        mpk = ZP - 2.5 * np.log10(fpk)
        if np.isfinite(dm):
            out[f'M_peak_{b}'] = mpk - dm - R_BAND[b] * ebv + 2.5 * np.log10(zz)
        lo_ok = t.min() - EXTRAPOLATION_MARGIN
        hi_ok = t.max() + EXTRAPOLATION_MARGIN
        for name, lev, before in [('rise_half', 0.5, True), ('rise_10', 0.1, True),
                                  ('fall_half', 0.5, False), ('fall_20', 0.2, False)]:
            dt = _crossing(tt, m, lev * fpk, i, before)
            if np.isfinite(dt):
                edge = tt[i] - dt if before else tt[i] + dt
                if lo_ok <= edge <= hi_ok:
                    out[f'{name}_{b}'] = dt / zz
        for d in DECLINE_PHASES:
            te = tt[i] + d * zz
            if te <= hi_ok:
                fe_ = float(villar(np.array([te]), **p)[0])
                if fe_ > 0:
                    out[f'dm{d}_{b}'] = 2.5 * np.log10(fpk / fe_)

        # model-free decline slopes from the detections themselves
        mag = ZP - 2.5 * np.log10(np.clip(f, 1e-12, None))
        ph = (t - tt[i]) / zz
        out[f'slope_early_{b}'] = _slope(ph, mag, 0, 30)
        out[f'slope_late_{b}'] = _slope(ph, mag, 30, 100)
        # residual bump: the SN Ia secondary maximum (r/i) and IIb/He shoulders sit
        # 15-40 rest-frame days after peak, where a smooth one-peak model under-predicts
        k = (ph >= 15) & (ph <= 40)
        if k.sum() >= 2:
            out[f'resid_bump_{b}'] = float(np.median((f[k] - villar(t[k], **p)) / fpk))

    ref = 'r' if 'r' in grids else 'g'
    tt_r, m_r, i_r = grids[ref]
    t_peak = tt_r[i_r]
    t_all = np.concatenate([bands[b][0] for b in bands])
    out['frac_prepeak'] = float(np.mean(t_all < t_peak))
    out['rest_span'] = float(np.ptp(t_all) / zz)
    if np.isfinite(out.get('slope_early_r', np.nan)) and np.isfinite(out.get('slope_late_r', np.nan)):
        se, sl = out['slope_early_r'], out['slope_late_r']
        out['plateau_ratio_r'] = float(sl / se) if abs(se) > 1e-3 else np.nan

    if 'g' in bands and 'r' in bands:
        pg, pr = bands['g'][3], bands['r'][3]
        tg, tr = bands['g'][0], bands['r'][0]
        lo = max(tg.min(), tr.min()) - EXTRAPOLATION_MARGIN
        hi = min(tg.max(), tr.max()) + EXTRAPOLATION_MARGIN
        tt_g, m_g, i_g = grids['g']
        out['dt_peak_gr'] = (tt_g[i_g] - tt_r[i_r]) / zz if ref == 'r' else np.nan
        cols = {}
        for ph in COLOR_PHASES:
            te = t_peak + ph * zz
            if lo <= te <= hi:
                fg = float(villar(np.array([te]), **pg)[0])
                fr = float(villar(np.array([te]), **pr)[0])
                if fg > 0 and fr > 0:
                    c = -2.5 * np.log10(fg / fr) - (R_BAND['g'] - R_BAND['r']) * ebv
                    cols[ph] = c
                    out[_phase_name(ph)] = c
        if 0 in cols and 30 in cols:
            out['gr_slope_0_30'] = (cols[30] - cols[0]) / 30.0
        # colour at the observed peak straight from the data: nearest g and r points
        # within 3 days of the r-band peak, no model involved
        dg = df[(df['fid'] == 1) & (np.abs(df['mjd'] - t_peak) < 3)]
        dr = df[(df['fid'] == 2) & (np.abs(df['mjd'] - t_peak) < 3)]
        if len(dg) and len(dr):
            out['gr_obs_peak'] = float(dg['magpsf'].median() - dr['magpsf'].median()
                                       - (R_BAND['g'] - R_BAND['r']) * ebv)
    return out


def features_from_csv(path, redshift, ebv=np.nan):
    df = pd.read_csv(path)
    if not {'mjd', 'fid', 'magpsf'}.issubset(df.columns):
        return None
    return v6_features(df, redshift, ebv)


# ---------------------------------------------------------------------------
# Joint 2-D Gaussian-process features (Boone 2019, "Avocado", AJ 158, 257: the
# approach that won PLAsTiCC). One GP over (time, wavelength) instead of two
# independent per-band fits, so a sparsely sampled g band borrows shape from r and
# colours at fixed phase are interpolated consistently. Matern nu=3/2 kernel, time
# length scale fitted, wavelength length scale fixed at 6000 A (Boone's choice),
# heteroscedastic noise from the photometric errors.
# ---------------------------------------------------------------------------
ZTF_WAVELENGTH = {1: 4746.48, 2: 6366.38}      # effective wavelengths, Angstrom
GP_PHASES = (-10, -5, 0, 5, 10, 20, 30, 50)
GP_FEATURES = (
    [f'gp_gr_{p:+d}'.replace('+', 'p').replace('-', 'm') for p in GP_PHASES] +
    ['gp_M_peak_g', 'gp_M_peak_r', 'gp_dt_peak_gr', 'gp_log_ls_rest',
     'gp_rise_half_g', 'gp_rise_half_r', 'gp_fall_half_g', 'gp_fall_half_r',
     'gp_fall_20_r', 'gp_dm15_g', 'gp_dm15_r', 'gp_dm30_r', 'gp_gr_slope_0_20']
)


def gp_features(df, redshift, ebv=np.nan, min_points=8):
    """GP_FEATURES for one object; NaN where undefined. Never raises."""
    out = {k: np.nan for k in GP_FEATURES}
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern
    except ImportError:          # pragma: no cover
        return out
    d = df[df['fid'].isin([1, 2]) & np.isfinite(df['magpsf'])].copy()
    if len(d) < min_points or d['fid'].nunique() < 2:
        return out
    sig = d['sigmapsf'].values if 'sigmapsf' in d else np.full(len(d), 0.1)
    f, fe = mag_to_flux(d['magpsf'].values, sig)
    scale = float(np.max(f))
    if not np.isfinite(scale) or scale <= 0:
        return out
    f, fe = f / scale, np.maximum(fe / scale, 1e-3)
    t = d['mjd'].values.astype(float)
    wl = d['fid'].map(ZTF_WAVELENGTH).values.astype(float)
    # wavelength in units of the fixed 6000 A length scale; sklearn cannot fix one
    # dimension of an anisotropic kernel, so its bounds are pinned to 1 +- 1e-3
    X = np.column_stack([t, wl / 6000.0])
    kernel = ConstantKernel(0.5, (1e-3, 1e2)) * Matern(length_scale=[20.0, 1.0],
                                                       length_scale_bounds=[(2.0, 300.0), (0.999, 1.001)],
                                                       nu=1.5)
    try:
        gp = GaussianProcessRegressor(kernel=kernel, alpha=fe ** 2, normalize_y=False,
                                      n_restarts_optimizer=0, random_state=0).fit(X, f)
    except Exception:
        return out
    z = float(redshift) if redshift is not None and np.isfinite(redshift) and redshift > 0 else np.nan
    zz = 1.0 + (z if np.isfinite(z) else 0.0)
    ebv = float(ebv) if ebv is not None and np.isfinite(ebv) else 0.0
    tt = np.arange(t.min() - 20.0, t.max() + 20.0, 0.5)
    curves = {}
    for fid, b in [(1, 'g'), (2, 'r')]:
        curves[b] = gp.predict(np.column_stack([tt, np.full_like(tt, ZTF_WAVELENGTH[fid] / 6000.0)]))
    ls = gp.kernel_.k2.length_scale
    out['gp_log_ls_rest'] = float(np.log10(ls[0] / zz))
    ir = int(np.argmax(curves['r'])); ig = int(np.argmax(curves['g']))
    t_peak = tt[ir]
    lo, hi = t.min() - 10.0, t.max() + 10.0
    dm = distance_modulus(z) if np.isfinite(z) else np.nan
    for b, i in [('g', ig), ('r', ir)]:
        c = curves[b]; pk = c[i]
        if pk <= 0:
            continue
        if np.isfinite(dm):
            out[f'gp_M_peak_{b}'] = (ZP - 2.5 * np.log10(pk * scale)) - dm - R_BAND[b] * ebv + 2.5 * np.log10(zz)
        pre = np.where(c[:i] < 0.5 * pk)[0]
        post = np.where(c[i:] < 0.5 * pk)[0]
        if len(pre) and tt[pre[-1]] >= lo:
            out[f'gp_rise_half_{b}'] = (tt[i] - tt[pre[-1]]) / zz
        if len(post) and tt[i + post[0]] <= hi:
            out[f'gp_fall_half_{b}'] = (tt[i + post[0]] - tt[i]) / zz
        if b == 'r':
            p20 = np.where(c[i:] < 0.2 * pk)[0]
            if len(p20) and tt[i + p20[0]] <= hi:
                out['gp_fall_20_r'] = (tt[i + p20[0]] - tt[i]) / zz
        for dd, key in [(15, f'gp_dm15_{b}'), (30, f'gp_dm30_{b}')]:
            if key not in out:
                continue
            j = np.searchsorted(tt, tt[i] + dd * zz)
            if j < len(tt) and tt[j] <= hi and c[j] > 0:
                out[key] = 2.5 * np.log10(pk / c[j])
    out['gp_dt_peak_gr'] = (tt[ig] - tt[ir]) / zz
    cols = {}
    for p in GP_PHASES:
        te = t_peak + p * zz
        if not (lo <= te <= hi):
            continue
        j = int(np.clip(np.searchsorted(tt, te), 0, len(tt) - 1))
        fg, fr = curves['g'][j], curves['r'][j]
        if fg > 0 and fr > 0:
            cols[p] = -2.5 * np.log10(fg / fr) - (R_BAND['g'] - R_BAND['r']) * ebv
            out[f'gp_gr_{p:+d}'.replace('+', 'p').replace('-', 'm')] = cols[p]
    if 0 in cols and 20 in cols:
        out['gp_gr_slope_0_20'] = (cols[20] - cols[0]) / 20.0
    return out
