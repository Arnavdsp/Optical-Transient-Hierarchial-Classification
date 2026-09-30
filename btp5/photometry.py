"""
Photometry, the Villar (2019) parametric fit, the quality gate, and feature extraction.

Every definition below is VERBATIM from the v4 notebook
(`notebooks/source/BTP_TNS_Hierarchical_v4_original.ipynb`), lifted by AST span by
`tools/extract_v5_photometry.py`. The v5 brief (Section 1) says to preserve v4's
pipeline except where explicitly changed, and none of these were.

They live in a module so the v5 additions can be tested against them offline and so
Phase 2/3 can run headless. `tools/build_notebook_v5.py` inlines this module back into
the notebook, and `tests5/test_photometry_parity.py` asserts the two stay identical.
"""

import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
from scipy.optimize import least_squares

from .config import QG


# ---------------------------------------------------------------------------
# Photometry helpers. ALeRCE returns PSF magnitudes; the parametric model is defined
# in flux, so convert once, up front, and carry the uncertainty through properly.
# ---------------------------------------------------------------------------
ZP = 27.5   # arbitrary but fixed zeropoint; only differences of magnitudes matter


def mag_to_flux(mag, sigma_mag=None):
    f = 10.0 ** (-0.4 * (np.asarray(mag, float) - ZP))
    if sigma_mag is None:
        return f
    fe = f * 0.4 * np.log(10.0) * np.asarray(sigma_mag, float)
    return f, fe


def clean_band(t, f, fe, min_points=5, sigma_clip=6.0):
    """Drop NaNs and gross outliers. Looser than v2's 5-sigma clip on magnitudes,
    because a real supernova peak IS a large excursion and v2's clip was sometimes
    eating it."""
    ok = np.isfinite(t) & np.isfinite(f) & np.isfinite(fe) & (fe > 0)
    t, f, fe = np.asarray(t)[ok], np.asarray(f)[ok], np.asarray(fe)[ok]
    if len(t) < min_points:
        return None
    # clip only on the residual to a running median, so the transient itself survives
    med = np.median(f)
    mad = np.median(np.abs(f - med)) * 1.4826 + 1e-12
    keep = (f - med) > -sigma_clip * mad          # one-sided: keep bright excursions
    t, f, fe = t[keep], f[keep], fe[keep]
    if len(t) < min_points:
        return None
    o = np.argsort(t)
    return t[o], f[o], fe[o]


def villar(t, A, beta, gamma, t0, tau_rise, tau_fall):
    tau_rise = max(tau_rise, 1e-3); tau_fall = max(tau_fall, 1e-3)
    sig = 1.0 / (1.0 + np.exp(-(t - t0) / tau_rise))
    out = np.empty_like(t, dtype=float)
    plateau = t < (gamma + t0)
    out[plateau]  = A * (1.0 - beta * (t[plateau] - t0))
    out[~plateau] = A * (1.0 - beta * gamma) * np.exp(-(t[~plateau] - (gamma + t0)) / tau_fall)
    return sig * out


def fit_villar(t, f, fe):
    """Returns (params dict, chi2/dof) or None. Weighted least squares with bounds."""
    t = np.asarray(t, float); f = np.asarray(f, float); fe = np.asarray(fe, float)
    fe = np.where(fe > 0, fe, np.nanmedian(fe[fe > 0]) if (fe > 0).any() else 1.0)
    i = int(np.argmax(f)); A0 = max(f[i], 1e-6); t_peak0 = t[i]
    lo = [1e-8, -0.5,   0.5, t.min() - 100,  0.05,   0.5]
    hi = [A0 * 20 + 1e-6, 0.5, 400.0, t.max() + 100, 100.0, 400.0]
    best = None
    for g0 in (10.0, 40.0, 120.0):
        p0 = np.clip([A0, 1e-3, g0, t_peak0, 5.0, 20.0], lo, hi)
        try:
            r = least_squares(lambda p: (villar(t, *p) - f) / fe, p0,
                              bounds=(lo, hi), method='trf', max_nfev=400)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        return None
    dof = max(len(t) - 6, 1)
    keys = ['A', 'beta', 'gamma', 't0', 'tau_rise', 'tau_fall']
    return dict(zip(keys, best.x)), float(2 * best.cost / dof)


def _log10(x, floor=1e-8):
    return float(np.log10(max(float(x), floor)))


# ---------------------------------------------------------------------------
# The quality gate. Returns (True, '') or (False, reason).
# Called from inside the retry loop, so a rejection costs one extra candidate and
# nothing else. Every rejection reason is counted so a stalled class explains itself.
#
# v4 additions, each one measured: min_obs_span, min_days_postpeak raised 5 -> 30,
# and require_two_bands. See the ablation table at the top of the notebook.
# ---------------------------------------------------------------------------
def quality_gate(bands, redshift, qg=QG):
    """`bands` maps 'g'/'r' -> (t, f, fe, params, chisq)."""
    if not bands:
        return False, 'no band survived cleaning'
    if qg.get('require_two_bands') and len(bands) < 2:
        return False, 'only one band'
    n_total = sum(len(v[0]) for v in bands.values())
    if n_total < qg['min_det_total']:
        return False, f'only {n_total} detections'
    best = max(bands, key=lambda b: len(bands[b][0]))
    if len(bands[best][0]) < qg['min_det_best_band']:
        return False, f'best band has only {len(bands[best][0])} detections'
    t, f, fe, p, chi = bands[best]
    span = float(t.max() - t.min())
    if span < qg.get('min_obs_span', 0.0):
        return False, f'baseline only {span:.0f} d'
    if float(np.max(f / fe)) < qg['min_peak_snr']:
        return False, 'peak SNR too low'
    if not np.isfinite(chi) or chi > qg['max_chisq_dof']:
        return False, f'fit chisq/dof {chi:.1f}'
    if qg['require_redshift'] and not (np.isfinite(redshift) and redshift > 0):
        return False, 'no redshift'
    if qg['require_bracket']:
        t0 = p['t0']
        if (t0 - t.min()) < qg['min_days_prepeak']:
            return False, 'no pre-peak coverage'
        if (t.max() - t0) < qg['min_days_postpeak']:
            return False, 'not enough post-peak coverage'
    return True, ''


COSMO = FlatLambdaCDM(H0=70, Om0=0.3)


def distance_modulus(z):
    if not (np.isfinite(z) and z > 0):
        return np.nan
    return float(COSMO.distmod(max(z, 1e-4)).value)


def process_ztf_object(path, oid, label, redshift=np.nan, extra=None):
    """CSV of ALeRCE detections -> one feature row, or None if the quality gate fails."""
    df = pd.read_csv(path)
    if not {'mjd', 'fid', 'magpsf'}.issubset(df.columns):
        return None
    if 'sigmapsf' not in df.columns:
        df['sigmapsf'] = 0.1

    bands = {}
    for fid, bname in [(1, 'g'), (2, 'r')]:
        s = df[df['fid'] == fid]
        if len(s) < 5:
            continue
        f, fe = mag_to_flux(s['magpsf'].values, s['sigmapsf'].values)
        c = clean_band(s['mjd'].values, f, fe)
        if c is None:
            continue
        out = fit_villar(*c)
        if out is None:
            continue
        p, chi = out
        bands[bname] = (c[0], c[1], c[2], p, chi)

    ok, reason = quality_gate(bands, redshift)
    if not ok:
        return None

    ref = 'r' if 'r' in bands else 'g'
    t, f, fe, p, chi = bands[ref]

    row = {'id': oid, 'label': label, 'survey': 'ZTF', 'ref_band': ref,
           'log_A':        _log10(p['A']),
           'beta':         float(p['beta']),
           'log_gamma':    _log10(p['gamma']),
           'log_tau_rise': _log10(p['tau_rise']),
           'log_tau_fall': _log10(p['tau_fall']),
           'log_chisq':    _log10(chi)}

    # colour evolution: ratios of every fit parameter between the two bands
    if 'g' in bands and 'r' in bands:
        pg = bands['g'][3]; pr = bands['r'][3]
        row.update({'gr_log_A':        _log10(pg['A']) - _log10(pr['A']),
                    'gr_dt0':          float(pg['t0'] - pr['t0']),
                    'gr_log_tau_rise': _log10(pg['tau_rise']) - _log10(pr['tau_rise']),
                    'gr_log_tau_fall': _log10(pg['tau_fall']) - _log10(pr['tau_fall']),
                    'gr_dbeta':        float(pg['beta'] - pr['beta']),
                    'gr_log_gamma':    _log10(pg['gamma']) - _log10(pr['gamma']),
                    'has_color': 1})
    else:
        for k in ['gr_log_A', 'gr_dt0', 'gr_log_tau_rise', 'gr_log_tau_fall',
                  'gr_dbeta', 'gr_log_gamma']:
            row[k] = np.nan
        row['has_color'] = 0

    # physical scale — the feature v2 threw away
    extra = extra or {}
    av = float(extra.get('bts_av') or 0.0)
    peak_mag_app = ZP - 2.5 * row['log_A']
    dm = distance_modulus(redshift)
    row['redshift']     = float(redshift) if np.isfinite(redshift) else np.nan
    row['peak_abs_mag'] = peak_mag_app - dm - av if np.isfinite(dm) else np.nan

    # model-derived, phase-anchored timescales (NOT window-anchored)
    tt = np.linspace(p['t0'] - 200, p['t0'] + 500, 2000)
    m = villar(tt, **p); pk = float(m.max())
    if pk > 0:
        above = tt[m >= pk / 2.0]
        row['duration_half'] = float(above.max() - above.min()) if len(above) else np.nan
        row['rise_half']     = float(p['t0'] - above.min()) if len(above) else np.nan
        row['fade_half']     = float(above.max() - p['t0']) if len(above) else np.nan
    else:
        row['duration_half'] = row['rise_half'] = row['fade_half'] = np.nan

    row['n_det']    = int(sum(len(v[0]) for v in bands.values()))
    row['max_snr']  = float(np.max(f / fe))
    row['obs_span'] = float(t.max() - t.min())
    return row


# ---------------------------------------------------------------------------
# Stellar flares from TESS. Same feature vocabulary so the columns line up, but the
# Villar model is a supernova model and does not describe a flare — so for flares the
# fit is deliberately run on the *relative* flux and the resulting parameters are still
# informative (a flare is an extremely fast rise and fall, which is exactly what
# tau_rise and tau_fall report). There is no redshift and no absolute magnitude, and
# `has_color` is 0, which is itself a real observational signal, not an artefact.
# ---------------------------------------------------------------------------
def process_tess_object(path, star_name, label):
    df = pd.read_csv(path)
    if not {'bjd', 'flux'}.issubset(df.columns):
        return None
    fe = df['flux_err'].values if 'flux_err' in df else np.full(len(df), np.nan)
    if not np.isfinite(fe).any():
        fe = np.full(len(df), float(np.nanstd(df['flux'].values)) or 1.0)
    c = clean_band(df['bjd'].values, df['flux'].values, fe, min_points=20)
    if c is None:
        return None
    t, f, ferr = c
    med = np.median(f)
    if med == 0:
        return None
    f = f / med; ferr = ferr / abs(med)
    out = fit_villar(t, f, ferr)
    if out is None:
        return None
    p, chi = out
    if len(t) < QG['min_det_best_band']:
        return None

    row = {'id': star_name, 'label': label, 'survey': 'TESS', 'ref_band': 'TESS',
           'log_A': _log10(p['A']), 'beta': float(p['beta']),
           'log_gamma': _log10(p['gamma']), 'log_tau_rise': _log10(p['tau_rise']),
           'log_tau_fall': _log10(p['tau_fall']), 'log_chisq': _log10(chi)}
    for k in ['gr_log_A', 'gr_dt0', 'gr_log_tau_rise', 'gr_log_tau_fall',
              'gr_dbeta', 'gr_log_gamma']:
        row[k] = np.nan
    row['has_color'] = 0
    row['redshift'] = 0.0
    row['peak_abs_mag'] = np.nan     # imputed later; flares are Galactic
    tt = np.linspace(p['t0'] - 20, p['t0'] + 60, 2000)
    m = villar(tt, **p); pk = float(m.max())
    above = tt[m >= pk / 2.0] if pk > 0 else np.array([])
    row['duration_half'] = float(above.max() - above.min()) if len(above) else np.nan
    row['rise_half']     = float(p['t0'] - above.min()) if len(above) else np.nan
    row['fade_half']     = float(above.max() - p['t0']) if len(above) else np.nan
    row['n_det']   = int(len(t))
    row['max_snr'] = float(np.nanmax(f / ferr))
    row['obs_span'] = float(t.max() - t.min())
    return row
