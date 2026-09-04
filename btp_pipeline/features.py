"""
Light-curve preprocessing and feature extraction.

Every function below is copied VERBATIM out of the previous notebook
(`notebooks/source/BTP_TNS_Hierarchical_2_original.ipynb`) by
`tools/extract_features_module.py`, which lifts them by AST span rather than
retyping them. The brief said to reuse these unchanged, and they are unchanged.

They live in a module rather than only in notebook cells so that Phase 2 can be
run headless (outside Colab) and so the retry loops can be tested against them.
`tests/test_features_parity.py` asserts this module and the generated notebook
still carry byte-identical definitions.
"""

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C

# `resolve_oid` calls a module-level `alerce` client. When this module is inlined
# into the notebook, cell 2 has already created one — so do not clobber it. When
# imported headless, the caller assigns it (see tools/run_phase2.py).
if 'alerce' not in globals():
    alerce = None


def classify_sn_subtype(t):
    """
    Maps a raw TNS `type` string to one of five SN subtype buckets, or None if it
    doesn't belong to any of them. SLSN-I/II/R are TNS's own top-level type strings
    (not prefixed 'SN '), so they're checked separately from the SN Ia/Ib/Ic/II family.
    Sub-subtypes are folded into their parent bucket for a manageable class count:
    e.g. 'SN Ia-91bg', 'SN Ia-pec', 'SN Iax' -> SN_Ia; 'SN Ic-BL', 'SN Icn' -> SN_Ic;
    'SN IIb', 'SN IIn', 'SN IIP', 'SN IIL' -> SN_II.
    """
    t = str(t)
    if 'SLSN' in t:
        return 'SLSN'
    if not t.startswith('SN '):
        return None
    rest = t[3:]
    if rest.startswith('Ia'):
        return 'SN_Ia'
    if rest.startswith('Ib'):
        return 'SN_Ib'
    if rest.startswith('Ic'):
        return 'SN_Ic'
    if rest.startswith('II'):
        return 'SN_II'
    return None


def extract_ztf_name(internal_names):
    if not isinstance(internal_names, str):
        return None
    for tok in internal_names.split(','):
        tok = tok.strip()
        if tok.startswith('ZTF'):
            return tok
    return None


def resolve_oid(row, radius_arcsec=2.0):
    ztf_name = extract_ztf_name(row.get('internal_names'))
    if ztf_name:
        return ztf_name
    try:
        res = alerce.query_objects(ra=row['ra'], dec=row['declination'],
                                    radius=radius_arcsec, format='pandas')
        if len(res):
            return res.iloc[0]['oid']
    except Exception:
        pass
    return None


# Cleaning helper
def clean_lightcurve(df, value_col, err_col=None, min_points=8, sigma_clip=5.0):
    df = df.dropna(subset=[value_col]).copy()
    if len(df) < min_points:
        return None
    med = df[value_col].median()
    mad = (df[value_col] - med).abs().median() * 1.4826 + 1e-6
    df = df[(df[value_col] - med).abs() < sigma_clip * mad]
    if len(df) < min_points:
        return None
    return df.sort_values(df.columns[0]).reset_index(drop=True)


MAX_GP_POINTS = 800   # tested: 18k points -> OOM killed instantly; 800 points -> fits in ~2s


def downsample_for_gp(t, y, max_points=MAX_GP_POINTS):
    if len(t) <= max_points:
        return t, y
    idx = np.linspace(0, len(t) - 1, max_points).astype(int)
    return t[idx], y[idx]


def gp_interpolate(t, y, n_points=200):
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    t, y = downsample_for_gp(t, y)
    t0 = t.min()
    X = (t - t0).reshape(-1, 1)
    span = max(t.max() - t.min(), 1.0)
    kernel = (C(1.0, (1e-3, 1e3))
              * RBF(length_scale=span / 10, length_scale_bounds=(1e-2, 1e3))
              + WhiteKernel(noise_level=0.05, noise_level_bounds=(1e-5, 2.0)))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=1)
    gp.fit(X, y)
    t_grid_rel = np.linspace(0, t.max() - t0, n_points)
    y_grid, y_std = gp.predict(t_grid_rel.reshape(-1, 1), return_std=True)
    return t_grid_rel + t0, y_grid


# Shape-feature extractor
def extract_shape_features(t_grid, y_grid):
    peak_val = float(y_grid.max())
    min_val  = float(y_grid.min())
    t_peak   = float(t_grid[np.argmax(y_grid)])
    rise_time  = t_peak - t_grid[0]
    decay_time = t_grid[-1] - t_peak
    amplitude  = peak_val - min_val
    return {'peak_val': peak_val, 'rise_time': rise_time, 'decay_time': decay_time, 'amplitude': amplitude}


# SN subtypes / AGN / TDE feature extraction (checkpointed, resumable)
def mag_to_relflux(mag, baseline_mag):
    return 10 ** (-0.4 * (mag - baseline_mag))


def process_ztf_object(path, oid, label):
    df = pd.read_csv(path)
    row = {'id': oid, 'label': label, 'survey': 'ZTF'}
    band_curves = {}
    for fid, band_name in [(1, 'g'), (2, 'r')]:
        band_df = clean_lightcurve(df[df['fid'] == fid][['mjd', 'magpsf']], 'magpsf', min_points=6)
        if band_df is None:
            continue
        try:
            t_grid, mag_grid = gp_interpolate(band_df['mjd'].values, band_df['magpsf'].values)
            band_curves[band_name] = (t_grid, mag_grid)
        except Exception:
            continue
    if 'r' in band_curves:
        primary_t, primary_mag = band_curves['r']
    elif 'g' in band_curves:
        primary_t, primary_mag = band_curves['g']
    else:
        return None
    baseline_mag = primary_mag.max()
    relflux = mag_to_relflux(primary_mag, baseline_mag)
    row.update(extract_shape_features(primary_t, relflux))
    if 'g' in band_curves and 'r' in band_curves:
        t_peak_idx = np.argmax(relflux)
        t_peak = primary_t[t_peak_idx]
        g_t, g_mag = band_curves['g']
        r_t, r_mag = band_curves['r']
        row['color_g_r'] = float(np.interp(t_peak, g_t, g_mag) - np.interp(t_peak, r_t, r_mag))
    else:
        row['color_g_r'] = np.nan
    return row


# Stellar flare feature extraction (checkpointed, resumable)
def process_tess_object(path, star_name, label):
    df = pd.read_csv(path)
    clean_df = clean_lightcurve(df[['bjd', 'flux']], 'flux', min_points=20)
    if clean_df is None:
        return None
    median_flux = clean_df['flux'].median()
    if median_flux == 0:
        return None
    try:
        t_grid, flux_grid = gp_interpolate(clean_df['bjd'].values, clean_df['flux'].values)
    except Exception:
        return None
    relflux = flux_grid / median_flux
    row = {'id': star_name, 'label': label, 'survey': 'TESS'}
    row.update(extract_shape_features(t_grid, relflux))
    row['color_g_r'] = np.nan
    return row
