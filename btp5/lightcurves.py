"""
Locate each object's saved light curve, and recover its sky position from it.

Why this exists: v4's feature rows (process_ztf_object) and acquisition manifests
(build_class_to_target) carry no coordinates at all. The host-offset feature needs
them, and so does augmentation (which re-reads the parent light curve). Without this,
a live run would give every object has_host_match=0 and the host feature would be
silently empty — a failure that no offline test with a coordinate-bearing fixture
could see.

The detection CSVs v4 saves DO keep per-detection ra/dec, and the median over an
object's detections is a robust position (ZTF alert astrometry scatters by
~0.1 arcsec; the median discards the occasional bad centroid). This works identically
for TNS-sourced and ALeRCE-sourced objects, since both are saved by the same loop.
"""

import os

import numpy as np
import pandas as pd


def index_lightcurves(base_dir, exclude_dirs=('tns_catalog',)):
    """{object id -> csv path} for every saved light curve under base_dir.

    Searching by filename rather than rebuilding v4's subdir naming keeps this
    correct whichever class folder an object was saved under.
    """
    idx = {}
    if not base_dir or not os.path.isdir(base_dir):
        return idx
    for root, dirs, files in os.walk(base_dir):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for fn in files:
            if fn.endswith('.csv') and not fn.endswith(('_manifest_v3.csv', '_failed_v3.csv')):
                idx.setdefault(fn[:-4], os.path.join(root, fn))
    return idx


def position_from_csv(path):
    """Median (ra, dec) over an object's detections, or (nan, nan)."""
    try:
        df = pd.read_csv(path, usecols=lambda c: c in ('ra', 'dec'))
    except Exception:
        return np.nan, np.nan
    if not {'ra', 'dec'}.issubset(df.columns) or df.empty:
        return np.nan, np.nan
    ra = df['ra'].to_numpy(float)
    dec = df['dec'].to_numpy(float)
    ok = np.isfinite(ra) & np.isfinite(dec)
    if not ok.any():
        return np.nan, np.nan
    ra, dec = ra[ok], dec[ok]
    # RA wraps at 360: an object at ra~0.0001 and ~359.9999 must not average to 180.
    if ra.max() - ra.min() > 180:
        ra = np.where(ra > 180, ra - 360, ra)
    r = float(np.median(ra)) % 360.0
    return r, float(np.median(dec))


def attach_coordinates(df, lc_index, verbose=True):
    """
    Add `ra`, `dec` and `lc_path` to a feature table.

    Rows that already have finite coordinates keep them. TESS rows get NaN, which is
    correct: a Galactic flare star has no host galaxy to measure an offset from.
    """
    out = df.copy()
    for c in ('ra', 'dec'):
        if c not in out.columns:
            out[c] = np.nan
    out['lc_path'] = out['id'].astype(str).map(lc_index)

    need = out['ra'].isna() | out['dec'].isna()
    if 'survey' in out.columns:
        need &= out['survey'].eq('ZTF')
    for i in out.index[need]:
        p = out.at[i, 'lc_path']
        if isinstance(p, str):
            out.at[i, 'ra'], out.at[i, 'dec'] = position_from_csv(p)

    if verbose:
        ztf = out['survey'].eq('ZTF') if 'survey' in out.columns else np.ones(len(out), bool)
        n_pos = int((ztf & out['ra'].notna()).sum())
        n_lc = int((ztf & out['lc_path'].notna()).sum())
        print(f'light curves found for {n_lc}/{int(ztf.sum())} ZTF objects; '
              f'positions recovered for {n_pos}')
        if n_lc < int(ztf.sum()):
            print('  (an object with no saved CSV was acquired in an earlier session '
                  'under a different BASE_DIR; its host offset will be NaN)')
    return out
