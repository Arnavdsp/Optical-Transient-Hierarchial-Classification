"""
ALeRCE light-curve features (Sanchez-Saez et al. 2021, the lc_classifier feature set)
as an optional extra block.

What they add that the v6 features do not: host and environment information. The set
includes the AllWISE colours of the counterpart (g-W2, r-W3, W1-W2, ...), the
star/galaxy score of the nearest PS1 source (sgscore1), the distance to the nearest
reference-image source, and Galactic latitude. Host environment is a known SN-type
discriminant (SLSNe-I in faint dwarf hosts, SNe Ia in all galaxy types, core-collapse
only in star-forming ones; e.g. Gagliano et al. 2021, Kisley et al. 2023).

One request per object. Features come back as (name, fid, value) rows, and objects
processed by several pipeline versions carry several copies, so one version is chosen
per object (the most recent) before pivoting to one wide row.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd


def _one(alerce, oid, survey_kwargs):
    try:
        f = alerce.query_features(oid, format='pandas', **survey_kwargs)
    except Exception:
        return oid, None
    if f is None or not len(f):
        return oid, None
    f = f.copy()
    if 'version' in f:
        f = f[f['version'] == sorted(f['version'].astype(str).unique())[-1]]
    f['key'] = 'al_' + f['name'].astype(str) + '_' + f['fid'].astype(str)
    s = pd.to_numeric(f.set_index('key')['value'], errors='coerce')
    return oid, s[~s.index.duplicated()]


def fetch_alerce_features(alerce, oids, cache_path=None, workers=8, survey_kwargs=None):
    """Wide frame indexed by oid. Cached to CSV; re-runs only fetch missing oids."""
    survey_kwargs = survey_kwargs or {}
    have = pd.DataFrame()
    if cache_path:
        try:
            have = pd.read_csv(cache_path, index_col=0)
        except (OSError, ValueError):
            have = pd.DataFrame()
    todo = [o for o in pd.unique(pd.Series(oids)) if o not in set(have.index)]
    got = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for oid, s in ex.map(lambda o: _one(alerce, o, survey_kwargs), todo):
            if s is not None:
                got[oid] = s
    new = pd.DataFrame(got).T
    out = pd.concat([have, new]) if len(have) else new
    if cache_path and len(out):
        out.to_csv(cache_path)
    return out


# Columns that are about the host / environment, or that are colour-like
# summaries ALeRCE computes over the full light curve. Purely identifying or
# survey-bookkeeping quantities are excluded.
EXCLUDE_PREFIXES = ('al_gal_', 'al_rb_', 'al_iqr_', 'al_n_det', 'al_n_pos', 'al_n_neg',
                    'al_last_', 'al_first_', 'al_delta_mjd', 'al_MHPS_non_zero',
                    'al_MHPS_PN_flag')


def usable_columns(frame, min_coverage=0.7):
    keep = []
    for c in frame.columns:
        if c.startswith(EXCLUDE_PREFIXES):
            continue
        if frame[c].notna().mean() >= min_coverage:
            keep.append(c)
    return keep
