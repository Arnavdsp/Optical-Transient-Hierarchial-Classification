"""
v6 acquisition: the v4 retry-until-target idea, run in parallel and to much larger
targets, writing into the same folders as the v5 loop.

* Same light-curve source (ALeRCE detections), same CSV layout
  (`<base_dir>/<subtype>/<oid>.csv`, columns mjd, fid, magpsf, sigmapsf, ra, dec), so
  every light curve the v5 run already saved on Drive is re-used, not re-downloaded.
* Two quality gates, both evaluated on the same v4 Villar fits: the v4 gate `QG`
  (recorded as `passes_v4_gate`) and a looser `QG_V6` that decides survival here.
  Whether the extra, shorter light curves help or hurt is then measured, not assumed:
  the modelling section trains on both populations and reports each.
* Own manifest, `<subtype>_manifest_v6.csv`, one line per attempted candidate with its
  outcome. Re-running resumes; raising the target continues from where it stopped.
* Threads, because the cost is network latency to ALeRCE, not CPU.
"""

import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

try:
    from btp5.photometry import quality_gate
    from btp5.config import QG
    from btp6.features import v6_features, fit_bands, gp_features
except ImportError:                     # pragma: no cover - notebook path
    pass

DET_COLS = ['mjd', 'fid', 'magpsf', 'sigmapsf', 'ra', 'dec']

# v6 survival gate. Same checks as v4's QG, with the coverage requirements relaxed:
# the v6 features are windowed (a decline at +60 d is only reported if the data reach
# +60 d) and NaN-tolerant, so a 35-day light curve no longer produces a fake number,
# it produces a missing one. Peak SNR, fit quality, redshift and two bands stay strict.
QG_V6 = dict(min_det_total=10, min_det_best_band=6, min_peak_snr=8.0, max_chisq_dof=20.0,
             require_redshift=True, require_bracket=True, min_days_prepeak=0.0,
             min_days_postpeak=15.0, min_obs_span=30.0, require_two_bands=True)
MIN_DOWNLOAD_DETECTIONS = 10


def _ebv(row):
    e = row.get('ebv', np.nan)
    try:
        e = float(e)
    except (TypeError, ValueError):
        e = np.nan
    if np.isfinite(e):
        return e
    av = row.get('bts_av', np.nan)
    try:
        av = float(av)
    except (TypeError, ValueError):
        av = np.nan
    return av / 3.1 if np.isfinite(av) else np.nan


def _one(row, label, out_dir, fetch, min_detections):
    """Download (unless cached), gate and featurise one candidate. Never raises."""
    oid = row.get('ztf_name')
    rec = {'ztf_name': oid, 'label': label, 'source': row.get('source'),
           'redshift': row.get('redshift'), 'ebv': _ebv(row), 'status': None}
    if not (isinstance(oid, str) and oid.startswith('ZTF')):
        rec['status'] = 'no ZTF id'
        return rec, None
    path = os.path.join(out_dir, f'{oid}.csv')
    if not os.path.exists(path):
        try:
            det = fetch(oid)
        except Exception:
            rec['status'] = 'ALeRCE error'
            return rec, None
        if det is None or len(det) < min_detections:
            rec['status'] = 'too few detections'
            return rec, None
        det[[c for c in DET_COLS if c in det.columns]].to_csv(path, index=False)
    try:
        z = float(row.get('redshift'))
    except (TypeError, ValueError):
        z = np.nan
    try:
        det = pd.read_csv(path)
        if 'sigmapsf' not in det:
            det['sigmapsf'] = 0.1
        bands = fit_bands(det)
        ok6, why6 = quality_gate(bands, z, qg=QG_V6)
        ok4, why4 = quality_gate(bands, z, qg=QG)
    except Exception:
        rec['status'] = 'fit error'
        return rec, None
    if not ok6:
        rec['status'] = 'gate: ' + re.sub(r'-?[\d.]+', 'N', why6)
        rec['gate_reason'] = why6
        return rec, None
    try:
        feat = {'id': oid, 'label': label, 'survey': 'ZTF'}
        feat.update(v6_features(det, z, rec['ebv'], bands=bands))
        feat.update(gp_features(det, z, rec['ebv']))
    except Exception:
        rec['status'] = 'v6 features failed'
        return rec, None
    feat['passes_v4_gate'] = bool(ok4)
    feat['v4_gate_reason'] = why4
    for c in ['bts_peakabs', 'bts_rise', 'bts_fade', 'bts_duration']:
        feat[c] = row.get(c, np.nan)
    feat['source'] = row.get('source')
    feat['type'] = row.get('type')          # the catalogue's own type string, e.g. 'SN IIb'
    feat['ebv'] = rec['ebv']
    rec['status'] = 'ok'
    return rec, feat


def acquire_to_target(label, pool, target, *, base_dir, fetch, out_subdir=None,
                      workers=8, min_detections=None, batch=None, verbose=True,
                      progress=None):
    """Run candidates from `pool` until `target` survive the v4 gate, or the pool ends.

    Returns (feature_frame, manifest_frame). The feature frame holds only survivors.
    """
    min_detections = min_detections or MIN_DOWNLOAD_DETECTIONS
    out_subdir = out_subdir or label.lower()
    out_dir = os.path.join(base_dir, out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    man_path = os.path.join(base_dir, f'{out_subdir}_manifest_v6.csv')
    feat_path = os.path.join(base_dir, f'{out_subdir}_features_v6.csv')

    manifest = pd.read_csv(man_path) if os.path.exists(man_path) else pd.DataFrame()
    feats = pd.read_csv(feat_path) if os.path.exists(feat_path) else pd.DataFrame()
    seen = set(manifest['ztf_name']) if len(manifest) else set()
    n_ok = len(feats)
    todo = [r for r in pool.to_dict('records') if r.get('ztf_name') not in seen]
    batch = batch or workers * 4
    lock = threading.Lock()
    new_man, new_feat = [], []
    bar = progress(total=target, initial=min(n_ok, target), desc=label) if progress else None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        i = 0
        while n_ok < target and i < len(todo):
            chunk = todo[i:i + batch]
            i += batch
            for rec, feat in ex.map(lambda r: _one(r, label, out_dir, fetch, min_detections), chunk):
                with lock:
                    new_man.append(rec)
                    if feat is not None:
                        new_feat.append(feat)
                        n_ok += 1
                        if bar is not None:
                            bar.update(1)
            manifest = pd.concat([manifest, pd.DataFrame(new_man)], ignore_index=True)
            feats = pd.concat([feats, pd.DataFrame(new_feat)], ignore_index=True)
            new_man, new_feat = [], []
            manifest.to_csv(man_path, index=False)
            feats.to_csv(feat_path, index=False)
    if bar is not None:
        bar.close()

    if verbose:
        reasons = manifest['status'].value_counts().to_dict() if len(manifest) else {}
        flag = 'reached' if n_ok >= target else 'SHORTFALL (pool exhausted)'
        print(f'  {label:<7s} {min(n_ok, target)}/{target} {flag}; '
              f'attempted {len(manifest)} of a pool of {len(pool)}; outcomes {reasons}')
    if len(feats):
        feats = feats.drop_duplicates('id').head(target).reset_index(drop=True)
    return feats, manifest
