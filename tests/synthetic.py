"""
Synthetic data mirroring the real pipeline's schema.

Same column names and dtypes as the real `feature_df`, physically plausible
orderings (flares are short and spiky, AGN are long and low-amplitude, SNe sit
in between, SLSN are the brightest and slowest), and — deliberately — the
ability to make one class too small, so the adaptive-fold and retry-loop
safeguards can be observed actually triggering rather than assumed.
"""

import numpy as np
import pandas as pd

SN_SUBTYPE_LABELS = ['SN_Ia', 'SN_Ib', 'SN_Ic', 'SN_II', 'SLSN']

# label -> (peak, rise, decay, amplitude, color) means; spreads deliberately wide
# enough that the classes overlap, so accuracies land in a realistic band rather
# than a meaningless 1.00.
_PROFILES = {
    'SN_Ia':         (1.9, 18.0,  40.0, 1.4,  0.05),
    'SN_Ib':         (1.7, 20.0,  45.0, 1.2,  0.15),
    'SN_Ic':         (1.7, 16.0,  38.0, 1.2,  0.10),
    'SN_II':         (1.6, 12.0,  70.0, 1.1,  0.30),
    'SLSN':          (2.6, 35.0,  95.0, 2.1, -0.10),
    'AGN':           (1.3, 60.0, 120.0, 0.5,  0.35),
    'TDE':           (2.0, 30.0, 110.0, 1.6, -0.25),
    'stellar_flare': (3.2,  0.4,   1.2, 2.6,  np.nan),
}


def make_feature_df(counts=None, seed=42, n_per_class=150, n_per_subtype=30):
    """
    Build a synthetic feature table.

    `counts` maps label -> n. Defaults to the balanced target
    (30 per SN subtype, 150 each for AGN/TDE/stellar_flare).
    """
    rng = np.random.default_rng(seed)
    if counts is None:
        counts = {s: n_per_subtype for s in SN_SUBTYPE_LABELS}
        counts.update({'AGN': n_per_class, 'TDE': n_per_class, 'stellar_flare': n_per_class})

    rows = []
    for label, n in counts.items():
        peak, rise, decay, amp, color = _PROFILES[label]
        survey = 'TESS' if label == 'stellar_flare' else 'ZTF'
        for i in range(n):
            rows.append({
                'id': f'{label}_{i:04d}',
                'label': label,
                'survey': survey,
                'peak_val':   float(peak  * rng.lognormal(0, 0.28)),
                'rise_time':  float(abs(rng.normal(rise,  rise  * 0.45)) + 0.05),
                'decay_time': float(abs(rng.normal(decay, decay * 0.45)) + 0.05),
                'amplitude':  float(abs(rng.normal(amp,   amp   * 0.40)) + 0.01),
                # TESS flares have no ZTF g-r colour at all — same as reality.
                'color_g_r': np.nan if label == 'stellar_flare' else float(rng.normal(color, 0.30)),
            })
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


def make_tns_pool(label, n, seed=0, ztf_fraction=0.8):
    """A fake TNS candidate pool with the columns `resolve_oid` reads."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        has_ztf = rng.random() < ztf_fraction
        rows.append({
            'name': f'{label}_TNS_{i:04d}',
            'ra': float(rng.uniform(0, 360)),
            'declination': float(rng.uniform(-30, 80)),
            'redshift': float(rng.uniform(0.01, 0.3)),
            'type': label,
            'internal_names': f'ZTF{i:06d}xyz' if has_ztf else 'ATLAS_something',
        })
    return pd.DataFrame(rows)


def make_detections(n_points=40, seed=0):
    """A fake ALeRCE detections frame with the real column names."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        'mjd': np.sort(rng.uniform(59000, 59100, n_points)),
        'fid': rng.choice([1, 2], n_points),
        'magpsf': rng.normal(19, 0.5, n_points),
        'sigmapsf': rng.uniform(0.02, 0.2, n_points),
        'ra': rng.uniform(0, 360, n_points),
        'dec': rng.uniform(-30, 80, n_points),
    })


def make_tess_frame(n_points=500, seed=0):
    """A fake TESS light-curve frame ('bjd','flux','flux_err')."""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(2458000, 2458027, n_points))
    flux = rng.normal(1.0, 0.01, n_points)
    flux[n_points // 2] += 0.5
    return pd.DataFrame({'bjd': t, 'flux': flux, 'flux_err': np.full(n_points, 0.005)})
