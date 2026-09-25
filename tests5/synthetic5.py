"""Synthetic fixtures mirroring the v5 schema — no network, no real API."""

import numpy as np
import pandas as pd

from btp5.config import FEATURE_COLS, SN_SUBTYPES

COARSE_OF = {'SN_Ia': 'SNe', 'SN_Ib': 'SNe', 'SN_Ic': 'SNe', 'SN_II': 'SNe',
             'SLSN': 'SNe', 'AGN': 'AGN', 'TDE': 'TDE', 'stellar_flare': 'stellar_flare'}

# label -> (log_A, log_tau_rise, log_tau_fall, peak_abs_mag) means
_PROFILE = {
    'SN_Ia':         (1.0, 0.75, 1.30, -19.3),
    'SN_Ib':         (0.9, 0.80, 1.35, -17.6),
    'SN_Ic':         (0.9, 0.72, 1.28, -17.8),
    'SN_II':         (0.8, 0.85, 1.60, -17.0),
    'SLSN':          (1.6, 1.20, 1.90, -21.5),
    'AGN':           (0.4, 1.50, 1.90, -20.0),
    'TDE':           (1.1, 1.10, 1.75, -19.8),
    'stellar_flare': (1.8, -0.60, -0.20, np.nan),
}


def make_feature_df(counts=None, seed=42, with_alerce=True, alerce_agreement=0.75):
    """A v5-schema feature table with both label sources attached."""
    rng = np.random.default_rng(seed)
    if counts is None:
        counts = {s: 30 for s in SN_SUBTYPES}
        counts.update({'AGN': 60, 'TDE': 40, 'stellar_flare': 60})

    rows = []
    for label, n in counts.items():
        la, ltr, ltf, pam = _PROFILE[label]
        survey = 'TESS' if label == 'stellar_flare' else 'ZTF'
        for i in range(n):
            two_band = survey == 'ZTF' and rng.random() < 0.9
            row = {'id': f'{label}_{i:04d}', 'label': label,
                   'coarse_label': COARSE_OF[label], 'survey': survey,
                   'ref_band': 'r' if survey == 'ZTF' else 'TESS',
                   'log_A': float(rng.normal(la, 0.35)),
                   'beta': float(rng.normal(0.01, 0.02)),
                   'log_gamma': float(rng.normal(1.3, 0.4)),
                   'log_tau_rise': float(rng.normal(ltr, 0.30)),
                   'log_tau_fall': float(rng.normal(ltf, 0.30)),
                   'log_chisq': float(rng.normal(0.3, 0.3)),
                   'has_color': int(two_band),
                   'duration_half': float(abs(rng.normal(40, 18)) + 1),
                   'rise_half': float(abs(rng.normal(14, 7)) + 0.5),
                   'fade_half': float(abs(rng.normal(30, 14)) + 0.5),
                   'n_det': int(rng.integers(15, 120)),
                   'max_snr': float(abs(rng.normal(25, 10)) + 8),
                   'obs_span': float(abs(rng.normal(150, 70)) + 60),
                   'redshift': (0.0 if survey == 'TESS'
                                else float(abs(rng.normal(0.08, 0.05)) + 0.005)),
                   'peak_abs_mag': (np.nan if not np.isfinite(pam)
                                    else float(rng.normal(pam, 0.8))),
                   'ra': float(rng.uniform(0, 360)) if survey == 'ZTF' else np.nan,
                   'dec': float(rng.uniform(-30, 80)) if survey == 'ZTF' else np.nan,
                   'is_synthetic': False}
            for c in ['gr_log_A', 'gr_dt0', 'gr_log_tau_rise', 'gr_log_tau_fall',
                      'gr_dbeta', 'gr_log_gamma']:
                row[c] = float(rng.normal(0, 0.3)) if two_band else np.nan
            # v4 builds its SN pools from BTS first, then tops up from TNS, so a real
            # feature table carries catalogue measurements for a large fraction of
            # ZTF objects and none for TESS. Mirror that, or the Phase 6 ablation
            # runs against an all-NaN feature group and never gets exercised.
            from_bts = survey == 'ZTF' and rng.random() < 0.55
            if from_bts:
                row['bts_peakabs'] = float(rng.normal(pam if np.isfinite(pam) else -19, 0.9))
                row['bts_rise'] = float(abs(rng.normal(15, 6)) + 1)
                row['bts_fade'] = float(abs(rng.normal(35, 15)) + 1)
                row['bts_duration'] = row['bts_rise'] + row['bts_fade']
            else:
                for c in ['bts_peakabs', 'bts_rise', 'bts_fade', 'bts_duration']:
                    row[c] = np.nan
            row['from_bts'] = int(from_bts)
            row['host_offset_arcsec'] = (
                np.nan if survey == 'TESS'
                else float(abs(rng.normal(0.2 if label == 'TDE' else 2.0, 1.0))))
            row['host_offset_norm'] = (np.nan if survey == 'TESS'
                                       else row['host_offset_arcsec'] / 3.0)
            row['has_host_match'] = 0 if survey == 'TESS' else 1
            rows.append(row)

    df = pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    if with_alerce:
        # ALeRCE agrees most of the time; TESS rows NEVER get a label, mirroring the
        # real structural gap (ALeRCE classifies ZTF, not TESS).
        fine_of = {'SN_Ia': 'SN_Ia', 'SN_Ib': 'SN_Ibc', 'SN_Ic': 'SN_Ibc',
                   'SN_II': 'SN_II', 'SLSN': 'SLSN'}
        ac, af, acls, ap = [], [], [], []
        alts = ['AGN', 'SNe', 'TDE']
        for _, r in df.iterrows():
            if r['survey'] == 'TESS':
                ac.append(np.nan); af.append(np.nan); acls.append(np.nan); ap.append(np.nan)
                continue
            agree = rng.random() < alerce_agreement
            coarse = r['coarse_label'] if agree else rng.choice(
                [a for a in alts if a != r['coarse_label']])
            ac.append(coarse)
            af.append(fine_of.get(r['label']) if coarse == 'SNe' else None)
            acls.append({'AGN': 'AGN', 'SNe': 'SNIa', 'TDE': 'TDE'}.get(coarse, 'AGN'))
            ap.append(float(rng.uniform(0.5, 0.99)))
        df['alerce_coarse'] = ac
        df['alerce_fine'] = af
        df['alerce_class'] = acls
        df['alerce_prob'] = ap
    return df


def make_probabilities_frame(oid='ZTFtest', seed=0, multi_version=True):
    """
    A query_probabilities-shaped frame. When multi_version=True it stacks TWO versions
    of the same classifier — the real behaviour that makes name-only grouping wrong.
    """
    rng = np.random.default_rng(seed)
    classes = ['SNIa', 'SESN', 'SNII', 'SNIIn', 'SLSN', 'TDE', 'AGN', 'QSO', 'CV/Nova']
    rows = []
    versions = ['2.1.0', '1.0.7'] if multi_version else ['2.1.0']
    for vi, ver in enumerate(versions):
        p = rng.dirichlet(np.ones(len(classes)) * (0.5 if vi == 0 else 2.0))
        if vi == 0:
            p = np.zeros(len(classes)); p[0] = 0.72
            p[1:] = 0.28 / (len(classes) - 1)
        for c, pr in zip(classes, p):
            rows.append({'classifier_name': 'lc_classifier_BHRF_forced_phot',
                         'classifier_version': ver, 'class_name': c,
                         'probability': float(pr), 'ranking': 0})
    return pd.DataFrame(rows)
