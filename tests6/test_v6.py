import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import pytest

from btp5.photometry import villar, ZP, distance_modulus
from btp6.osc import osc_frame, osc_subtype, union_pool
from btp6.features import v6_features, V6_FEATURES, R_BAND
from btp6.model import (add_sesn_label, Hierarchical4, SoftVote, cap_majority, coverage_curve,
                        cross_validate, scheme_frame, make_models)


def classify(t):
    t = str(t)
    if 'SLSN' in t: return 'SLSN'
    if not t.startswith('SN '): return None
    r = t[3:]
    if r.startswith('Ib/c') or r.startswith('Ibc'): return None
    for p, s in [('Ia', 'SN_Ia'), ('Ib', 'SN_Ib'), ('Ic', 'SN_Ic'), ('II', 'SN_II')]:
        if r.startswith(p): return s
    return None


def test_osc_subtype_uses_tns_rule():
    assert osc_subtype('Ic-BL', classify) == 'SN_Ic'
    assert osc_subtype('Ib/c', classify) is None
    assert osc_subtype('SLSN-I', classify) == 'SLSN'
    assert osc_subtype('IIb', classify) == 'SN_II'
    assert osc_subtype('Ia-91bg', classify) == 'SN_Ia'
    assert osc_subtype('', classify) is None


def _ev(name, ztf, types, z='0.05'):
    return {'name': name, 'alias': [{'value': name}] + ([{'value': ztf}] if ztf else []),
            'claimedtype': [{'value': t} for t in types], 'redshift': [{'value': z}]}


def test_osc_frame_drops_conflicts_and_missing_ztf():
    cat = [_ev('A', 'ZTF1', ['Ic']), _ev('B', 'ZTF2', ['Ic', 'Ib']),
           _ev('C', None, ['Ic']), _ev('D', 'ZTF4', ['Ic', 'Ic-BL']), _ev('E', 'ZTF5', ['Ib/c'])]
    f = osc_frame(cat, classify)
    assert list(f['ztf_name']) == ['ZTF1', 'ZTF4']
    assert (f['sn_subtype'] == 'SN_Ic').all()
    assert np.isclose(f['redshift'].iloc[0], 0.05)


def test_union_pool_removes_cross_source_conflicts_and_fills_gaps():
    tns = pd.DataFrame({'ztf_name': ['Z1', 'Z2', 'Z3'], 'sn_subtype': ['SN_Ib', 'SN_Ib', 'SN_Ib'],
                        'redshift': [0.02, np.nan, 0.03]})
    osc = pd.DataFrame({'ztf_name': ['Z2', 'Z3', 'Z4'], 'sn_subtype': ['SN_Ib', 'SN_Ic', 'SN_Ib'],
                        'redshift': [0.04, 0.03, 0.05], 'ebv': [0.1, 0.2, 0.3]})
    pool, st = union_pool('SN_Ib', [('TNS', tns), ('OSC', osc)])
    assert set(pool['ztf_name']) == {'Z1', 'Z2', 'Z4'}          # Z3 is conflicted
    z2 = pool.set_index('ztf_name').loc['Z2']
    assert np.isclose(z2['redshift'], 0.04) and np.isclose(z2['ebv'], 0.1)
    assert z2['source'] == 'TNS+OSC'
    assert st['conflicted_ids'] == 1


def _synthetic_lc(z, t_peak_rest_offset=0.0, color=0.3, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    zz = 1 + z
    p = dict(A=10 ** (-0.4 * (18.5 - ZP)), beta=0.005 / zz, gamma=10 * zz, t0=59000.0,
             tau_rise=3.0 * zz, tau_fall=25.0 * zz)
    for fid, off in [(1, color), (2, 0.0)]:
        t = np.sort(rng.uniform(58980, 59120, 200))
        f = villar(t, **p) * 10 ** (-0.4 * off)
        m = ZP - 2.5 * np.log10(np.clip(f, 1e-9, None))
        k = m < 21
        rows.append(pd.DataFrame({'mjd': t[k], 'fid': fid, 'magpsf': m[k] + rng.normal(0, 0.02, k.sum()),
                                  'sigmapsf': 0.02}))
    return pd.concat(rows, ignore_index=True), p


def test_v6_features_are_rest_frame_and_recover_colour():
    d1, _ = _synthetic_lc(0.02)
    d2, _ = _synthetic_lc(0.30)
    f1 = v6_features(d1, 0.02)
    f2 = v6_features(d2, 0.30)
    assert set(V6_FEATURES) <= set(f1)
    # the same rest-frame light curve at two redshifts gives the same rest timescale
    assert abs(f1['fall_half_r'] - f2['fall_half_r']) / f1['fall_half_r'] < 0.15
    # colour at peak recovers the injected 0.3 mag
    assert abs(f1['gr_p0'] - 0.3) < 0.08
    # absolute magnitude: m_peak - DM + 2.5log(1+z), within the fit noise
    assert np.isfinite(f1['M_peak_r'])
    assert abs((f1['M_peak_r'] + distance_modulus(0.02) - 2.5 * np.log10(1.02)) - 18.5) < 0.3


def test_v6_extinction_correction_is_applied():
    d, _ = _synthetic_lc(0.05)
    a = v6_features(d, 0.05, ebv=0.0)
    b = v6_features(d, 0.05, ebv=0.1)
    assert np.isclose(a['M_peak_r'] - b['M_peak_r'], R_BAND['r'] * 0.1, atol=1e-6)
    assert np.isclose(a['gr_p0'] - b['gr_p0'], (R_BAND['g'] - R_BAND['r']) * 0.1, atol=1e-6)


def test_v6_features_do_not_extrapolate_past_data():
    d, _ = _synthetic_lc(0.05)
    d = d[d['mjd'] < 59030]                       # ~25 d after peak only
    f = v6_features(d, 0.05)
    assert np.isnan(f['dm60_r'])


def _toy(n=60, seed=0):
    rng = np.random.default_rng(seed)
    labs = ['SN_Ia'] * (3 * n) + ['SN_II'] * n + ['SN_Ib'] * (n // 2) + ['SN_Ic'] * (n // 2) + ['SLSN'] * n
    centers = {'SN_Ia': 0, 'SN_II': 3, 'SN_Ib': 6, 'SN_Ic': 6, 'SLSN': 9}
    X = np.array([[centers[l] + rng.normal(), rng.normal()] for l in labs])
    return X, np.array(labs)


def test_cap_majority():
    X, y = _toy()
    Xc, yc = cap_majority(X, y, 2.0, np.random.default_rng(0))
    _, c = np.unique(yc, return_counts=True)
    assert c.max() <= 2 * c.min()


def test_hierarchical_and_cv_on_separable_toy():
    X, y = _toy()
    d = scheme_frame(pd.DataFrame({'label': y}), '4-way')
    yy = d['y'].values
    base = make_models()['HistGB (balanced)']
    r = cross_validate(Hierarchical4(base), X, yy, n_splits=3, n_repeats=1, return_oof=True)
    assert r['bacc_mean'] > 0.85
    cc = coverage_curve(yy, r['oof_prob'], r['classes'])
    assert cc['accuracy'].iloc[-1] >= cc['accuracy'].iloc[0]


def test_softvote_probabilities_sum_to_one():
    X, y = _toy()
    ms = make_models()
    sv = SoftVote([ms['HistGB (balanced)'], ms['Logistic Regression']]).fit(X, y)
    assert np.allclose(sv.predict_proba(X).sum(1), 1)


def test_sesn_label_moves_only_iib():
    d = pd.DataFrame({'label': ['SN_II', 'SN_II', 'SN_II', 'SN_Ib', 'SN_Ia'],
                      'type': ['SN IIb', 'IIb', 'SN IIn', 'SN Ib', 'SN Ia']})
    out = add_sesn_label(d)
    assert list(out['label_sesn']) == ['SN_Ib', 'SN_Ib', 'SN_II', 'SN_Ib', 'SN_Ia']
    s = scheme_frame(out, '4-way SESN')
    assert list(s['y']) == ['SN_Ibc', 'SN_Ibc', 'SN_II', 'SN_Ibc', 'SN_Ia']


def test_gp_features_recover_colour():
    from btp6.features import gp_features
    d, _ = _synthetic_lc(0.05)
    g = gp_features(d, 0.05)
    assert abs(g['gp_gr_p0'] - 0.3) < 0.1
    assert np.isfinite(g['gp_M_peak_r']) and np.isfinite(g['gp_fall_half_r'])


def test_custom_models_work_with_sklearn_inspection():
    from sklearn.inspection import permutation_importance
    X, y = _toy()
    yy = scheme_frame(pd.DataFrame({'label': y}), '4-way')['y'].values
    ms = make_models()
    for m in [Hierarchical4(ms['HistGB (balanced)']),
              SoftVote([ms['HistGB (balanced)'], ms['Logistic Regression']])]:
        m.fit(X, yy)
        r = permutation_importance(m, X, yy, scoring='f1_macro', n_repeats=2, random_state=0)
        assert r.importances_mean[0] > r.importances_mean[1]
