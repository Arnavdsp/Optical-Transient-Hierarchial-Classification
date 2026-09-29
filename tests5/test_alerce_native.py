"""ALeRCE-native acquisition: the pool that makes the three pools genuinely distinct."""
import numpy as np
import pandas as pd
import pytest

from btp5 import alerce_native as AN
from btp5.config import QG, SN_SUBTYPES

LIVE = ['AGN', 'Blazar', 'CEP', 'CV/Nova', 'DSCT', 'EA', 'EB/EW', 'LPV', 'Microlensing',
        'Periodic-Other', 'QSO', 'RRLab', 'RRLc', 'RSCVn', 'SESN', 'SLSN', 'SNII',
        'SNIIn', 'SNIa', 'TDE', 'YSO']


def v4_sn_subtype(t):
    """Stand-in with v4's behaviour for the cases exercised here."""
    t = str(t)
    if 'SLSN' in t:
        return 'SLSN'
    if not t.startswith('SN '):
        return None
    r = t[3:]
    if r.startswith(('Ib/c', 'Ibc')):
        return None
    for pre, lab in (('Ia', 'SN_Ia'), ('Ib', 'SN_Ib'), ('Ic', 'SN_Ic'), ('II', 'SN_II')):
        if r.startswith(pre):
            return lab
    return None


class FakeAlerce:
    """Mimics the live API: ascending probability, deterministic pages, count=True."""
    def __init__(self, n=4000):
        self.n = n
        self.calls = 0

    def _all(self, class_name, probability, classifier):
        p = np.linspace(probability, 0.95, self.n)
        return pd.DataFrame({'oid': [f'ZTF{class_name.replace("/", "")}{i:05d}'
                                     for i in range(self.n)],
                             'meanra': np.linspace(0, 359, self.n),
                             'meandec': np.linspace(-20, 70, self.n),
                             'probability': p, 'class': class_name,
                             'classifier': classifier})

    def query_objects(self, classifier=None, class_name=None, probability=0.5,
                      page_size=100, page=1, count=False, format='pandas', **kw):
        self.calls += 1
        full = self._all(class_name, probability, classifier)
        if count:
            return {'total': len(full), 'items': []}
        lo = (page - 1) * page_size
        return full.iloc[lo:lo + page_size].reset_index(drop=True)


def test_groups_follow_the_live_crosswalk():
    g = AN.native_query_groups(LIVE)
    assert g['SN_Ia'] == ['SNIa']
    assert g['SN_Ibc'] == ['SESN']
    assert sorted(g['SN_II']) == ['SNII', 'SNIIn']
    assert g['SLSN'] == ['SLSN'] and g['TDE'] == ['TDE']
    assert sorted(g['AGN']) == ['AGN', 'Blazar', 'QSO']
    assert sorted(g['stellar_flare']) == ['CV/Nova', 'YSO']
    for periodic in ('LPV', 'RRLab', 'Microlensing'):
        assert not any(periodic in v for v in g.values())


def test_pool_is_a_uniform_sample_not_the_least_confident_first():
    rng = np.random.default_rng(0)
    fake = FakeAlerce(n=4000)
    pools, summary = AN.query_native_candidates(fake, 'bhrf', ['SNIa'], 30, 0.5,
                                                rng, oversample=4, page_size=500,
                                                verbose=False)
    pool = pools['SN_Ia']
    assert len(pool) == 120
    # A uniform sample of 0.5..0.95 has median ~0.725. One truncated page (the first
    # 500 of 4000, ascending) would have median ~0.53 — the bias being guarded.
    assert pool['probability'].median() == pytest.approx(0.725, abs=0.06)
    assert pool['probability'].max() > 0.85, 'high-confidence objects never reached'
    assert (pool['probability'] >= 0.5).all()


def test_multi_class_groups_are_allocated_in_proportion_to_totals():
    class Skewed(FakeAlerce):
        def _all(self, class_name, probability, classifier):
            n = {'AGN': 3000, 'QSO': 900, 'Blazar': 100}[class_name]
            return pd.DataFrame({'oid': [f'Z{class_name}{i}' for i in range(n)],
                                 'meanra': 0.0, 'meandec': 0.0,
                                 'probability': np.linspace(probability, 0.9, n)})
    pools, _ = AN.query_native_candidates(Skewed(), 'bhrf', ['AGN', 'QSO', 'Blazar'],
                                          50, 0.5, np.random.default_rng(0), oversample=8,
                                          verbose=False)
    share = pools['AGN']['alerce_query_class'].value_counts(normalize=True)
    assert share['AGN'] == pytest.approx(0.75, abs=0.02)
    assert share['Blazar'] == pytest.approx(0.025, abs=0.01)


def test_already_acquired_objects_are_excluded():
    rng = np.random.default_rng(1)
    excl = [f'ZTFSNIa{i:05d}' for i in range(0, 4000, 2)]
    pools, _ = AN.query_native_candidates(FakeAlerce(), 'bhrf', ['SNIa'], 50, 0.5, rng,
                                          exclude_oids=excl, verbose=False)
    assert not set(pools['SN_Ia']['oid']) & set(excl)


def test_crossmatch_labels_with_v4_rules_and_keeps_non_tns_objects_unlabelled():
    cands = pd.DataFrame({'oid': ['ZTFa', 'ZTFb', 'ZTFc', 'ZTFd', 'ZTFe'],
                          'native_group': ['SN_Ia'] * 5})
    tns_full = pd.DataFrame({
        'ztf_name': ['ZTFa', 'ZTFb', 'ZTFc', 'ZTFd'],
        'name': ['2020a', '2020b', '2020c', '2020d'],
        'type': ['SN Ia-91bg', 'QSO', 'CV', 'SN Ib/c'],
        'redshift': [0.05, 0.9, np.nan, 0.02]})
    out = AN.crossmatch_tns(cands, tns_full, v4_sn_subtype).set_index('oid')
    assert out.loc['ZTFa', 'tns_label'] == 'SN_Ia' and out.loc['ZTFa', 'has_tns_label']
    assert out.loc['ZTFb', 'tns_label'] == 'AGN'
    # Spectroscopically typed, but not in our scheme -> recorded, not counted.
    assert out.loc['ZTFc', 'tns_type'] == 'CV' and not out.loc['ZTFc', 'has_tns_label']
    # v4 deliberately excludes the ambiguous 'SN Ib/c'
    assert not out.loc['ZTFd', 'has_tns_label']
    # Not in TNS at all -> the photometric-only subset
    assert not out.loc['ZTFe', 'has_tns_label'] and pd.isna(out.loc['ZTFe', 'tns_type'])


def test_acquisition_pool_matches_what_v4_resolve_oid_needs():
    cands = pd.DataFrame({'oid': ['ZTF1'], 'meanra': [10.0], 'meandec': [5.0],
                          'tns_redshift': [np.nan], 'tns_type': [None]})
    pool = AN.to_acquisition_pool(cands)
    row = pool.iloc[0]
    assert row['ztf_name'].startswith('ZTF'), 'v4 resolve_oid short-circuits on ztf_name'
    assert row['source'] == AN.SOURCE_NATIVE
    assert {'name', 'ra', 'declination', 'redshift'} <= set(pool.columns)


def test_redshift_relaxation_is_scoped_and_always_restored():
    assert QG['require_redshift'] is True
    with AN.redshift_optional(QG):
        assert QG['require_redshift'] is False
        assert QG['min_obs_span'] == 60.0, 'every other threshold must stand'
    assert QG['require_redshift'] is True
    with pytest.raises(RuntimeError):
        with AN.redshift_optional(QG):
            raise RuntimeError('acquisition blew up')
    assert QG['require_redshift'] is True, 'must be restored even on an exception'


def test_finalized_rows_never_carry_alerce_opinion_as_the_tns_label():
    cands = pd.DataFrame({'oid': ['ZTFa', 'ZTFe'], 'tns_label': ['SN_Ia', None],
                          'native_group': ['SN_Ia', 'SN_Ia']})
    feats = [{'id': 'ZTFa', 'label': 'ALERCE_SN_Ia', 'redshift': 0.05},
             {'id': 'ZTFe', 'label': 'ALERCE_SN_Ia', 'redshift': np.nan}]
    df = AN.finalize_native_rows(feats, cands, SN_SUBTYPES).set_index('id')
    assert df.loc['ZTFa', 'label'] == 'SN_Ia' and df.loc['ZTFa', 'coarse_label'] == 'SNe'
    assert pd.isna(df.loc['ZTFe', 'label']) and not df.loc['ZTFe', 'has_tns_label']
    assert df.loc['ZTFa', 'passes_v4_gate'] and not df.loc['ZTFe', 'passes_v4_gate']
    assert (df['source'] == AN.SOURCE_NATIVE).all()
