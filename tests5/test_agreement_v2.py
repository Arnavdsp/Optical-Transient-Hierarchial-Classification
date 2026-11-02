"""Both-stage agreement from out-of-fold predictions, the subset split, and the
Section 3.1 TDE cross-check."""
import numpy as np
import pandas as pd
import pytest

from btp5 import agreement as AG
from btp5.alerce_labels import describe_probabilities, collapse_tns_fine
from btp5.config import FEATURE_COLS
from btp5.modeling import oof_predictions, predict_unseen
import synthetic5


def test_describe_probabilities_keeps_top1_even_below_the_cut():
    p = synthetic5.make_probabilities_frame(multi_version=False)
    kw = dict(classifier_name='lc_classifier_BHRF_forced_phot', classifier_version='2.1.0')
    hi = describe_probabilities(p, threshold=0.5, **kw)
    lo = describe_probabilities(p, threshold=0.95, **kw)
    assert hi['alerce_coarse'] == 'SNe' and hi['alerce_top1_class'] == 'SNIa'
    assert lo['alerce_coarse'] is None, 'below the cut: not in the ALeRCE-confident pool'
    assert lo['alerce_top1_class'] == 'SNIa', 'but its top-1 is still recorded'


def test_oof_predicts_every_real_row_exactly_once_and_never_a_synthetic_one():
    df = synthetic5.make_feature_df()
    kids = df.sample(40, random_state=0).copy()
    kids['parent_id'], kids['id'] = kids['id'], kids['id'] + '__aug0'
    kids['is_synthetic'] = True
    df['parent_id'] = np.nan
    full = pd.concat([df, kids], ignore_index=True)

    oof = oof_predictions(full, FEATURE_COLS, 'coarse_label', 'Random Forest', verbose=False)
    real_ids = set(df['id'])
    assert set(oof['id']) == real_ids
    assert oof['id'].is_unique
    assert not oof['id'].str.contains('__aug').any()
    assert (oof['pred'] == oof['true']).mean() > 0.4


def test_oof_never_trains_on_a_copy_of_the_object_being_predicted(monkeypatch):
    """Instrument the fit: whenever object X is in a fold's test set, no row with
    parent_id == X may be in that fold's training set."""
    import btp5.modeling as M
    df = synthetic5.make_feature_df(counts={'SN_Ia': 20, 'SN_II': 20, 'SLSN': 20})
    kids = df.copy()
    kids['parent_id'], kids['id'] = kids['id'], kids['id'] + '__aug0'
    kids['is_synthetic'] = True
    # make each copy trivially identifiable from its features
    df['parent_id'] = np.nan
    full = pd.concat([df, kids], ignore_index=True)
    seen = []
    real_impute = M.impute

    def spy(Xtr, Xte):
        seen.append((Xtr.copy(), Xte.copy()))
        return real_impute(Xtr, Xte)
    monkeypatch.setattr(M, 'impute', spy)
    oof_predictions(full, FEATURE_COLS, 'label', 'Random Forest', verbose=False)
    # Copies here are exact duplicates of their parents, so any leak shows up as a
    # test row appearing verbatim in the training matrix.
    for Xtr, Xte in seen:
        tr = {tuple(np.round(np.nan_to_num(r), 9)) for r in Xtr}
        for r in Xte:
            assert tuple(np.round(np.nan_to_num(r), 9)) not in tr, \
                'a test object was trained on via its synthetic copy'


def _agreement_frame(seed=0, n=240, alerce_acc=0.8, ours_acc=0.75, phot_frac=0.4):
    rng = np.random.default_rng(seed)
    cls = ['AGN', 'SNe', 'TDE']
    truth = rng.choice(cls, n)
    def noisy(acc):
        return np.where(rng.random(n) < acc, truth, rng.choice(cls, n))
    has_tns = rng.random(n) > phot_frac
    return pd.DataFrame({'truth': np.where(has_tns, truth, None),
                         'ours': noisy(ours_acc), 'alerce': noisy(alerce_acc),
                         'has_tns_label': has_tns}), cls


def test_agreement_report_has_both_subsets_and_one_baseline():
    f, cls = _agreement_frame()
    rep = AG.agreement_report(f, 'ours', 'alerce', 'truth', cls, 'RF', 'Stage 1')
    assert set(rep['subsets']) == {AG.SUBSET_SPEC, AG.SUBSET_PHOT}
    for s in rep['subsets'].values():
        assert 0 <= s['actual_rate'] <= 1 and s['n'] > 0 and 'text' in s
    assert rep['expected_matrix'].shape == (3, 3)
    # the photometric subset has no truth, so the baseline must come from TNS objects
    assert rep['purity_ours'].values.sum() > 0


def test_subset_contrast_reports_direction_either_way():
    f, cls = _agreement_frame()
    # make the photometric subset systematically disagree
    phot = ~f['has_tns_label']
    f.loc[phot, 'alerce'] = f.loc[phot, 'ours'].map({'AGN': 'SNe', 'SNe': 'TDE', 'TDE': 'AGN'})
    rep = AG.agreement_report(f, 'ours', 'alerce', 'truth', cls, 'RF', 'Stage 1')
    msg = AG.contrast_subsets(rep)
    assert 'Lower on the photometric-only subset' in msg and '82% vs 72%' in msg


def test_report_degrades_gracefully_with_no_truth():
    f, cls = _agreement_frame()
    f['truth'] = None
    rep = AG.agreement_report(f, 'ours', 'alerce', 'truth', cls, 'RF', 'Stage 2')
    assert 'note' in rep and not rep['subsets']


def test_stage2_comparison_collapses_ib_ic_only_for_the_comparison():
    ours = pd.Series(['SN_Ib', 'SN_Ic', 'SN_Ia', 'SN_II', 'SLSN'])
    assert list(ours.map(collapse_tns_fine)) == ['SN_Ibc', 'SN_Ibc', 'SN_Ia', 'SN_II', 'SLSN']


def test_tde_crosscheck_is_the_table3_analogue():
    f = pd.DataFrame({'label': ['TDE'] * 10 + ['SLSN'] * 4 + ['SN_Ia'] * 5,
                      'alerce_top1_class': ['TDE'] * 4 + ['SLSN'] * 5 + ['SNIa']
                                           + ['SLSN'] * 4 + ['SNIa'] * 5})
    tab, text = AG.tde_crosscheck(f)
    assert tab.loc['TDE', 'n'] == 10
    assert tab.loc['TDE', 'TDE'] == pytest.approx(0.4)
    assert tab.loc['TDE', 'SLSN'] == pytest.approx(0.5)
    assert '40%' in text and 'SLSN' in text and '54.9%' in text
    none, msg = AG.tde_crosscheck(f.iloc[:0])
    assert none is None and 'not meaningful' in msg


def test_predict_unseen_covers_objects_without_a_label():
    df = synthetic5.make_feature_df()
    target = df.sample(30, random_state=3).copy()
    out = predict_unseen(df, FEATURE_COLS, 'coarse_label', 'Logistic Regression', target)
    assert len(out) == 30 and set(out['pred']) <= set(df['coarse_label'])
    assert out['confidence'].between(0, 1).all()
