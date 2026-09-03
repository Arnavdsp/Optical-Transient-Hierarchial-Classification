"""
Unified split, the 4 models, and Stage 2 Option A vs Option B — on synthetic data.

The headline assertions here are the structural ones: that Stage 2's rows are
subsets of Stage 1's partitions by construction (so the leakage that inflated an
earlier version of this pipeline cannot recur), and that the adaptive-fold
safeguard actually fires when a class is too thin.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from btp_pipeline.config import MODEL_NAMES, SN_SUBTYPE_LABELS, X_COLS  # noqa: E402
from btp_pipeline.modeling import (  # noqa: E402
    add_coarse_label, assert_split_nesting, build_models, make_global_split,
    option_ab_comparison, run_stage1, run_stage2_option_a, run_stage2_option_b,
    safe_cv_folds)
import synthetic  # noqa: E402


def _prepared(counts=None, seed=42):
    from btp_pipeline.acquisition import assemble_feature_df
    df = synthetic.make_feature_df(counts=counts, seed=seed)
    df = assemble_feature_df([df.to_dict('records')], verbose=False)
    df = add_coarse_label(df, SN_SUBTYPE_LABELS)
    split = make_global_split(df, X_COLS, verbose=False)
    return df, split


# ---------------------------------------------------------------- split

def test_single_global_split_is_stratified_and_covers_everything():
    df, split = _prepared()
    tr, te = split['idx_train'], split['idx_test']
    assert len(tr) + len(te) == len(df)
    assert not set(tr) & set(te), 'train and test overlap'
    assert set(tr) | set(te) == set(range(len(df))), 'split lost rows'
    # Stratification: each coarse class keeps its 25% share within a row or two.
    for cls in df['coarse_label'].unique():
        share_test = (df['coarse_label'].iloc[te] == cls).sum() / (df['coarse_label'] == cls).sum()
        assert abs(share_test - 0.2) < 0.05, f'{cls} test share {share_test:.3f} is not ~0.2'


def test_stage2_rows_are_subsets_of_stage1_partitions_no_leakage():
    df, split = _prepared()
    s2_train, s2_test = assert_split_nesting(split, SN_SUBTYPE_LABELS)

    assert set(s2_train).issubset(set(split['idx_train']))
    assert set(s2_test).issubset(set(split['idx_test']))
    assert not set(s2_train) & set(split['idx_test'])
    assert len(s2_train) + len(s2_test) == int(df['label'].isin(SN_SUBTYPE_LABELS).sum())
    # And every Stage 2 row really is an SN.
    assert df['label'].iloc[s2_train].isin(SN_SUBTYPE_LABELS).all()
    assert df['label'].iloc[s2_test].isin(SN_SUBTYPE_LABELS).all()


def test_leakage_would_be_caught_if_reintroduced():
    """Sanity-check the guard itself: hand it a split that DOES leak and confirm it fails."""
    df, split = _prepared()
    bad = dict(split)
    # Move one SN test row into the training partition — the exact mistake being guarded.
    sn_test = [i for i in split['idx_test'] if df['label'].iloc[i] in SN_SUBTYPE_LABELS]
    bad['idx_train'] = np.append(split['idx_train'], sn_test[0])
    try:
        assert_split_nesting(bad, SN_SUBTYPE_LABELS)
    except AssertionError:
        return
    raise AssertionError('leakage guard did not fire on a deliberately leaky split')


# ---------------------------------------------------------------- adaptive folds

def test_adaptive_folds_trigger_on_a_too_small_class(capsys):
    y_fine = np.array([0] * 30 + [1] * 30 + [2] * 3)   # class 2 is deliberately tiny
    folds = safe_cv_folds(y_fine, desired=5, verbose=True)
    assert folds == 3, f'expected folds to drop to 3, got {folds}'
    assert 'reducing CV folds' in capsys.readouterr().out

    assert safe_cv_folds(np.array([0] * 30 + [1] * 30), desired=5, verbose=False) == 5
    # Never drops below 2, even for a pathologically thin class.
    assert safe_cv_folds(np.array([0] * 30 + [1]), desired=5, verbose=False) == 2


def test_end_to_end_runs_with_a_deliberately_thin_subtype(capsys):
    """A thin SN_Ib must degrade folds, not crash the run."""
    counts = {'SN_Ia': 30, 'SN_Ib': 6, 'SN_Ic': 30, 'SN_II': 30, 'SLSN': 30,
              'AGN': 150, 'TDE': 150, 'stellar_flare': 150}
    df, split = _prepared(counts=counts)
    s1 = run_stage1(split, verbose=False)
    s2a = run_stage2_option_a(split, SN_SUBTYPE_LABELS, verbose=True)
    assert 'reducing CV folds' in capsys.readouterr().out
    assert len(s2a['results']) == 4


# ---------------------------------------------------------------- models

def test_four_models_with_canonical_names_and_svm_config():
    models = build_models()
    assert list(models) == MODEL_NAMES, f'model display names drifted: {list(models)}'

    svm_bag, tree_bag = models['Bagging (SVM)'], models['Bagging (Trees)']
    # Design correction D: the SVM variant must NOT inherit the tree config.
    assert svm_bag.n_estimators == 50
    assert tree_bag.n_estimators == 300
    # probability must stay OFF: enabling it triggers a 5-fold Platt-scaling CV
    # inside every base estimator. It is not passed explicitly, because sklearn 1.9
    # deprecated the keyword (its default there is the sentinel 'deprecated', and
    # False on older versions) — either way, what matters is that it is not True.
    assert svm_bag.estimator.probability is not True
    assert svm_bag.estimator.kernel == 'rbf'
    assert svm_bag.n_jobs == -1


def test_stage1_trains_all_four_and_beats_chance():
    df, split = _prepared()
    s1 = run_stage1(split, verbose=False)
    assert list(s1['models']) == MODEL_NAMES
    assert len(s1['results']) == 4
    for name in MODEL_NAMES:
        acc = float(s1['results'].set_index('Model').loc[name, 'Test accuracy'])
        assert acc > 0.25, f'{name} at chance or below on 4 balanced classes ({acc:.3f})'
        assert s1['confusions'][name].shape == (4, 4)
        assert s1['confusions'][name].sum() == len(split['idx_test'])


# ---------------------------------------------------------------- Option A / B

def test_option_a_evaluates_only_true_sn_test_rows():
    df, split = _prepared()
    s2a = run_stage2_option_a(split, SN_SUBTYPE_LABELS, verbose=False)
    assert len(s2a['le'].classes_) == 5
    assert set(s2a['le'].classes_) == set(SN_SUBTYPE_LABELS)
    assert len(s2a['y_test']) == len(s2a['idx_test'])
    assert df['label'].iloc[s2a['idx_test']].isin(SN_SUBTYPE_LABELS).all()
    for name in MODEL_NAMES:
        assert s2a['confusions'][name].shape == (5, 5)


def test_option_b_cascade_metrics_are_coherent():
    df, split = _prepared()
    s1 = run_stage1(split, verbose=False)
    s2a = run_stage2_option_a(split, SN_SUBTYPE_LABELS, verbose=False)
    s2b = run_stage2_option_b(split, s1, s2a, SN_SUBTYPE_LABELS, verbose=False)

    n_classes = df['label'].nunique()
    for name in MODEL_NAMES:
        r = s2b['per_model'][name]
        # Objects Stage 1 did not call SNe keep their coarse label as the final answer.
        assert set(np.unique(r['final_pred'][~r['sne_mask']])) <= {'AGN', 'TDE', 'stellar_flare'}
        # Objects Stage 1 DID call SNe get a fine subtype, never the coarse 'SNe'.
        assert 'SNe' not in set(np.unique(r['final_pred'][r['sne_mask']]))
        assert s2b['confusions'][name].shape == (n_classes, n_classes)
        assert s2b['confusions'][name].sum() == len(split['idx_test'])

    s = s2b['summary'].set_index('Model')
    for name in MODEL_NAMES:
        cond = s.loc[name, 'Conditional accuracy (correctly-routed SNe)']
        comp = s.loc[name, 'Option-A-comparable accuracy (true SNe)']
        # Mis-routed SNe are counted wrong in `comparable` but excluded from
        # `conditional`, so conditional can only ever be the higher of the two.
        assert cond + 1e-9 >= comp, f'{name}: conditional {cond} < comparable {comp}'
        for col in ['End-to-end accuracy (all test objects)',
                    'Option-A-comparable accuracy (true SNe)']:
            assert 0.0 <= float(s.loc[name, col]) <= 1.0


def test_option_ab_comparison_table_shape():
    df, split = _prepared()
    s1 = run_stage1(split, verbose=False)
    s2a = run_stage2_option_a(split, SN_SUBTYPE_LABELS, verbose=False)
    s2b = run_stage2_option_b(split, s1, s2a, SN_SUBTYPE_LABELS, verbose=False)
    comp = option_ab_comparison(s2a, s2b)

    assert len(comp) == 4
    assert set(comp['Model']) == set(MODEL_NAMES)
    assert 'Routing cost (A - B comparable)' in comp.columns
    # Ground-truth routing is a ceiling: it cannot be beaten by the cascade on the
    # same population, since the cascade adds Stage 1 errors on top.
    assert (comp['Routing cost (A - B comparable)'] >= -1e-9).all(), \
        'Option B beat its own ceiling — the two are not being scored on the same population'


def test_repeated_cv_gives_a_tighter_estimate_than_the_small_holdout():
    """Guards the small-sample caveat: the repeated-CV estimate must be built from
    many more fits than the single 30-object split, and must never touch Stage 1's
    test partition."""
    from btp_pipeline.modeling import repeated_cv_estimate, single_split_resolution

    df, split = _prepared()
    s2a = run_stage2_option_a(split, SN_SUBTYPE_LABELS, verbose=False)

    rep = repeated_cv_estimate(s2a['X_train'], s2a['y_train'], s2a['le'],
                               'Stage 2 / A', n_repeats=3, verbose=False)
    assert set(rep['Model']) == set(MODEL_NAMES)
    assert (rep['n_fits'] >= 15).all(), 'repeated CV should aggregate many fits'
    assert (rep['CI 2.5%'] <= rep['Repeated-CV mean']).all()
    assert (rep['Repeated-CV mean'] <= rep['CI 97.5%']).all()

    # It is computed purely from Stage 2 training rows, which are inside Stage 1's
    # training partition — so no Stage 1 test object can influence it.
    assert len(s2a['X_train']) == len(s2a['idx_train'])
    assert not set(s2a['idx_train']) & set(split['idx_test'])

    res = single_split_resolution(len(s2a['y_test']), len(s2a['le'].classes_))
    assert res['n_test'] == 30
    assert res['worst_case_std_error'] > 0.08, 'a 30-object test set really is this coarse'
    assert res['per_class_recall_step'] > 0.15
