"""
Phase 4 components on synthetic data: permutation importance across all four
model families, native importance where it exists, misclassification analysis,
and the 2D decision-boundary projection.
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')  # headless
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from btp_pipeline.acquisition import assemble_feature_df  # noqa: E402
from btp_pipeline.config import FEATURES, MODEL_NAMES, SN_SUBTYPE_LABELS, X_COLS  # noqa: E402
from btp_pipeline.modeling import (  # noqa: E402
    add_coarse_label, build_models, make_global_split, run_stage1)
from btp_pipeline.interpret import (  # noqa: E402
    class_separation_ranking, misclassification_table, native_importances,
    per_class_recall, permutation_importances, physics_summary,
    plot_confusions, plot_decision_boundaries, plot_permutation_importance,
    rank_discriminative_features)
import synthetic  # noqa: E402


def _stage1():
    df = synthetic.make_feature_df()
    df = assemble_feature_df([df.to_dict('records')], verbose=False)
    df = add_coarse_label(df, SN_SUBTYPE_LABELS)
    split = make_global_split(df, X_COLS, verbose=False)
    return df, split, run_stage1(split, verbose=False)


def test_permutation_importance_covers_all_four_models_including_svm():
    df, split, s1 = _stage1()
    perm = permutation_importances(s1['models'], s1['X_test'], s1['y_test'], X_COLS, n_repeats=5)

    assert set(perm['Model']) == set(MODEL_NAMES), \
        'permutation importance must cover all four families — that is the whole point'
    assert len(perm) == 4 * len(X_COLS)
    assert perm['perm_importance'].notna().all()
    # The bagged RBF SVM has no native importance at all, yet still gets a score here.
    svm = perm[perm['Model'] == 'Bagging (SVM)']
    assert len(svm) == len(X_COLS) and svm['perm_importance'].notna().all()

    ranked = rank_discriminative_features(perm)
    assert list(ranked.columns) == ['mean_perm_importance', 'across_model_std']
    assert len(ranked) == len(X_COLS)
    assert ranked['mean_perm_importance'].is_monotonic_decreasing


def test_native_importance_present_for_three_families_absent_for_svm():
    df, split, s1 = _stage1()
    nat = native_importances(s1['models'], X_COLS)
    present = set(nat['Model'])

    assert 'Random Forest' in present
    assert 'Logistic Regression' in present
    assert 'Bagging (Trees)' in present
    assert 'Bagging (SVM)' not in present, \
        'an RBF-kernel bagged SVM has no native importance; it must be omitted, not faked'

    rf = nat[nat['Model'] == 'Random Forest']['native_importance']
    assert abs(rf.sum() - 1.0) < 1e-6, 'Gini importances should sum to 1'
    bag = nat[nat['Model'] == 'Bagging (Trees)']
    assert len(bag) == len(X_COLS) and (bag['native_importance'] >= 0).all()


def test_synthetic_physics_ordering_is_recovered():
    """The synthetic flares are built short-and-spiky and the AGN long-and-flat.
    A sane importance method should notice the timescale features."""
    df, split, s1 = _stage1()
    perm = permutation_importances(s1['models'], s1['X_test'], s1['y_test'], X_COLS, n_repeats=5)
    ranked = rank_discriminative_features(perm)
    top3 = ranked.index[:3].tolist()
    assert {'rise_time', 'decay_time'} & set(top3), \
        f'timescale features should dominate this synthetic set, got {top3}'

    sep = class_separation_ranking(df, 'coarse_label', FEATURES)
    assert len(sep) == len(FEATURES)
    assert sep['separation_ratio'].is_monotonic_decreasing


def test_misclassification_and_recall_tables():
    df, split, s1 = _stage1()
    model = s1['models']['Random Forest']
    y_true = s1['le'].inverse_transform(s1['y_test'])
    y_pred = s1['le'].inverse_transform(model.predict(s1['X_test']))

    tab = misclassification_table(y_true, y_pred)
    assert set(tab.columns) == {'true', 'pred', 'n', 'share_of_errors'}
    assert (tab['true'] != tab['pred']).all(), 'correct predictions leaked into the error table'
    if len(tab):
        assert tab['n'].is_monotonic_decreasing

    rec = per_class_recall(y_true, y_pred)
    assert rec['n_test'].sum() == len(y_true)
    assert ((rec['recall'] >= 0) & (rec['recall'] <= 1)).all()

    # A perfect prediction produces an empty, well-formed error table.
    empty = misclassification_table(y_true, y_true)
    assert len(empty) == 0 and 'share_of_errors' in empty.columns


def test_plots_build_without_error(tmp_path):
    df, split, s1 = _stage1()
    perm = permutation_importances(s1['models'], s1['X_test'], s1['y_test'], X_COLS, n_repeats=3)

    f1 = plot_permutation_importance(perm, 'Stage 1 — permutation importance',
                                     out_path=str(tmp_path / 'perm.png'))
    assert len(f1.axes) >= 4
    assert (tmp_path / 'perm.png').exists()

    accs = dict(zip(s1['results']['Model'], s1['results']['Test accuracy']))
    f2 = plot_confusions(s1['confusions'], list(s1['le'].classes_),
                         'Stage 1 — confusion matrices', accuracies=accs,
                         out_path=str(tmp_path / 'cm.png'))
    assert (tmp_path / 'cm.png').exists()

    top2 = rank_discriminative_features(perm).index[:2].tolist()
    f3 = plot_decision_boundaries(df, split['idx_train'], split['idx_test'], 'coarse_label',
                                  top2, build_models, 'Stage 1 decision boundaries',
                                  out_path=str(tmp_path / 'db.png'), grid_steps=60)
    assert (tmp_path / 'db.png').exists()
    assert len(f3.axes) == 4, 'a decision-boundary panel is required for each of the 4 models'


def test_decision_boundary_fits_on_training_rows_only():
    """Guards the fix to the earlier version, which fit the visualisation model on
    the whole dataset and so drew a boundary that had already seen the test points."""
    import inspect
    from btp_pipeline import interpret
    src = inspect.getsource(interpret.plot_decision_boundaries)
    assert 'fit(X2_train' in src
    assert 'scaler.fit_transform(X2[idx_train])' in src


def test_physics_summary_shape():
    df, split, s1 = _stage1()
    summ = physics_summary(df, 'coarse_label', FEATURES)
    assert len(summ) == 4
    assert set(summ.columns.get_level_values(0)) == set(FEATURES)
