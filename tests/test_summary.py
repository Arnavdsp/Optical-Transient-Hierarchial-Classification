"""The results summary must be generated from the real result objects, so it can
never state a number the tables above it do not support."""

import os
import sys

import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from btp_pipeline.acquisition import assemble_feature_df, verify_counts  # noqa: E402
from btp_pipeline.config import SN_SUBTYPE_LABELS, X_COLS  # noqa: E402
from btp_pipeline.modeling import (  # noqa: E402
    add_coarse_label, make_global_split, option_ab_comparison, repeated_cv_estimate,
    run_stage1, run_stage2_option_a, run_stage2_option_b, single_split_resolution)
from btp_pipeline.interpret import (  # noqa: E402
    misclassification_table, per_class_recall, permutation_importances)
from btp_pipeline.summary import build_results_markdown  # noqa: E402
import synthetic  # noqa: E402


def test_summary_renders_and_is_grounded_in_the_numbers():
    df = assemble_feature_df([synthetic.make_feature_df().to_dict('records')], verbose=False)
    ok, report = verify_counts(df, 150, 30, SN_SUBTYPE_LABELS, verbose=False)
    df = add_coarse_label(df, SN_SUBTYPE_LABELS)
    split = make_global_split(df, X_COLS, verbose=False)

    s1 = run_stage1(split, verbose=False)
    s2a = run_stage2_option_a(split, SN_SUBTYPE_LABELS, verbose=False)
    s2b = run_stage2_option_b(split, s1, s2a, SN_SUBTYPE_LABELS, verbose=False)
    comp = option_ab_comparison(s2a, s2b)

    perm1 = permutation_importances(s1['models'], s1['X_test'], s1['y_test'], X_COLS, n_repeats=3)
    perm2 = permutation_importances(s2a['models'], s2a['X_test'], s2a['y_test'], X_COLS, n_repeats=3)

    def yt_yp(st):
        return (st['le'].inverse_transform(st['y_test']),
                st['le'].inverse_transform(st['models']['Random Forest'].predict(st['X_test'])))

    yt1, yp1 = yt_yp(s1)
    yt2, yp2 = yt_yp(s2a)
    rep = repeated_cv_estimate(s2a['X_train'], s2a['y_train'], s2a['le'], 'S2', n_repeats=2, verbose=False)

    md = build_results_markdown(
        report, split, s1, s2a, s2b, comp, perm1, perm2,
        misclassification_table(yt1, yp1), misclassification_table(yt2, yp2),
        per_class_recall(yt1, yp1), per_class_recall(yt2, yp2),
        single_split_resolution(len(s1['y_test']), 4),
        single_split_resolution(len(s2a['y_test']), 5),
        repeated_cv_s2=rep, feature_df=df)

    # Structure
    for section in ['# Results summary', '## Phase 2', 'Stage 1', 'Option A', 'Option B',
                    'Phase 4', 'Honest caveats']:
        assert section in md, f'missing section: {section}'

    # Grounded: the best Stage 1 accuracy actually printed must be the one computed.
    best = s1['results']['Test accuracy'].max()
    assert f"{100 * best:.1f}%" in md, 'summary does not quote the accuracy it computed'

    # The small-sample caveat must be present and must quote the real test size.
    assert f"only {len(s2a['y_test'])} objects" in md
    assert 'load-bearing' in md

    # No unfilled placeholders anywhere.
    for bad in ['TODO', 'XXX', 'FIXME', '{}', 'nan%']:
        assert bad not in md, f'placeholder {bad!r} leaked into the summary'
    assert len(md) > 3000


def test_summary_reports_shortfalls_when_they_exist():
    counts = {'SN_Ia': 30, 'SN_Ib': 12, 'SN_Ic': 30, 'SN_II': 30, 'SLSN': 30,
              'AGN': 150, 'TDE': 150, 'stellar_flare': 150}
    df = assemble_feature_df([synthetic.make_feature_df(counts=counts).to_dict('records')], verbose=False)
    ok, report = verify_counts(df, 150, 30, SN_SUBTYPE_LABELS, verbose=False)
    assert not ok

    df = add_coarse_label(df, SN_SUBTYPE_LABELS)
    split = make_global_split(df, X_COLS, verbose=False)
    s1 = run_stage1(split, verbose=False)
    s2a = run_stage2_option_a(split, SN_SUBTYPE_LABELS, verbose=False)
    s2b = run_stage2_option_b(split, s1, s2a, SN_SUBTYPE_LABELS, verbose=False)
    perm1 = permutation_importances(s1['models'], s1['X_test'], s1['y_test'], X_COLS, n_repeats=2)
    perm2 = permutation_importances(s2a['models'], s2a['X_test'], s2a['y_test'], X_COLS, n_repeats=2)
    yt1 = s1['le'].inverse_transform(s1['y_test'])
    yp1 = s1['le'].inverse_transform(s1['models']['Random Forest'].predict(s1['X_test']))
    yt2 = s2a['le'].inverse_transform(s2a['y_test'])
    yp2 = s2a['le'].inverse_transform(s2a['models']['Random Forest'].predict(s2a['X_test']))

    md = build_results_markdown(
        report, split, s1, s2a, s2b, option_ab_comparison(s2a, s2b), perm1, perm2,
        misclassification_table(yt1, yp1), misclassification_table(yt2, yp2),
        per_class_recall(yt1, yp1), per_class_recall(yt2, yp2),
        single_split_resolution(len(s1['y_test']), 4),
        single_split_resolution(len(s2a['y_test']), 5))

    assert 'Shortfalls remain' in md
    assert 'SN_Ib' in md and 'short by 18' in md
