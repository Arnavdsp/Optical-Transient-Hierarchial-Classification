"""
Phases 3 and 4 on the real feature table, headless.

Runs exactly what the notebook runs — same modules — and writes every deliverable
(tables, plots, the generated results summary) to outputs/real_run/.
"""

import os
import sys
import warnings

warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from btp_pipeline.acquisition import verify_counts  # noqa: E402
from btp_pipeline.config import (  # noqa: E402
    FEATURES, N_PER_CLASS, N_PER_SN_SUBTYPE, SN_SUBTYPE_LABELS, X_COLS)
from btp_pipeline.modeling import (  # noqa: E402
    add_coarse_label, assert_split_nesting, build_models, make_global_split,
    option_ab_comparison, repeated_cv_estimate, run_stage1, run_stage2_option_a,
    run_stage2_option_b, single_split_resolution)
from btp_pipeline.interpret import (  # noqa: E402
    class_separation_ranking, feature_diagnostics, format_feature_diagnostics,
    misclassification_table, native_importances,
    per_class_recall, permutation_importances, physics_summary, plot_confusions,
    plot_decision_boundaries, plot_permutation_importance, rank_discriminative_features)
from btp_pipeline.summary import build_results_markdown  # noqa: E402

FEATURE_CSV = os.path.join(ROOT, 'data', 'features', 'phase3_features_balanced.csv')
OUT = os.path.join(ROOT, 'outputs', 'real_run')
os.makedirs(OUT, exist_ok=True)
pd.set_option('display.width', 220, 'display.max_columns', 60)


def hdr(t):
    print('\n' + '=' * 90 + f'\n{t}\n' + '=' * 90)


feature_df = pd.read_csv(FEATURE_CSV)
hdr('PHASE 2 RESULT — final sample')
ok, count_report = verify_counts(feature_df, N_PER_CLASS, N_PER_SN_SUBTYPE, SN_SUBTYPE_LABELS)
count_report.to_csv(os.path.join(OUT, 'count_report.csv'), index=False)

hdr('ONE GLOBAL SPLIT')
feature_df = add_coarse_label(feature_df, SN_SUBTYPE_LABELS)
split = make_global_split(feature_df, X_COLS)
s2_train_idx, s2_test_idx = assert_split_nesting(split, SN_SUBTYPE_LABELS)
print(f'\nStage 2 inherits {len(s2_train_idx)} train / {len(s2_test_idx)} test rows.')
print('Leakage check PASSED.')

hdr('STAGE 1 — coarse')
stage1 = run_stage1(split)
print('\n' + stage1['results'].round(3).to_string(index=False))
stage1['results'].to_csv(os.path.join(OUT, 'stage1_results.csv'), index=False)

hdr('STAGE 2 OPTION A — ground-truth routing')
stage2a = run_stage2_option_a(split, SN_SUBTYPE_LABELS)
print('\n' + stage2a['results'].round(3).to_string(index=False))
stage2a['results'].to_csv(os.path.join(OUT, 'stage2a_results.csv'), index=False)

hdr('STAGE 2 OPTION B — cascaded')
stage2b = run_stage2_option_b(split, stage1, stage2a, SN_SUBTYPE_LABELS)
print('\n' + stage2b['summary'].round(3).to_string(index=False))
stage2b['summary'].to_csv(os.path.join(OUT, 'stage2b_results.csv'), index=False)

hdr('OPTION A vs OPTION B')
comparison = option_ab_comparison(stage2a, stage2b)
print(comparison.round(3).to_string(index=False))
comparison.to_csv(os.path.join(OUT, 'option_ab_comparison.csv'), index=False)

hdr('MEASUREMENT RESOLUTION + repeated CV')
res_s1 = single_split_resolution(len(stage1['y_test']), len(stage1['le'].classes_))
res_s2 = single_split_resolution(len(stage2a['y_test']), len(stage2a['le'].classes_))
for tag, r in [('Stage 1', res_s1), ('Stage 2', res_s2)]:
    print(f"{tag}: n_test={r['n_test']}, step {r['accuracy_step']:.3f}, "
          f"worst-case SE {r['worst_case_std_error']:.3f}, "
          f"{r['mean_test_objects_per_class']:.1f}/class")
print()
repeated_cv_s2 = repeated_cv_estimate(stage2a['X_train'], stage2a['y_train'],
                                      stage2a['le'], 'Stage 2 / A')
repeated_cv_s2.to_csv(os.path.join(OUT, 'stage2_repeated_cv.csv'), index=False)

hdr('PHASE 4 — importance')
perm_s1 = permutation_importances(stage1['models'], stage1['X_test'], stage1['y_test'],
                                  X_COLS, n_repeats=30)
perm_s2 = permutation_importances(stage2a['models'], stage2a['X_test'], stage2a['y_test'],
                                  X_COLS, n_repeats=30)
ranked_s1, ranked_s2 = rank_discriminative_features(perm_s1), rank_discriminative_features(perm_s2)
print('Stage 1:'); print(ranked_s1.round(4).to_string())
print('\nStage 2:'); print(ranked_s2.round(4).to_string())
perm_s1.to_csv(os.path.join(OUT, 'stage1_permutation_importance.csv'), index=False)
perm_s2.to_csv(os.path.join(OUT, 'stage2_permutation_importance.csv'), index=False)
nat_s1 = native_importances(stage1['models'], X_COLS)
nat_s1.to_csv(os.path.join(OUT, 'stage1_native_importance.csv'), index=False)
print('\nNative (Stage 1):')
print(nat_s1.pivot(index='feature', columns='Model', values='native_importance').round(4).to_string())

hdr('PHASE 4 — misclassification')
yt_s1 = stage1['le'].inverse_transform(stage1['y_test'])
yp_s1 = stage1['le'].inverse_transform(stage1['models']['Random Forest'].predict(stage1['X_test']))
yt_s2 = stage2a['le'].inverse_transform(stage2a['y_test'])
yp_s2 = stage2a['le'].inverse_transform(stage2a['models']['Random Forest'].predict(stage2a['X_test']))
recall_s1, recall_s2 = per_class_recall(yt_s1, yp_s1), per_class_recall(yt_s2, yp_s2)
misclass_s1, misclass_s2 = misclassification_table(yt_s1, yp_s1), misclassification_table(yt_s2, yp_s2)
print('Stage 1 recall:'); print(recall_s1.round(3).to_string())
print('Stage 1 confusions:'); print(misclass_s1.to_string(index=False))
print('\nStage 2 recall:'); print(recall_s2.round(3).to_string())
print('Stage 2 confusions:'); print(misclass_s2.to_string(index=False))

hdr('PHASE 4 — separability and physics')
sep1 = class_separation_ranking(feature_df, 'coarse_label', FEATURES)
sn_rows = np.flatnonzero(feature_df['label'].isin(SN_SUBTYPE_LABELS).values)
sn_df = feature_df.iloc[sn_rows].reset_index(drop=True)
sep2 = class_separation_ranking(sn_df, 'label', FEATURES)
print('Stage 1 separability:'); print(sep1.round(3).to_string(index=False))
print('\nStage 2 separability:'); print(sep2.round(3).to_string(index=False))
phys1 = physics_summary(feature_df, 'coarse_label', FEATURES)
phys2 = physics_summary(sn_df, 'label', FEATURES)
phys1.to_csv(os.path.join(OUT, 'stage1_physics_summary.csv'))
phys2.to_csv(os.path.join(OUT, 'stage2_physics_summary.csv'))
print('\n' + phys1.to_string())
print('\n' + phys2.to_string())

hdr('PHASE 4 — plots')
plot_permutation_importance(perm_s1, 'Stage 1 — permutation importance',
                            os.path.join(OUT, 'stage1_perm_importance.png'))
plot_permutation_importance(perm_s2, 'Stage 2 — permutation importance',
                            os.path.join(OUT, 'stage2_perm_importance.png'))
plot_confusions(stage1['confusions'], list(stage1['le'].classes_), 'Stage 1 — confusion matrices',
                dict(zip(stage1['results']['Model'], stage1['results']['Test accuracy'])),
                os.path.join(OUT, 'stage1_confusion.png'))
plot_confusions(stage2a['confusions'], list(stage2a['le'].classes_),
                'Stage 2 Option A — confusion matrices',
                dict(zip(stage2a['results']['Model'], stage2a['results']['Test accuracy'])),
                os.path.join(OUT, 'stage2a_confusion.png'))
plot_confusions(stage2b['confusions'], stage2b['all_classes'],
                'Stage 2 Option B — cascaded, all fine-grained classes', None,
                os.path.join(OUT, 'stage2b_confusion.png'))
plot_decision_boundaries(feature_df, split['idx_train'], split['idx_test'], 'coarse_label',
                         ranked_s1.index[:2].tolist(), build_models, 'Stage 1',
                         os.path.join(OUT, 'stage1_decision_boundary.png'))
pos = {o: n for n, o in enumerate(sn_rows)}
plot_decision_boundaries(sn_df, np.array([pos[i] for i in s2_train_idx]),
                         np.array([pos[i] for i in s2_test_idx]), 'label',
                         ranked_s2.index[:2].tolist(), build_models, 'Stage 2',
                         os.path.join(OUT, 'stage2_decision_boundary.png'))
plt.close('all')

hdr('PHASE 4 — feature-construction diagnostics')
diagnostics = feature_diagnostics(feature_df, FEATURES)
print(format_feature_diagnostics(diagnostics))

hdr('RESULTS SUMMARY')
summary_md = build_results_markdown(
    count_report, split, stage1, stage2a, stage2b, comparison, perm_s1, perm_s2,
    misclass_s1, misclass_s2, recall_s1, recall_s2, res_s1, res_s2,
    repeated_cv_s2=repeated_cv_s2, feature_df=feature_df, diagnostics=diagnostics)
with open(os.path.join(OUT, 'results_summary.md'), 'w') as f:
    f.write(summary_md)
print(f"Wrote {os.path.join(OUT, 'results_summary.md')} ({len(summary_md)} chars)")
print('\nPHASES 3 & 4 COMPLETE')
