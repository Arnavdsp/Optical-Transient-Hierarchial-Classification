"""
Full Phase 3 + Phase 4 dry run on a synthetic 600-object dataset.

Same schema, same code paths, no network. This is the gate that must pass before
any real TNS/ALeRCE/MAST quota is spent.
"""
import os, sys
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'tests'))

from btp_pipeline.acquisition import assemble_feature_df, verify_counts
from btp_pipeline.config import FEATURES, SN_SUBTYPE_LABELS, X_COLS, N_PER_CLASS, N_PER_SN_SUBTYPE
from btp_pipeline.modeling import (add_coarse_label, assert_split_nesting, build_models,
                                   make_global_split, option_ab_comparison, run_stage1,
                                   run_stage2_option_a, run_stage2_option_b)
from btp_pipeline.interpret import (class_separation_ranking, misclassification_table,
                                    native_importances, per_class_recall,
                                    permutation_importances, physics_summary,
                                    plot_confusions, plot_decision_boundaries,
                                    plot_permutation_importance, rank_discriminative_features)
import synthetic

OUT = os.path.join(ROOT, 'outputs', 'synthetic_dry_run')
os.makedirs(OUT, exist_ok=True)
pd.set_option('display.width', 200, 'display.max_columns', 50)

def hdr(t): print('\n' + '=' * 78 + f'\n{t}\n' + '=' * 78)

hdr('PHASE 2 (synthetic stand-in) — balanced counts')
df = assemble_feature_df([synthetic.make_feature_df().to_dict('records')])
ok, report = verify_counts(df, N_PER_CLASS, N_PER_SN_SUBTYPE, SN_SUBTYPE_LABELS)

hdr('ONE GLOBAL SPLIT')
df = add_coarse_label(df, SN_SUBTYPE_LABELS)
split = make_global_split(df, X_COLS)
s2_tr, s2_te = assert_split_nesting(split, SN_SUBTYPE_LABELS)
print(f"\nStage 2 inherits {len(s2_tr)} train / {len(s2_te)} test rows from Stage 1's partitions.")
print("Leakage check: PASSED (Stage 2 rows are subsets of Stage 1's by construction).")

hdr('PHASE 3 — STAGE 1 (coarse: SNe / AGN / TDE / stellar_flare)')
s1 = run_stage1(split)
print('\n' + s1['results'].round(3).to_string(index=False))

hdr('PHASE 3 — STAGE 2 OPTION A (ceiling: ground-truth routing)')
s2a = run_stage2_option_a(split, SN_SUBTYPE_LABELS)
print('\n' + s2a['results'].round(3).to_string(index=False))

hdr('PHASE 3 — STAGE 2 OPTION B (realistic: Stage-1-predicted SNe)')
s2b = run_stage2_option_b(split, s1, s2a, SN_SUBTYPE_LABELS)
print('\n' + s2b['summary'].round(3).to_string(index=False))

hdr('OPTION A vs OPTION B — side by side')
comp = option_ab_comparison(s2a, s2b)
print(comp.round(3).to_string(index=False))
comp.to_csv(os.path.join(OUT, 'option_ab_comparison.csv'), index=False)

hdr('PHASE 4 — permutation importance (all 4 models), Stage 1')
perm1 = permutation_importances(s1['models'], s1['X_test'], s1['y_test'], X_COLS, n_repeats=20)
print(perm1.pivot(index='feature', columns='Model', values='perm_importance').round(4).to_string())
print('\nMost discriminative features (mean across the 4 models):')
print(rank_discriminative_features(perm1).round(4).to_string())
print('\nNative importances where the family provides one:')
print(native_importances(s1['models'], X_COLS).pivot(
    index='feature', columns='Model', values='native_importance').round(4).to_string())
print('  (Bagging (SVM) absent by design — an RBF SVM exposes no native importance.)')

hdr('PHASE 4 — permutation importance (all 4 models), Stage 2')
perm2 = permutation_importances(s2a['models'], s2a['X_test'], s2a['y_test'], X_COLS, n_repeats=20)
print(rank_discriminative_features(perm2).round(4).to_string())

hdr('PHASE 4 — misclassification analysis (Random Forest)')
for tag, st in [('Stage 1', s1), ('Stage 2 / A', s2a)]:
    yt = st['le'].inverse_transform(st['y_test'])
    yp = st['le'].inverse_transform(st['models']['Random Forest'].predict(st['X_test']))
    print(f'\n--- {tag} per-class recall:'); print(per_class_recall(yt, yp).round(3).to_string())
    print(f'--- {tag} dominant confusions:'); print(misclassification_table(yt, yp).round(3).to_string(index=False))

hdr('PHASE 4 — data-side separability and physics table')
print(class_separation_ranking(df, 'coarse_label', FEATURES).round(3).to_string(index=False))
print('\n' + physics_summary(df, 'coarse_label', FEATURES).to_string())

hdr('PHASE 4 — plots')
top2_s1 = rank_discriminative_features(perm1).index[:2].tolist()
top2_s2 = rank_discriminative_features(perm2).index[:2].tolist()
plot_permutation_importance(perm1, 'Stage 1 — permutation importance', os.path.join(OUT, 's1_perm.png'))
plot_confusions(s1['confusions'], list(s1['le'].classes_), 'Stage 1 — confusion matrices',
                dict(zip(s1['results']['Model'], s1['results']['Test accuracy'])), os.path.join(OUT, 's1_cm.png'))
plot_decision_boundaries(df, split['idx_train'], split['idx_test'], 'coarse_label', top2_s1,
                         build_models, 'Stage 1', os.path.join(OUT, 's1_boundary.png'), grid_steps=120)
plot_permutation_importance(perm2, 'Stage 2 — permutation importance', os.path.join(OUT, 's2_perm.png'))
plot_confusions(s2a['confusions'], list(s2a['le'].classes_), 'Stage 2 Option A — confusion matrices',
                dict(zip(s2a['results']['Model'], s2a['results']['Test accuracy'])), os.path.join(OUT, 's2a_cm.png'))
plot_confusions(s2b['confusions'], s2b['all_classes'], 'Stage 2 Option B — cascaded, all 8 classes',
                None, os.path.join(OUT, 's2b_cm.png'))
plot_decision_boundaries(df[df['label'].isin(SN_SUBTYPE_LABELS)].reset_index(drop=True),
                         *[np.array([j for j, i in enumerate(sorted(np.concatenate([s2_tr, s2_te])))
                                     if i in set(idx)]) for idx in (s2_tr, s2_te)],
                         'label', top2_s2, build_models, 'Stage 2', os.path.join(OUT, 's2_boundary.png'),
                         grid_steps=120)
print('Wrote plots to', OUT)
for f in sorted(os.listdir(OUT)): print('  ', f)

hdr('DRY RUN COMPLETE — no network calls were made')
