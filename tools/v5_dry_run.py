"""
Full v5 Phase 4-6 dry run on synthetic data. No network, no API quota.
This is the gate before the real dual-track run.
"""
import os, sys, warnings
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'tests5'))

from btp5 import agreement as AG, calibration as CAL, pools as PL
from btp5.alerce_labels import build_crosswalk, collapse_tns_fine
from btp5.config import FEATURE_COLS
from btp5.modeling import run_track, track_summary
import synthetic5

pd.set_option('display.width', 220, 'display.max_columns', 40)
def hdr(t): print('\n' + '=' * 88 + f'\n{t}\n' + '=' * 88)

LIVE = ['AGN','Blazar','CEP','CV/Nova','DSCT','EA','EB/EW','LPV','Microlensing',
        'Periodic-Other','QSO','RRLab','RRLc','RSCVn','SESN','SLSN','SNII','SNIIn',
        'SNIa','TDE','YSO']

hdr('PHASE 2B — crosswalk (built against the live class list)')
build_crosswalk(LIVE)

hdr('PHASE 2C — three pools')
df = synthetic5.make_feature_df()
df['has_tns_label'] = True
df, counts = PL.build_pools(df)

hdr('PHASE 4 — dual-track training (one parameterized code path)')
tns = run_track(df, FEATURE_COLS, 'coarse_label', 'label', 'TNS')
al_pool = PL.get_pool(df, PL.POOL_ALERCE)
alerce = run_track(al_pool, FEATURE_COLS, 'alerce_coarse', 'alerce_fine', 'ALeRCE')

hdr('Section 6.3 — accuracy / macro-F1 per model, per track, with the pool named')
print(track_summary([tns, alerce]).round(3).to_string(index=False))

hdr('PHASE 5 — agreement on the OVERLAP pool')
ov = PL.get_pool(df, PL.POOL_OVERLAP)
classes = sorted(set(ov['coarse_label']) & set(ov['alerce_coarse'].dropna()))
print(f'overlap n={len(ov)}, comparable classes={classes}')
print('(stellar_flare is structurally absent — ALeRCE classifies ZTF, not TESS)\n')

s1 = tns['stage1']
le, sp = s1['split']['le'], s1['split']
test_df = sp['df'].iloc[sp['idx_test']]
for model_name in ['Random Forest', 'Logistic Regression']:
    pred_ours = le.inverse_transform(s1['models'][model_name].predict(sp['X_test']))
    m = test_df['alerce_coarse'].notna().values & np.isin(pred_ours, classes)
    m &= np.isin(test_df['alerce_coarse'].fillna('').values, classes)
    if m.sum() < 5:
        print(f'{model_name}: too few overlap test objects'); continue
    ours = pred_ours[m]
    theirs = test_df['alerce_coarse'].values[m]
    truth = test_df['coarse_label'].values[m]
    P = AG.purity_matrix(truth, ours, classes)
    C = AG.completeness_matrix(truth, theirs, classes)
    res = AG.compare_expected_actual(ours, theirs, P, C, classes)
    print(f'--- {model_name} ---')
    print('expected agreement matrix:'); print(res['expected_matrix'].round(3).to_string())
    print('actual agreement matrix:');   print(res['actual_matrix'].round(3).to_string())
    print()
    print(AG.interpret_agreement(res, model_name, PL.POOL_OVERLAP))
    print()

hdr('Section 6.4 — external corroboration')
print(AG.EXTERNAL_RESULTS.to_string(index=False))
print('\n' + AG.TDE_SLSN_STATEMENT)
print('\n' + AG.TNS_LABEL_NOISE_CAVEAT)

hdr('PHASE 6B — calibration, with and without redshift')
s2 = tns['stage2']
if s2 is not None:
    sp2 = s2['split']
    zi = [FEATURE_COLS.index(c) for c in ['redshift', 'peak_abs_mag']]
    keep = [i for i in range(len(FEATURE_COLS)) if i not in zi]
    for name in ['Random Forest', 'Logistic Regression']:
        with_z = CAL.calibration_for_model(s2['models'][name], sp2['X_test'], sp2['y_test'])
        from btp5.modeling import build_models
        m2 = build_models()[name]
        m2.fit(sp2['X_train'][:, keep], sp2['y_train'])
        without_z = CAL.calibration_for_model(m2, sp2['X_test'][:, keep], sp2['y_test'])
        print(CAL.compare_with_without_redshift(with_z, without_z, name)); print()
else:
    print('Stage 2 unavailable in this synthetic run.')

hdr('DRY RUN COMPLETE — no network calls were made')
