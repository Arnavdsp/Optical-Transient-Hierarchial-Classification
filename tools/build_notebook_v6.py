"""
Build the v6 notebook: the v5 notebook, unchanged, plus PART B (v6 optimisation).

    python tools/build_notebook_v6.py <v5_notebook.ipynb> <out.ipynb> [--keep-credentials]

Every v5 cell is kept as it is, outputs included, so the v5 results stay reproducible
and comparable. PART B is appended after them and only needs v5's setup cells (Drive,
alerce client, TNS + BTS catalogues, the v4 photometry functions).

The v6 modules are inlined verbatim from btp6/, which tests6/ exercises.

Credentials: by default the TNS credentials cell is replaced by the Colab-secrets
version, so the output is safe to commit to a public repository. --keep-credentials
leaves the source notebook's credentials cell as it is (for a private copy only).
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def md(t):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': t.strip('\n').splitlines(True)}


def code(t):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
            'source': t.strip('\n').splitlines(True)}


def module_cell(name, header):
    with open(os.path.join(ROOT, 'btp6', f'{name}.py')) as f:
        body = f.read()
    return code(f"# {header}\n"
                f"# ---------------------------------------------------------------\n"
                f"# Inlined verbatim from btp6/{name}.py (tested by tests6/). The\n"
                f"# `try: from btp5... except ImportError` lines are for the repository's\n"
                f"# tests; in Colab those names already exist from the v5 cells above.\n"
                f"# ---------------------------------------------------------------\n\n" + body)


SAFE_CREDENTIALS = '''# TNS credentials, read at run time from Colab Secrets (key icon in the left sidebar:
# TNS_BOT_ID, TNS_BOT_NAME, TNS_API_KEY, with notebook access on), from environment
# variables, or from a prompt.
import os

def _get_secret(name, prompt=None):
    try:
        from google.colab import userdata
        val = userdata.get(name)
        if val:
            return val
    except Exception:
        pass
    val = os.environ.get(name)
    if val:
        return val
    import getpass
    return getpass.getpass(prompt or f'{name}: ')

TNS_BOT_ID   = _get_secret('TNS_BOT_ID')
TNS_BOT_NAME = _get_secret('TNS_BOT_NAME')
TNS_API_KEY  = _get_secret('TNS_API_KEY')
TNS_MARKER = ('tns_marker{"tns_id":' + str(TNS_BOT_ID) +
              ',"type": "bot", "name":"' + TNS_BOT_NAME + '"}')
HEADERS = {'user-agent': TNS_MARKER}
print(f'TNS credentials loaded for bot id {TNS_BOT_ID} (key not shown)')
'''

HEADER_V6 = '''
# BTP v6 — more data, physical features, honest imbalance handling

**PART A (cells up to "PART B")** is the v5 notebook exactly as it ran, outputs kept.
**PART B (at the end)** is v6. It re-uses PART A's setup, catalogues, light curves and
v4 photometry, and replaces only the SN sub-classification.

| | v5 (PART A) | v6 (PART B) |
|---|---|---|
| SN label sources | TNS + BTS | TNS + BTS + **Open Supernova Catalog (sne.space, Harvard CfA)**, cross-source label conflicts removed |
| Objects per subtype | 30 / 30 / 30 / 30 / 30 | every Ib, Ic and SLSN the catalogues can supply that survives the gate; Ia and II capped at 600 |
| Light-curve gate | v4 QG (60-d baseline, 30 d post-peak) | same checks, coverage relaxed; v4-gate pass/fail kept per object and the effect measured |
| Features | Villar parameters, observer frame, A-based peak magnitude | rest-frame timescales, curve-peak absolute magnitude with MW extinction and the (1+z) K-term, g-r colour at fixed rest-frame phases, decline rates, plateau and secondary-bump indicators, plus a joint 2-D (time x wavelength) Gaussian-process fit (Boone 2019) and template matching against 66 real core-collapse SNe (Vincenzi 2019, CfA/CSP data) and SALT2 |
| Imbalance | balanced by truncation at 30 | class-weighted models, in-fold majority capping and SMOTE compared, balanced accuracy and macro-F1 reported next to accuracy |
| Evaluation | one 80/20 split, n_test = 41 | repeated stratified 5-fold CV for model choice, then one untouched 20 % test set; plus a 90 % selective-classification operating point fixed on training data |

Run order in Colab: run all of PART A once (or at least STEP 1 to STEP 3: setup, TNS,
BTS and the photometry cell), then PART B. PART B checkpoints to Drive and resumes.
'''

PART_B = []

PART_B.append(md('''
# PART B — v6 SN sub-classification

What limited v5's Stage 2, from the v5 diagnostics (STEP 11 above):

1. **Sample size.** The learning curve was still rising at 30 per class (0.50 at n=15, 0.59
   at n=30), and the test set had 41 objects, so the 95 % CI on accuracy was ±15 points.
2. **Features in the wrong frame.** Timescales were observer-frame, so redshift and
   shape were entangled; `peak_abs_mag` came from the Villar amplitude, ~7x noisier than
   the curve peak; `rise_half` was measured from t0, not the peak.
3. **No colour at a fixed phase**, though colour near peak is the strongest photometric
   Ia / stripped-envelope / H-rich discriminant in the literature.
4. **Ib vs Ic** (median |Cohen's d| = 0.13 over all features): the difference is the
   presence of He I lines in the spectrum, which g and r photometry does not see. The
   4-way scheme (Ib + Ic = SESN), which ALeRCE also uses, is therefore the main target.
'''))

PART_B.append(code('''
# ---- v6 configuration ---------------------------------------------------------
import os, sys, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
try:
    import imblearn  # noqa: F401  (SMOTE; preinstalled on Colab)
except ImportError:
    get_ipython().system('pip install -q imbalanced-learn')

V6_DIR = os.path.join(BASE_DIR, 'v6')
os.makedirs(V6_DIR, exist_ok=True)

# Targets per subtype. The three rare classes ask for more than the catalogues hold,
# so they run to pool exhaustion; Ia and II are capped so the majority-to-minority
# ratio stays near 5:1 instead of the ~40:1 of the raw TNS counts. The learning curve
# was still rising at 400 per class (4-way SESN: 0.805 / 0.831 / 0.857 at 40 / 70 / 100 %).
V6_TARGETS = {'SN_Ia': 600, 'SN_II': 600, 'SN_Ic': 5000, 'SN_Ib': 5000, 'SLSN': 5000}
V6_WORKERS = 8          # parallel ALeRCE requests
USE_ALERCE_FEATURES = False   # see the ALeRCE-features cell for why this is off
USE_CONE_MATCH = True         # recover Ib / Ic / SLSN that TNS lists without a ZTF name
TARGET_ACCURACY = 0.90        # for the selective-classification operating point
CV_SPLITS, CV_REPEATS = 5, 2   # 10 held-out folds per number; raise for tighter error bars

need = ['BASE_DIR', 'alerce', 'tns_full', 'classify_sn_subtype', 'fit_villar', 'villar',
        'clean_band', 'mag_to_flux', 'quality_gate', 'QG', 'ZP', 'distance_modulus',
        'process_ztf_object']
missing = [n for n in need if n not in globals()]
assert not missing, f'run PART A setup cells first; missing: {missing}'
print('v6 output folder:', V6_DIR)
'''))

PART_B.append(module_cell('osc', 'v6 — Open Supernova Catalog loader and pool union'))

PART_B.append(code('''
# ---- three label sources -> one pool per subtype --------------------------------
osc_cat = load_osc_catalog(os.path.join(V6_DIR, 'osc_catalog.min.json'))
osc_df = osc_frame(osc_cat, classify_sn_subtype)
print(f'OSC events: {len(osc_cat):,}; with a ZTF alias and one unambiguous subtype:')
print(osc_df['sn_subtype'].value_counts().to_string())

tns_sn = tns_full[tns_full['sn_subtype'].notna() & tns_full['ztf_name'].notna()]
if USE_CONE_MATCH:
    # TNS records a ZTF name only when ZTF or a broker reported the object. For the
    # three rare classes, objects discovered since ZTF began (2018-03), north of
    # dec -30 and with a redshift, are matched to ALeRCE by a 2" cone search.
    _cand = tns_full[tns_full['ztf_name'].isna() & tns_full['redshift'].notna()
                     & (tns_full['declination'] > -30)
                     & (pd.to_datetime(tns_full['discoverydate'], errors='coerce') >= '2018-03-01')
                     & tns_full['sn_subtype'].isin(['SN_Ib', 'SN_Ic', 'SLSN'])]
    _cone = resolve_ztf_ids(_cand, alerce, workers=V6_WORKERS, survey_kwargs=_survey_kwargs(),
                            cache_path=os.path.join(V6_DIR, 'cone_matches.csv'))
    _hit = _cone[_cone['ztf_name'].notna()]
    print('recovered by cone search:', _hit['sn_subtype'].value_counts().to_dict())
    tns_sn = pd.concat([tns_sn, _hit], ignore_index=True)
sources = []
if 'bts' in globals() and bts is not None and len(bts):
    sources.append(('BTS', bts[bts['sn_subtype'].notna()]))
sources += [('TNS', tns_sn), ('OSC', osc_df)]

v6_pools, pool_rows = {}, []
for st in SN_SUBTYPES:
    v6_pools[st], stats = union_pool(st, sources, random_state=RANDOM_STATE)
    pool_rows.append({'subtype': st, **stats})
pool_table = pd.DataFrame(pool_rows).set_index('subtype')
print()
print(pool_table.to_string())
print('\\nColumns BTS/TNS/OSC count the source that supplied each row first; the OSC column '
      'is what it adds beyond the others. conflicted_ids: ZTF ids given different '
      'subtypes by different sources, removed from every pool.')
'''))

PART_B.append(module_cell('features', 'v6 — rest-frame, phase-anchored light-curve features'))
PART_B.append(module_cell('acquire', 'v6 — parallel retry-until-target acquisition'))

PART_B.append(code('''
# ---- acquisition --------------------------------------------------------------
# Re-uses every light curve already on Drive from PART A (same folders, same CSVs).
# Checkpointed after every batch; re-running resumes.
_fetch = lambda oid: alerce.query_detections(oid, format='pandas', **_survey_kwargs())
v6_parts, v6_manifests = [], []
for st in SN_SUBTYPES:
    t0 = time.time()
    f, m = acquire_to_target(st, v6_pools[st], V6_TARGETS[st], base_dir=BASE_DIR,
                             fetch=_fetch, out_subdir=st.lower(), workers=V6_WORKERS)
    v6_parts.append(f); v6_manifests.append(m)
    print(f'           {time.time() - t0:.0f} s')
sn_v6 = pd.concat(v6_parts, ignore_index=True).drop_duplicates('id').reset_index(drop=True)
sn_v6.to_csv(os.path.join(V6_DIR, 'sn_features_v6.csv'), index=False)

counts = pd.DataFrame({'v5 (PART A)': {s: 30 for s in SN_SUBTYPES},
                       'v6 total': sn_v6['label'].value_counts(),
                       'v6, also passes v4 gate': sn_v6.groupby('label')['passes_v4_gate'].sum()})
counts['v6 gain'] = (counts['v6 total'] / counts['v5 (PART A)']).round(1).astype(str) + 'x'
print(counts.to_string())
_c = sn_v6['label'].value_counts()
print(f'\\nmajority : minority ratio  v6 = {_c.max() / _c.min():.1f} : 1   '
      f'(raw TNS = {tns_sn["sn_subtype"].value_counts().max() / tns_sn["sn_subtype"].value_counts().min():.0f} : 1)')
'''))

PART_B.append(module_cell('templates', 'v6 — template matching against real CfA/CSP SNe (Vincenzi 2019) and SALT2'))

PART_B.append(code('''
# ---- template-matching features -----------------------------------------------
# ~3 s per object per CPU (66 core-collapse templates x 4 host reddenings + SALT2).
# Parallel over Colab's CPUs and checkpointed to Drive every 100 objects.
from concurrent.futures import ProcessPoolExecutor
_tmpl_path = os.path.join(V6_DIR, 'template_features.csv')
_bank = None
def _tmpl_one(args):
    global _bank
    if _bank is None:
        _bank = TemplateBank()
    oid, label, z, ebv = args
    try:
        d = pd.read_csv(os.path.join(BASE_DIR, label.lower(), f'{oid}.csv'))
        o = template_features(d, z, _bank, ebv)
    except Exception:
        o = {}
    o['id'] = oid
    return o

# fetch every template file once in this process (sncosmo caches them on disk), so the
# worker processes do not race each other downloading the same files
_pre = TemplateBank()
for _i in range(len(_pre.catalogue)):
    _pre.curves(_i, 0.0, 0.05)
tmpl = pd.read_csv(_tmpl_path) if os.path.exists(_tmpl_path) else pd.DataFrame(columns=['id'])
_todo = sn_v6[~sn_v6['id'].isin(tmpl['id'])]
print(f'template fits to run: {len(_todo)} objects on {os.cpu_count()} CPUs')
_rows, t0 = [], time.time()
with ProcessPoolExecutor(os.cpu_count()) as ex:
    for o in ex.map(_tmpl_one, list(zip(_todo['id'], _todo['label'], _todo['redshift'], _todo['ebv'])),
                    chunksize=8):
        _rows.append(o)
        if len(_rows) % 100 == 0:
            tmpl = pd.concat([tmpl, pd.DataFrame(_rows)], ignore_index=True); _rows = []
            tmpl.to_csv(_tmpl_path, index=False)
            print(f'  {len(tmpl)} done, {time.time() - t0:.0f} s', flush=True)
tmpl = pd.concat([tmpl, pd.DataFrame(_rows)], ignore_index=True)
tmpl.to_csv(_tmpl_path, index=False)
sn_v6 = sn_v6.drop(columns=[c for c in TEMPLATE_FEATURES if c in sn_v6]).merge(
    tmpl[['id'] + TEMPLATE_FEATURES], on='id', how='left')
print('objects with a template fit:', int(sn_v6['tmpl_lchi2_II'].notna().sum()), '/', len(sn_v6))
print('\\nmedian log10 reduced chi^2 of the best template, by true class (lower = better fit):')
print(sn_v6.groupby('label')[[f'tmpl_lchi2_{c}' for c in TEMPLATE_CLASSES]].median().round(2).to_string())
'''))

PART_B.append(module_cell('alerce_features', 'v6 — optional ALeRCE light-curve feature block'))

PART_B.append(code('''
# ALeRCE's own lc_classifier features. Tested on the 2026-10-31 run: on their own they
# reached 0.695 (4-way, balanced HistGB) against 0.829 for the v6 features, and the
# AllWISE host colours the set includes cover only 13 % of these objects. Off by default.
if USE_ALERCE_FEATURES:
    al_feats = fetch_alerce_features(alerce, sn_v6['id'], workers=V6_WORKERS,
                                     cache_path=os.path.join(V6_DIR, 'alerce_features.csv'),
                                     survey_kwargs=_survey_kwargs())
    AL_COLS = usable_columns(al_feats)
    sn_v6 = sn_v6.join(al_feats[AL_COLS], on='id')
else:
    AL_COLS = []
print('ALeRCE feature columns in use:', len(AL_COLS))
'''))

PART_B.append(module_cell('model', 'v6 — schemes, imbalance handling, models, evaluation'))

PART_B.append(md('''
### Where the improvement comes from

One change at a time, all scored by the same repeated stratified 5-fold CV. Rows A to C
use only objects that pass the v4 gate, so v4 and v6 features are compared on
identical objects. Row A reproduces v5's setup: 30 per class, v4 features, a
default random forest.
'''))

PART_B.append(code('''
from sklearn.ensemble import RandomForestClassifier
sn_v6 = add_sesn_label(sn_v6)            # label_sesn: SN IIb counted as stripped-envelope
print('SN IIb moved to the stripped-envelope class for the "4-way SESN" scheme:',
      int((sn_v6['label_sesn'] != sn_v6['label']).sum()))
V4_COLS = [c for c in LIGHTCURVE_COLS + PHYSICAL_COLS]

# v4 feature rows for the v4-gate objects, from the same CSVs, with v4's own function
v4_path = os.path.join(V6_DIR, 'v4_rows_for_v6_objects.csv')
v4_rows = pd.read_csv(v4_path) if os.path.exists(v4_path) else pd.DataFrame(columns=['id'])
_rows = []
for r in sn_v6[sn_v6['passes_v4_gate'] & ~sn_v6['id'].isin(v4_rows['id'])].itertuples():
    row = process_ztf_object(os.path.join(BASE_DIR, r.label.lower(), f'{r.id}.csv'),
                             r.id, r.label, redshift=r.redshift)
    if row:
        _rows.append(row)
if _rows:
    v4_rows = pd.concat([v4_rows, pd.DataFrame(_rows)], ignore_index=True)
    v4_rows.to_csv(v4_path, index=False)
v4_rows = v4_rows[v4_rows['id'].isin(sn_v6['id'])].reset_index(drop=True)   # this run's objects only
both = v4_rows.set_index('id')[V4_COLS + ['label']].join(
    sn_v6.set_index('id')[V6_FEATURES], how='inner', rsuffix='_v6').reset_index()
V6_IN_BOTH = [c + '_v6' if c in V4_COLS else c for c in V6_FEATURES]
FEATS = V6_FEATURES + GP_FEATURES + TEMPLATE_FEATURES + AL_COLS

rf_v5 = make_pipeline(SimpleImputer(strategy='median'),
                      RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1))
zoo = make_models(RANDOM_STATE)
sub30 = v4_rows.groupby('label', group_keys=False).apply(
    lambda g: g.sample(min(30, len(g)), random_state=RANDOM_STATE))

LADDER = [
    ('A  v5 setup: v4 features, 30 per class, RF',       sub30,  V4_COLS,           rf_v5),
    ('B  + all v4-gate objects',                          v4_rows, V4_COLS,          rf_v5),
    ('C  + v6 features (same objects as B)',              both,   V6_IN_BOTH,        rf_v5),
    ('D  + relaxed gate (all v6 objects)',                sn_v6,  V6_FEATURES,       rf_v5),
    ('E  + class-weighted gradient boosting',             sn_v6,  V6_FEATURES + AL_COLS, zoo['HistGB (balanced)']),
    ('F  + joint 2-D GP features',                        sn_v6,  V6_FEATURES + GP_FEATURES + AL_COLS, zoo['HistGB (balanced)']),
    ('G  + template-matching features',                   sn_v6,  FEATS,             zoo['HistGB (balanced)']),
]
ladder_rows = []
for scheme in ['5-way', '4-way', '3-way']:
    for name, d, cols, mdl in LADDER:
        s = scheme_frame(d, scheme)
        r = cross_validate(mdl, s[cols].values, s['y'].values, n_splits=CV_SPLITS,
                           n_repeats=CV_REPEATS, random_state=RANDOM_STATE)
        ladder_rows.append({'scheme': scheme, 'step': name, 'n': len(s),
                            'accuracy': r['acc_mean'], 'acc_sd': r['acc_sd'],
                            'balanced_acc': r['bacc_mean'], 'macro_F1': r['f1_mean']})
        print(f"{scheme} | {name:48s} n={len(s):5d}  acc {r['acc_mean']:.3f} ± {r['acc_sd']:.3f}"
              f"  bal {r['bacc_mean']:.3f}  F1 {r['f1_mean']:.3f}", flush=True)
ladder = pd.DataFrame(ladder_rows)
ladder.to_csv(os.path.join(V6_DIR, 'improvement_ladder.csv'), index=False)

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, scheme in zip(axes, ['5-way', '4-way', '3-way']):
    d = ladder[ladder['scheme'] == scheme]
    x = np.arange(len(d))
    ax.bar(x - 0.2, d['accuracy'], 0.4, yerr=d['acc_sd'], label='accuracy', color='#4C72B0')
    ax.bar(x + 0.2, d['balanced_acc'], 0.4, label='balanced accuracy', color='#DD8452')
    ax.axhline(0.9, ls='--', c='k', lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([s.split()[0] for s in d['step']])
    ax.set_title(scheme); ax.set_ylim(0.4, 1.0)
axes[0].legend(loc='lower right'); axes[0].set_ylabel('CV score')
plt.suptitle('v6 improvement ladder (steps A to G, see the table above)'); plt.tight_layout(); plt.show()
'''))

PART_B.append(md('''
### Imbalance handling and model choice

Compared on the 4-way scheme by repeated CV on the **training 80 % only**; the 20 % test
set is split off first and not looked at until the final cell. Every resampling step
happens inside the training fold.
'''))

PART_B.append(code('''
from sklearn.model_selection import train_test_split
SCHEMES_RUN = ['5-way', '4-way', '4-way SESN', '3-way']
sel_rows = []
CANDIDATES = {
    'HistGB (balanced)':          (zoo['HistGB (balanced)'], {}),
    'HistGB + cap 2:1':           (zoo['HistGB (balanced)'], {'cap_ratio': 2.0}),
    'HistGB + SMOTE':             (zoo['HistGB (balanced)'], {'use_smote': True}),
    'Random Forest (balanced)':   (zoo['Random Forest (balanced)'], {}),
    'Extra Trees (balanced)':     (zoo['Extra Trees (balanced)'], {}),
    'Logistic Regression':        (zoo['Logistic Regression'], {}),
    'Soft vote HGB+RF+ET':        (SoftVote([zoo['HistGB (balanced)'], zoo['Random Forest (balanced)'],
                                             zoo['Extra Trees (balanced)']]), {}),
    'Hierarchical (HGB nodes)':   (Hierarchical4(zoo['HistGB (balanced)']), {}),
}
# The full zoo was run on 2026-11-01 (see the repository's results/v6/): Logistic
# Regression, Extra Trees alone and majority capping never won any scheme, so by
# default they are skipped to keep the Colab run time down (~40 % fewer fits).
FULL_MODEL_ZOO = False
if not FULL_MODEL_ZOO:
    for _k in ['HistGB + cap 2:1', 'Extra Trees (balanced)', 'Logistic Regression']:
        CANDIDATES.pop(_k)
splits = {}
for scheme in SCHEMES_RUN:
    s = scheme_frame(sn_v6, scheme)
    tr, te = train_test_split(np.arange(len(s)), test_size=0.2, stratify=s['y'],
                              random_state=RANDOM_STATE)
    splits[scheme] = (s, tr, te)
    for name, (mdl, kw) in CANDIDATES.items():
        if name.startswith('Hierarchical') and not scheme.startswith('4-way'):
            continue
        r = cross_validate(mdl, s[FEATS].values[tr], s['y'].values[tr], n_splits=CV_SPLITS,
                           n_repeats=CV_REPEATS, random_state=RANDOM_STATE, **kw)
        sel_rows.append({'scheme': scheme, 'model': name, 'cv_accuracy': r['acc_mean'],
                         'cv_acc_sd': r['acc_sd'], 'cv_balanced_acc': r['bacc_mean'],
                         'cv_macro_F1': r['f1_mean']})
        print(f"{scheme} | {name:28s} acc {r['acc_mean']:.3f} ± {r['acc_sd']:.3f}  "
              f"bal {r['bacc_mean']:.3f}  F1 {r['f1_mean']:.3f}", flush=True)
selection = pd.DataFrame(sel_rows)
selection.to_csv(os.path.join(V6_DIR, 'model_selection_cv.csv'), index=False)
# chosen per scheme by CV macro-F1 (not accuracy, which would reward the majority class)
best = selection.loc[selection.groupby('scheme')['cv_macro_F1'].idxmax()].set_index('scheme')
print('\\nchosen by CV macro-F1:'); print(best[['model', 'cv_accuracy', 'cv_balanced_acc', 'cv_macro_F1']].round(3))
'''))

PART_B.append(md('''
### Final, untouched test set

Each scheme's chosen model is refitted on the full training 80 % and scored once on the
20 % test set, with a bootstrap 95 % interval. Accuracy on the confident subset is also
shown: a photometric classifier is used by trusting its confident calls and sending
the rest for spectroscopy, so "accuracy at X % completeness" is the operational number
(de Soto et al. 2024 report theirs the same way).
'''))

PART_B.append(code('''
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
def bootstrap_ci(y, p, fn, n=2000, seed=0):
    rng = np.random.default_rng(seed); y = np.asarray(y); p = np.asarray(p)
    v = [fn(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return np.percentile(v, [2.5, 97.5])

final_rows, final_models, coverage, selective_rows = [], {}, {}, []
fig, axes = plt.subplots(1, 4, figsize=(22, 4.8))
for ax, scheme in zip(axes, SCHEMES_RUN):
    s, tr, te = splits[scheme]
    name = best.loc[scheme, 'model']; mdl, kw = CANDIDATES[name]
    X = s[FEATS].values.astype(float); y = s['y'].values
    rng = np.random.default_rng(RANDOM_STATE)
    Xtr, ytr = cap_majority(X[tr], y[tr], kw.get('cap_ratio'), rng)
    Xte = X[te]
    if kw.get('use_smote'):
        imp = SimpleImputer(strategy='median').fit(X[tr]); Xtr, ytr = smote(Xtr, ytr, rng); Xte = imp.transform(Xte)
    fitted = clone(mdl).fit(Xtr, ytr); final_models[scheme] = fitted
    pred = fitted.predict(Xte); prob = fitted.predict_proba(Xte)
    cls = np.array(list(fitted.classes_))
    lo, hi = bootstrap_ci(y[te], pred, accuracy_score)
    final_rows.append({'scheme': scheme, 'model': name, 'n_test': len(te),
                       'test_accuracy': accuracy_score(y[te], pred), 'acc_95CI': f'{lo:.3f}-{hi:.3f}',
                       'test_balanced_acc': balanced_accuracy_score(y[te], pred),
                       'test_macro_F1': f1_score(y[te], pred, average='macro')})
    coverage[scheme] = coverage_curve(y[te], prob, cls)
    # selective classification: the confidence cut is chosen on out-of-fold
    # predictions over the TRAINING 80 % only, then applied once to the test set.
    # Rule: Geifman & El-Yaniv (2017) -- the 95 % upper bound on the kept set's error
    # must be within the target, so the cut has a margin for sampling noise.
    oof = cross_validate(mdl, X[tr], y[tr], n_splits=CV_SPLITS, n_repeats=1, return_oof=True,
                         random_state=RANDOM_STATE, **kw)
    thr, cov_tr, acc_tr = confidence_threshold(y[tr], oof['oof_prob'], oof['classes'], TARGET_ACCURACY)
    rep = selective_report(y[te], prob, cls, thr)
    selective_rows.append({'scheme': scheme, 'threshold (from train)': round(thr, 3),
                           'train-CV coverage': cov_tr, 'train-CV accuracy': acc_tr,
                           'TEST coverage': rep['coverage'], 'TEST accuracy': rep['accuracy'],
                           'TEST balanced acc': rep['balanced_accuracy'], 'TEST macro-F1': rep['macro_F1'],
                           **{f'kept {c}': v for c, v in rep['per_class_coverage'].items()}})
    cm = confusion_table(y[te], pred, cls)
    norm = cm.values / cm.values.sum(1, keepdims=True)
    ax.imshow(norm, cmap='Blues', vmin=0, vmax=1)
    for i in range(len(cls)):
        for j in range(len(cls)):
            ax.text(j, i, f'{cm.values[i, j]}\\n{norm[i, j]:.0%}', ha='center', va='center',
                    color='white' if norm[i, j] > 0.6 else 'black', fontsize=8)
    ax.set_xticks(range(len(cls))); ax.set_xticklabels(cls, rotation=30)
    ax.set_yticks(range(len(cls))); ax.set_yticklabels(cls)
    ax.set_xlabel('predicted'); ax.set_ylabel('true'); ax.set_title(f'{scheme}: {name}')
plt.tight_layout(); plt.show()
final = pd.DataFrame(final_rows)
final.to_csv(os.path.join(V6_DIR, 'final_test_results.csv'), index=False)
display(final.round(3))
for scheme, cc in coverage.items():
    print(f'\\n{scheme} — accuracy on the most confident fraction of the test set')
    print(cc.round(3).to_string(index=False))
selective = pd.DataFrame(selective_rows)
selective.to_csv(os.path.join(V6_DIR, 'selective_classification.csv'), index=False)
print(f'\\nSelective classification at a {TARGET_ACCURACY:.0%} accuracy target. The cut is fixed on the '
      'training data; the TEST columns are the honest out-of-sample result. Objects below the '
      'cut are the ones to send for a spectrum.')
display(selective.round(3))
'''))

PART_B.append(code('''
# Which features carry the 4-way result: permutation importance on the test set.
from sklearn.inspection import permutation_importance
s, tr, te = splits['4-way SESN']
pi = permutation_importance(final_models['4-way SESN'], s[FEATS].values[te].astype(float), s['y'].values[te],
                            scoring='f1_macro', n_repeats=20, random_state=RANDOM_STATE, n_jobs=-1)
imp = pd.Series(pi.importances_mean, index=FEATS).sort_values(ascending=False)
imp.to_csv(os.path.join(V6_DIR, 'perm_importance_4way.csv'))
imp.head(20)[::-1].plot.barh(figsize=(7, 6), color='#4C72B0')
plt.xlabel('drop in test macro-F1 when shuffled'); plt.title('4-way SESN: top 20 features'); plt.tight_layout(); plt.show()
'''))

PART_B.append(code('''
# ---- written summary, generated from the numbers above ---------------------------
L = ['## v6 results\\n']
L.append(f'Objects: {len(sn_v6)} SNe ({", ".join(f"{k} {v}" for k, v in sn_v6["label"].value_counts().items())}), '
         f'against 150 in v5.\\n')
for scheme in SCHEMES_RUN:
    a = ladder[(ladder.scheme == scheme)]
    f = final[final.scheme == scheme].iloc[0]
    cc = coverage[scheme]
    c80 = cc[cc.kept_fraction == 0.8].iloc[0]
    cv = (f'CV accuracy {a.accuracy.iloc[0]:.3f} (v5 setup) -> {a.accuracy.iloc[-1]:.3f} (v6); ' if len(a) else '')
    sv = selective[selective.scheme == scheme].iloc[0]
    L.append(f'**{scheme}**: {cv}'
             f'held-out test with {f.model}: accuracy {f.test_accuracy:.3f} (95 % CI {f.acc_95CI}), '
             f'balanced accuracy {f.test_balanced_acc:.3f}, macro-F1 {f.test_macro_F1:.3f}. '
             f'With the {TARGET_ACCURACY:.0%} confidence cut fixed on training data: test accuracy '
             f'{sv["TEST accuracy"]:.3f} on {sv["TEST coverage"]:.0%} of test objects.\\n')
L.append('Caveat on redshift: redshift and absolute magnitude are legitimate physical features, '
         'but part of their power is the selection function of spectroscopic follow-up (SLSNe '
         'are found far away, Ib/c nearby). v5 STEP 11 measured this; it applies to any '
         'classifier trained on TNS-classified objects, including ALeRCE and Superphot+.\\n')
summary_v6 = '\\n'.join(L)
from IPython.display import Markdown
display(Markdown(summary_v6))
with open(os.path.join(V6_DIR, 'results_summary_v6.md'), 'w') as fh:
    fh.write(summary_v6)
'''))


def build(src, out, keep_credentials=False):
    with open(src) as f:
        nb = json.load(f)
    cells = nb['cells']
    while cells and not ''.join(cells[-1]['source']).strip():
        cells.pop()
    if not keep_credentials:
        for c in cells:
            s = ''.join(c['source'])
            if c['cell_type'] == 'code' and 'TNS_API_KEY' in s and 'HEADERS' in s and 'requests.post' not in s:
                c['source'] = SAFE_CREDENTIALS.splitlines(True)
                c['outputs'] = []
    cells.insert(0, md(HEADER_V6))
    cells.extend(PART_B)
    nb['cells'] = cells
    with open(out, 'w') as f:
        json.dump(nb, f, indent=1)
    return nb


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    nb = build(args[0], args[1], keep_credentials='--keep-credentials' in sys.argv)
    print(f'wrote {args[1]}: {len(nb["cells"])} cells')
