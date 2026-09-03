"""
Build the deliverable notebook.

Two rules drive this script:

1. Anything the brief said to reuse (`classify_sn_subtype`, `extract_ztf_name`,
   `resolve_oid`, `clean_lightcurve`, `gp_interpolate`, `extract_shape_features`,
   `process_ztf_object`, `process_tess_object`) is copied VERBATIM out of the
   original notebook. It is not retyped, so it cannot drift.

2. Anything new is inlined verbatim from `btp_pipeline/`, which is what the test
   suite exercises. The notebook therefore stays self-contained — it runs in
   Colab with no repo clone — while the code in it is the code that was tested.
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The previous notebook is vendored into the repo rather than read from wherever it
# happened to be uploaded, so the build is reproducible on any machine (CI included)
# and the provenance of every verbatim-reused cell stays auditable.
SRC_NB = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    ROOT, 'notebooks', 'source', 'BTP_TNS_Hierarchical_2_original.ipynb')
OUT_NB = os.path.join(ROOT, 'notebooks', 'BTP_TNS_Hierarchical_Balanced.ipynb')


def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.strip('\n').splitlines(True)}


def code(text):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None,
            'outputs': [], 'source': text.strip('\n').splitlines(True)}


def module_cell(name, header):
    """Inline a tested module as a notebook cell."""
    path = os.path.join(ROOT, 'btp_pipeline', f'{name}.py')
    with open(path) as f:
        body = f.read()
    return code(f"# {header}\n"
                f"# ---------------------------------------------------------------\n"
                f"# Inlined verbatim from btp_pipeline/{name}.py, which is covered by the\n"
                f"# repository's synthetic-data test suite. Edit it there and rebuild this\n"
                f"# notebook (tools/build_notebook.py) rather than editing it here.\n"
                f"# ---------------------------------------------------------------\n\n"
                + body)


with open(SRC_NB) as f:
    orig = json.load(f)


def orig_src(i):
    return ''.join(orig['cells'][i]['source'])


def orig_upto(i, marker):
    """Verbatim prefix of an original cell, cut at `marker` — used where a cell
    mixed a function definition (keep) with a driver loop (superseded)."""
    s = orig_src(i)
    assert marker in s, f'marker {marker!r} not found in cell {i}'
    return s[:s.index(marker)].rstrip() + '\n'


cells = []
A = cells.append

# ============================================================ title
A(md("""
# BTP: Optical Transient Classification — Balanced Hierarchical Pipeline

**Classes (8):** SN Ia, SN Ib, SN Ic, SN II, SLSN, AGN, TDE (TNS, spectroscopically
confirmed, photometry via ZTF/ALeRCE) and Stellar Flares (TESS, via lightkurve).

**Two-stage hierarchy**
- **Stage 1 (coarse):** SNe vs AGN vs TDE vs Stellar Flare — 150 objects each, 600 total.
- **Stage 2 (fine):** for SNe, which subtype — Ia / Ib / Ic / II / SLSN, 30 each.

**Four models at both stages:** Random Forest, Logistic Regression, Bagging (Trees),
Bagging (SVM).

---

### What changed in this rebuild, and why

1. **Balanced acquisition by retry, not by hope.** The previous pipeline sampled a
   fixed number of candidates per class and accepted whatever survived cross-match,
   download and GP fitting. Survival is not random — sparser, noisier classes fail
   more often — so the SN subtype set collapsed to 95 objects (Ia 18, Ib 13, Ic 15,
   II 21, SLSN 28) with the rare classes hit hardest, which is the direct cause of
   Stage 2 landing near 47-53%. Every class now uses a *keep pulling candidates
   until N survive* loop, so the final counts are exact.

2. **One global split, not two independent ones.** Earlier versions split Stage 1
   (over the full dataset) and Stage 2 (over the SN subset) independently, which let
   an SN object in Stage 1's test set turn up in Stage 2's training set and quietly
   inflate the cascaded accuracy. There is now exactly **one** stratified split;
   Stage 2's train and test rows are subsets of Stage 1's own partitions **by
   construction**, and that nesting is asserted in code. No "leakage-free retrain"
   cleanup step is needed, because leakage is structurally impossible.

3. **A fourth model, configured for what it is.** Bagging (SVM) does not inherit the
   tree-bagging settings — see the model-building cell for the reasoning.

4. **One importance method that spans all four families.** Permutation importance,
   computed identically for every model on held-out data, because an RBF-kernel
   bagged SVM has neither coefficients nor impurity gains.

Code that the rebuild did not need to change — `classify_sn_subtype`,
`extract_ztf_name`, `resolve_oid`, `clean_lightcurve`, `gp_interpolate`,
`extract_shape_features`, `process_ztf_object`, `process_tess_object` — is reused
unchanged from the previous notebook.
"""))

# ============================================================ PHASE 2
A(md("# PHASE 2 — Data Acquisition (TNS + ZTF/ALeRCE + TESS)"))
A(code(orig_src(2)))                       # setup / config / constants  (verbatim)
# TNS credentials — NOT reused verbatim. The original cell hard-coded a live TNS bot
# API key, which is a secret and must not sit in a notebook that gets committed or
# shared. Same interface (TNS_MARKER / TNS_API_KEY / HEADERS), sourced safely.
A(code("""
# TNS credentials — read from the environment, never hard-coded.
#
# The previous notebook had the bot id and API key written into this cell. Anything
# committed to git or shared as a notebook carries them along, so they are read at
# run time instead. In Colab, put them in the Secrets panel (the key icon in the left
# sidebar) as TNS_BOT_ID, TNS_BOT_NAME and TNS_API_KEY, with notebook access enabled.
import os

def _get_secret(name, prompt=None):
    # Colab Secrets first, then the environment, then an interactive prompt.
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

assert TNS_API_KEY and TNS_BOT_ID, 'TNS credentials are not set'
print(f'TNS credentials loaded for bot id {TNS_BOT_ID} (key not shown)')
"""))
A(code(orig_src(4)))                       # TNS bulk CSV                (verbatim)
A(code(orig_src(5)))                       # classify_sn_subtype + pools (verbatim)
A(code(orig_src(6)))                       # extract_ztf_name/resolve_oid(verbatim)
A(code(orig_src(8)))                       # FLARE_STAR_NAMES            (verbatim)

A(md("""
## Preprocessing and feature extraction

These are unchanged from the previous notebook. They are defined *before*
acquisition now, because the retry loop calls the feature extractor inline: an
object only counts towards the target once it has actually produced a feature row.
That is the whole mechanism by which the final counts come out exact.

GP interpolation downsamples to at most 800 points before fitting — real TESS light
curves run ~13,000-20,000 points/sector, which was the root cause of an earlier OOM
crash (verified: 18k points killed the kernel instantly, 800 points fits in ~2s).
"""))
A(code(orig_src(13)))                                          # clean_lightcurve
A(code(orig_src(14)))                                          # gp_interpolate
A(code(orig_src(15)))                                          # extract_shape_features
A(code(orig_upto(16, 'def extract_ztf_features_checkpointed')))  # mag_to_relflux + process_ztf_object
A(code(orig_upto(17, 'def extract_flare_features_checkpointed')))  # process_tess_object

A(md("""
## Balanced acquisition — retry until the target is met

The generalisation of `build_subtype_to_target` to every class. Two loops, one
contract: *a candidate that fails at any stage is replaced, not absorbed as a loss.*

- `build_tns_class_to_target` covers all seven TNS/ALeRCE classes (the 5 SN
  subtypes plus AGN and TDE — same source, so it is close to a direct reuse).
- `build_flares_to_target` applies the same contract to how flares are actually
  sourced: TESS sectors via lightkurve, keyed by star + sector rather than by a TNS
  row. Flares are **not** forced through the TNS/ALeRCE path.

Both checkpoint to disk and resume, and both record failed candidates so a resumed
run does not re-attempt what already failed.

> **Note on superseded cells.** This replaces the previous notebook's
> `fetch_lightcurves_from_tns` and `fetch_tess_flares` driver loops and the
> `trim_to_target` helper. All three existed to cope with under-filled classes by
> accepting or trimming whatever arrived; the retry loop returns exactly the target,
> so there is nothing left to trim. The functions they *called* are reused unchanged.
"""))
A(module_cell('acquisition', 'Balanced acquisition — retry until target'))

A(code('''
# Wire the generalised loops to the real APIs. Each adapter is a thin shim over the
# existing, unchanged functions — this is where dependency injection meets reality.
import lightkurve as lk

def _resolve_fn(row):
    return resolve_oid(row)

def _detections_fn(oid):
    return alerce.query_detections(oid, format='pandas', sort='mjd')

def _process_ztf_fn(csv_path, oid, label):
    return process_ztf_object(csv_path, oid, label)

def _flare_search_fn(star_name):
    return lk.search_lightcurve(star_name, mission='TESS', author='SPOC', exptime=120)

def _flare_download_fn(entry):
    lc = entry.download().remove_nans()
    df = lc.to_pandas().reset_index()[['time', 'flux', 'flux_err']]
    return df.rename(columns={'time': 'bjd'})

def _process_tess_fn(csv_path, star_name, label):
    return process_tess_object(csv_path, star_name, label)

# How deep a candidate pool to draw per class. Attrition through cross-match +
# download + cleaning + GP has historically run 40-60%, so ~8x the target leaves
# comfortable headroom; the loop stops as soon as the target is met, so an
# over-generous pool costs nothing but a bigger shuffle.
CANDIDATE_POOL_MULTIPLIER = 8

def _pool_for(mask, target, seed=42):
    pool = tns_full[mask][[c for c in KEEP_COLS if c in tns_full.columns]].copy()
    n = min(len(pool), target * CANDIDATE_POOL_MULTIPLIER)
    return pool.sample(n, random_state=seed).reset_index(drop=True)

# Label and feature vocabulary, defined once and used by both phases.
SN_SUBTYPE_LABELS_LIST = ['SN_Ia', 'SN_Ib', 'SN_Ic', 'SN_II', 'SLSN']
FEATURES = ['peak_val', 'rise_time', 'decay_time', 'amplitude', 'color_g_r']
X_COLS = FEATURES + ['has_color']

print(f"Targets: {N_PER_SN_SUBTYPE} per SN subtype x 5 = {5 * N_PER_SN_SUBTYPE} SNe, "
      f"{N_PER_CLASS} each for AGN / TDE / stellar_flare")
'''))

A(code('''
# ---- The 5 SN subtypes + AGN + TDE, all through the same retry loop ----
tns_features, tns_manifests = {}, {}

for subtype in SN_SUBTYPES:
    pool = _pool_for(tns_full['sn_subtype'] == subtype, N_PER_SN_SUBTYPE)
    tns_manifests[subtype], tns_features[subtype] = build_tns_class_to_target(
        subtype, pool, N_PER_SN_SUBTYPE,
        base_dir=BASE_DIR, feature_dir=FEATURE_DIR,
        resolve_oid_fn=_resolve_fn, query_detections_fn=_detections_fn,
        process_fn=_process_ztf_fn, progress=tqdm)

for label, mask in [('AGN', agn_mask), ('TDE', tde_mask)]:
    pool = _pool_for(mask, N_PER_CLASS)
    tns_manifests[label], tns_features[label] = build_tns_class_to_target(
        label, pool, N_PER_CLASS,
        base_dir=BASE_DIR, feature_dir=FEATURE_DIR,
        resolve_oid_fn=_resolve_fn, query_detections_fn=_detections_fn,
        process_fn=_process_ztf_fn, progress=tqdm)
'''))

A(code('''
# ---- Stellar flares, same contract, TESS-native path ----
flare_manifest_rows, flare_features = build_flares_to_target(
    FLARE_STAR_NAMES, N_PER_CLASS,
    base_dir=BASE_DIR, feature_dir=FEATURE_DIR,
    search_fn=_flare_search_fn, download_fn=_flare_download_fn,
    process_fn=_process_tess_fn, progress=tqdm)

for label in list(tns_features) + ['stellar_flare']:
    n = len(flare_features) if label == 'stellar_flare' else len(tns_features[label])
    target = N_PER_CLASS if label in ('AGN', 'TDE', 'stellar_flare') else N_PER_SN_SUBTYPE
    print(f"  {label:<16} {n}/{target}")
'''))

A(code('''
# Persist the balanced manifests under the same names the previous notebook used, so
# a kernel restart can reload them and the sanity-check plots below have something to
# point at.
manifest_dfs = {label: pd.DataFrame(rows) for label, rows in tns_manifests.items()}
manifest_dfs['stellar_flare'] = pd.DataFrame(flare_manifest_rows)

for label, mdf in manifest_dfs.items():
    mdf.to_csv(os.path.join(BASE_DIR, f'{label.lower()}_manifest.csv'), index=False)

sn_ia_manifest = manifest_dfs['SN_Ia']
flare_manifest = manifest_dfs['stellar_flare']
print({k: len(v) for k, v in manifest_dfs.items()})
'''))

A(code('''
# ---- Combine and verify against the definition of done ----
feature_df = assemble_feature_df(
    [tns_features[s] for s in SN_SUBTYPES] +
    [tns_features['AGN'], tns_features['TDE'], flare_features])

feature_df.to_csv(os.path.join(FEATURE_DIR, 'phase3_features_balanced.csv'), index=False)

counts_ok, count_report = verify_counts(
    feature_df, N_PER_CLASS, N_PER_SN_SUBTYPE, SN_SUBTYPE_LABELS_LIST)
'''))

A(code(orig_upto(20, '# Check a ZTF object')))   # plot_gp_sanity_check (verbatim)
A(code('''
# Spot-check one ZTF light curve and one TESS light curve against their GP fits.
if len(sn_ia_manifest):
    plot_gp_sanity_check(sn_ia_manifest.iloc[0]['oid'], 'sn_ia')

if len(flare_manifest):
    plot_gp_sanity_check(flare_manifest.iloc[0]['file'], 'stellar_flares',
                         file_col_is_oid=False, value_col='flux', is_mag=False)
'''))
A(code(orig_src(22)))   # feature distributions (verbatim)

# ============================================================ PHASE 3
A(md("""
# PHASE 3 — Hierarchical Classification

### The split, stated once

There is exactly one stratified 80/20 split, computed over the full 600-row
`feature_df` and stratified on the coarse label:

- **Stage 1** trains and evaluates on it directly.
- **Stage 2** trains only on the SN-labelled rows *inside Stage 1's training
  partition*, and evaluates only on the SN-labelled rows *inside Stage 1's test
  partition*.

An SN object in Stage 1's test set therefore cannot appear in Stage 2's training
set. `assert_split_nesting` checks this and fails loudly if it is ever violated.
"""))
A(module_cell('modeling', 'Phase 3 — split, models, Stage 1 / Stage 2'))

A(code('''
feature_df = add_coarse_label(feature_df, SN_SUBTYPE_LABELS_LIST)
print(feature_df['coarse_label'].value_counts().to_string())

split = make_global_split(feature_df, X_COLS, test_size=0.2, random_state=42)

# The guard. If this ever fires, the split has been recomputed somewhere it should not be.
s2_train_idx, s2_test_idx = assert_split_nesting(split, SN_SUBTYPE_LABELS_LIST)
print(f"\\nStage 2 inherits {len(s2_train_idx)} train / {len(s2_test_idx)} test rows "
      f"from Stage 1's own partitions.")
print("Leakage check PASSED — Stage 2 rows are subsets of Stage 1's by construction.")
'''))

A(md("### Stage 1 — coarse classification (SNe / AGN / TDE / Stellar Flare)"))
A(code('''
stage1 = run_stage1(split, random_state=42)
print()
display(stage1['results'].round(3))
'''))

A(md("""
### Stage 2 — Option A: ground-truth routing (the ceiling)

Subtype the **true** SN objects directly, with no Stage 1 errors in the way. This is
the upper bound on what Stage 2 can do; it is not achievable in deployment, because
in deployment nothing tells you which objects are really SNe.
"""))
A(code('''
stage2a = run_stage2_option_a(split, SN_SUBTYPE_LABELS_LIST, random_state=42)
print()
display(stage2a['results'].round(3))
'''))

A(md("""
### Stage 2 — Option B: Stage-1-predicted SNe (the realistic path)

Subtype whatever **Stage 1 predicted** was an SN — including Stage 1's mistakes.
Three numbers, because they answer three different questions:

| metric | population | what it tells you |
| --- | --- | --- |
| End-to-end accuracy | all test objects, 8 classes | what the whole pipeline does — but flattered by the easy non-SN classes |
| Conditional accuracy | true SNe that Stage 1 routed correctly | Stage 2's own skill, isolated from routing |
| Option-A-comparable | true SN test rows, mis-route counted wrong | the **only** number directly comparable to Option A |
""" ))
A(code('''
stage2b = run_stage2_option_b(split, stage1, stage2a, SN_SUBTYPE_LABELS_LIST)
print()
display(stage2b['summary'].round(3))
'''))

A(md("### Option A vs Option B — side by side"))
A(code('''
comparison = option_ab_comparison(stage2a, stage2b)
display(comparison.round(3))

print("\\nThe honest comparison is Option A vs 'Option B comparable': same population,")
print("same scoring, the only difference being that Option B must survive Stage 1 routing.")
print("'Routing cost' is the price of the hierarchy, per model.")
comparison.to_csv(os.path.join(FEATURE_DIR, 'option_ab_comparison.csv'), index=False)
'''))

A(md("""
### How much of this is signal? — measurement resolution

Balancing the dataset removed the *bias* caused by uneven attrition. It could not
remove the *variance*: 30 objects per subtype is a small sample, and after an 80/20
split Stage 2's held-out set is about 30 objects in total, ~6 per subtype.

The cell below states the error bar explicitly, and adds a repeated stratified-CV
estimate over the Stage 2 **training** partition — many more fits, and still no
contact with Stage 1's test partition, so it adds no leakage. Read the held-out
number as the untouched evaluation and the repeated-CV number when judging whether
two models genuinely differ.
"""))
A(code('''
res_s1 = single_split_resolution(len(stage1['y_test']), len(stage1['le'].classes_))
res_s2 = single_split_resolution(len(stage2a['y_test']), len(stage2a['le'].classes_))
for tag, r in [('Stage 1', res_s1), ('Stage 2', res_s2)]:
    print(f"{tag}: n_test={r['n_test']}, accuracy resolution {r['accuracy_step']:.3f}, "
          f"worst-case SE {r['worst_case_std_error']:.3f}, "
          f"~{r['mean_test_objects_per_class']:.1f} objects/class "
          f"(per-class recall moves in steps of {r['per_class_recall_step']:.2f})")

print()
repeated_cv_s2 = repeated_cv_estimate(stage2a['X_train'], stage2a['y_train'],
                                      stage2a['le'], 'Stage 2 / A',
                                      n_splits=5, n_repeats=10)
display(repeated_cv_s2.round(3))
'''))

A(code('''
# Persist everything needed to reproduce or write up the results later.
import joblib
for tag, st in [('stage1', stage1), ('stage2a', stage2a)]:
    for name, model in st['models'].items():
        safe = name.replace(' ', '_').replace('(', '').replace(')', '')
        joblib.dump(model, os.path.join(FEATURE_DIR, f'{tag}_{safe}.pkl'))
    joblib.dump(st['scaler'], os.path.join(FEATURE_DIR, f'{tag}_scaler.pkl'))
    joblib.dump(st['le'], os.path.join(FEATURE_DIR, f'{tag}_label_encoder.pkl'))
np.save(os.path.join(FEATURE_DIR, 'split_idx_train.npy'), split['idx_train'])
np.save(os.path.join(FEATURE_DIR, 'split_idx_test.npy'), split['idx_test'])

for name, r in stage2b['per_model'].items():
    safe = name.replace(' ', '_').replace('(', '').replace(')', '')
    out = feature_df.iloc[split['idx_test']][['id', 'label', 'coarse_label']].copy()
    out['stage1_pred'] = r['coarse_pred']
    out['final_pred'] = r['final_pred']
    out['routed_to_stage2'] = r['sne_mask']
    out.to_csv(os.path.join(FEATURE_DIR, f'cascade_predictions_{safe}.csv'), index=False)
print('Saved models, split indices and cascade predictions to', FEATURE_DIR)
'''))

# ============================================================ PHASE 4
A(md("""
# PHASE 4 — Interpretation and Physical Insights

Applied to **both** stages, per the project brief: feature importance, the most
discriminative features, misclassification analysis, confusion matrices, decision
boundaries, and the physics connection.

**On importance methods.** Random Forest offers Gini importance, Logistic Regression
offers coefficients, Bagging (Trees) can average importances over its estimators —
and a bagged RBF SVM offers nothing at all. Those three are also not on comparable
scales. So **permutation importance on the held-out test set** is computed
identically for all four models, and native importances are reported alongside it
where they exist, as a cross-check rather than as the comparison.
"""))
A(module_cell('interpret', 'Phase 4 — interpretation'))

A(md("### Stage 1 interpretation"))
A(code('''
perm_s1 = permutation_importances(stage1['models'], stage1['X_test'], stage1['y_test'],
                                  X_COLS, n_repeats=30)
display(perm_s1.pivot(index='feature', columns='Model', values='perm_importance').round(4))

print('Most discriminative features, averaged across the 4 models:')
ranked_s1 = rank_discriminative_features(perm_s1)
display(ranked_s1.round(4))

plot_permutation_importance(perm_s1, 'Stage 1 — permutation importance (held-out test set)',
                            os.path.join(FEATURE_DIR, 'stage1_perm_importance.png'))
plt.show()
'''))
A(code('''
native_s1 = native_importances(stage1['models'], X_COLS)
display(native_s1.pivot(index='feature', columns='Model', values='native_importance').round(4))
print("Bagging (SVM) is absent above by design: an RBF-kernel SVM exposes neither")
print("coefficients nor impurity gains. That is exactly why permutation importance is")
print("the method used for the cross-model comparison.")
'''))
A(code('''
plot_confusions(stage1['confusions'], list(stage1['le'].classes_),
                'Stage 1 — confusion matrices (held-out test set)',
                dict(zip(stage1['results']['Model'], stage1['results']['Test accuracy'])),
                os.path.join(FEATURE_DIR, 'stage1_confusion.png'))
plt.show()

yt_s1 = stage1['le'].inverse_transform(stage1['y_test'])
yp_s1 = stage1['le'].inverse_transform(stage1['models']['Random Forest'].predict(stage1['X_test']))
print('Per-class recall (Random Forest):')
display(per_class_recall(yt_s1, yp_s1).round(3))
print('Dominant confusions:')
misclass_s1 = misclassification_table(yt_s1, yp_s1)
display(misclass_s1)
recall_s1 = per_class_recall(yt_s1, yp_s1)
'''))
A(code('''
top2_s1 = ranked_s1.index[:2].tolist()
print('Stage 1 decision boundary projected onto:', top2_s1)
plot_decision_boundaries(feature_df, split['idx_train'], split['idx_test'], 'coarse_label',
                         top2_s1, build_models, 'Stage 1',
                         os.path.join(FEATURE_DIR, 'stage1_decision_boundary.png'))
plt.show()
print("Note: the visualisation models are refit on these two features using TRAINING rows")
print("only, and the scattered points are held-out TEST objects. Fitting the visualisation")
print("on the full dataset — as an earlier version did — draws a boundary that has already")
print("seen the points it is judged against.")
'''))

A(md("### Stage 2 interpretation"))
A(code('''
perm_s2 = permutation_importances(stage2a['models'], stage2a['X_test'], stage2a['y_test'],
                                  X_COLS, n_repeats=30)
ranked_s2 = rank_discriminative_features(perm_s2)
display(ranked_s2.round(4))
plot_permutation_importance(perm_s2, 'Stage 2 — permutation importance (held-out test set)',
                            os.path.join(FEATURE_DIR, 'stage2_perm_importance.png'))
plt.show()

plot_confusions(stage2a['confusions'], list(stage2a['le'].classes_),
                'Stage 2 Option A — confusion matrices (true SN test rows)',
                dict(zip(stage2a['results']['Model'], stage2a['results']['Test accuracy'])),
                os.path.join(FEATURE_DIR, 'stage2a_confusion.png'))
plt.show()

plot_confusions(stage2b['confusions'], stage2b['all_classes'],
                'Stage 2 Option B — cascaded, all original fine-grained classes',
                None, os.path.join(FEATURE_DIR, 'stage2b_confusion.png'))
plt.show()
'''))
A(code('''
yt_s2 = stage2a['le'].inverse_transform(stage2a['y_test'])
yp_s2 = stage2a['le'].inverse_transform(stage2a['models']['Random Forest'].predict(stage2a['X_test']))
recall_s2 = per_class_recall(yt_s2, yp_s2)
misclass_s2 = misclassification_table(yt_s2, yp_s2)
print('Per-class recall (Random Forest, Option A):')
display(recall_s2.round(3))
print('Dominant subtype confusions:')
display(misclass_s2)
print(f"\\nRead these with the sample size attached: {res_s2['n_test']} test objects total,")
print(f"~{res_s2['mean_test_objects_per_class']:.0f} per subtype. A recall of 0.00 or 1.00 here")
print("is not evidence of anything.")
'''))
A(code('''
sn_rows = np.flatnonzero(feature_df['label'].isin(SN_SUBTYPE_LABELS_LIST).values)
sn_df = feature_df.iloc[sn_rows].reset_index(drop=True)
sn_positions = {orig: new for new, orig in enumerate(sn_rows)}
sn_train_local = np.array([sn_positions[i] for i in s2_train_idx])
sn_test_local = np.array([sn_positions[i] for i in s2_test_idx])

top2_s2 = ranked_s2.index[:2].tolist()
print('Stage 2 decision boundary projected onto:', top2_s2)
plot_decision_boundaries(sn_df, sn_train_local, sn_test_local, 'label', top2_s2,
                         build_models, 'Stage 2',
                         os.path.join(FEATURE_DIR, 'stage2_decision_boundary.png'))
plt.show()
'''))
A(code('''
# Data-side separability — independent of any model — and the physics table.
print('Feature separability, Stage 1 (between-class variance / within-class variance):')
display(class_separation_ranking(feature_df, 'coarse_label', FEATURES).round(3))
print('Feature separability, Stage 2 (SN subtypes only):')
display(class_separation_ranking(sn_df, 'label', FEATURES).round(3))

phys_s1 = physics_summary(feature_df, 'coarse_label', FEATURES)
phys_s2 = physics_summary(sn_df, 'label', FEATURES)
phys_s1.to_csv(os.path.join(FEATURE_DIR, 'stage1_physics_summary.csv'))
phys_s2.to_csv(os.path.join(FEATURE_DIR, 'stage2_physics_summary.csv'))
display(phys_s1)
display(phys_s2)
'''))

# ============================================================ SUMMARY
A(md("""
# Results summary

The cell below generates the summary **from the result objects computed above** —
counts, accuracies, the Option A/B gap, the permutation-importance ranking and the
dominant confusions — and attaches the physics interpretation to whatever actually
came out on top. Nothing in it is typed in by hand, so it cannot disagree with the
tables it sits under.
"""))
A(module_cell('summary', 'Results summary generator'))
A(code('''
from IPython.display import Markdown

summary_md = build_results_markdown(
    count_report, split, stage1, stage2a, stage2b, comparison,
    perm_s1, perm_s2, misclass_s1, misclass_s2, recall_s1, recall_s2,
    res_s1, res_s2, repeated_cv_s2=repeated_cv_s2, feature_df=feature_df)

with open(os.path.join(FEATURE_DIR, 'results_summary.md'), 'w') as f:
    f.write(summary_md)
print('Written to', os.path.join(FEATURE_DIR, 'results_summary.md'))

Markdown(summary_md)
'''))

nb = {
    'cells': cells,
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.11'},
        'colab': {'provenance': [], 'toc_visible': True},
    },
    'nbformat': 4,
    'nbformat_minor': 5,
}

os.makedirs(os.path.dirname(OUT_NB), exist_ok=True)
with open(OUT_NB, 'w') as f:
    json.dump(nb, f, indent=1)
    f.write('\n')

print(f'Wrote {OUT_NB}')
print(f'  {len(cells)} cells '
      f'({sum(c["cell_type"] == "code" for c in cells)} code, '
      f'{sum(c["cell_type"] == "markdown" for c in cells)} markdown)')
