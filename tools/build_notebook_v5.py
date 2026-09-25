"""
Build the v5 deliverable notebook.

Same contract as the v4 build: anything the brief said to preserve is copied VERBATIM
from the vendored v4 notebook, and anything new is inlined verbatim from `btp5/`,
which the test suite exercises. The notebook stays self-contained for Colab while the
code inside it is the code under test.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'notebooks', 'source', 'BTP_TNS_Hierarchical_v4_original.ipynb')
OUT = os.path.join(ROOT, 'notebooks', 'BTP_TNS_Hierarchical_v5.ipynb')


def md(t):
    return {'cell_type': 'markdown', 'metadata': {},
            'source': t.strip('\n').splitlines(True)}


def code(t):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None,
            'outputs': [], 'source': t.strip('\n').splitlines(True)}


def module_cell(name, header):
    with open(os.path.join(ROOT, 'btp5', f'{name}.py')) as f:
        body = f.read()
    # Strip intra-package imports; in the notebook everything shares one namespace.
    body = '\n'.join(l for l in body.split('\n')
                     if not l.startswith('from .') and not l.startswith('from btp5'))
    return code(
        f"# {header}\n"
        f"# ---------------------------------------------------------------\n"
        f"# Inlined verbatim from btp5/{name}.py, which the repository's synthetic\n"
        f"# test suite (tests5/) exercises. Edit it there and rebuild this notebook\n"
        f"# with tools/build_notebook_v5.py rather than editing it here.\n"
        f"# ---------------------------------------------------------------\n\n" + body)


with open(SRC) as f:
    v4 = json.load(f)


def v4_src(i):
    return ''.join(v4['cells'][i]['source'])


cells = []
A = cells.append

# ============================================================ title
A(md(r"""
# BTP v5 — Hierarchical Optical Transient Classification
## TNS vs ALeRCE ground-truth comparison

Two-stage hierarchical classification of ZTF optical transients:

* **Stage 1 (coarse):** AGN / SNe / TDE / Stellar Flare
* **Stage 2 (fine, SNe branch):** SN Ia / Ib / Ic / II / SLSN

run **twice** — once with **TNS spectroscopic labels** as ground truth, once with
**ALeRCE's own classifier output** as ground truth — and then compared using the
expected-vs-actual agreement framework of de Soto et al. (2024, Superphot+,
arXiv:2403.07975), which was built to solve exactly this problem.

Three models at both stages, both tracks: Logistic Regression, Random Forest,
Bagging Classifier.

---

## What v5 changes, and what it keeps

**Kept from v4, unchanged:** TNS acquisition, the Villar (2019) parametric fit, the
quality gate, the retry-until-target loop, the 22-feature vocabulary, 5-fold CV with
adaptive folds, bootstrap 95% CIs, and the Phase 6 diagnostics.

**New in v5:** the ALeRCE label track with a live-fetched taxonomy and an explicit
crosswalk; three separately-tracked pools; a host-offset feature; noise-model
augmentation; the agreement framework; and a calibration diagnostic.

---

## Three findings from building this, stated up front

**1. ALeRCE's TDE class does not live where the brief assumed.** Verified live via
`query_classifiers()`: the classic `lc_classifier` (hierarchical_rf_1.1.0) has **15
classes and no TDE at all**. TDE exists only in the 2025
`lc_classifier_BHRF_forced_phot` (2.1.0), whose 21 classes are
`[SNIa, SESN, SNII, SNIIn, SLSN, TDE, Microlensing, QSO, AGN, Blazar, YSO, CV/Nova,
LPV, EA, EB/EW, Periodic-Other, RSCVn, CEP, RRLab, RRLc, DSCT]`. Since Section 3.1 of
the brief wants ALeRCE's 2025 TDE output cross-checked, the BHRF model is the primary
label source here, with the classic model kept as a sensitivity check.

**2. The stated justification for merging SN IIn into SN II is factually wrong, though
the decision still stands.** The brief justifies the merge by saying ALeRCE's 2025
taxonomy "also has no separate IIn class in its 5-way SN output
(SNIa/SNIbc/SNII/SLSN/TDE)". The live BHRF transient branch is actually
`[SNIa, SESN, SNII, SNIIn, SLSN, TDE]` — it **does** have a separate `SNIIn`, and it
calls the stripped-envelope class `SESN`, not `SNIbc`. The merge is kept, because the
real reason for it is sound on its own (our TNS IIn count is far too small to support
a separate class), and ALeRCE's `SNIIn` is merged into `SN_II` symmetrically so both
tracks apply the same collapse. But the justification in the notebook is the
sample-size one, not the incorrect appeal to ALeRCE's taxonomy.

**Note on ordering.** The brief's Section 9 lists the ALeRCE label step (its Phase 2B)
before preprocessing and acquisition. It runs *after* acquisition here, because ALeRCE
is queried per ZTF object id and those ids only exist once `resolve_oid` has run inside
the acquisition loop. Every step below cites its corresponding phase in the brief so
the mapping stays traceable.

**3. The ALeRCE track structurally cannot contain our stellar flares.** Our flares come
from **TESS**; ALeRCE classifies **ZTF** alerts. A TESS M-dwarf has no ZTF object id
and therefore no ALeRCE probability vector — not a low-confidence one, none at all.
So the overlap pool has no `stellar_flare` class, and the ALeRCE-labelled Stage 1 is
a **3-way** problem (AGN/SNe/TDE) against the TNS track's 4-way. This is reported
throughout rather than hidden by scoring a class one side could never predict.
"""))

# ============================================================ PHASE 1
A(md('# STEP 1 — Setup and configuration  *(brief Phase 1)*'))
A(code(v4_src(2).replace(
    "!pip install -q alerce lightkurve astroquery pandas numpy matplotlib tqdm requests scikit-learn",
    "# astro-ghost added for the v5 host-offset feature (Section 4.1).\n"
    "!pip install -q alerce lightkurve astroquery pandas numpy matplotlib tqdm requests scikit-learn astro-ghost")))
# v4's credentials cell defines a _get_secret() helper and then overrides it with
# hard-coded literals, so the helper is dead code and the key ships in the notebook.
# v5 keeps the helper and actually uses it. This is a deliberate change to a
# preserved cell, made because the alternative is committing a live credential.
A(code(v4_src(3).split('TNS_MARKER =')[0].rstrip() + """


TNS_BOT_ID   = _get_secret('TNS_BOT_ID')
TNS_BOT_NAME = _get_secret('TNS_BOT_NAME')
TNS_API_KEY  = _get_secret('TNS_API_KEY')

TNS_MARKER = ('tns_marker{"tns_id":' + str(TNS_BOT_ID) +
              ',"type": "bot", "name":"' + TNS_BOT_NAME + '"}')
HEADERS = {'user-agent': TNS_MARKER}

assert TNS_API_KEY and TNS_BOT_ID, 'TNS credentials are not set'
print(f'TNS credentials loaded for bot id {TNS_BOT_ID} (key not shown)')
"""))

A(md("""
### alerce client version — checked, not assumed

v1.x and v2.x of the client differ: v2.x requires an explicit `survey='ztf'` argument
where v1.x does not accept one. The version is printed and pinned below so a future
re-run cannot silently change call semantics.
"""))
A(module_cell('config', 'v5 configuration'))
A(module_cell('alerce_labels', 'Section 3.2 — ALeRCE label track'))
A(code('''
print('alerce client version:', client_version())
print('survey kwargs for this client:', _survey_kwargs())
print()
print('Primary label source :', ALERCE_PRIMARY_CLASSIFIER, ALERCE_PRIMARY_VERSION)
print('Sensitivity check    :', ALERCE_SECONDARY_CLASSIFIER)
print('Confidence threshold :', ALERCE_CONF_THRESHOLD,
      '(matches the cut Superphot+ used for their own ALeRCE comparison set)')
'''))

# ============================================================ PHASE 2A
A(md('# STEP 2 — TNS candidate pools  *(brief Phase 2A)*'))
for i in [5, 6, 7, 8, 9, 10, 11]:
    A(code(v4_src(i)))

# ============================================================ PHASE 3
A(md('# STEP 3 — Preprocessing, the Villar fit, and the quality gate  *(brief Phase 3)*\n\nUnchanged from v4. The description below is v4\'s own.'))
A(md(v4_src(12).replace('# PHASE 3 — Preprocessing, the Villar fit, and the quality gate', '').lstrip()))
A(module_cell('photometry', 'v4 photometry, Villar fit, quality gate, features'))

# ============================================================ PHASE 4A — acquisition
A(md("""
# STEP 4 — Balanced acquisition  *(brief Phase 4)*

Run before the ALeRCE label step, which the brief's Section 9 lists earlier. The
reason is mechanical: ALeRCE is queried per **ZTF object id**, and those ids only
exist once `resolve_oid` has run inside the acquisition loop. Fetching ALeRCE labels
before acquisition would have nothing to fetch them for.
"""))
for _i in [19, 20, 21, 22, 23, 24]:
    A(code(v4_src(_i)))

# ============================================================ PHASE 2B
A(md("""
# STEP 5 — ALeRCE-labelled ground truth  *(brief Phase 2B, Section 3.2)*

**The crosswalk below is built against the class list fetched live and printed before
any label is used downstream.** This is a required step, not a diagnostic: ALeRCE's
taxonomy has already changed once (the 2025 TDE addition), the version strings in the
classifier catalogue do not always match those attached to per-object probabilities,
and the mapping from their tree onto our four classes involves real judgement — most
of all that their "stochastic" branch is **not** our AGN class. It also contains YSO
and CV/Nova, which are Galactic.

One implementation detail that is easy to get wrong and silent when you do:
`query_probabilities` returns **every version it has ever run** for an object, stacked
in one frame. Measured on ZTF18abtmbaz, grouping by `classifier_name` alone makes
`LC_classifier_ATAT_forced_phot(beta)` sum to 4.0 across four versions, and the
argmax then crosses between distributions. Every lookup here is keyed on
`(classifier_name, classifier_version)`.
"""))
A(code('''
# Fetch the live taxonomy. Do NOT hardcode this — confirm it at run time.
taxonomy = fetch_taxonomy(alerce)
'''))
A(code('''
live_classes = fetch_classes(alerce, ALERCE_PRIMARY_CLASSIFIER, ALERCE_PRIMARY_VERSION)
print()
crosswalk_table, unmapped_classes = build_crosswalk(live_classes)
'''))
A(code('''
# Per-object ALeRCE labels for every ZTF object already acquired.
# TESS rows are skipped: ALeRCE classifies ZTF alerts, so they have no entry at all.
alerce_label_rows = []
ztf_ids = [r['id'] for _, r in feature_df.iterrows() if r.get('survey') == 'ZTF']
print(f'querying ALeRCE for {len(ztf_ids)} ZTF objects '
      f'({int((feature_df["survey"] == "TESS").sum())} TESS rows skipped by construction)')

for oid in tqdm(ztf_ids, desc='ALeRCE probabilities'):
    try:
        probs = alerce.query_probabilities(oid, format='pandas', **_survey_kwargs())
    except Exception:
        continue
    lab = label_from_probabilities(probs, ALERCE_PRIMARY_CLASSIFIER,
                                   ALERCE_PRIMARY_VERSION,
                                   threshold=ALERCE_CONF_THRESHOLD)
    if lab is not None:
        lab['id'] = oid
        alerce_label_rows.append(lab)

print(f'{len(alerce_label_rows)} objects passed the '
      f'{ALERCE_CONF_THRESHOLD} confidence cut')
'''))

# ============================================================ PHASE 2C
A(md("""
# STEP 6 — Pool construction  *(brief Phase 2C, Section 3.3)*

Three pools, kept separate for the rest of the notebook. Every table and plot below
names the pool it was computed on.
"""))
A(module_cell('pools', 'Section 3.3 — the three pools'))
A(code('''
feature_df = attach_alerce_labels(feature_df, alerce_label_rows)
feature_df['has_tns_label'] = feature_df['label'].notna()
feature_df, pool_counts = build_pools(feature_df)
'''))

# ============================================================ PHASE 3B
A(md("""
# STEP 7 — Host offset  *(brief Phase 3B, Section 4.1)*

ALeRCE's own 2025 TDE update added exactly this feature type, because nuclear offset
directly targets the TDE/AGN/nuclear-transient boundary — which v4's diagnostics
identified as Stage 1's weakest link. A TDE is nuclear by definition; a core-collapse
SN is displaced into its host's disc.

Missing data follows v4's existing convention exactly: NaN plus a presence flag, the
same pattern as `color_g_r`/`has_color`. A failed host association is expected
regularly (hostless transients, crowded fields, catalogue gaps) and is not an error.
TESS flare stars are Galactic and correctly have no host galaxy at all.
"""))
A(module_cell('hostoffset', 'Section 4.1 — host offset via astro-ghost'))
A(code('''
# ra/dec come from the acquisition manifests; objects without coordinates (TESS
# flare stars) correctly get has_host_match=0 rather than a guessed host.
feature_df = add_host_features(feature_df, progress=tqdm)
'''))

# ============================================================ PHASE 3C
A(md("""
# STEP 8 — Noise-model augmentation  *(brief Phase 3C, Section 5)*

Townsend et al. 2026 (NoiZTF, arXiv:2602.13036) fitted an empirical ZTF flux
uncertainty model; their published constants are used verbatim to synthesize fainter
/ higher-redshift copies of objects in the smallest classes.

**Not done, deliberately:** timestamp jitter. That is a directly evidenced negative
result in the same paper — 0.1-day jitter measurably degraded SN Ia precision by
corrupting inferred colour evolution.

**Deferred:** full SNANA simulation. Mature templates exist for Ia/II/Ibc but not
obviously for SLSN or TDE, which are precisely the classes that need help, so the
engineering cost is badly matched to the bottleneck.

Synthetic rows are flagged and held in the training partition — a synthetic copy
shares its parent's light curve, so allowing one into a test set is leakage.
"""))
A(module_cell('augment', 'Section 5 — ZTF noise-model augmentation'))
A(code('''
if 'is_synthetic' not in feature_df.columns:
    feature_df['is_synthetic'] = False

counts_now = feature_df['label'].value_counts().to_dict()
aug_plan = plan_augmentation(counts_now, N_HEADLINE_SN_SUBTYPE, AUGMENT_CLASSES)
ok, msg = augmented_rows_are_flagged(feature_df)
print('\\n' + msg)
'''))

# ============================================================ PHASE 4
A(md("""
# STEP 9 — Model training, both tracks  *(brief Phase 4, Section 7)*

One parameterized code path, called twice. Not two copy-pasted blocks: a fix applied
to one track then cannot fail to reach the other.

Stage 2's split is **inherited** from Stage 1's partitions rather than recomputed, so
an SN object held out of Stage 1 cannot appear in Stage 2's training set.
"""))
A(module_cell('modeling', 'Phase 4 — dual-track training'))
A(code('''
tns_track = run_track(feature_df, [c for c in FEATURE_COLS if c in feature_df.columns],
                      'coarse_label', 'label', 'TNS')

alerce_pool = get_pool(feature_df, POOL_ALERCE)
alerce_track = run_track(alerce_pool, [c for c in FEATURE_COLS if c in alerce_pool.columns],
                         'alerce_coarse', 'alerce_fine', 'ALeRCE')
'''))
A(code('''
summary_table = track_summary([tns_track, alerce_track])
display(summary_table.round(3))
print('\\nStage 1 class counts differ by track by construction — the ALeRCE side has '
      'no stellar_flare, because TESS objects carry no ZTF alerts to classify.')
'''))

# ============================================================ PHASE 5
A(md("""
# STEP 10 — The comparison  *(brief Phase 5, Section 6)*

Putting the two tracks' accuracies side by side as bare numbers would be confounded by
sample size, label reliability and class definition all at once. Instead, after
de Soto et al. (2024):

$$A_{\\rm expected} = P_{\\rm ours}^{\\rm T}\\cdot C_{\\rm ALeRCE}$$

with $P$ the purity (precision) matrix and $C$ the completeness (recall) matrix. That
is what agreement would look like if the two pipelines' errors were independent.
Comparing the **actual** agreement matrix against it is the informative move — and if
actual exceeds expected, that indicates **shared underlying signal**, not independent
verification.
"""))
A(module_cell('agreement', 'Section 6 — agreement framework'))
A(code('''
overlap = get_pool(feature_df, POOL_OVERLAP)
comparable_classes = sorted(set(overlap['coarse_label'].dropna())
                            & set(overlap['alerce_coarse'].dropna()))
print(f'overlap pool: n={len(overlap)}, comparable Stage 1 classes: {comparable_classes}')
stage1_class_set(overlap, 'alerce_coarse')
'''))
A(code('''
# Expected vs actual agreement, per model, on the overlap pool.
s1 = tns_track['stage1']; sp = s1['split']; le = sp['le']
test_rows = sp['df'].iloc[sp['idx_test']]
agreement_results = {}

for model_name in HEADLINE_MODELS:
    pred_ours = le.inverse_transform(s1['models'][model_name].predict(sp['X_test']))
    theirs_all = test_rows['alerce_coarse'].values
    truth_all = test_rows['coarse_label'].values
    m = (pd.notna(theirs_all) & np.isin(pred_ours, comparable_classes)
         & np.isin(pd.Series(theirs_all).fillna('').values, comparable_classes))
    if m.sum() < 5:
        print(f'{model_name}: only {int(m.sum())} overlap test objects — skipped')
        continue
    P = purity_matrix(truth_all[m], pred_ours[m], comparable_classes)
    C = completeness_matrix(truth_all[m], theirs_all[m], comparable_classes)
    res = compare_expected_actual(pred_ours[m], theirs_all[m], P, C, comparable_classes)
    agreement_results[model_name] = res

    print(f'--- {model_name} ---')
    print('expected (independent errors):')
    display(res['expected_matrix'].round(3))
    print('actual:')
    display(res['actual_matrix'].round(3))
    print(interpret_agreement(res, model_name, POOL_OVERLAP))
    print()
'''))
A(code('''
# Section 6.2 asks for the spectroscopic / non-spectroscopic split separately.
# Here every overlap object is TNS-confirmed by construction, so the comparison is
# reported on the ALeRCE-confident pool split by whether a TNS label also exists.
alerce_only = feature_df[feature_df['in_alerce_pool'] & ~feature_df['in_tns_pool']]
print(f'ALeRCE-confident WITH a TNS label : {int(feature_df["in_overlap_pool"].sum())}')
print(f'ALeRCE-confident WITHOUT one      : {len(alerce_only)}')
if len(alerce_only) == 0:
    print('\\nNo non-TNS-confirmed objects: acquisition is TNS-driven, so every object '
          'in the ALeRCE pool also carries a TNS label. The spectroscopic vs '
          'photometric-only contrast of de Soto et al. Section 6.2 therefore cannot '
          'be reproduced from this sample — reported rather than approximated.')
'''))
A(code('''
print('Section 6.4 — external corroboration:')
display(EXTERNAL_RESULTS)
print(TDE_SLSN_STATEMENT)
print()
print(TNS_LABEL_NOISE_CAVEAT)
'''))

# ============================================================ PHASE 6
A(md('''
# STEP 11 — v4 diagnostics  *(brief Phase 6)*

These are v4's diagnostics, preserved. They expect three names that v4 created in its
own modelling block: `X_COLS`, `feature_df_full`, and the two extra estimators. The
bridging cell below supplies them from the v5 objects instead of importing v4's whole
modelling scaffolding, which would otherwise overwrite the v5 three-model headline
builder with v4's five-model one.
'''))
A(code('''
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score

if 'coarse_label' not in feature_df.columns:
    feature_df['coarse_label'] = feature_df['label'].apply(
        lambda l: 'SNe' if l in SN_SUBTYPES else l)

X_COLS = [c for c in FEATURE_COLS if c in feature_df.columns]

# v4 kept every acquired object here, so the ablation / learning curve / class-scheme
# comparison can use the surplus above the balanced headline set.
feature_df_full = feature_df.copy()

print(f'{len(X_COLS)} feature columns available to the diagnostics')
print(f'feature_df_full: {feature_df_full.shape[0]} rows')
print('Five-model set available for cross-checking via build_diagnostic_models(); '
      'the headline tables stay on the three the brief specified.')
'''))
# v4 cell 32 (the ablation) evaluates 'catalogue columns ALONE'. If no BTS-sourced
# object survives acquisition, those columns are entirely NaN and
# HistGradientBoosting's binner raises 'window shape cannot be larger than input
# array shape' — crashing the notebook mid-run. One guard is added at the top of
# evaluate_feature_set to return NaNs for a degenerate feature set instead. This is
# the only v4 cell whose behaviour v5 changes, and only in the failure path.
_abl = v4_src(32).replace(
    "    d = feature_df_full if classes is None else feature_df_full[feature_df_full['label'].isin(classes)]",
    "    d = feature_df_full if classes is None else feature_df_full[feature_df_full['label'].isin(classes)]\n"
    "    # v5 guard: a feature group that is entirely missing (e.g. the BTS catalogue\n"
    "    # columns when no BTS-sourced object survived acquisition) cannot be binned,\n"
    "    # and previously crashed the whole cell. Report it as unavailable instead.\n"
    "    usable = [c for c in cols if c in d.columns and d[c].notna().any()\n"
    "              and d[c].nunique(dropna=True) > 1]\n"
    "    if len(usable) == 0:\n"
    "        print(f'  (skipped: no usable columns in this group)')\n"
    "        return dict(acc=np.nan, acc_sd=np.nan, bacc=np.nan, f1=np.nan)\n"
    "    cols = usable")
assert 'v5 guard' in _abl
A(code(_abl))
for i in [33, 34, 35, 36, 37]:
    A(code(v4_src(i)))

# ============================================================ PHASE 6B
A(md("""
# STEP 12 — Calibration  *(brief Phase 6B, Section 8)*

v4 already asks whether redshift helps accuracy. This asks the sharper question from
de Soto et al. (2024, Fig. 11): does redshift make the model **more confident without
making it more correct**? Reported either way.
"""))
A(module_cell('calibration', 'Section 8 — calibration diagnostic'))
A(code('''
s2 = tns_track['stage2']
if s2 is not None:
    sp2 = s2['split']
    z_idx = [sp2['feature_cols'].index(c) for c in ['redshift', 'peak_abs_mag']
             if c in sp2['feature_cols']]
    keep = [i for i in range(len(sp2['feature_cols'])) if i not in z_idx]
    curves = {}
    for name in HEADLINE_MODELS:
        with_z = calibration_for_model(s2['models'][name], sp2['X_test'], sp2['y_test'])
        m2 = build_models()[name]
        m2.fit(sp2['X_train'][:, keep], sp2['y_train'])
        without_z = calibration_for_model(m2, sp2['X_test'][:, keep], sp2['y_test'])
        curves[f'{name} (with z)'] = with_z
        curves[f'{name} (no z)'] = without_z
        print(compare_with_without_redshift(with_z, without_z, name))
        print()
    plot_calibration(curves, 'Stage 2 calibration, with and without redshift')
    plt.show()
else:
    print('Stage 2 unavailable — calibration skipped.')
'''))

# ============================================================ PHASE 7
A(md("""
# STEP 13 — Written summary  *(brief Phase 7)*

Everything below is generated from the objects computed above, so it cannot disagree
with the tables it sits under.
"""))
A(code('''
lines = []
lines.append('## Final sample and pools\\n')
lines.append(pool_counts.to_string(index=False))
lines.append('\\n\\n## Headline results\\n')
lines.append(summary_table.round(3).to_string(index=False))

lines.append('\\n\\n## Agreement (Section 6)\\n')
for name, res in agreement_results.items():
    lines.append(interpret_agreement(res, name, POOL_OVERLAP) + '\\n')

lines.append('\\n## Deferred items, stated as limitations with their reasons\\n')
lines.append(
    '* **AllWISE infrared colour (Section 4.2) — not built.** ALeRCE uses it, but it '
    'needs a separate cross-match query and lacks the direct evidentiary link to our '
    'specific weakest boundary that host offset has. A plausible avenue for closing '
    'part of the remaining Stage-1 TDE/AGN confusion; left for future work.\\n')
lines.append(
    '* **Colour variability over time (Section 4.3) — not built.** Would require '
    're-architecting feature extraction to track colour across epochs rather than at a '
    'single peak epoch. This is specifically why ALeRCE\\'s TDE update works: TDEs stay '
    'blue while SNe redden. Real and evidenced, but a bigger lift than 4.1/4.2.\\n')
lines.append(
    '* **ParSNIP-style non-parametric features (Section 4.4) — explicitly not built.** '
    'This is the single most strongly evidenced accuracy lever in the research log: '
    'three independent groups, two surveys, ParSNIP/SuperRAENN beating Villar-family '
    'parametric fits by 8-10 F1 points every time it has been tested. It is not built '
    'because it requires training a variational autoencoder — a substantially heavier '
    'engineering task than anything else here — and the evidence comes from datasets '
    '15-1000x larger than we can acquire (6,061 for Superphot+ and 6,123 for ParSNIP, '
    'against our low hundreds).\\n')
lines.append(
    '* **SNANA synthetic simulation (Section 5) — deferred.** Aleo et al. 2023 / '
    'YSE-DR1 use ~60,000 objects per class. Mature SN templates exist for Ia/II/Ibc '
    'but not obviously for SLSN or TDE — exactly the classes that need help — so the '
    'engineering cost is not well matched to our actual bottleneck.\\n')

lines.append('\\n## External corroboration\\n')
lines.append(EXTERNAL_RESULTS.to_string(index=False))
lines.append('\\n\\n' + TDE_SLSN_STATEMENT)
lines.append('\\n\\n## Ground-truth caveat\\n')
lines.append(TNS_LABEL_NOISE_CAVEAT)

lines.append('\\n\\n## Structural limits found while building v5\\n')
lines.append(
    '* ALeRCE\\'s TDE class exists only in lc_classifier_BHRF_forced_phot 2.1.0, not in '
    'the classic lc_classifier, which has no TDE at all. Verified live.\\n')
lines.append(
    '* The BHRF transient branch is [SNIa, SESN, SNII, SNIIn, SLSN, TDE] — it DOES '
    'separate SNIIn and calls the stripped-envelope class SESN. The SN IIn merge in '
    'this notebook is justified by our own sample size, not by ALeRCE\\'s taxonomy.\\n')
lines.append(
    '* The ALeRCE track cannot contain stellar flares: our flares are TESS objects and '
    'ALeRCE classifies ZTF alerts. The ALeRCE-labelled Stage 1 is therefore 3-way '
    'against the TNS track\\'s 4-way, and the overlap pool has no flare class.\\n')
lines.append(
    '* Ib and Ic are kept separate on the TNS side and merged on the ALeRCE side, '
    'because ALeRCE never emits them separately. Their Stage 2 is one class coarser, '
    'stated rather than papered over — and it connects directly to the published '
    'finding that Ib/Ic is the least separable pair in the scheme.\\n')

summary_md = '\\n'.join(lines)
with open(os.path.join(FEATURE_DIR, 'v5_results_summary.md'), 'w') as f:
    f.write(summary_md)
print('written to', os.path.join(FEATURE_DIR, 'v5_results_summary.md'))

from IPython.display import Markdown
Markdown(summary_md)
'''))

nb = {'cells': cells,
      'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python',
                                  'name': 'python3'},
                   'language_info': {'name': 'python', 'version': '3.11'},
                   'colab': {'provenance': [], 'toc_visible': True}},
      'nbformat': 4, 'nbformat_minor': 5}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w') as f:
    json.dump(nb, f, indent=1)
    f.write('\n')
print(f'Wrote {OUT}')
print(f'  {len(cells)} cells ({sum(c["cell_type"]=="code" for c in cells)} code, '
      f'{sum(c["cell_type"]=="markdown" for c in cells)} markdown)')
