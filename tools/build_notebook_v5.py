"""
Build the v5 deliverable notebook.

Contract, unchanged from the first v5 build:
  * anything the brief says to preserve is copied VERBATIM from the vendored v4
    notebook (notebooks/source/BTP_TNS_Hierarchical_v4_original.ipynb);
  * anything new is inlined verbatim from btp5/, which tests5/ exercises;
  * the notebook is self-contained for Colab.

Every deliberate change to a preserved v4 cell is made here, in one place, with the
reason next to it: the credentials cell, the ablation guard, and one printed string
in the permutation-importance cell.
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'notebooks', 'source', 'BTP_TNS_Hierarchical_v4_original.ipynb')
OUT = os.path.join(ROOT, 'notebooks', 'BTP_TNS_Hierarchical_v5.ipynb')


def md(t):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': t.strip('\n').splitlines(True)}


def code(t):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
            'source': t.strip('\n').splitlines(True)}


def module_cell(name, header):
    with open(os.path.join(ROOT, 'btp5', f'{name}.py')) as f:
        body = f.read()
    lines = [l for l in body.split('\n')
             if not l.startswith('from .') and not l.startswith('from btp5')]
    body = '\n'.join(lines)
    assert 'from .' not in body, f'{name}: a relative import survived inlining'
    return code(f"# {header}\n"
                f"# ---------------------------------------------------------------\n"
                f"# Inlined verbatim from btp5/{name}.py, which the repository's test suite\n"
                f"# (tests5/) exercises. Edit it there and rebuild this notebook with\n"
                f"# tools/build_notebook_v5.py rather than editing it here.\n"
                f"# ---------------------------------------------------------------\n\n" + body)


with open(SRC) as f:
    v4 = json.load(f)


def v4_src(i):
    return ''.join(v4['cells'][i]['source'])


# ------------------------------------------------------------------ v4 cells that change
# 1. Credentials: v4 defines a _get_secret helper and then OVERRIDES it with literal
#    values, so the helper is dead code and the key ships in the notebook. v5 uses it.
CREDENTIALS = v4_src(3).split('TNS_MARKER =')[0].rstrip() + '''


TNS_BOT_ID   = _get_secret('TNS_BOT_ID')
TNS_BOT_NAME = _get_secret('TNS_BOT_NAME')
TNS_API_KEY  = _get_secret('TNS_API_KEY')

TNS_MARKER = ('tns_marker{"tns_id":' + str(TNS_BOT_ID) +
              ',"type": "bot", "name":"' + TNS_BOT_NAME + '"}')
HEADERS = {'user-agent': TNS_MARKER}

assert TNS_API_KEY and TNS_BOT_ID, 'TNS credentials are not set'
print(f'TNS credentials loaded for bot id {TNS_BOT_ID} (key not shown)')
'''
assert 'TNS_API_KEY  = _get_secret' in CREDENTIALS

# 2. Ablation guard: with no BTS-sourced survivor the catalogue feature group is all-NaN
#    and HistGradientBoosting's binner raises mid-notebook. Only the failure path changes.
_line = ("    d = feature_df_full if classes is None else "
         "feature_df_full[feature_df_full['label'].isin(classes)]")
ABLATION = v4_src(32).replace(_line, _line + '''
    # v5 guard: a feature group that is entirely missing (e.g. the BTS catalogue
    # columns when no BTS-sourced object survived acquisition) cannot be binned,
    # and previously crashed the whole cell. Report it as unavailable instead.
    usable = [c for c in cols if c in d.columns and d[c].notna().any()
              and d[c].nunique(dropna=True) > 1]
    if len(usable) == 0:
        print('  (skipped: no usable columns in this group)')
        return dict(acc=np.nan, acc_sd=np.nan, bacc=np.nan, f1=np.nan)
    cols = usable''')
assert 'v5 guard' in ABLATION

# 3. Permutation importance prints "the five models"; v5's headline set is three.
PERM = v4_src(41).replace('averaged across the five models',
                          'averaged across the headline models')
assert PERM != v4_src(41)

# The v4 diagnostics, in the exact text shown in the notebook. The ALeRCE pass
# re-executes THIS list, so the two passes cannot drift apart.
DIAG_ALL = [(32, ABLATION), (33, v4_src(33)), (34, v4_src(34)), (35, v4_src(35)),
            (36, v4_src(36)), (37, v4_src(37)), (39, v4_src(39)), (40, v4_src(40)),
            (41, PERM), (42, v4_src(42))]
TNS_ONLY = {35, 37}      # see btp5/diagnostics.py TNS_SCHEME_ONLY
DIAG_ALERCE = [(i, s) for i, s in DIAG_ALL if i not in TNS_ONLY]

cells = []
A = cells.append

# ================================================================== title
A(md(r"""
# BTP v5 — Hierarchical Optical Transient Classification
## TNS vs ALeRCE ground truth, compared with the de Soto et al. (2024) agreement framework

Two-stage hierarchical classification of ZTF optical transients —
**Stage 1:** AGN / SNe / TDE / Stellar Flare, **Stage 2 (SNe):** Ia / Ib / Ic / II / SLSN —
run twice, once against **TNS spectroscopic labels** and once against **ALeRCE's own
classifier output**, then compared with the expected-vs-actual agreement framework of
de Soto et al. (2024, Superphot+, arXiv:2403.07975). Three models at both stages, both
tracks: Logistic Regression, Random Forest, Bagging Classifier.

**Preserved from v4:** TNS/BTS acquisition, the Villar (2019) fit, the quality gate, the
retry-until-target loop, the feature vocabulary, CV / bootstrap CIs, all diagnostics.

**New:** a live-verified ALeRCE crosswalk; an ALeRCE-native acquisition so the three
pools are genuinely distinct; host offset; Townsend et al. (2026) augmentation; agreement
at both stages; the diagnostics on both tracks; calibration.

---

### Findings from checking ALeRCE live, stated before anything depends on them

1. **TDE lives only in the 2025 classifier.** `lc_classifier` (hierarchical_rf_1.1.0) has
   15 classes and **no TDE**. `lc_classifier_BHRF_forced_phot` 2.1.0 has 21, including
   TDE. Section 3.1 wants ALeRCE's TDE output cross-checked, so BHRF is the primary source.
2. **Probabilities are stacked across versions.** `query_probabilities` returns every
   version it has run for an object in one frame (one classifier summed to 4.0 across four
   versions on ZTF18abtmbaz). Every lookup is keyed on (name, version).
3. **The brief's stated reason for merging SN IIn is not what the live API shows.** The
   brief says ALeRCE's 2025 SN output has no separate IIn. The live BHRF transient branch
   is `[SNIa, SESN, SNII, SNIIn, SLSN, TDE]` — it **does** separate IIn, and names the
   stripped-envelope class `SESN`. The merge stands on its own reason (sample size), and
   ALeRCE's SNIIn is merged symmetrically so both tracks collapse identically.
4. **ALeRCE's "stochastic" branch is not our AGN class**, as the brief warned: it also
   holds YSO and CV/Nova, which are Galactic. Only QSO/AGN/Blazar map to AGN.
5. **ALeRCE returns the least confident objects first.** At p ≥ 0.5, `query_objects`
   answers 0.500, 0.500, 0.501… and large classes outgrow a page (AGN+QSO+Blazar: 62,645
   objects). The native pool is a uniform random sample over the whole set above the cut.
6. **ALeRCE's confident set is tiny for the rare classes:** 33 SESN, 43 SLSN and 23 TDE in
   its entire BHRF catalogue at p ≥ 0.5 (checked 2026-09-29). That bounds pool 2.
7. **The ALeRCE track cannot contain our stellar flares.** They are TESS objects; ALeRCE
   classifies ZTF alerts. The overlap pool has no flare class.

**On ordering.** The brief's Section 9 puts the ALeRCE label step (its Phase 2B) before
preprocessing and acquisition. It runs after acquisition here: ALeRCE is queried per ZTF
object id, and those ids only exist once acquisition has resolved them. Every step cites
its phase in the brief.
"""))

# ================================================================== STEP 1
A(md('# STEP 1 — Setup and configuration  *(brief Phase 1)*'))
A(code(v4_src(2).replace(
    "!pip install -q alerce lightkurve astroquery pandas numpy matplotlib tqdm requests scikit-learn",
    "# v5 adds astro-ghost (host offset, Section 4.1) and sncosmo (template\n"
    "# K-corrections for augmentation, Section 5).\n"
    "!pip install -q alerce lightkurve astroquery pandas numpy matplotlib tqdm requests "
    "scikit-learn astro-ghost sncosmo tabulate")))
A(md("""
### TNS credentials

v4 defined a `_get_secret()` helper and then overrode it with the bot id and API key as
literal strings, so the helper never ran and the key travelled with every copy of the
notebook. This cell uses the helper. Put `TNS_BOT_ID`, `TNS_BOT_NAME` and `TNS_API_KEY`
in Colab's Secrets panel (key icon, left sidebar).
"""))
A(code(CREDENTIALS))
A(md("""
### alerce client version — checked, not assumed

v1.x and v2.x of the client differ: v2.x needs an explicit `survey='ztf'`, v1.x rejects
it. The version is printed so a re-run cannot silently change call semantics.
"""))
A(module_cell('config', 'v5 configuration'))
A(module_cell('alerce_labels', 'Section 3.2 — ALeRCE label track'))
A(code('''
print('alerce client version:', client_version())
print('survey kwargs for this client:', _survey_kwargs())
print('Primary label source :', ALERCE_PRIMARY_CLASSIFIER, ALERCE_PRIMARY_VERSION)
print('Sensitivity check    :', ALERCE_SECONDARY_CLASSIFIER)
print('Confidence threshold :', ALERCE_CONF_THRESHOLD,
      '(the cut Superphot+ used for their own ALeRCE-labelled comparison set)')
'''))

# ================================================================== STEP 2
A(md('# STEP 2 — TNS candidate pools  *(brief Phase 2A)*'))
A(code(v4_src(5)))
A(md("""
### Why SN IIn stays merged into SN II

`classify_sn_subtype` (below, unchanged from v4) folds IIn into SN_II. The brief resolves
this and it is kept. **The justification is sample size**: TNS lists too few spectroscopic
IIn to support a class once the quality gate has run.

The brief also cites ALeRCE's 2025 taxonomy as having no separate IIn. Checked live, that
is not so: `lc_classifier_BHRF_forced_phot` 2.1.0's transient branch is
`[SNIa, SESN, SNII, SNIIn, SLSN, TDE]`. So the merge is not "what ALeRCE also does" — and
to keep the two tracks comparable, ALeRCE's `SNIIn` is merged into `SN_II` as well
(`FINE_CROSSWALK`, Step 1).

`SN Ib/c` (undecided between Ib and Ic) is excluded, as in v4: training on it teaches the
model that the two overlap, which is the confusion Stage 2 is trying to resolve.
"""))
A(code(v4_src(6)))
A(md("""
### TDE: now cross-checkable against ALeRCE  *(brief Section 3.1)*

The TDE pool below is v4's TNS + BTS union, unchanged. What is new is that ALeRCE's 2025
classifier emits a TDE class, so every TNS TDE can be checked against it — done in
Step 10, as the direct analogue of Superphot+ Table 3.
"""))
for i in (7, 8, 9, 10, 11):
    A(code(v4_src(i)))

# ================================================================== STEP 3
A(md('# STEP 3 — Preprocessing, the Villar fit, and the quality gate  *(brief Phase 3)*'))
A(md(v4_src(12).replace('# PHASE 3 — Preprocessing, the Villar fit, and the quality gate', '').lstrip()))
A(module_cell('photometry', 'v4 photometry, Villar fit, quality gate, features (verbatim)'))
A(md("""
### Finding the light curves again

v4's feature rows and acquisition manifests carry **no sky coordinates**. Host offset
(Step 7) needs them and augmentation (Step 8) needs to re-read each light curve. v4's
detection CSVs do keep per-detection `ra`/`dec`, so both are recovered from those. The
first v5 build missed this: its tests supplied coordinates that real v4 rows do not
have, so in a live run every object would have come back `has_host_match = 0`.
"""))
A(module_cell('lightcurves', 'light-curve index and coordinate recovery'))

# ================================================================== STEP 4
A(md("""
# STEP 4 — Balanced acquisition  *(brief Phase 4)*

v4's retry-until-target loop, unchanged.
"""))
for i in (19, 20, 21, 22, 23, 24):
    A(code(v4_src(i)))
A(code('''
# Columns v4 created later, in its modelling cells, which v5 does not include. The first
# v5 build omitted `coarse_label` here and would have failed at its first training call.
feature_df['coarse_label']   = feature_df['label'].apply(lambda l: 'SNe' if l in SN_SUBTYPES else l)
feature_df['is_synthetic']   = False
feature_df['passes_v4_gate'] = True          # every row here cleared the full v4 gate
feature_df['source']         = np.where(feature_df['survey'] == 'TESS', 'TESS (curated list)', 'TNS/BTS')
# "TNS-confirmed" means a TNS spectroscopic label. TESS flares come from a curated star
# list, not TNS: they train the TNS track (as in v4) but are not TNS-confirmed objects.
feature_df['has_tns_label']  = feature_df['label'].notna() & feature_df['survey'].eq('ZTF')
print(feature_df['source'].value_counts().to_string())
'''))

# ================================================================== STEP 5
A(md("""
# STEP 5 — ALeRCE-labelled ground truth  *(brief Phase 2B, Section 3.2)*

**The crosswalk is built against the class list fetched live, and printed, before any
ALeRCE label is used.** Mapping decisions:

* **AGN** ← `AGN`, `QSO`, `Blazar` — all accretion onto a supermassive black hole.
* **Not AGN** — ALeRCE's stochastic branch also holds `YSO` (young stellar objects) and
  `CV/Nova` (accreting white-dwarf binaries). Both are Galactic. They map to
  `stellar_flare` as the nearest analogue, which is an approximation: they are not the
  UV-Ceti M-dwarf population of our TESS sample.
* **SNe** ← `SNIa`, `SESN`, `SNII`, `SNIIn`, `SLSN`; for Stage 2, `SESN` → `SN_Ibc`
  (ALeRCE never separates Ib from Ic) and `SNIIn` → `SN_II`.
* **TDE** ← `TDE`.
* **Excluded, never reassigned:** every periodic class and `Microlensing`.
"""))
A(code('''
taxonomy = fetch_taxonomy(alerce)
'''))
A(code('''
live_classes = fetch_classes(alerce, ALERCE_PRIMARY_CLASSIFIER, ALERCE_PRIMARY_VERSION)
print()
crosswalk_table, unmapped_classes = build_crosswalk(live_classes)
'''))
A(md("""
### Pool 2 sourced from ALeRCE itself

Section 3.3 defines the ALeRCE-confident pool as objects ALeRCE classifies confidently,
**independent of whether they also have a TNS label**. So it has to be sourced from
ALeRCE, not only by labelling objects TNS already supplied — otherwise it is a subset of
the TNS pool and the spectroscopic-vs-photometric comparison of Section 6.2 is impossible
(the first v5 build had exactly this problem).

* The sample is **uniform over every object above the cut**, drawn by random rank across
  ALeRCE's pages, because ALeRCE returns the least confident objects first.
* Objects already acquired from TNS are excluded here (they are already in the overlap).
* Each is cross-matched to TNS on its ZTF id and labelled with v4's own rules; one with a
  TNS type counts as TNS-confirmed, one without is photometric-only.
* **Only v4's `require_redshift` is relaxed**, and only inside this acquisition. An object
  nobody took a spectrum of has no spectroscopic redshift, so under v4's gate the
  photometric-only subset would be empty by construction. Every other threshold stands.
  These objects have NaN redshift and peak absolute magnitude — like de Soto et al.'s
  photometric-only subset — and that is one reason to expect lower agreement there.
"""))
A(module_cell('alerce_native', 'Section 3.3 pool 2 — ALeRCE-native acquisition'))
A(code('''
native_rng = np.random.default_rng(RANDOM_STATE)
native_pools, native_summary = query_native_candidates(
    alerce, ALERCE_PRIMARY_CLASSIFIER, live_classes, N_ALERCE_NATIVE_PER_GROUP,
    ALERCE_CONF_THRESHOLD, native_rng, exclude_oids=feature_df['id'].astype(str))
'''))
A(code('''
# Acquire through v4's own loop, with ONLY the redshift requirement relaxed.
native_features, native_cands = [], []
with redshift_optional(QG):
    for group, cands in native_pools.items():
        cands = crossmatch_tns(cands, tns_full, classify_sn_subtype)
        native_cands.append(cands)
        _, feats = build_class_to_target(f'ALERCE_{group}', to_acquisition_pool(cands),
                                         N_ALERCE_NATIVE_PER_GROUP,
                                         out_subdir=f'alerce_{group.lower()}')
        native_features.extend(feats)
print('require_redshift restored to', QG['require_redshift'])

native_cands = pd.concat(native_cands, ignore_index=True) if native_cands else pd.DataFrame()
native_df = finalize_native_rows(native_features, native_cands, SN_SUBTYPES)
if len(native_df):
    print(f"{len(native_df)} ALeRCE-native objects acquired: "
          f"{int(native_df['has_tns_label'].sum())} also TNS-confirmed, "
          f"{int((~native_df['has_tns_label']).sum())} photometric-only")
    print(native_df.groupby('native_group').size().to_string())
feature_df = pd.concat([feature_df, native_df], ignore_index=True)
'''))
A(md("""
### ALeRCE labels for every ZTF object

Both the thresholded label (for the ALeRCE-confident pool) and ALeRCE's raw top-1 class
(for the Section 3.1 TDE cross-check, which must see what ALeRCE thinks of every TNS TDE,
not only the confident ones). TESS rows are skipped: they have no ZTF alerts.
"""))
A(code('''
alerce_label_rows = []
ztf_ids = feature_df.loc[feature_df['survey'] == 'ZTF', 'id'].astype(str).unique()
print(f'querying ALeRCE for {len(ztf_ids)} ZTF objects '
      f'({int((feature_df["survey"] == "TESS").sum())} TESS rows skipped by construction)')
for oid in tqdm(ztf_ids, desc='ALeRCE probabilities'):
    try:
        probs = alerce.query_probabilities(oid, format='pandas', **_survey_kwargs())
    except Exception:
        continue
    d = describe_probabilities(probs, ALERCE_PRIMARY_CLASSIFIER, ALERCE_PRIMARY_VERSION,
                               threshold=ALERCE_CONF_THRESHOLD)
    if d is not None:
        d['id'] = oid
        alerce_label_rows.append(d)
n_conf = sum(r['alerce_coarse'] is not None for r in alerce_label_rows)
print(f'{len(alerce_label_rows)} objects have an ALeRCE classification; '
      f'{n_conf} pass the {ALERCE_CONF_THRESHOLD} cut and map into our scheme')
'''))

# ================================================================== STEP 6
A(md("""
# STEP 6 — Pool construction  *(brief Phase 2C, Section 3.3)*

Three pools, tagged on every row and never collapsed. Every result below names its pool.
"""))
A(module_cell('pools', 'Section 3.3 — the three pools'))
A(code('''
feature_df = attach_alerce_labels(feature_df, alerce_label_rows)
feature_df, pool_counts = build_pools(feature_df)
'''))

# ================================================================== STEP 7
A(md("""
# STEP 7 — Host offset  *(brief Phase 3B, Section 4.1)*

ALeRCE's 2025 TDE update added exactly this feature type, because nuclear offset targets
the TDE / AGN / nuclear-transient boundary — Stage 1's weakest link in v4. A TDE is
nuclear by definition; a core-collapse SN sits in its host's disc.

Missing data follows v4's existing convention: NaN plus a presence flag, as with
colour / `has_color`. TESS flare stars are Galactic and correctly have no host galaxy.
"""))
A(module_cell('hostoffset', 'Section 4.1 — host offset via astro-ghost'))
A(code('''
lc_index = index_lightcurves(BASE_DIR)
feature_df = attach_coordinates(feature_df, lc_index)
feature_df = add_host_features(feature_df, progress=tqdm)
'''))

# ================================================================== STEP 8
A(md("""
# STEP 8 — Noise-model augmentation  *(brief Phase 3C, Section 5)*

Townsend et al. (2026, NoiZTF, arXiv:2602.13036), implemented as their Sects. 3.1–3.3
describe it — checked against the paper, and different in five places from the first v5
build, which had followed a paraphrase:

| | Townsend et al. | first v5 build |
|---|---|---|
| flux units | Jy scaled to AB → **µJy, ZP 23.9** | ZP 27.5 (noise ~25x too small) |
| observation dates | **unchanged** | stretched by (1+z) |
| z_sim | **[z_true, z_true + 0.1]**, p ∝ z² | z_true × 1.2–2.5 |
| error | original σ **rescaled by their Eq. 4** | fresh draw |
| flux | D·f + N(0, σ_z²) | noise on a fresh σ |

The unit is an **inference**: the paper says "janskys scaled to be consistent with the AB
magnitude system" without printing a zero point, but its flux bins (f < 400,
800–1200, > 1600) only make sense in µJy.

* **K-correction:** sncosmo templates (SALT2 at Townsend's [x1, c] = [1.0, 0.2] for Ia;
  Nugent templates for Ib/c and II); the Hogg et al. (2002) form for SLSN, as Townsend
  do. TDEs are outside Townsend's SN-only scope, so the Hogg form applied to them is an
  **extrapolation**. Townsend's own template list (their Table 7) was not verified.
* **Not done: timestamp jitter** — a directly evidenced negative result in the same paper
  (0.1 d jitter degraded SN Ia precision by corrupting colour evolution).
* **Departure:** copies keep only detections at SNR ≥ 5, because this pipeline trains on
  the alert stream, which only ever contains ≥ 5σ detections. Each copy must then pass
  v4's own fit and quality gate, like a real object.
* **Leakage:** Townsend split before augmenting so copies never straddle train/test. Here
  every copy carries `parent_id` and the split drops any copy whose parent is held out.
* **Deferred: full SNANA simulation** (Aleo et al. 2023 / YSE-DR1, ~60,000 per class).
  Mature templates exist for Ia/II/Ibc but not obviously for SLSN or TDE — exactly the
  classes that need help — so the cost is badly matched to the bottleneck.
"""))
A(module_cell('augment', 'Section 5 — Townsend et al. (2026) noise-model augmentation'))
A(code('''
print('sncosmo available:', sncosmo_available())
aug_plan = plan_augmentation(feature_df[feature_df['is_synthetic'] == False]['label']
                             .value_counts().to_dict(),
                             AUGMENT_COPIES_PER_PARENT, AUGMENT_CLASSES)
synthetic_df = generate_synthetic_rows(
    feature_df, AUGMENT_CLASSES, AUGMENT_COPIES_PER_PARENT, process_ztf_object,
    os.path.join(BASE_DIR, 'synthetic'), np.random.default_rng(RANDOM_STATE),
    dz_max=AUGMENT_DZ_MAX, z_scale=AUGMENT_Z_SCALE,
    subsampling_rate=AUGMENT_SUBSAMPLING_RATE, use_sncosmo=True)
'''))
A(code('''
# Copies are training material only: they belong to no pool, and ALeRCE never saw them.
if len(synthetic_df):
    synthetic_df['source'] = 'synthetic'
    synthetic_df['passes_v4_gate'] = True
    for c in ('in_tns_pool', 'in_alerce_pool', 'in_overlap_pool'):
        synthetic_df[c] = False
    augmentation_check = augmentation_consistency(synthetic_df, feature_df)
    feature_df = pd.concat([feature_df, synthetic_df], ignore_index=True)
else:
    augmentation_check = None
ok, msg = augmented_rows_are_flagged(feature_df)
print(msg)
assert ok, msg
'''))

# ================================================================== STEP 9
A(md("""
# STEP 9 — Model training, both tracks  *(brief Phase 4, Section 7)*

One parameterised code path, called once per label source. Three headline models;
HistGradientBoosting and Bagging (SVM) remain available via `build_diagnostic_models()`
only, because v4 measured all five converging to within ~1 point at Stage 1.

* **TNS track:** every row with a TNS label (plus the curated TESS flares, as in v4) that
  clears v4's *full* gate, with synthetic copies in training only.
* **ALeRCE track:** the ALeRCE-confident pool, with `label` / `coarse_label` replaced by
  ALeRCE's crosswalked output. Its Stage 2 is 4-way (Ia / Ibc / II / SLSN) against the
  TNS track's 5-way — by design; Ib and Ic are not merged on the TNS side to match.

Stage 2's split is inherited from Stage 1's partitions, so an SN held out of Stage 1 can
never train Stage 2.
"""))
A(module_cell('modeling', 'Phase 4 — dual-track training'))
A(module_cell('diagnostics', 'track frames, v4 adapters and the two-track diagnostics runner'))
A(code('''
TRACK_FEATURES = [c for c in FEATURE_COLS if c in feature_df.columns]
tns_frame = tns_training_frame(feature_df)
tns_track = run_track(tns_frame, TRACK_FEATURES, 'coarse_label', 'label', 'TNS')

alerce_frame = alerce_track_frame(feature_df)
alerce_track = run_track(alerce_frame, TRACK_FEATURES, 'coarse_label', 'label', 'ALeRCE')
'''))
A(code('''
summary_table = track_summary([tns_track, alerce_track])
summary_table.insert(1, 'Pool', summary_table['Track'].map(
    {'TNS': 'TNS-confirmed (+ TESS flares)', 'ALeRCE': 'ALeRCE-confident'}))
display(summary_table.round(3))
print('Scored against TNS truth on the TNS track and against ALeRCE labels on the ALeRCE '
      'track. These two columns are NOT comparable as bare numbers — different samples, '
      'label reliability and Stage 2 class definitions. Step 10 is the comparison.')
'''))
A(md("""
### Did augmentation help?

The same TNS track with and without the synthetic copies. The real test partitions are
identical in both runs — asserted, not assumed — so this is a paired comparison.
"""))
A(code('''
tns_track_noaug = run_track(tns_frame[is_real(tns_frame)], TRACK_FEATURES,
                            'coarse_label', 'label', 'TNS (no augmentation)', verbose=False)
aug_ablation = None
if tns_track['stage2'] and tns_track_noaug['stage2']:
    a, b = tns_track['stage2']['split'], tns_track_noaug['stage2']['split']
    same = (set(a['df'].iloc[a['idx_test']]['id']) == set(b['df'].iloc[b['idx_test']]['id']))
    assert same, 'augmentation changed the test set — the comparison would be meaningless'
    aug_ablation = (tns_track['stage2']['results'].set_index('Model')[['Test accuracy', 'Macro-F1']]
                    .join(tns_track_noaug['stage2']['results'].set_index('Model')
                          [['Test accuracy', 'Macro-F1']], rsuffix=' (no aug)'))
    aug_ablation['delta acc'] = aug_ablation['Test accuracy'] - aug_ablation['Test accuracy (no aug)']
    display(aug_ablation.round(3))
    print(f"Stage 2 test set: {len(a['idx_test'])} objects. A difference smaller than the "
          'bootstrap CI width is not evidence either way.')
'''))

# ================================================================== STEP 10
A(md("""
# STEP 10 — The comparison  *(brief Phase 5, Section 6)*

After de Soto et al. (2024): $A_{\\rm expected} = P_{\\rm ours}^{\\rm T}\\,C_{\\rm ALeRCE}$ —
our purity matrix times ALeRCE's completeness matrix — is what agreement would be if the
two pipelines' errors were independent. Actual agreement **above** it means shared signal
(both key off the same light-curve physics), not independent verification.

* **Our predictions are out-of-fold.** Every object is predicted by a model that never saw
  it; objects with no TNS label are predicted by a model trained on all TNS objects. The
  held-out test split alone is too small at Stage 2 for a per-class matrix.
* **P_ours** comes from the full TNS validation set, as de Soto et al. build theirs.
  **C_ALeRCE** can only come from objects with both a TNS truth and an ALeRCE label.
* **How Section 6.2's split is read.** It asks for the overlap pool split into
  TNS-confirmed and non-confirmed subsets — but Section 3.3 defines the overlap pool as
  objects that *are* TNS-confirmed, so the second subset would be empty. de Soto et al.
  compare every object **both** pipelines classify; here that is the ALeRCE-confident pool,
  split into its **TNS-confirmed part (which is exactly the overlap pool)** and its
  **photometric-only part**.
"""))
A(module_cell('agreement', 'Section 6 — agreement framework'))
A(code('''
# Section 3.1 — what ALeRCE's 2025 classifier makes of TNS TDEs (and SLSNe).
tde_table, tde_text = tde_crosscheck(feature_df[is_real(feature_df)])
if tde_table is not None:
    display(tde_table.round(2))
print(tde_text)
'''))
A(code('''
# ---- Stage 1 agreement, per model ------------------------------------------------
alerce_real = alerce_track_frame(feature_df, real_only=True)
tns_truth = feature_df.drop_duplicates('id').set_index('id')
tns_coarse = tns_truth['coarse_label'].where(tns_truth['has_tns_label'].astype(bool))
S1_CLASSES = [c for c in ['AGN', 'SNe', 'TDE'] if c in set(alerce_real['coarse_label'])]
print('Stage 1 comparison classes:', S1_CLASSES,
      '(stellar_flare cannot appear: no TNS-confirmed object in the comparison is a flare)')

stage1_ours, stage1_reports = {}, {}
for name in HEADLINE_MODELS:
    oof = oof_predictions(tns_frame, TRACK_FEATURES, 'coarse_label', name)
    v = oof[oof['true'].isin(S1_CLASSES) & oof['pred'].isin(S1_CLASSES)]
    P = purity_matrix(v['true'], v['pred'], S1_CLASSES)
    ours = dict(zip(oof['id'], oof['pred']))
    unseen = alerce_real[~alerce_real['id'].astype(str).isin(ours)]
    up = predict_unseen(tns_frame, TRACK_FEATURES, 'coarse_label', name, unseen)
    ours.update(zip(up['id'], up['pred']))
    stage1_ours[name] = ours
    comp = pd.DataFrame({'id': alerce_real['id'].astype(str).values,
                         'ours': alerce_real['id'].astype(str).map(ours).values,
                         'alerce': alerce_real['coarse_label'].values,
                         'truth': alerce_real['id'].map(tns_coarse).values,
                         'has_tns_label': alerce_real['has_tns_label'].astype(bool).values})
    rep = agreement_report(comp, 'ours', 'alerce', 'truth', S1_CLASSES, name, 'Stage 1',
                           purity_ours=P)
    stage1_reports[name] = rep
    print(f'\\n=== {name}, Stage 1 ===')
    if 'expected_matrix' in rep:
        print('expected (independent errors):'); display(rep['expected_matrix'].round(3))
    for sub, r in rep['subsets'].items():
        if 'actual_matrix' in r:
            print(f'actual — {sub}:'); display(r['actual_matrix'].round(3))
            print(r['text'])
        else:
            print(f'{sub}: n={r.get("n", 0)} — {r.get("note", "")}')
    print(contrast_subsets(rep))
'''))
A(code('''
# ---- Stage 2 agreement, per model ------------------------------------------------
# Population: objects ALeRCE calls SN-like AND our own Stage 1 routes to SNe — the
# cascade-realistic comparison. Our 5-way Ib/Ic are collapsed to Ibc ONLY here, to meet
# ALeRCE's vocabulary; the TNS track's own Stage 2 stays 5-way.
sn_tns = tns_frame[tns_frame['label'].isin(SN_SUBTYPES)]
tns_fine_c = tns_truth['label'].where(tns_truth['has_tns_label'].astype(bool)
                                      & tns_truth['label'].isin(SN_SUBTYPES)).map(collapse_tns_fine)
stage2_reports = {}
for name in HEADLINE_MODELS:
    oof2 = oof_predictions(sn_tns, TRACK_FEATURES, 'label', name)
    oof2['pred_c'], oof2['true_c'] = oof2['pred'].map(collapse_tns_fine), oof2['true'].map(collapse_tns_fine)
    P2 = purity_matrix(oof2['true_c'], oof2['pred_c'], ALERCE_SN_CLASSES)
    pop = alerce_real[alerce_real['label'].isin(ALERCE_SN_CLASSES)]
    pop = pop[pop['id'].astype(str).map(stage1_ours[name]) == 'SNe']
    ours2 = dict(zip(oof2['id'], oof2['pred_c']))
    unseen = pop[~pop['id'].astype(str).isin(ours2)]
    up = predict_unseen(sn_tns, TRACK_FEATURES, 'label', name, unseen)
    ours2.update(zip(up['id'], up['pred'].map(collapse_tns_fine)))
    comp2 = pd.DataFrame({'ours': pop['id'].astype(str).map(ours2).values,
                          'alerce': pop['label'].values,
                          'truth': pop['id'].map(tns_fine_c).values,
                          'has_tns_label': pop['has_tns_label'].astype(bool).values})
    rep2 = agreement_report(comp2, 'ours', 'alerce', 'truth', ALERCE_SN_CLASSES, name,
                            'Stage 2', purity_ours=P2)
    stage2_reports[name] = rep2
    print(f'\\n=== {name}, Stage 2 (n={len(comp2)}) ===')
    if 'note' in rep2:
        print(rep2['note']); continue
    for sub, r in rep2['subsets'].items():
        print(r['text'] if 'text' in r else f'{sub}: n={r.get("n", 0)} — {r.get("note", "")}')
    print(contrast_subsets(rep2))
'''))
A(code('''
print('Section 6.4 — external corroboration:')
display(EXTERNAL_RESULTS)
print(TDE_SLSN_STATEMENT)
print()
print(TNS_LABEL_NOISE_CAVEAT)
'''))

# ================================================================== STEP 11
A(md(f"""
# STEP 11 — v4 diagnostics, on both tracks  *(brief Phase 6)*

v4's diagnostics, preserved. Two things differ from v4, both about what they are pointed at:

* **Real objects only.** They draw their own random splits; given synthetic copies, a copy
  could train while its parent is tested, inflating every number.
* **Imputed the way v4 imputed.** v4 filled every feature gap before its diagnostics ran
  (class-independent medians; a 0.0 sentinel for peak absolute magnitude). v5 adds
  columns after that point — host offset, redshift-less native objects — which can be
  entirely empty, and v4's HistGradientBoosting fits crash on an all-empty column. The
  diagnostic view is imputed with v4's own rules; training is not (it imputes inside
  each split).
* **Both tracks.** The TNS pass runs as ordinary cells below. The ALeRCE pass re-executes
  the **same source** (a repository test asserts it is identical) with the track rebound.
  Cells {sorted(TNS_ONLY)} — the class-scheme comparison and the class-imbalance test — run
  on the TNS track only: they are written for the 5-way vocabulary, and ALeRCE's SN scheme
  is already the class-scheme comparison's own "4-way Ia / Ib-c / II / SLSN" row.

One deliberate change to a preserved cell: the ablation (below) gains a guard for a
feature group that is entirely missing, which otherwise crashed it when no BTS-sourced
object survived acquisition.
"""))
A(code('''
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# TNS pass: point v4's names at the TNS track, real objects only.
_feature_df_all = feature_df
X_COLS = list(tns_track['stage1']['split']['feature_cols'])
# v4's diagnostics were written for v4's fully imputed table; give them one.
feature_df_full = v4_impute(tns_diagnostic_frame(feature_df), X_COLS)
feature_df      = feature_df_full                  # v4's separability cell reads this name
split, stage1, stage2a = v4_adapters(tns_track)
print(f'TNS diagnostics on {len(feature_df_full)} real objects, {len(X_COLS)} features')
'''))
for i, src in DIAG_ALL:
    A(code(src))
A(code('''
feature_df = _feature_df_all      # restore the full table for the steps below
'''))
A(code('# ALeRCE pass: the SAME v4 diagnostic source as the cells above (asserted\n'
       '# byte-identical by tests5/test_notebook_structure.py), re-executed with the\n'
       '# ALeRCE track bound in place of the TNS one, and everything restored afterwards.\n'
       '_V4_DIAGNOSTIC_SOURCES = ' + repr(DIAG_ALERCE) + '\n'
       '''
_a_split, _a_stage1, _a_stage2a = v4_adapters(alerce_track)
if _a_stage2a is None:
    print('ALeRCE track has no Stage 2 model — diagnostics skipped on that track.')
else:
    _a_cols = list(alerce_track['stage1']['split']['feature_cols'])
    _a_view = v4_impute(alerce_track_frame(feature_df, real_only=True), _a_cols)
    run_v4_cells(_V4_DIAGNOSTIC_SOURCES, globals(), {
        'feature_df_full': _a_view,
        'feature_df': _a_view,
        'SN_SUBTYPES': ALERCE_SN_CLASSES,
        'X_COLS': _a_cols,
        'split': _a_split, 'stage1': _a_stage1, 'stage2a': _a_stage2a,
    }, label='ALeRCE track', min_per_class=8)
for _i, _why in TNS_SCHEME_ONLY.items():
    print(f'v4 cell {_i} not run on the ALeRCE track — {_why}')
'''))

# ================================================================== STEP 12
A(md("""
# STEP 12 — Calibration  *(brief Phase 6B, Section 8)*

After de Soto et al. (2024, Fig. 11): does redshift make Stage 2 **more confident without
making it more correct**? Reported either way.
"""))
A(module_cell('calibration', 'Section 8 — calibration diagnostic'))
A(code('''
calibration_texts = []
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
        calibration_texts.append(compare_with_without_redshift(with_z, without_z, name))
        print(calibration_texts[-1]); print()
    plot_calibration(curves, 'Stage 2 calibration, with and without redshift')
    plt.show()
else:
    print('Stage 2 unavailable — calibration skipped.')
'''))

# ================================================================== STEP 13
A(md("""
# STEP 13 — Written summary  *(brief Phase 7)*

Generated from the objects computed above, so it cannot disagree with them.
"""))
A(code(r'''
L = []
L.append('## Samples and pools\n')
L.append(feature_df['source'].value_counts().rename('rows').to_frame().to_markdown())
L.append('\n\n' + pool_counts.to_markdown(index=False))
if 'native_summary' in globals():
    L.append('\n\n**What ALeRCE offers above the confidence cut, per group:**\n\n'
             + native_summary.to_markdown(index=False))

L.append('\n\n## Headline results (Section 6.3)\n')
L.append(summary_table.round(3).to_markdown(index=False))
L.append('\n\nThese are scored against different truths on different samples and are not '
         'comparable as bare numbers; the agreement analysis below is the comparison.')

if aug_ablation is not None:
    L.append('\n\n## Augmentation (Section 5)\n')
    L.append(aug_ablation.round(3).to_markdown())
if augmentation_check:
    L.append(f"\n\nSynthetic copies fit {augmentation_check['median_residual_mag']:+.3f} mag "
             f"brighter than their parents predict (median; spread "
             f"{augmentation_check['residual_spread_mag']:.3f}). Negative values are the "
             'truncation bias of near-limit detection, which faint real objects share.')

L.append('\n\n## Section 3.1 — TDE cross-check\n')
L.append(tde_text)
if tde_table is not None:
    L.append('\n\n' + tde_table.round(2).to_markdown())

L.append('\n\n## Agreement (Section 6)\n')
for stage, reps in (('Stage 1', stage1_reports), ('Stage 2', stage2_reports)):
    L.append(f'\n### {stage}\n')
    for name, rep in reps.items():
        for sub, r in rep.get('subsets', {}).items():
            if 'text' in r:
                L.append('- ' + r['text'])
        L.append('- ' + contrast_subsets(rep))

if calibration_texts:
    L.append('\n\n## Calibration (Section 8)\n')
    L.extend('- ' + t for t in calibration_texts)

L.append('\n\n## External corroboration (Section 6.4)\n')
L.append(EXTERNAL_RESULTS.to_markdown(index=False))
L.append('\n\n' + TDE_SLSN_STATEMENT)

L.append('\n\n## Ground-truth caveat (Section 6.5)\n')
L.append(TNS_LABEL_NOISE_CAVEAT)

L.append('\n\n## Deferred items — limitations, with their reasons\n')
L.append('- **AllWISE infrared colour (Section 4.2) — not built.** ALeRCE\'s feature set '
         'includes AllWISE infrared colours, which this pipeline does not; this is a plausible '
         'avenue for closing part of the remaining Stage-1 TDE/AGN confusion, left for future '
         'work. It needs a separate cross-match and lacks host offset\'s direct link to that '
         'boundary.')
L.append('- **Colour variability over time (Section 4.3) — not built.** It would need feature '
         'extraction to track colour across epochs rather than at one. It is specifically why '
         'ALeRCE\'s TDE update works: TDEs stay blue while SNe redden. A bigger lift than 4.1.')
L.append('- **ParSNIP-style non-parametric features (Section 4.4) — deliberately not built.** '
         'The most strongly evidenced accuracy lever in the research log: across three groups '
         'and two surveys, ParSNIP/SuperRAENN beat Villar-family fits by 8-10 F1 points every '
         'time tested. Not built because it requires training a variational autoencoder, a far '
         'heavier task than anything here, and the evidence comes from samples 15-1000x larger '
         'than ours (6,061 for Superphot+, 6,123 for ParSNIP, against our low hundreds). For '
         'scale: Superphot+ reached 83% / 0.61 F1 on 5-way SNe with N=6061; SuperRAENN 87% / '
         '0.66 purity with N~2885.')
L.append('- **SNANA simulation (Section 5) — deferred.** Aleo et al. 2023 / YSE-DR1 use '
         '~60,000 objects per class. Mature templates exist for Ia/II/Ibc but not obviously '
         'for SLSN or TDE, exactly the classes that need help.')
L.append('- **ALeRCE\'s total feature count** is cited in the research log as unverified. It '
         'is not quoted anywhere in this notebook, and no comparison here depends on it.')

L.append('\n\n## Structural limits found while building v5\n')
L.append('- ALeRCE\'s TDE class exists only in lc_classifier_BHRF_forced_phot 2.1.0; the '
         'classic lc_classifier has none. Verified live.')
L.append('- The BHRF transient branch separates SNIIn and calls the stripped-envelope class '
         'SESN. The IIn merge here rests on our own sample size, not on ALeRCE\'s taxonomy.')
L.append('- The ALeRCE track cannot contain stellar flares (TESS objects have no ZTF alerts). '
         'Its CV/Nova + YSO objects stand in for that class and are not the same population.')
L.append('- ALeRCE-native objects without a TNS spectrum have no redshift; they were acquired '
         'with only that part of v4\'s gate relaxed, and are excluded from the TNS track.')
L.append('- Ib and Ic stay separate on the TNS side and are merged only to compare with '
         'ALeRCE, which never separates them — the pair the literature finds least separable.')
L.append('- v4\'s peak_abs_mag is derived from the Villar A parameter, which is degenerate with '
         't0 and tau_rise: under an identical noise realization it scattered 0.255 mag against '
         '0.035 mag for the fitted curve\'s actual peak. Preserved, as v4 behaviour; flagged as '
         'the cheapest improvement available to v4\'s features.')

summary_md = '\n'.join(L)
with open(os.path.join(FEATURE_DIR, 'v5_results_summary.md'), 'w') as f:
    f.write(summary_md)
print('written to', os.path.join(FEATURE_DIR, 'v5_results_summary.md'))

from IPython.display import Markdown
Markdown(summary_md)
'''))
A(code('''
# Everything needed to reproduce a number without re-running acquisition.
feature_df.to_csv(os.path.join(FEATURE_DIR, 'v5_feature_table.csv'), index=False)
summary_table.to_csv(os.path.join(FEATURE_DIR, 'v5_headline_results.csv'), index=False)
pool_counts.to_csv(os.path.join(FEATURE_DIR, 'v5_pool_counts.csv'), index=False)
for stage, reps in (('stage1', stage1_reports), ('stage2', stage2_reports)):
    for name, rep in reps.items():
        tag = name.replace(' ', '_')
        if 'expected_matrix' in rep:
            rep['expected_matrix'].to_csv(os.path.join(FEATURE_DIR, f'v5_{stage}_{tag}_expected.csv'))
        for sub, r in rep.get('subsets', {}).items():
            if 'actual_matrix' in r:
                s = 'spec' if sub == SUBSET_SPEC else 'phot'
                r['actual_matrix'].to_csv(os.path.join(FEATURE_DIR, f'v5_{stage}_{tag}_actual_{s}.csv'))
print('saved to', FEATURE_DIR)
'''))

nb = {'cells': cells,
      'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                   'language_info': {'name': 'python', 'version': '3.11'},
                   'colab': {'provenance': [], 'toc_visible': True}},
      'nbformat': 4, 'nbformat_minor': 5}
with open(OUT, 'w') as f:
    json.dump(nb, f, indent=1)
    f.write('\n')
print(f'Wrote {OUT}: {len(cells)} cells '
      f'({sum(c["cell_type"] == "code" for c in cells)} code, '
      f'{sum(c["cell_type"] == "markdown" for c in cells)} markdown)')
