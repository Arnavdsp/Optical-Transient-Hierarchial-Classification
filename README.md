# Optical Transient Hierarchical Classification

A two-stage machine-learning pipeline that classifies optical transients from
light-curve shape features.

- **Stage 1 (coarse):** SNe vs AGN vs TDE vs Stellar Flare — 150 objects each, 600 total.
- **Stage 2 (fine):** for SNe, the subtype — Ia / Ib / Ic / II / SLSN, 30 each.

Data comes from TNS (spectroscopically confirmed labels) with ZTF photometry via
ALeRCE, plus TESS light curves via `lightkurve` for stellar flares. Four models are
trained at each stage: Random Forest, Logistic Regression, Bagging (Trees) and
Bagging (SVM).

## v6 — SN sub-classification: more data, physical features, imbalance handled

`notebooks/BTP_TNS_Hierarchical_v6.ipynb` is the v5 notebook with its outputs, plus a
PART B that replaces Stage 2. Built by `tools/build_notebook_v6.py` from `btp6/`
(tested by `tests6/`); `tools/validate_notebook_v6.py` executes PART B for real
outside Colab.

What changed, and why:

- **Labels from three catalogues.** TNS + ZTF BTS + the Open Supernova Catalog
  (sne.space). OSC events are kept only when all their claimed types agree, and a ZTF
  id given different subtypes by different catalogues is dropped everywhere. The OSC
  adds few objects beyond TNS (it aggregates TNS), but it cross-checks the labels.
- **All of every rare class.** Ib, Ic and SLSN run until their pools are exhausted;
  Ia and II are capped at 600. Objects per subtype (Ia / Ib / Ic / II / SLSN): 600 / 96 / 183 / 600 / 202, against 30 each.
  The imbalance is handled in the models (class weights, SMOTE inside the training
  fold), not by throwing data away.
- **A looser gate, measured.** The v4 gate's 60-day baseline rejected ~70 % of SNe Ia.
  The v6 features are windowed and NaN-tolerant, so shorter light curves give missing
  values rather than wrong ones. Whether each object passes the v4 gate is still recorded.
- **Rest-frame physical features.** Timescales divided by (1+z); absolute peak
  magnitude from the fitted curve's maximum, corrected for Milky Way extinction
  (Schlafly & Finkbeiner 2011) and the (1+z) bandwidth term; g−r colour at fixed
  rest-frame phases; decline rates; plateau and secondary-bump indicators; plus a
  joint time × wavelength Gaussian process (Boone 2019, the PLAsTiCC winner), which
  gives consistent colours when one band is sparse.
- **4-way SESN scheme.** TNS files SN IIb under SN II, but IIb are stripped-envelope
  SNe (Modjaz et al. 2019), and 15 of 22 were predicted Ib/c. Reported alongside,
  not instead of, the TNS-label 4-way scheme.
- **Evaluation.** Model choice by repeated stratified 5-fold CV on 80 % of the data,
  then one score on an untouched 20 % test set (332 objects; v5 had 41).

Two further additions:

- **Template matching** (`btp6/templates.py`). Every light curve is fitted at its known
  redshift with the 66 Vincenzi et al. (2019) core-collapse templates (real SNe built
  from CfA and Carnegie Supernova Project data: II, IIn, IIb, Ib, Ic, Ic-BL) and with
  SALT2 for Ia. Host reddening is free, and one amplitude is shared by g and r so colour
  is constrained. The best reduced chi^2 per subclass becomes a feature (PSNID-style).
  Each class fits its own templates best (median table in the notebook).
- **TNS objects with no ZTF name** are matched to ALeRCE by a 2" cone search for the
  three rare classes (+4 Ib, +4 Ic, +17 SLSN).

Results (run 2026-11-02, `results/v6/`; untouched test set of 337 objects):

| scheme | v5 setup, CV | v6, CV | v6 test accuracy, all objects (95 % CI) | test balanced acc. |
|---|---|---|---|---|
| 3-way Ia / CCSN / SLSN | 0.917 | 0.927 | **0.929** (0.899–0.955) | 0.915 |
| 4-way Ia / Ibc / II / SLSN | 0.773 | 0.860 | **0.861** (0.822–0.896) | 0.847 |
| 4-way SESN (IIb with Ibc) | — | 0.874* | 0.846 (0.804–0.884) | 0.839 |
| 5-way | 0.607 | 0.821 | 0.804 (0.757–0.846) | 0.657 |

CV columns are rows A and G of the improvement ladder (same objects, repeated 5-fold
CV). *SESN has no ladder row; its number is the best model's CV on the training 80 %.

**90 % operating point (selective classification).** A confidence cut is fixed on
out-of-fold predictions over the training 80 % only, by the guaranteed-risk rule of
Geifman & El-Yaniv (2017): the 95 % upper bound on the kept set's error must be within
10 %. Applied once to the test set:

| scheme | test objects classified | test accuracy | test balanced acc. |
|---|---|---|---|
| 3-way | 100 % | **0.929** | 0.915 |
| 4-way | 85.5 % | **0.913** | 0.905 |
| 4-way SESN | 82.8 % | 0.900 | 0.888 |
| 5-way | 79.2 % | 0.903 | 0.701 |

The objects below the cut are the ones to send for a spectrum. A disclosure: the first
version of this step chose the cut where training accuracy was exactly 0.90, without
the bound, and gave 0.888 on the test set at 90 % coverage. The bound was added after
seeing that result, so the 4-way 0.913 is not a fully blind number.

**What limits 4-way on all objects.** It is SN II versus stripped-envelope.
Spectroscopically those differ by hydrogen and helium lines, and in g and r photometry
they overlap. Several further changes each moved 4-way CV by under half a point:
template matching, the cone-matched objects, all 81 TNS SN IIb, and the ALeRCE
feature set. All eight model families tried, and three imbalance strategies, land
within about a point of each other.

## v5 — TNS vs ALeRCE ground truth

`notebooks/BTP_TNS_Hierarchical_v5.ipynb` builds on the v4 pipeline (Villar fit,
quality gate, retry loop — preserved verbatim) and runs the whole two-stage hierarchy
twice: once against TNS spectroscopic labels, once against ALeRCE's own classifier
output. The two are then compared with the expected-vs-actual agreement framework of
de Soto et al. (2024, arXiv:2403.07975), at both stages, on three separately tracked
pools (TNS-confirmed, ALeRCE-confident, overlap).

```
btp5/
  config.py          v4 settings + v5 additions
  photometry.py      v4's Villar fit / quality gate / features, lifted verbatim by AST
  lightcurves.py     finds each saved light curve and recovers its sky position
  alerce_labels.py   live taxonomy, crosswalk, version-keyed labels
  alerce_native.py   pool 2 sourced from ALeRCE itself, uniformly sampled
  pools.py           the three pools
  hostoffset.py      host offset via astro-ghost (NaN + flag on failure)
  augment.py         Townsend et al. (2026) noise-model augmentation
  modeling.py        one dual-track runner; out-of-fold predictions
  diagnostics.py     v4's diagnostics on both tracks, via adapters
  agreement.py       expected vs actual agreement, subset contrast, TDE cross-check
  calibration.py     reliability curves with and without redshift
tests5/              74 offline tests
tools/build_notebook_v5.py      builds the notebook from btp5/ + the vendored v4 notebook
tools/validate_notebook_v5.py   executes the notebook's own cells offline, end to end
```

Everything the v5 notebook says about ALeRCE was checked against the live API, and a
few of those checks contradicted the brief it was built from. They are listed at the
top of the notebook: TDE exists only in the 2025 BHRF classifier; probabilities come
back stacked across classifier versions; the live SN branch does separate SNIIn;
ALeRCE returns its least confident objects first; and its confident set holds only 33
SESN, 43 SLSN and 23 TDE.

## Layout

```
notebooks/BTP_TNS_Hierarchical_Balanced.ipynb   the deliverable — runs top to bottom in Colab
btp_pipeline/                                   the logic, unit-tested offline
  config.py         targets, labels, feature and model names
  acquisition.py    Phase 2 — retry-until-target acquisition
  modeling.py       Phase 3 — the one global split, the 4 models, Stage 1 / Stage 2 A+B
  interpret.py      Phase 4 — permutation importance, confusions, boundaries, physics
  summary.py        results write-up, generated from the actual numbers
tests/                                          synthetic-data test suite (no network)
tools/build_notebook.py                         rebuilds the notebook from the modules
tools/synthetic_dry_run.py                      full Phase 3+4 run on synthetic data
tools/validate_notebook.py                      executes the notebook's own cells offline
```

 `btp_pipeline/` is the single source
of truth; `tools/build_notebook.py` inlines those modules verbatim into notebook
cells and copies the reused functions verbatim out of the previous notebook. The
result is self-contained — it needs no repo clone at run time — while the code
inside it is the code the test suite exercises. To change pipeline logic: edit the
module, run the tests, rebuild the notebook.

## Running

```bash
pip install -r requirements.txt

python -m pytest tests/ -q          # 26 tests, no network
python tools/synthetic_dry_run.py   # full Phase 3+4 on synthetic data
python tools/validate_notebook.py   # execute the notebook's own Phase 3/4 cells
python tools/build_notebook.py      # rebuild the notebook from the modules

python -m pytest tests5/ -q            # v5: 74 tests, no network
python tools/build_notebook_v5.py      # v5: rebuild the notebook
python tools/validate_notebook_v5.py   # v5: execute the notebook offline, end to end
```

Phase 2 (the real TNS/ALeRCE/TESS download) runs either in the notebook under
Colab, or headless:

```bash
python tools/run_phase2.py          # balanced acquisition; resumes if interrupted
python tools/run_phase2.py --classes AGN,TDE
python tools/run_phase3.py          # Phases 3 and 4 -> results/
```

Both paths run the same code: `btp_pipeline/features.py` holds the reused
extraction functions, lifted verbatim from the original notebook by
`tools/extract_features_module.py`, and `tests/test_features_parity.py` asserts the
module and the notebook still define them identically.

Headless acquisition needs `alerce` and `lightkurve`, whose legacy dependencies
(`fbpca`, `memoization`) will not build against modern setuptools. A venv pinned to
`setuptools<60` installs them cleanly.

## Results (real run)

`results/` holds the output of a full run against TNS, ALeRCE and TESS —
`results_summary.md` is the generated write-up, alongside the per-stage tables,
plots and the 549-row feature table.

Final sample: **150 SNe** (30 each of Ia/Ib/Ic/II/SLSN), **125 AGN**, **124 TDE**,
**150 stellar flares**. The two shortfalls are not fixable by pulling harder: the
entire TNS catalogue (206,621 rows) contains only 176 AGN-family and 153 TDE
objects, so 150 survivors of each is above what the source can supply. See
`results/count_report.csv`.

Headline: Stage 1 reaches 0.791 (Random Forest and Bagging (Trees)); Stage 2
Option A reaches 0.633 (Random Forest), up from the ~0.47-0.53 of the unbalanced
build. Read `results/results_summary.md` for the numbers with their error bars —
Stage 2's held-out set is 30 objects, so the repeated-CV estimate (0.607 +/- 0.091)
is the more reliable figure.

Two feature-construction problems surfaced only once real data was in hand, and
are documented in the summary: `amplitude` is exactly `peak_val - 1` for every ZTF
object, and `rise_time`/`decay_time` partly measure the observing baseline rather
than the transient.

## Credentials

TNS credentials are **not** stored in the notebook. It reads `TNS_BOT_ID`,
`TNS_BOT_NAME` and `TNS_API_KEY` at run time — from the Colab Secrets panel (key
icon in the left sidebar, with notebook access enabled), from environment
variables, or by interactive prompt as a last resort.

The previous notebook hard-coded a live TNS bot API key. That key reached this
public repository before it was removed, so **it must be rotated at
<https://www.wis-tns.org/> under the bot's settings.** `tests/test_no_secrets.py`
scans the repo on every CI run to stop a credential being committed again.

## Design decisions worth knowing

**One global split.** Stage 1 and Stage 2 are *not* split independently. There is
one stratified 80/20 split over all 600 rows; Stage 2's train and test rows are the
SN-labelled subsets of Stage 1's own partitions. An earlier version split them
separately, which let an SN object in Stage 1's test set appear in Stage 2's
training set and inflate the cascaded accuracy. `assert_split_nesting` now checks
this in code, so no post-hoc "leakage-free retrain" step is needed.

**Retry until the target is met.** Objects are lost at every step — TNS rows with no
ZTF cross-match, ALeRCE objects with too few detections, light curves that fail
cleaning or GP fitting. That attrition is not uniform across classes, so accepting
whatever survives systematically guts the rarest classes. Both acquisition loops
count an object only once it has produced a feature row, and keep pulling candidates
until the target is reached — or report the shortfall explicitly.

**Bagging (SVM) is configured as an SVM, not as a forest.** 50 estimators rather
than 300, `probability` left off, `n_jobs=-1`. Enabling `probability` triggers a
5-fold Platt-scaling CV *inside every base estimator*; nothing downstream needs
calibrated probabilities, since the confusion matrices, the cascade and the decision
boundaries all call `.predict`.

**Permutation importance for the cross-model comparison.** Gini importance,
regression coefficients and bagged-tree averages are not comparable to each other,
and a bagged RBF SVM has none of them. Permutation importance on the held-out test
set is computed identically for all four models; native importances are reported
alongside as a cross-check.

**Small samples are stated, not hidden.** Stage 2's held-out set is ~30 objects,
~6 per subtype — a worst-case standard error near 9 percentage points, with
per-subtype recall moving in steps of ~17 points. Balancing the dataset removed the
*bias* from uneven attrition; it could not remove that variance. Repeated stratified
CV over the Stage 2 training partition is reported alongside the held-out number as
the better-resolved estimate.
