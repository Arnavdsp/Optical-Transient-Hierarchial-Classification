# Optical Transient Hierarchical Classification

A two-stage machine-learning pipeline that classifies optical transients from
light-curve shape features.

- **Stage 1 (coarse):** SNe vs AGN vs TDE vs Stellar Flare — 150 objects each, 600 total.
- **Stage 2 (fine):** for SNe, the subtype — Ia / Ib / Ic / II / SLSN, 30 each.

Data comes from TNS (spectroscopically confirmed labels) with ZTF photometry via
ALeRCE, plus TESS light curves via `lightkurve` for stellar flares. Four models are
trained at each stage: Random Forest, Logistic Regression, Bagging (Trees) and
Bagging (SVM).

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
