# Optical Transient Hierarchical Classification

A two-stage machine-learning pipeline that classifies optical transients from
light-curve shape features.

- **Stage 1 (coarse):** SNe vs AGN vs TDE vs Stellar Flare — 150 objects each, 600 total.
- **Stage 2 (fine):** for SNe, the subtype — Ia / Ib / Ic / II / SLSN, 30 each.

Data comes from TNS (spectroscopically confirmed labels) with ZTF photometry via
ALeRCE, plus TESS light curves via `lightkurve` for stellar flares. Four models are
trained at each stage: Random Forest, Logistic Regression, Bagging (Trees) and
Bagging (SVM).

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

**The notebook is generated, not hand-edited.** `btp_pipeline/` is the single source
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
```

Phase 2 (the real TNS/ALeRCE/TESS download) runs in the notebook, in Colab. It is
checkpointed and resumable throughout.

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
