"""
v6 Stage 2 modelling: class schemes, imbalance handling, model zoo, evaluation.

Evaluation protocol (the same for every number this module reports):

* Repeated stratified K-fold over every real object (default 5 folds x 5 repeats), so
  each number is a mean over 25 held-out folds and comes with its spread. With ~100
  objects per minority class this resolves differences of ~2 points; v5's single
  41-object test split could not resolve 10.
* Every preprocessing step that learns from data (majority capping, SMOTE, scaling,
  imputation) is fitted inside the training fold only.
* Three metrics, always together: accuracy, balanced accuracy (mean per-class recall)
  and macro-F1. With an imbalanced test set accuracy alone rewards predicting SN Ia.
* Two test compositions: 'natural' (the folds as they fall) and 'balanced' (each test
  fold subsampled to equal class counts before scoring). Balanced-test accuracy is the
  number that answers "how well does it separate the classes".
"""

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SCHEMES = {
    '5-way': {'SN_Ia': 'SN_Ia', 'SN_Ib': 'SN_Ib', 'SN_Ic': 'SN_Ic', 'SN_II': 'SN_II', 'SLSN': 'SLSN'},
    '4-way': {'SN_Ia': 'SN_Ia', 'SN_Ib': 'SN_Ibc', 'SN_Ic': 'SN_Ibc', 'SN_II': 'SN_II', 'SLSN': 'SLSN'},
    '3-way': {'SN_Ia': 'SN_Ia', 'SN_Ib': 'CCSN', 'SN_Ic': 'CCSN', 'SN_II': 'CCSN', 'SLSN': 'SLSN'},
}
# '4-way SESN': the 4-way scheme with SN IIb counted as stripped-envelope. TNS (and
# `classify_sn_subtype`) files IIb under SN II because a little hydrogen is seen early,
# but IIb progenitors have lost almost all of their H envelope and their light curves
# are those of Ib (Filippenko 1997; Modjaz, Gutierrez & Arcavi 2019, Nat. Astron. 3,
# 717, where "stripped-envelope SNe" are IIb + Ib + Ic). On the v6 run 15 of 22 IIb were
# predicted Ib/c under the TNS labels. This scheme reads the `label_sesn` column.
SCHEMES['4-way SESN'] = SCHEMES['4-way']
SCHEME_LABEL_COL = {'4-way SESN': 'label_sesn'}
IIB = re.compile(r'^\s*(SN\s*)?IIb', re.I)


def add_sesn_label(df, type_col='type'):
    out = df.copy()
    is_iib = out[type_col].astype(str).str.match(IIB) & out['label'].eq('SN_II')
    out['label_sesn'] = np.where(is_iib, 'SN_Ib', out['label'])
    return out


def scheme_frame(df, scheme, label_col='label'):
    if isinstance(scheme, str):
        label_col = SCHEME_LABEL_COL.get(scheme, label_col)
    mp = SCHEMES[scheme] if isinstance(scheme, str) else scheme
    d = df[df[label_col].isin(mp)].copy()
    d['y'] = d[label_col].map(mp)
    return d.reset_index(drop=True)


# --------------------------------------------------------------------------- models
def make_models(random_state=42):
    """Name -> unfitted estimator. Tree models take NaN natively except RF/ET/LR,
    which get a median imputer fitted inside the fold."""
    imp = lambda: SimpleImputer(strategy='median')
    m = {
        'HistGB (balanced)': HistGradientBoostingClassifier(
            # chosen by a small sweep on the v6 run (4-way SESN, 5-fold CV): 0.868 vs 0.857
            # for the 500-iteration, 15-leaf setting, and ~2x faster
            max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=10,
            l2_regularization=0.0, class_weight='balanced', random_state=random_state),
        'Random Forest (balanced)': make_pipeline(imp(), RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2, max_features='sqrt',
            class_weight='balanced_subsample', n_jobs=-1, random_state=random_state)),
        'Extra Trees (balanced)': make_pipeline(imp(), ExtraTreesClassifier(
            n_estimators=400, min_samples_leaf=2, max_features=0.5,
            class_weight='balanced', n_jobs=-1, random_state=random_state)),
        'Logistic Regression': make_pipeline(imp(), StandardScaler(), LogisticRegression(
            C=0.5, max_iter=4000, class_weight='balanced')),
    }
    return m


class SoftVote(ClassifierMixin, BaseEstimator):
    """Average of predict_proba over member models (each fitted in-fold)."""

    def __init__(self, members, weights=None):
        self.members = members
        self.weights = weights

    def fit(self, X, y, sample_weight=None):
        self.fitted_ = [clone(m).fit(X, y) for m in self.members]
        self.classes_ = self.fitted_[0].classes_
        return self

    def predict_proba(self, X):
        w = self.weights or [1.0] * len(self.fitted_)
        P = sum(wi * m.predict_proba(X) for wi, m in zip(w, self.fitted_))
        return P / sum(w)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class Hierarchical4(ClassifierMixin, BaseEstimator):
    """SLSN vs rest -> Ia vs CC -> II vs Ibc, each node its own classifier.

    The physically motivated order: SLSNe are separated by luminosity and timescale,
    Ia from core-collapse by colour and decline, and II from Ibc by the hydrogen
    plateau and colour evolution. Probabilities multiply down the tree.
    """

    TREE = [('SLSN', None), ('SN_Ia', None), ('SN_II', 'SN_Ibc')]

    def __init__(self, base):
        self.base = base

    def fit(self, X, y, sample_weight=None):
        y = np.asarray(y)
        self.classes_ = np.array(sorted(set(y)))
        self.n1 = clone(self.base).fit(X, (y == 'SLSN').astype(int))
        k = y != 'SLSN'
        self.n2 = clone(self.base).fit(X[k], (y[k] == 'SN_Ia').astype(int))
        k2 = k & (y != 'SN_Ia')
        self.n3 = clone(self.base).fit(X[k2], (y[k2] == 'SN_II').astype(int))
        return self

    def predict_proba(self, X):
        p1 = self.n1.predict_proba(X)[:, 1]
        p2 = self.n2.predict_proba(X)[:, 1]
        p3 = self.n3.predict_proba(X)[:, 1]
        r = 1 - p1
        P = {'SLSN': p1, 'SN_Ia': r * p2, 'SN_II': r * (1 - p2) * p3,
             'SN_Ibc': r * (1 - p2) * (1 - p3)}
        return np.column_stack([P[c] for c in self.classes_])

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# ----------------------------------------------------------------------- imbalance
def cap_majority(X, y, ratio, rng):
    """Subsample every class to at most `ratio` x the smallest class (in-fold only)."""
    if ratio is None:
        return X, y
    vals, cnt = np.unique(y, return_counts=True)
    cap = int(np.ceil(cnt.min() * ratio))
    keep = []
    for v in vals:
        idx = np.where(y == v)[0]
        if len(idx) > cap:
            idx = rng.choice(idx, cap, replace=False)
        keep.append(idx)
    keep = np.sort(np.concatenate(keep))
    return X[keep], y[keep]


def smote(X, y, rng):
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        return X, y
    Xi = SimpleImputer(strategy='median').fit(X).transform(X)
    k = max(1, min(5, np.unique(y, return_counts=True)[1].min() - 1))
    return SMOTE(k_neighbors=k, random_state=int(rng.integers(1 << 30))).fit_resample(Xi, y)


# ---------------------------------------------------------------------- evaluation
def _balanced_subsample(y, rng):
    vals, cnt = np.unique(y, return_counts=True)
    n = cnt.min()
    return np.sort(np.concatenate([rng.choice(np.where(y == v)[0], n, replace=False) for v in vals]))


def cross_validate(model, X, y, *, n_splits=5, n_repeats=5, cap_ratio=None, use_smote=False,
                   random_state=42, return_oof=False):
    """Repeated stratified K-fold. Returns a dict of mean/sd metrics (+ OOF if asked).

    OOF predictions come from the first repeat only, so each object has exactly one.
    """
    X = np.asarray(X, float)
    y = np.asarray(y)
    rng = np.random.default_rng(random_state)
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    rec = {k: [] for k in ['acc', 'bacc', 'f1', 'acc_baltest']}
    oof_pred = np.empty(len(y), dtype=object)
    oof_prob = None
    classes = np.array(sorted(set(y)))
    for k, (tr, te) in enumerate(cv.split(X, y)):
        Xtr, ytr = cap_majority(X[tr], y[tr], cap_ratio, rng)
        if use_smote:
            Xtr, ytr = smote(Xtr, ytr, rng)
        m = clone(model).fit(Xtr, ytr)
        Xte = X[te]
        if use_smote:
            Xte = SimpleImputer(strategy='median').fit(X[tr]).transform(Xte)
        p = m.predict(Xte)
        rec['acc'].append(accuracy_score(y[te], p))
        rec['bacc'].append(balanced_accuracy_score(y[te], p))
        rec['f1'].append(f1_score(y[te], p, average='macro'))
        b = _balanced_subsample(y[te], rng)
        rec['acc_baltest'].append(accuracy_score(y[te][b], p[b]))
        if return_oof and k < n_splits:
            oof_pred[te] = p
            if hasattr(m, 'predict_proba'):
                pr = m.predict_proba(Xte)
                if oof_prob is None:
                    oof_prob = np.zeros((len(y), len(classes)))
                cl = list(m.classes_)
                oof_prob[te] = pr[:, [cl.index(c) for c in classes]]
    out = {f'{k}_mean': float(np.mean(v)) for k, v in rec.items()}
    out.update({f'{k}_sd': float(np.std(v)) for k, v in rec.items()})
    if return_oof:
        out['oof_pred'] = oof_pred
        out['oof_prob'] = oof_prob
        out['classes'] = classes
    return out


def coverage_curve(y, prob, classes, fractions=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5)):
    """Accuracy on the most confident fraction of objects (OOF probabilities).

    The standard way photometric classifiers are used in practice: spectroscopic
    follow-up is spent on the uncertain ones, and the confident ones are trusted.
    """
    y = np.asarray(y)
    conf = prob.max(axis=1)
    pred = classes[np.argmax(prob, axis=1)]
    order = np.argsort(-conf)
    rows = []
    for f in fractions:
        k = order[:max(1, int(round(f * len(y))))]
        rows.append({'kept_fraction': f, 'n': len(k),
                     'min_confidence': float(conf[k].min()),
                     'accuracy': accuracy_score(y[k], pred[k]),
                     'balanced_accuracy': balanced_accuracy_score(y[k], pred[k]),
                     'macro_F1': f1_score(y[k], pred[k], average='macro')})
    return pd.DataFrame(rows)


def confusion_table(y, pred, classes):
    cm = confusion_matrix(y, pred, labels=classes)
    return pd.DataFrame(cm, index=[f'true {c}' for c in classes],
                        columns=[f'pred {c}' for c in classes])


def confidence_threshold(y, prob, classes, target=0.90, delta=0.05):
    """Confidence cut for selective classification, by the "selection with guaranteed
    risk" rule of Geifman & El-Yaniv (2017, NeurIPS): keep as many objects as possible
    such that the one-sided (1 - delta) Clopper-Pearson UPPER bound on the error rate of
    the kept set is at most 1 - target. Chosen on out-of-fold TRAINING predictions only,
    then applied unchanged to the test set.

    Returns (threshold, coverage, accuracy) on the data the cut was chosen on.
    """
    from scipy.stats import beta
    y = np.asarray(y)
    conf = prob.max(axis=1)
    wrong = classes[np.argmax(prob, axis=1)] != y
    order = np.argsort(-conf)
    n = np.arange(1, len(y) + 1)
    e = np.cumsum(wrong[order])
    upper = np.where(e < n, beta.ppf(1 - delta, e + 1, n - e), 1.0)
    ok = np.where((upper <= 1 - target) & (n >= 20))[0]
    if not len(ok):
        return 1.0, 0.0, np.nan
    k = ok.max()
    return float(conf[order][k]), float((k + 1) / len(y)), float(1 - e[k] / (k + 1))


def selective_report(y, prob, classes, threshold):
    y = np.asarray(y)
    conf = prob.max(axis=1)
    pred = classes[np.argmax(prob, axis=1)]
    keep = conf >= threshold
    if not keep.any():
        return {'coverage': 0.0, 'n_kept': 0, 'accuracy': np.nan, 'balanced_accuracy': np.nan,
                'macro_F1': np.nan, 'per_class_coverage': {}}
    return {'coverage': float(keep.mean()), 'n_kept': int(keep.sum()),
            'accuracy': accuracy_score(y[keep], pred[keep]),
            'balanced_accuracy': balanced_accuracy_score(y[keep], pred[keep]),
            'macro_F1': f1_score(y[keep], pred[keep], average='macro'),
            'per_class_coverage': {c: float(keep[y == c].mean()) for c in classes}}
