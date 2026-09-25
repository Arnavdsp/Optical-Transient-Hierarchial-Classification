"""
Phase 4 — model training, run once per label source.

The brief is explicit that the two tracks must share one parameterized code path
rather than two copy-pasted blocks, so `run_track` takes the label source as an
argument and everything else is identical by construction. That also means a fix
applied to one track cannot fail to reach the other.

Models, evaluation protocol and the one-global-split discipline are carried over from
v4 unchanged: three headline models, class_weight='balanced', 5-fold CV with an
adaptive fold count, and a bootstrap 95% CI on test accuracy.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .config import RANDOM_STATE


def build_models(random_state=RANDOM_STATE, n_jobs=-1):
    """The three headline models, keyed by the display names used in every table."""
    return {
        'Logistic Regression': LogisticRegression(
            max_iter=2000, class_weight='balanced', random_state=random_state),
        'Random Forest': RandomForestClassifier(
            n_estimators=300, class_weight='balanced',
            random_state=random_state, n_jobs=n_jobs),
        'Bagging Classifier': BaggingClassifier(
            estimator=DecisionTreeClassifier(class_weight='balanced',
                                             random_state=random_state),
            n_estimators=300, max_samples=0.8, max_features=0.7,
            random_state=random_state, n_jobs=n_jobs),
    }


def safe_cv_folds(y, desired=5, verbose=True):
    """v4's adaptive fold count: k-fold needs at least k members of the rarest class."""
    counts = pd.Series(y).value_counts()
    folds = int(min(desired, counts.min()))
    if folds < desired and verbose:
        print(f'  NOTE: CV folds {desired} -> {max(folds, 2)} '
              f'(rarest class has {int(counts.min())} training examples)')
    return max(folds, 2)


def bootstrap_ci(y_true, y_pred, n_boot=2000, random_state=RANDOM_STATE):
    """Percentile bootstrap 95% CI on accuracy — v4's protocol, unchanged."""
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    n = len(y_true)
    if n == 0:
        return (float('nan'), float('nan'))
    accs = [float((y_true[i] == y_pred[i]).mean())
            for i in (rng.integers(0, n, n) for _ in range(n_boot))]
    return (float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5)))


def impute(X_train, X_test):
    """Median imputation fitted on train only. LogReg and SVMs cannot take NaN, and
    our feature table has structural NaNs (no colour, no host, no redshift, and the
    bts_* columns whenever a candidate came from TNS rather than BTS).

    A column that is entirely NaN in training has no median at all; numpy warns and
    returns NaN, so those fall back to 0.0 after standardisation-safe handling. That
    is expected for whole-column gaps, not a data problem, so the warning is
    suppressed here rather than printed once per fit."""
    with np.errstate(invalid='ignore'):
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter('ignore', RuntimeWarning)
            med = np.nanmedian(X_train, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    return (np.where(np.isfinite(X_train), X_train, med),
            np.where(np.isfinite(X_test), X_test, med))


def prepare_split(df, feature_cols, label_col, test_size=0.2,
                  random_state=RANDOM_STATE, exclude_synthetic_from_test=True,
                  verbose=True):
    """
    One stratified split for a track.

    Synthetic (augmented) rows are forced into the training partition: a synthetic
    copy shares its parent's light curve, so letting one into the test set is leakage
    dressed up as extra data.
    """
    sub = df[df[label_col].notna()].reset_index(drop=True)
    y_raw = sub[label_col].astype(str).values

    # Classes with a single member cannot be stratified or scored; drop with a notice.
    vc = pd.Series(y_raw).value_counts()
    too_small = vc[vc < 2].index.tolist()
    if too_small:
        if verbose:
            print(f'  dropping {too_small} — fewer than 2 objects, cannot split')
        keep = ~pd.Series(y_raw).isin(too_small).values
        sub, y_raw = sub[keep].reset_index(drop=True), y_raw[keep]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    X = sub[feature_cols].astype(float).values
    idx = np.arange(len(sub))

    synth = (sub['is_synthetic'].astype(bool).values
             if exclude_synthetic_from_test and 'is_synthetic' in sub.columns
             else np.zeros(len(sub), bool))

    if synth.any():
        real_idx = idx[~synth]
        tr_r, te_r = train_test_split(real_idx, test_size=test_size,
                                      stratify=y[real_idx], random_state=random_state)
        idx_train = np.sort(np.concatenate([tr_r, idx[synth]]))
        idx_test = np.sort(te_r)
    else:
        idx_train, idx_test = train_test_split(idx, test_size=test_size,
                                               stratify=y, random_state=random_state)
        idx_train, idx_test = np.sort(idx_train), np.sort(idx_test)

    Xtr, Xte = impute(X[idx_train], X[idx_test])
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    if verbose:
        print(f'  {len(idx_train)} train / {len(idx_test)} test, '
              f'{len(le.classes_)} classes'
              + (f', {int(synth.sum())} synthetic rows held in train'
                 if synth.any() else ''))
    return {'df': sub, 'X': X, 'le': le, 'y': y, 'scaler': scaler,
            'idx_train': idx_train, 'idx_test': idx_test,
            'X_train': Xtr_s, 'X_test': Xte_s,
            'y_train': y[idx_train], 'y_test': y[idx_test],
            'feature_cols': list(feature_cols)}


def prepare_nested_split(parent, label_col, row_mask, verbose=True):
    """
    Stage 2's split, INHERITED from Stage 1 rather than recomputed.

    `row_mask` selects the Stage 2 rows (the SN-labelled ones) as positions into the
    parent split's frame. Training rows are the masked rows inside the parent's
    training partition; evaluation rows are the masked rows inside its test partition.
    Calling prepare_split() on the SN subset instead would draw a fresh split, and an
    SN object held out of Stage 1 could land in Stage 2's training set — leakage that
    inflates the cascade without being visible in any single number.
    """
    pdf = parent['df']
    mask = np.asarray(row_mask, bool)
    tr = np.array([i for i in parent['idx_train'] if mask[i]], dtype=int)
    te = np.array([i for i in parent['idx_test'] if mask[i]], dtype=int)

    labels = pdf[label_col].astype(str).values
    keep_tr = np.array([i for i in tr if labels[i] not in ('nan', 'None')], dtype=int)
    keep_te = np.array([i for i in te if labels[i] not in ('nan', 'None')], dtype=int)
    if len(keep_tr) < 5 or len(np.unique(labels[keep_tr])) < 2:
        return None

    le = LabelEncoder()
    le.fit(labels[np.concatenate([keep_tr, keep_te])] if len(keep_te)
           else labels[keep_tr])
    # An unseen class in test would break inverse_transform; drop those rows loudly.
    known = set(le.classes_)
    keep_te = np.array([i for i in keep_te if labels[i] in known], dtype=int)

    X = parent['X']
    Xtr, Xte = impute(X[keep_tr], X[keep_te] if len(keep_te) else X[keep_tr][:0])
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte) if len(keep_te) else Xte

    if verbose:
        print(f'  {len(keep_tr)} train / {len(keep_te)} test SN rows, '
              f'inherited from Stage 1 partitions ({len(le.classes_)} classes)')
    return {'df': pdf, 'X': X, 'le': le, 'y': None, 'scaler': scaler,
            'idx_train': keep_tr, 'idx_test': keep_te,
            'X_train': Xtr_s, 'X_test': Xte_s,
            'y_train': le.transform(labels[keep_tr]),
            'y_test': le.transform(labels[keep_te]) if len(keep_te) else np.array([], int),
            'feature_cols': list(parent['feature_cols'])}


def evaluate(split, stage_name, random_state=RANDOM_STATE, verbose=True):
    """Fit the three models on a prepared split and score them."""
    models = build_models(random_state=random_state)
    folds = safe_cv_folds(split['y_train'], verbose=verbose)
    labels_idx = np.arange(len(split['le'].classes_))

    fitted, rows, confusions = {}, [], {}
    for name, model in models.items():
        cv = cross_val_score(model, split['X_train'], split['y_train'],
                             cv=folds, scoring='accuracy')
        model.fit(split['X_train'], split['y_train'])
        pred = model.predict(split['X_test'])
        acc = accuracy_score(split['y_test'], pred)
        f1 = f1_score(split['y_test'], pred, average='macro', zero_division=0)
        lo, hi = bootstrap_ci(split['y_test'], pred, random_state=random_state)
        fitted[name] = model
        confusions[name] = confusion_matrix(split['y_test'], pred, labels=labels_idx)
        rows.append({'Model': name, 'CV folds': folds,
                     'CV accuracy': cv.mean(), 'CV std': cv.std(),
                     'Test accuracy': acc, 'Macro-F1': f1,
                     'CI low': lo, 'CI high': hi, 'n_test': len(split['y_test'])})
        if verbose:
            print(f'  [{stage_name}] {name:<22} acc {acc:.3f} '
                  f'[{lo:.3f}-{hi:.3f}]  macro-F1 {f1:.3f}')
    return fitted, pd.DataFrame(rows), confusions


def run_track(df, feature_cols, coarse_label_col, fine_label_col, track_name,
              sn_coarse_value='SNe', random_state=RANDOM_STATE, verbose=True):
    """
    One full two-stage run for one label source.

    Stage 2 trains on the SN-labelled rows inside Stage 1's TRAINING partition and is
    evaluated on the SN-labelled rows inside Stage 1's TEST partition, so Stage 2
    cannot see a Stage 1 test object. Same nesting discipline as v4.
    """
    if verbose:
        print(f'\n=== {track_name}: Stage 1 (coarse) ===')
    s1 = prepare_split(df, feature_cols, coarse_label_col,
                       random_state=random_state, verbose=verbose)
    s1_models, s1_results, s1_cm = evaluate(s1, f'{track_name}/S1',
                                            random_state=random_state, verbose=verbose)

    if verbose:
        print(f'\n=== {track_name}: Stage 2 (SN subtypes) ===')
    sn_mask = ((s1['df'][coarse_label_col] == sn_coarse_value)
               & s1['df'][fine_label_col].notna()).values
    s2 = s2_models = s2_results = s2_cm = None
    if sn_mask.sum() >= 10:
        s2 = prepare_nested_split(s1, fine_label_col, sn_mask, verbose=verbose)
    if s2 is not None and len(s2['y_test']) > 0:
        s2_models, s2_results, s2_cm = evaluate(s2, f'{track_name}/S2',
                                                random_state=random_state,
                                                verbose=verbose)
    else:
        s2 = None
        if verbose:
            print(f'  Stage 2 skipped: {int(sn_mask.sum())} SN-labelled rows is too '
                  'few to inherit a usable train/test split from Stage 1.')

    return {'track': track_name,
            'stage1': {'split': s1, 'models': s1_models,
                       'results': s1_results, 'confusions': s1_cm},
            'stage2': ({'split': s2, 'models': s2_models,
                        'results': s2_results, 'confusions': s2_cm}
                       if s2 is not None else None)}


def track_summary(tracks):
    """Section 6.3's first two bullets: accuracy and macro-F1 per model, per track,
    per stage, with the pool each was computed on named."""
    rows = []
    for t in tracks:
        for stage in ('stage1', 'stage2'):
            st = t.get(stage)
            if st is None or st['results'] is None:
                continue
            for _, r in st['results'].iterrows():
                rows.append({'Track': t['track'], 'Stage': stage,
                             'Model': r['Model'], 'Accuracy': r['Test accuracy'],
                             'Macro-F1': r['Macro-F1'],
                             '95% CI': f"{r['CI low']:.3f}-{r['CI high']:.3f}",
                             'n_test': int(r['n_test'])})
    return pd.DataFrame(rows)


def build_diagnostic_models(random_state=RANDOM_STATE, n_jobs=-1):
    """
    The five-model set, for the diagnostics section only.

    Section 7 of the brief: the headline comparison is deliberately three models,
    because v4 measured all five converging to within ~1 point at Stage 1, so the
    extra two add noise to the headline table without adding information. They stay
    available here for cross-checking, under a separate name so they can never
    silently replace the three-model headline set.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.svm import SVC

    models = build_models(random_state=random_state, n_jobs=n_jobs)
    models['Bagging (SVM)'] = BaggingClassifier(
        estimator=SVC(kernel='rbf', class_weight='balanced', random_state=random_state),
        n_estimators=50, max_samples=0.8, max_features=0.7,
        random_state=random_state, n_jobs=n_jobs)
    models['HistGradientBoosting'] = HistGradientBoostingClassifier(
        random_state=random_state)
    return models
