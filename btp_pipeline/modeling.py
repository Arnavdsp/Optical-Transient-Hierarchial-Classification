"""
Phase 3 — hierarchical model training and evaluation.

Design correction A (the important one): there is exactly ONE stratified split,
computed over the full combined feature table and stratified on the coarse label.
Stage 1 trains and evaluates on it directly. Stage 2's training rows are the
SN-labelled rows *inside Stage 1's training partition*, and its evaluation rows
are the SN-labelled rows *inside Stage 1's test partition*. Stage 2 therefore
cannot see a Stage 1 test object during training — leakage is impossible by
construction, so no "leakage-free retrain" cleanup step is needed afterwards.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# --------------------------------------------------------------------------
# Labels and the one global split
# --------------------------------------------------------------------------

def add_coarse_label(feature_df, sn_subtype_labels):
    """Collapse the 5 SN subtypes into a single 'SNe' coarse label, keeping
    the fine-grained subtype untouched in 'label'."""
    df = feature_df.copy()
    df['coarse_label'] = df['label'].apply(lambda l: 'SNe' if l in sn_subtype_labels else l)
    return df


def make_global_split(feature_df, x_cols, test_size=0.2, random_state=42, verbose=True):
    """
    The single source of truth for who is 'train' and who is 'test' anywhere in
    this project. Stratified on the coarse label so all four coarse classes are
    proportionally represented on both sides.

    Returns a dict carrying positional indices into `feature_df`; every later
    stage slices these rather than splitting again.
    """
    X = feature_df[x_cols].values
    le_coarse = LabelEncoder()
    y_coarse = le_coarse.fit_transform(feature_df['coarse_label'].values)

    idx_all = np.arange(len(feature_df))
    idx_train, idx_test = train_test_split(
        idx_all, test_size=test_size, stratify=y_coarse, random_state=random_state
    )
    idx_train.sort()
    idx_test.sort()

    split = {
        'X': X,
        'x_cols': list(x_cols),
        'idx_train': idx_train,
        'idx_test': idx_test,
        'y_coarse': y_coarse,
        'le_coarse': le_coarse,
        'feature_df': feature_df,
    }
    if verbose:
        print(f"Global split: {len(idx_train)} train / {len(idx_test)} test "
              f"(stratified on coarse_label, test_size={test_size})")
        print(pd.crosstab(feature_df['coarse_label'],
                          np.where(np.isin(idx_all, idx_train), 'train', 'test')))
    return split


def assert_split_nesting(split, sn_subtype_labels):
    """
    Guard the design correction. Fails loudly if Stage 2's rows are ever anything
    other than a subset of Stage 1's own train/test partitions.
    """
    df = split['feature_df']
    sn_mask = df['label'].isin(sn_subtype_labels).values
    s2_train = np.array([i for i in split['idx_train'] if sn_mask[i]])
    s2_test = np.array([i for i in split['idx_test'] if sn_mask[i]])

    assert set(s2_train).issubset(set(split['idx_train'])), "Stage 2 train escaped Stage 1 train"
    assert set(s2_test).issubset(set(split['idx_test'])), "Stage 2 test escaped Stage 1 test"
    assert not (set(s2_train) & set(split['idx_test'])), "LEAKAGE: Stage 2 trains on a Stage 1 test object"
    assert not (set(s2_test) & set(split['idx_train'])), "LEAKAGE: Stage 2 evaluates on a Stage 1 train object"
    return s2_train, s2_test


def safe_cv_folds(y_arr, desired=5, verbose=True):
    """
    Adaptive fold count (design correction C). Even at a clean 30 per subtype,
    an 80/20 split leaves ~24 per class in training and the split does not always
    land evenly, so a fixed cv=5 is not guaranteed safe. Degrade gracefully
    instead of crashing cross_val_score.
    """
    counts = pd.Series(y_arr).value_counts()
    min_count = int(counts.min())
    folds = min(desired, min_count)
    if folds < desired and verbose:
        print(f"NOTE: reducing CV folds {desired} -> {max(folds, 2)}; smallest class in this "
              f"training split has only {min_count} examples.")
    return max(folds, 2)


# --------------------------------------------------------------------------
# The 4 models
# --------------------------------------------------------------------------

def build_models(random_state=42, n_jobs=-1):
    """
    Fresh, unfitted instances of the four models, keyed by their canonical
    display names.

    Bagging (SVM) deliberately does NOT copy the tree-bagging configuration
    (design correction D):
      * n_estimators=50 rather than 300. Each base SVC is O(n^2)-ish to fit and
        there is no cheap warm start, so 300 bagged SVCs is minutes of compute
        for no measurable accuracy gain at n=600.
      * probability left at its default (off). Enabling it triggers an internal
        5-fold Platt-scaling CV *inside every base estimator* — 50x (or 300x) a
        5-fold refit. It is not passed explicitly because scikit-learn 1.9
        deprecated the keyword; the default is already what we want. Nothing
        downstream needs calibrated probabilities from this model: the confusion
        matrices, the cascade and the decision-boundary plots all call .predict,
        and feature importance comes from permutation importance, which also only
        needs .predict. If calibrated probabilities are ever needed, flip it on
        for that analysis alone.
      * n_jobs=-1 to parallelise across base estimators.
    """
    return {
        'Random Forest': RandomForestClassifier(
            n_estimators=300, class_weight='balanced', random_state=random_state, n_jobs=n_jobs),
        'Logistic Regression': LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=random_state),
        'Bagging (Trees)': BaggingClassifier(
            estimator=DecisionTreeClassifier(class_weight='balanced', random_state=random_state),
            n_estimators=300, max_samples=0.8, max_features=0.7,
            random_state=random_state, n_jobs=n_jobs),
        'Bagging (SVM)': BaggingClassifier(
            estimator=SVC(kernel='rbf', class_weight='balanced', random_state=random_state),
            n_estimators=50, max_samples=0.8, max_features=0.7,
            random_state=random_state, n_jobs=n_jobs),
    }


def _scale(X_train, X_test):
    scaler = StandardScaler()
    return scaler, scaler.fit_transform(X_train), scaler.transform(X_test)


def train_and_evaluate(X_train, y_train, X_test, y_test, le, stage_name,
                       random_state=42, cv_desired=5, verbose=True):
    """
    Fit all four models on one (already-scaled) split and score them.

    Returns (fitted_models, results_df, confusions) where results_df carries CV
    mean/std on the training partition and accuracy on the held-out test rows.
    """
    models = build_models(random_state=random_state)
    folds = safe_cv_folds(y_train, desired=cv_desired, verbose=verbose)

    fitted, rows, confusions = {}, [], {}
    labels_idx = np.arange(len(le.classes_))
    for name, model in models.items():
        cv = cross_val_score(model, X_train, y_train, cv=folds, scoring='accuracy')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        fitted[name] = model
        confusions[name] = confusion_matrix(y_test, y_pred, labels=labels_idx)
        rows.append({'Model': name, 'CV folds': folds, 'CV accuracy': cv.mean(),
                     'CV std': cv.std(), 'Test accuracy': acc})
        if verbose:
            print(f"[{stage_name}] {name:<22} CV {cv.mean():.3f} +/- {cv.std():.3f} | "
                  f"test {acc:.3f}")
    return fitted, pd.DataFrame(rows), confusions


# --------------------------------------------------------------------------
# Stage 1 / Stage 2
# --------------------------------------------------------------------------

def run_stage1(split, random_state=42, verbose=True):
    """Coarse classification: SNe / AGN / TDE / stellar_flare, on the global split."""
    X, idx_train, idx_test = split['X'], split['idx_train'], split['idx_test']
    y = split['y_coarse']
    le = split['le_coarse']

    scaler, X_train_s, X_test_s = _scale(X[idx_train], X[idx_test])
    y_train, y_test = y[idx_train], y[idx_test]

    fitted, results, confusions = train_and_evaluate(
        X_train_s, y_train, X_test_s, y_test, le, 'Stage 1',
        random_state=random_state, verbose=verbose)

    return {'stage': 'Stage 1', 'scaler': scaler, 'le': le, 'models': fitted,
            'results': results, 'confusions': confusions,
            'X_train': X_train_s, 'X_test': X_test_s,
            'y_train': y_train, 'y_test': y_test,
            'idx_train': idx_train, 'idx_test': idx_test}


def run_stage2_option_a(split, sn_subtype_labels, random_state=42, verbose=True):
    """
    Option A — the 'ceiling'. Subtype the TRUE SN rows directly, i.e. perfect
    ground-truth routing with no Stage 1 errors in the way. Train rows are the
    SN rows of Stage 1's training partition; test rows are the SN rows of Stage
    1's test partition. Nothing is re-split.
    """
    df, X = split['feature_df'], split['X']
    s2_train, s2_test = assert_split_nesting(split, sn_subtype_labels)

    le = LabelEncoder()
    le.fit(df['label'].iloc[np.concatenate([s2_train, s2_test])].values)
    y_train = le.transform(df['label'].iloc[s2_train].values)
    y_test = le.transform(df['label'].iloc[s2_test].values)

    scaler, X_train_s, X_test_s = _scale(X[s2_train], X[s2_test])
    if verbose:
        print(f"Stage 2 (Option A): {len(s2_train)} train / {len(s2_test)} test true-SN rows")
        print(pd.Series(df['label'].iloc[s2_train].values).value_counts().to_string())

    fitted, results, confusions = train_and_evaluate(
        X_train_s, y_train, X_test_s, y_test, le, 'Stage 2 / A',
        random_state=random_state, verbose=verbose)

    return {'stage': 'Stage 2 Option A', 'scaler': scaler, 'le': le, 'models': fitted,
            'results': results, 'confusions': confusions,
            'X_train': X_train_s, 'X_test': X_test_s,
            'y_train': y_train, 'y_test': y_test,
            'idx_train': s2_train, 'idx_test': s2_test}


def cascade_predict(stage1_model, stage2_model, X_raw, scaler1, scaler2, le1, le2,
                    sn_coarse_label='SNe'):
    """
    Stage 1 decides SNe vs not. Only objects Stage 1 *predicted* to be SNe get a
    Stage 2 subtype; everything else keeps Stage 1's coarse answer as final.
    """
    coarse_pred = le1.inverse_transform(stage1_model.predict(scaler1.transform(X_raw)))
    final = np.array(coarse_pred, dtype=object)
    sne_mask = coarse_pred == sn_coarse_label
    if sne_mask.any():
        fine = le2.inverse_transform(stage2_model.predict(scaler2.transform(X_raw[sne_mask])))
        final[sne_mask] = fine
    return {'coarse_pred': coarse_pred, 'final_pred': final, 'sne_mask': sne_mask}


def run_stage2_option_b(split, stage1, stage2a, sn_subtype_labels, verbose=True):
    """
    Option B — the 'realistic' path. Subtype whatever Stage 1 *predicted* was an
    SN, including Stage 1's mistakes. Reuses the Option A models: they were
    trained on true SN rows from Stage 1's training partition, which is the only
    subtype-labelled training data that exists. What differs is the population
    they are applied to at test time.

    Three metrics, because they answer three different questions:
      * End-to-end accuracy — final 8-class prediction vs true fine label, over
        ALL test objects. The honest "what does the whole pipeline do" number,
        but flattered by the easy non-SN classes.
      * Conditional accuracy — subtype accuracy among true SNe that Stage 1
        actually routed correctly. Isolates Stage 2's own skill.
      * Option-A-comparable accuracy — over true SN test rows only, counting a
        Stage 1 mis-route as wrong. This is the only number directly comparable
        to Option A, because it is the same population scored the same way.
    """
    df, X = split['feature_df'], split['X']
    idx_test = split['idx_test']
    X_test_raw = X[idx_test]
    true_fine = df['label'].iloc[idx_test].values
    true_coarse = df['coarse_label'].iloc[idx_test].values
    is_true_sn = np.isin(true_fine, sn_subtype_labels)

    results, rows = {}, []
    for name in stage1['models']:
        r = cascade_predict(stage1['models'][name], stage2a['models'][name], X_test_raw,
                            stage1['scaler'], stage2a['scaler'],
                            stage1['le'], stage2a['le'])
        final, coarse_pred, sne_mask = r['final_pred'], r['coarse_pred'], r['sne_mask']

        end_to_end = float((final == true_fine).mean())

        routed_ok = is_true_sn & (coarse_pred == 'SNe')
        conditional = float((final[routed_ok] == true_fine[routed_ok]).mean()) if routed_ok.any() else np.nan

        comparable = float((final[is_true_sn] == true_fine[is_true_sn]).mean()) if is_true_sn.any() else np.nan

        fp_leak = int((~is_true_sn & sne_mask).sum())     # non-SNe wrongly sent into Stage 2
        fn_loss = int((is_true_sn & ~sne_mask).sum())     # true SNe Stage 1 never routed

        r.update({'true_fine': true_fine, 'true_coarse': true_coarse, 'is_true_sn': is_true_sn})
        results[name] = r
        rows.append({'Model': name,
                     'End-to-end accuracy (all test objects)': end_to_end,
                     'Conditional accuracy (correctly-routed SNe)': conditional,
                     'Option-A-comparable accuracy (true SNe)': comparable,
                     'Non-SNe mis-routed into Stage 2': fp_leak,
                     'True SNe lost before Stage 2': fn_loss})
        if verbose:
            print(f"[Stage 2 / B] {name:<22} end-to-end {end_to_end:.3f} | "
                  f"conditional {conditional:.3f} | comparable {comparable:.3f} | "
                  f"FP-in {fp_leak}, FN-lost {fn_loss}")

    summary = pd.DataFrame(rows)
    all_classes = sorted(pd.unique(df['label']))
    confusions = {name: confusion_matrix(true_fine, results[name]['final_pred'], labels=all_classes)
                  for name in results}
    return {'stage': 'Stage 2 Option B', 'per_model': results, 'summary': summary,
            'confusions': confusions, 'all_classes': all_classes, 'idx_test': idx_test}


def option_ab_comparison(stage2a, stage2b):
    """Side-by-side Option A vs Option B, per model, on matched populations."""
    a = stage2a['results'].set_index('Model')['Test accuracy']
    b = stage2b['summary'].set_index('Model')
    out = pd.DataFrame({
        'Option A (ground-truth routing)': a,
        'Option B (comparable: true SNe, mis-route = wrong)':
            b['Option-A-comparable accuracy (true SNe)'],
        'Option B (conditional: correctly-routed only)':
            b['Conditional accuracy (correctly-routed SNe)'],
        'Option B (end-to-end, all 8 classes)':
            b['End-to-end accuracy (all test objects)'],
    })
    out['Routing cost (A - B comparable)'] = (
        out['Option A (ground-truth routing)']
        - out['Option B (comparable: true SNe, mis-route = wrong)'])
    return out.reset_index()


def repeated_cv_estimate(X_train, y_train, le, stage_name, n_splits=5, n_repeats=10,
                         random_state=42, verbose=True):
    """
    A better-resolved accuracy estimate than a single small held-out set.

    Why this exists. Stage 2's held-out test set is the SN rows inside Stage 1's
    20% test partition: 30 objects, ~6 per subtype. A single accuracy on 30 draws
    has a binomial standard error of ~9 percentage points, and a per-subtype
    recall computed on 6 objects moves in steps of 17 points. That resolution is
    coarser than the effect the rebuild is trying to measure, so the single-split
    number cannot by itself tell you whether Stage 2 improved.

    Repeated stratified K-fold over the Stage 2 TRAINING partition reuses every
    training object as a validation object across repeats, giving a mean and a
    spread. It never touches Stage 1's test partition, so it introduces no
    leakage. Report it alongside — not instead of — the held-out number: the
    held-out set remains the only fully untouched evaluation.
    """
    from sklearn.model_selection import RepeatedStratifiedKFold

    n_splits = int(min(n_splits, pd.Series(y_train).value_counts().min()))
    n_splits = max(n_splits, 2)
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                 random_state=random_state)
    rows = []
    for name, model in build_models(random_state=random_state).items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        lo, hi = np.percentile(scores, [2.5, 97.5])
        rows.append({'Model': name, 'Repeated-CV mean': scores.mean(),
                     'Repeated-CV std': scores.std(),
                     'CI 2.5%': lo, 'CI 97.5%': hi,
                     'n_fits': len(scores)})
        if verbose:
            print(f"[{stage_name}] {name:<22} repeated CV {scores.mean():.3f} "
                  f"+/- {scores.std():.3f}  (95% range {lo:.3f}-{hi:.3f}, {len(scores)} fits)")
    return pd.DataFrame(rows)


def single_split_resolution(n_test, n_classes):
    """
    The honest error bar on an accuracy measured from `n_test` objects, plus how
    coarse a per-class recall is at this sample size. Printed next to every
    headline number so nobody over-reads a 30-object result.
    """
    per_class = n_test / n_classes if n_classes else np.nan
    return {
        'n_test': n_test,
        'worst_case_std_error': float(0.5 / np.sqrt(n_test)) if n_test else np.nan,
        'accuracy_step': float(1.0 / n_test) if n_test else np.nan,
        'mean_test_objects_per_class': float(per_class),
        'per_class_recall_step': float(1.0 / per_class) if per_class else np.nan,
    }
