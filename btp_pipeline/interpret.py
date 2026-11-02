"""
Phase 4 — interpretation and physical insight, applied to BOTH stages.

Covers the brief's Phase 4 deliverables: feature importance, most discriminative
features, misclassification analysis, confusion matrices, decision boundaries,
and the physics connection.

Design correction E: permutation importance is computed uniformly for all four
models on the held-out test set. Native importances (RF Gini, LR coefficients,
mean Gini across bagged trees) are reported alongside where they exist, but they
are not comparable across model families and an RBF-kernel bagged SVM has none
at all — so permutation importance is the one method that puts all four on the
same axis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay


# --------------------------------------------------------------------------
# Feature importance
# --------------------------------------------------------------------------

def permutation_importances(models, X_test, y_test, x_cols, n_repeats=20,
                            random_state=42, n_jobs=None):
    """
    Permutation importance for every model, on the held-out test set.
    Returns a tidy frame: one row per (model, feature) with mean and std.
    """
    rows = []
    for name, model in models.items():
        r = permutation_importance(model, X_test, y_test, n_repeats=n_repeats,
                                   random_state=random_state, scoring='accuracy',
                                   n_jobs=n_jobs)
        for feat, mean, std in zip(x_cols, r.importances_mean, r.importances_std):
            rows.append({'Model': name, 'feature': feat,
                         'perm_importance': float(mean), 'perm_std': float(std)})
    return pd.DataFrame(rows)


def native_importances(models, x_cols, le=None):
    """
    Each family's own importance measure, where one exists:
      Random Forest    -> Gini importance
      Bagging (Trees)  -> mean Gini across the bagged trees (features are
                          subsampled per estimator, so importances are mapped
                          back onto the full feature vector before averaging)
      LogReg           -> mean |standardised coefficient| across classes
      Bagging (SVM)    -> none; RBF SVMs expose neither coefficients nor
                          impurity gains. This is exactly why permutation
                          importance is the cross-family method.
    """
    rows = []
    for name, model in models.items():
        if hasattr(model, 'feature_importances_'):
            vals = np.asarray(model.feature_importances_, dtype=float)
            kind = 'Gini importance'
        elif hasattr(model, 'coef_'):
            vals = np.abs(np.atleast_2d(model.coef_)).mean(axis=0)
            kind = 'mean |standardised coef|'
        elif hasattr(model, 'estimators_') and hasattr(model.estimators_[0], 'feature_importances_'):
            acc = np.zeros(len(x_cols), dtype=float)
            cnt = np.zeros(len(x_cols), dtype=float)
            feats_per_est = getattr(model, 'estimators_features_', None)
            for est, feats in zip(model.estimators_, feats_per_est):
                acc[list(feats)] += est.feature_importances_
                cnt[list(feats)] += 1
            vals = np.divide(acc, np.maximum(cnt, 1))
            kind = 'mean Gini across bagged trees'
        else:
            continue
        for feat, v in zip(x_cols, vals):
            rows.append({'Model': name, 'feature': feat,
                         'native_importance': float(v), 'native_kind': kind})
    return pd.DataFrame(rows)


def plot_permutation_importance(perm_df, title, out_path=None):
    """One horizontal-bar panel per model, shared feature ordering."""
    models = list(dict.fromkeys(perm_df['Model']))
    order = (perm_df.groupby('feature')['perm_importance'].mean()
             .sort_values(ascending=False).index.tolist())
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, models):
        sub = perm_df[perm_df['Model'] == name].set_index('feature').loc[order]
        ax.barh(sub.index, sub['perm_importance'], xerr=sub['perm_std'],
                color='seagreen', alpha=0.85)
        ax.invert_yaxis()
        ax.axvline(0, color='0.4', lw=0.8)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel('Δ accuracy when shuffled')
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=140, bbox_inches='tight')
    return fig


def rank_discriminative_features(perm_df):
    """Mean permutation importance across the four models — the headline
    'most discriminative features' answer for the brief."""
    return (perm_df.groupby('feature')['perm_importance']
            .agg(['mean', 'std']).sort_values('mean', ascending=False)
            .rename(columns={'mean': 'mean_perm_importance', 'std': 'across_model_std'}))


# --------------------------------------------------------------------------
# Confusion matrices
# --------------------------------------------------------------------------

def plot_confusions(confusions, class_names, title, accuracies=None, out_path=None):
    """A row of confusion matrices, one per model."""
    names = list(confusions)
    size = max(4.5, len(class_names) * 0.95)
    fig, axes = plt.subplots(1, len(names), figsize=(size * len(names), size))
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        disp = ConfusionMatrixDisplay(confusions[name], display_labels=class_names)
        disp.plot(ax=ax, cmap='Blues', colorbar=False, xticks_rotation=45, values_format='d')
        sub = f"{name}"
        if accuracies is not None and name in accuracies:
            sub += f"  (acc {accuracies[name]:.3f})"
        ax.set_title(sub, fontsize=11)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=140, bbox_inches='tight')
    return fig


# --------------------------------------------------------------------------
# Misclassification analysis
# --------------------------------------------------------------------------

def misclassification_table(y_true_labels, y_pred_labels, top_n=10):
    """Which true->predicted confusions actually dominate the error budget."""
    err = pd.DataFrame({'true': y_true_labels, 'pred': y_pred_labels})
    err = err[err['true'] != err['pred']]
    if err.empty:
        return pd.DataFrame(columns=['true', 'pred', 'n', 'share_of_errors'])
    tab = (err.groupby(['true', 'pred']).size().reset_index(name='n')
           .sort_values('n', ascending=False))
    tab['share_of_errors'] = tab['n'] / tab['n'].sum()
    return tab.head(top_n).reset_index(drop=True)


def per_class_recall(y_true_labels, y_pred_labels):
    df = pd.DataFrame({'true': y_true_labels, 'pred': y_pred_labels})
    return (df.assign(correct=df['true'] == df['pred'])
            .groupby('true')['correct'].agg(['mean', 'size'])
            .rename(columns={'mean': 'recall', 'size': 'n_test'})
            .sort_values('recall'))


# --------------------------------------------------------------------------
# Decision boundaries (2D projection)
# --------------------------------------------------------------------------

def plot_decision_boundaries(feature_df, idx_train, idx_test, label_col, two_features,
                             build_models_fn, title, out_path=None, grid_steps=250,
                             random_state=42):
    """
    2D decision-boundary panels, one per model, on the two most discriminative
    features (we have 6, so this is a projection, not the real boundary).

    The models here are refit on those two features using the TRAINING rows only,
    and the TEST rows are scattered on top. Fitting the visualisation model on the
    whole dataset — as an earlier version of this notebook did — draws a boundary
    that has already seen the points it is being judged against, which makes the
    picture look tidier than the classifier actually is.
    """
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    f1, f2 = two_features
    X2 = feature_df[[f1, f2]].values.astype(float)
    le = LabelEncoder()
    y2 = le.fit_transform(feature_df[label_col].values)

    scaler = StandardScaler()
    X2_train = scaler.fit_transform(X2[idx_train])
    X2_test = scaler.transform(X2[idx_test])
    y_train, y_test = y2[idx_train], y2[idx_test]

    pad = 0.6
    x_min, x_max = X2_train[:, 0].min() - pad, X2_train[:, 0].max() + pad
    y_min, y_max = X2_train[:, 1].min() - pad, X2_train[:, 1].max() + pad
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_steps),
                         np.linspace(y_min, y_max, grid_steps))
    grid = np.c_[xx.ravel(), yy.ravel()]

    models = build_models_fn(random_state=random_state)
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.8))
    axes = np.atleast_1d(axes)
    cmap = plt.get_cmap('tab10')
    for ax, (name, model) in zip(axes, models.items()):
        model.fit(X2_train, y_train)
        zz = model.predict(grid).reshape(xx.shape)
        ax.contourf(xx, yy, zz, alpha=0.25, levels=np.arange(-0.5, len(le.classes_)), cmap='tab10')
        for k, cls in enumerate(le.classes_):
            m = y_test == k
            ax.scatter(X2_test[m, 0], X2_test[m, 1], s=26, color=cmap(k % 10),
                       edgecolor='k', linewidth=0.4, label=cls)
        acc = float((model.predict(X2_test) == y_test).mean())
        ax.set_title(f"{name}  (2-feature test acc {acc:.2f})", fontsize=10)
        ax.set_xlabel(f'{f1} (standardised)')
        ax.set_ylabel(f'{f2} (standardised)')
    axes[0].legend(fontsize=8, loc='best')
    fig.suptitle(f"{title} — projection onto {f1} vs {f2}; points are held-out test objects",
                 fontsize=12)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=140, bbox_inches='tight')
    return fig


# --------------------------------------------------------------------------
# Physics summary
# --------------------------------------------------------------------------

def physics_summary(feature_df, label_col, features):
    """Per-class mean/std of every physical feature — the table the write-up
    has to be consistent with."""
    return feature_df.groupby(label_col)[features].agg(['mean', 'std']).round(3)


def class_separation_ranking(feature_df, label_col, features):
    """
    A crude but useful 'how separable is this feature' score: between-class
    variance of the class means divided by the mean within-class variance
    (a one-way F-statistic in spirit). Complements permutation importance,
    which is model-mediated, with something purely about the data.
    """
    rows = []
    for f in features:
        g = feature_df.groupby(label_col)[f]
        between = g.mean().var(ddof=1)
        within = g.var(ddof=1).mean()
        rows.append({'feature': f, 'between_class_var': float(between),
                     'mean_within_class_var': float(within),
                     'separation_ratio': float(between / within) if within else np.nan})
    return pd.DataFrame(rows).sort_values('separation_ratio', ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# Feature-construction diagnostics
# --------------------------------------------------------------------------

def feature_diagnostics(feature_df, features, survey_col='survey'):
    """
    Check what the features actually measure, as opposed to what they are named.

    Two problems in this feature set were found only by looking at the real
    numbers, and both are properties of how the features are *constructed*, so
    they will recur on any re-run until the extractor changes:

    1. **Redundancy.** `process_ztf_object` sets the flux baseline to the faintest
       magnitude in the light curve, so relative flux has a minimum of exactly 1.0
       for every ZTF object. `amplitude = peak - min` is then identically
       `peak_val - 1`: the two columns are the same feature twice, and the ZTF part
       of the dataset really has five features, not six.

    2. **Timescales are observing baselines.** `extract_shape_features` measures
       rise and decay across the whole interpolated light curve, so for an object
       with years of archival ZTF coverage `rise_time` is "time from the first
       archival detection to peak", not the photometric rise of the transient. That
       quantity correlates with class through the observing strategy rather than
       the astrophysics — persistently variable AGN accumulate long histories,
       while TESS flares are capped at the ~27-day sector length.

    Returns a dict of measured evidence; it asserts nothing on its own.
    """
    out = {'collinear_pairs': [], 'baseline': {}}

    # 1. exact linear dependence between feature pairs, per survey
    surveys = ([None] if survey_col not in feature_df else
               list(pd.unique(feature_df[survey_col])))
    for sv in surveys:
        sub = feature_df if sv is None else feature_df[feature_df[survey_col] == sv]
        if len(sub) < 3:
            continue
        for i, a in enumerate(features):
            for b in features[i + 1:]:
                diff = (sub[a] - sub[b]).round(9)
                r = sub[a].corr(sub[b])
                if diff.nunique() == 1:
                    out['collinear_pairs'].append({
                        'survey': sv, 'feature_a': a, 'feature_b': b,
                        'relation': f'{a} = {b} + {float(diff.iloc[0]):.6g}',
                        'pearson_r': float(r), 'n': int(len(sub)),
                    })

    # 2. how much of the "timescale" is really observing baseline
    if {'rise_time', 'decay_time'}.issubset(feature_df.columns):
        span = feature_df['rise_time'] + feature_df['decay_time']
        out['baseline'] = {
            'median_days': float(span.median()),
            'max_days': float(span.max()),
            'frac_over_1yr': float((span > 365).mean()),
            'frac_over_3yr': float((span > 1095).mean()),
            'by_class': feature_df.assign(_span=span).groupby('label')['_span']
                        .median().sort_values(ascending=False).to_dict(),
        }
    return out


def format_feature_diagnostics(diag):
    """Render `feature_diagnostics` output as markdown for the results summary."""
    lines = ['\n## Feature-construction issues found in the real data\n']
    pairs = diag.get('collinear_pairs', [])
    if pairs:
        lines.append("**Two features are one feature.** Measured on the real table:\n")
        for p in pairs:
            lines.append(f"- On {p['survey']} objects (n={p['n']}): `{p['relation']}`, "
                         f"Pearson r = {p['pearson_r']:.6f}.")
        lines.append(
            "\nThis is structural, not a coincidence: `process_ztf_object` uses the faintest "
            "magnitude as the flux baseline, so relative flux bottoms out at exactly 1.0 and "
            "`amplitude = peak - min` reduces to `peak_val - 1`. The ZTF half of the dataset "
            "therefore carries five independent features, not six. Nothing here is *wrong* — "
            "the models are not harmed by a duplicated column — but any statement of the form "
            "'peak brightness and amplitude both matter' is counting one quantity twice, and "
            "their permutation importances should be read as a single shared contribution.\n")
    b = diag.get('baseline') or {}
    if b:
        lines.append("**`rise_time` and `decay_time` partly measure the observing baseline, "
                     "not the transient.**\n")
        lines.append(f"- Median interpolated span: {b['median_days']:.0f} days; "
                     f"longest: {b['max_days']:.0f} days.")
        lines.append(f"- {100 * b['frac_over_1yr']:.1f}% of objects span more than a year, "
                     f"{100 * b['frac_over_3yr']:.1f}% more than three years.")
        lines.append("- Median span by class: " + ', '.join(
            f"{k} {v:.0f} d" for k, v in list(b['by_class'].items())))
        lines.append(
            "\nA supernova rises in roughly 10-30 days, so a span of years is archival "
            "coverage of the position, not the event. `extract_shape_features` measures "
            "across the whole interpolated curve, so for objects with long ZTF histories "
            "these columns encode how long the field has been monitored. That correlates "
            "with class through observing strategy rather than physics: persistently "
            "variable AGN accumulate the longest histories, while TESS flares are capped "
            "at the ~27-day sector length. Some of Stage 1's separability on these features "
            "is therefore survey signature.\n"
            "\n**The fix**, if you want these features to mean what they are named: window "
            "the light curve around the detected peak (say peak minus 50 days to peak plus "
            "150 days) before fitting the GP, and measure rise and decay inside that window. "
            "That is a change to `extract_shape_features` / `process_ztf_object`, which this "
            "rebuild deliberately left untouched, so it is flagged rather than applied.\n")
    return '\n'.join(lines)
