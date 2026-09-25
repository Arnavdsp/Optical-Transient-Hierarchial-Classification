"""
Section 8 — prediction calibration, after de Soto et al. 2024 Figure 11.

The specific claim worth testing: adding redshift can make a model *more confident
without making it more correct*. v4 already has a redshift-ablation diagnostic; this
sharpens it by asking what redshift does to the confidence-vs-accuracy relationship
rather than only to the accuracy.
"""

import numpy as np
import pandas as pd


def calibration_curve(confidences, correct, n_bins=10):
    """
    Binned confidence vs observed fraction correct.

    Returns one row per non-empty bin. A perfectly calibrated model has
    fraction_correct == mean_confidence in every bin.
    """
    conf = np.asarray(confidences, float)
    ok = np.asarray(correct).astype(bool)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
        if not m.any():
            continue
        rows.append({'bin_low': lo, 'bin_high': hi, 'n': int(m.sum()),
                     'mean_confidence': float(conf[m].mean()),
                     'fraction_correct': float(ok[m].mean())})
    return pd.DataFrame(rows)


def expected_calibration_error(curve):
    """ECE: sample-weighted mean |confidence - accuracy| across bins."""
    if curve.empty:
        return float('nan')
    w = curve['n'] / curve['n'].sum()
    return float((w * (curve['mean_confidence'] - curve['fraction_correct']).abs()).sum())


def overconfidence(curve):
    """Signed mean (confidence - accuracy). Positive = overconfident."""
    if curve.empty:
        return float('nan')
    w = curve['n'] / curve['n'].sum()
    return float((w * (curve['mean_confidence'] - curve['fraction_correct'])).sum())


def calibration_for_model(model, X_test, y_test, n_bins=10):
    """Curve + summary stats for a fitted classifier that exposes predict_proba."""
    if not hasattr(model, 'predict_proba'):
        return None
    proba = model.predict_proba(X_test)
    conf = proba.max(axis=1)
    pred = model.classes_[proba.argmax(axis=1)]
    correct = (pred == np.asarray(y_test))
    curve = calibration_curve(conf, correct, n_bins=n_bins)
    return {'curve': curve,
            'ece': expected_calibration_error(curve),
            'overconfidence': overconfidence(curve),
            'mean_confidence': float(conf.mean()),
            'accuracy': float(correct.mean()),
            'n': int(len(conf))}


def compare_with_without_redshift(with_z, without_z, model_name):
    """
    The specific de Soto et al. claim, stated as a testable comparison and answered
    from the numbers rather than asserted.
    """
    if with_z is None or without_z is None:
        return f'{model_name}: calibration unavailable (no predict_proba).'
    d_acc = with_z['accuracy'] - without_z['accuracy']
    d_conf = with_z['mean_confidence'] - without_z['mean_confidence']
    msg = (f"**{model_name}.** With redshift: accuracy {with_z['accuracy']:.3f}, mean "
           f"confidence {with_z['mean_confidence']:.3f}, ECE {with_z['ece']:.3f}. "
           f"Without: accuracy {without_z['accuracy']:.3f}, confidence "
           f"{without_z['mean_confidence']:.3f}, ECE {without_z['ece']:.3f}. ")
    if d_conf > 0.02 and d_acc <= 0.02:
        msg += (f"Confidence rises {d_conf:+.3f} while accuracy moves only {d_acc:+.3f} "
                "— this reproduces de Soto et al. (2024, Fig. 11): redshift makes the "
                "model more confident without making it more correct.")
    elif d_conf > 0.02 and d_acc > 0.02:
        msg += (f"Both confidence ({d_conf:+.3f}) and accuracy ({d_acc:+.3f}) rise, so "
                "here redshift carries real information rather than only inflating "
                "confidence — the de Soto pattern does NOT reproduce in our data.")
    else:
        msg += ("Redshift does not materially change confidence, so the de Soto "
                "overconfidence pattern does not arise either way here.")
    return msg


def plot_calibration(curves, title, out_path=None):
    """Reliability diagram: one line per variant, diagonal = perfect calibration."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    ax.plot([0, 1], [0, 1], ls='--', color='0.5', lw=1, label='perfectly calibrated')
    for name, c in curves.items():
        if c is None or c['curve'].empty:
            continue
        cv = c['curve']
        ax.plot(cv['mean_confidence'], cv['fraction_correct'], marker='o', label=name)
    ax.set_xlabel('mean predicted confidence')
    ax.set_ylabel('observed fraction correct')
    ax.set_title(title, fontsize=11)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=140, bbox_inches='tight')
    return fig
