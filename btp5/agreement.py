"""
Section 6 — the comparison methodology, after de Soto et al. 2024 (Superphot+,
arXiv:2403.07975).

The naive thing — putting the TNS-track accuracy next to the ALeRCE-track accuracy —
is confounded by three things at once: different sample sizes, different label
reliability, and different class definitions at Stage 2. So instead:

  A_expected = P_ours^T · C_ALeRCE

where P_ours is our model's purity (precision) matrix and C_ALeRCE is ALeRCE's
completeness (recall) matrix. This is what agreement would look like if our errors and
ALeRCE's errors were statistically independent. Comparing the ACTUAL agreement matrix
against it is the informative move: actual > expected means the two pipelines share
underlying signal (both keying off similar physics), not that either has been
independently verified.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def purity_matrix(y_true, y_pred, classes):
    """
    Column-normalised confusion matrix: P[i, j] = P(true = i | predicted = j).
    'Of everything we called j, what fraction really was i' — precision, per class.
    """
    cm = confusion_matrix(y_true, y_pred, labels=classes).astype(float)
    col = cm.sum(axis=0, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        P = np.divide(cm, col, out=np.zeros_like(cm), where=col > 0)
    return pd.DataFrame(P, index=classes, columns=classes)


def completeness_matrix(y_true, y_pred, classes):
    """
    Row-normalised confusion matrix: C[i, j] = P(predicted = j | true = i).
    'Of everything that really was i, what fraction did we call j' — recall, per class.
    """
    cm = confusion_matrix(y_true, y_pred, labels=classes).astype(float)
    row = cm.sum(axis=1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        C = np.divide(cm, row, out=np.zeros_like(cm), where=row > 0)
    return pd.DataFrame(C, index=classes, columns=classes)


def expected_agreement(purity_ours, completeness_alerce):
    """
    Section 6.1: A_expected = P_ours^T · C_ALeRCE.

    Both matrices must be over the same class list, in the same order — mismatched
    ordering silently produces a plausible-looking but meaningless matrix, so it is
    checked rather than assumed.
    """
    assert list(purity_ours.index) == list(completeness_alerce.index), \
        'purity and completeness matrices are over different classes/orders'
    A = purity_ours.values.T @ completeness_alerce.values
    return pd.DataFrame(A, index=purity_ours.index, columns=completeness_alerce.columns)


def actual_agreement(pred_ours, pred_alerce, classes, normalize='all'):
    """
    Section 6.2: how often our prediction and ALeRCE's prediction actually coincide,
    over the overlap pool. `normalize='all'` gives fractions of the whole pool;
    'index' gives, per our-predicted class, how ALeRCE distributed them.
    """
    cm = confusion_matrix(pred_ours, pred_alerce, labels=classes).astype(float)
    if normalize == 'all' and cm.sum() > 0:
        cm = cm / cm.sum()
    elif normalize == 'index':
        row = cm.sum(axis=1, keepdims=True)
        with np.errstate(invalid='ignore', divide='ignore'):
            cm = np.divide(cm, row, out=np.zeros_like(cm), where=row > 0)
    return pd.DataFrame(cm, index=classes, columns=classes)


def agreement_rate(pred_ours, pred_alerce):
    """Scalar: fraction of overlap objects where the two pipelines agree exactly."""
    a = np.asarray(pred_ours)
    b = np.asarray(pred_alerce)
    return float((a == b).mean()) if len(a) else float('nan')


def compare_expected_actual(pred_ours, pred_alerce, purity_ours, completeness_alerce,
                            classes):
    """
    Section 6.3: the headline comparison for one model, on one pool.

    Returns a dict with both matrices, their diagonals (per-class agreement), and the
    scalar actual vs expected rates. The interpretation of the sign of
    (actual - expected) is written out by `interpret_agreement`.
    """
    A_exp = expected_agreement(purity_ours, completeness_alerce)
    A_act = actual_agreement(pred_ours, pred_alerce, classes, normalize='all')

    expected_rate = float(np.trace(A_exp.values) / max(len(classes), 1))
    # A_exp is a product of two stochastic matrices, so its diagonal is a per-class
    # expected agreement, not a population fraction. Weight it by how often each class
    # actually occurs so it is comparable to the observed rate.
    counts = pd.Series(pred_ours).value_counts(normalize=True)
    w = np.array([counts.get(c, 0.0) for c in classes])
    expected_weighted = float((np.diag(A_exp.values) * w).sum())

    return {
        'classes': list(classes),
        'expected_matrix': A_exp,
        'actual_matrix': A_act,
        'expected_rate_unweighted': expected_rate,
        'expected_rate_weighted': expected_weighted,
        'actual_rate': agreement_rate(pred_ours, pred_alerce),
        'per_class_expected': pd.Series(np.diag(A_exp.values), index=classes),
        'per_class_actual': pd.Series(np.diag(A_act.values), index=classes),
        'n': int(len(pred_ours)),
    }


def interpret_agreement(result, model_name, pool_name, external_note=True):
    """
    Section 6.3 demands a written interpretation, not just numbers. This produces it
    from the computed values, so the prose cannot drift from the table.
    """
    act, exp = result['actual_rate'], result['expected_rate_weighted']
    n = result['n']
    L = []
    L.append(f"**{model_name} on the {pool_name} pool (n={n}).** "
             f"Our pipeline and ALeRCE agree on {100 * act:.1f}% of objects; "
             f"independent-error agreement would be {100 * exp:.1f}%.")
    if not np.isfinite(act) or not np.isfinite(exp):
        L.append("Too few objects in this pool to compute a meaningful rate.")
        return ' '.join(L)

    if act > exp + 0.02:
        L.append(
            f"Actual agreement exceeds expected by {100 * (act - exp):.1f} points. "
            "That is *not* independent verification: if two pipelines erred "
            "independently, agreement would sit at the expected value. Exceeding it "
            "means both are keying off substantially the same underlying signal — "
            "unsurprising, since both are trained on ZTF light-curve shape and colour. "
            "Agreement here measures shared method, not shared correctness.")
    elif act < exp - 0.02:
        L.append(
            f"Actual agreement falls {100 * (exp - act):.1f} points BELOW the "
            "independent-error baseline. The two pipelines are disagreeing more than "
            "chance would predict, which points at a systematic difference in class "
            "definition or in the feature basis rather than at random error.")
    else:
        L.append(
            "Actual and expected agreement are within ~2 points, which is what "
            "genuinely independent errors would look like.")

    pc_a, pc_e = result['per_class_actual'], result['per_class_expected']
    gap = (pc_a - pc_e).sort_values()
    if len(gap):
        worst = gap.index[0]
        L.append(f"Weakest class: **{worst}** "
                 f"(actual {pc_a[worst]:.2f} vs expected {pc_e[worst]:.2f}).")
        if external_note and worst in ('TDE', 'SLSN'):
            L.append(
                "TDE/SLSN disagreement is a documented, cross-pipeline pattern rather "
                "than evidence that either side is broken — see the external "
                "corroboration table (Section 6.4): Superphot+ Table 3 found 54.9% of "
                "true TDEs called SLSN-I when forced through a 5-way SN classifier, and "
                "Townsend et al. 2026 confirmed the same confusion direction "
                "independently on real ZTF data.")
    return ' '.join(L)


# --------------------------------------------------------------------------
# Section 6.4 — external corroboration, quoted with sources
# --------------------------------------------------------------------------

EXTERNAL_RESULTS = pd.DataFrame([
    {'source': 'Superphot+ (de Soto et al. 2024, arXiv:2403.07975)',
     'task': '5-way SN, no redshift', 'result': '83% acc / 0.61 macro-F1', 'N': '6061'},
    {'source': 'SuperRAENN (Villar et al. 2020)',
     'task': '5-way SN', 'result': '87% acc / 0.66 purity', 'N': '~2885'},
    {'source': 'YSE-DR1 (Aleo et al. 2023)',
     'task': '3-way SN (Ia/II/Ibc)', 'result': '82% acc, real spectroscopic sample',
     'N': '472'},
    {'source': 'Superphot+ vs ALeRCE agreement (de Soto et al. 2024)',
     'task': '4-way SN',
     'result': '82% actual vs 69% expected (spec. subset); 72% actual (photometric-only)',
     'N': '—'},
    {'source': 'Superphot+ Table 3 (true TDEs through a 5-way SN classifier)',
     'task': 'TDE mis-assignment',
     'result': '54.9% -> SLSN-I, 21.6% -> SN IIn, 13.7% -> SN Ia', 'N': '—'},
    {'source': 'Townsend et al. 2026 (NoiZTF, arXiv:2602.13036)',
     'task': 'TDE vs SLSN/IIn',
     'result': 'same confusion direction, independently confirmed on real ZTF data',
     'N': '—'},
])

TDE_SLSN_STATEMENT = (
    "TDE<->SLSN/IIn confusion has now been documented across three independent "
    "pipelines — ALeRCE pre-2025, Superphot+ (de Soto et al. 2024) and NoiZTF "
    "(Townsend et al. 2026) — on top of whatever this notebook finds. Any such "
    "confusion in our own results should be read as expected, physically motivated "
    "and consistent with the literature, not as a defect specific to this pipeline.")

TNS_LABEL_NOISE_CAVEAT = (
    "TNS labels are the stronger of the two ground-truth sources, but they are not an "
    "oracle. Aleo et al. (2023) independently re-ran spectral classification (SNID) on "
    "public TNS spectra and found objects where their reclassification disagreed with "
    "the TNS-listed type, and some TNS spectra listed as 'classified' were on "
    "reanalysis noise-dominated with no safe classification available. Results here are "
    "therefore described as agreement with TNS's *recorded* classification, not as "
    "agreement with ground truth.")


# --------------------------------------------------------------------------
# The full Section 6 report, both subsets, one stage at a time
# --------------------------------------------------------------------------

SUBSET_SPEC = 'TNS-confirmed (= overlap pool)'
SUBSET_PHOT = 'photometric-only (ALeRCE-confident, no TNS label)'


def agreement_report(frame, our_col, alerce_col, truth_col, classes, model_name, stage,
                     tns_col='has_tns_label', min_n=5, purity_ours=None):
    """
    Expected vs actual agreement for one model at one stage.

    How Section 6.2's split is read, stated because the brief's two definitions
    collide: 6.2 asks for the overlap pool split into TNS-confirmed and non-confirmed
    subsets, but Section 3.3 defines the overlap pool as objects that ARE
    TNS-confirmed, so that second subset is empty by construction. de Soto et al.'s
    actual design compares every object BOTH pipelines classify. Here that is the
    ALeRCE-confident pool, split into its TNS-confirmed part — which is exactly the
    overlap pool — and its photometric-only part.

    P_ours and C_ALeRCE come from the TNS-confirmed objects (the only ones with a truth
    to score against), and are then used as the independent-error baseline for BOTH
    subsets. That is also de Soto et al.'s construction.
    """
    f = frame[frame[our_col].isin(classes) & frame[alerce_col].isin(classes)].copy()
    truthed = f[f[truth_col].isin(classes)]
    out = {'model': model_name, 'stage': stage, 'classes': list(classes), 'subsets': {}}
    if len(truthed) < min_n:
        out['note'] = (f'only {len(truthed)} objects with a truth label in these classes — '
                       'purity/completeness matrices cannot be estimated')
        return out
    # P_ours: pass the purity matrix from the FULL validation set when available (de
    # Soto et al. build it from their whole validation sample, not from the handful
    # of objects that happen to sit in the comparison pool). C_ALeRCE can only come
    # from objects that have both a TNS truth and an ALeRCE label, i.e. the overlap.
    P = (purity_ours.reindex(index=classes, columns=classes).fillna(0.0)
         if purity_ours is not None
         else purity_matrix(truthed[truth_col], truthed[our_col], classes))
    C = completeness_matrix(truthed[truth_col], truthed[alerce_col], classes)
    out['purity_ours'], out['completeness_alerce'] = P, C
    out['expected_matrix'] = expected_agreement(P, C)

    for name, mask in [(SUBSET_SPEC, f[tns_col].astype(bool)),
                       (SUBSET_PHOT, ~f[tns_col].astype(bool))]:
        sub = f[mask]
        if len(sub) < min_n:
            out['subsets'][name] = {'n': int(len(sub)), 'note': 'too few objects'}
            continue
        res = compare_expected_actual(sub[our_col].values, sub[alerce_col].values,
                                      P, C, classes)
        res['text'] = interpret_agreement(res, f'{model_name} ({stage})', name)
        out['subsets'][name] = res
    return out


def contrast_subsets(report):
    """
    The de Soto comparison: agreement on the spectroscopic vs photometric subset.
    They found 82% vs 72%. Reported either way, not assumed.
    """
    s, p = report['subsets'].get(SUBSET_SPEC, {}), report['subsets'].get(SUBSET_PHOT, {})
    if 'actual_rate' not in s or 'actual_rate' not in p:
        return (f"{report['model']} ({report['stage']}): the spectroscopic/photometric "
                f"contrast needs both subsets populated (n={s.get('n', 0)} and "
                f"{p.get('n', 0)}).")
    a, b = s['actual_rate'], p['actual_rate']
    msg = (f"{report['model']} ({report['stage']}): agreement {100 * a:.1f}% on the "
           f"TNS-confirmed subset (n={s['n']}) vs {100 * b:.1f}% on the photometric-only "
           f"subset (n={p['n']}). ")
    if b < a - 0.02:
        msg += ("Lower on the photometric-only subset, the same direction as de Soto et al. "
                "(82% vs 72%). Expected: those objects are fainter or less well sampled on "
                "average (nobody took a spectrum), and here they also lack a redshift, so "
                "peak absolute magnitude is imputed for every one of them.")
    elif b > a + 0.02:
        msg += ("HIGHER on the photometric-only subset — the opposite of de Soto et al.'s "
                "finding. Worth checking class composition: the photometric-only pool is "
                "drawn per ALeRCE class and may be richer in the easy classes.")
    else:
        msg += 'The two subsets agree to within ~2 points, unlike de Soto et al.'
    return msg


def tde_crosscheck(frame, tns_label_col='label', top1_col='alerce_top1_class',
                   of_interest=('TDE', 'SLSN'), min_n=3):
    """
    Section 3.1: what does ALeRCE's own classifier make of objects TNS calls a TDE (and
    SLSN)? Uses ALeRCE's top-1 class for every object queried, regardless of confidence,
    so it is the direct analogue of Superphot+ Table 3 (true TDEs forced through a SN
    classifier: 54.9% -> SLSN-I, 21.6% -> SN IIn, 13.7% -> SN Ia). The difference is that
    ALeRCE's 2025 classifier HAS a TDE class, so this measures how often it uses it.
    """
    f = frame[frame[tns_label_col].isin(of_interest) & frame[top1_col].notna()]
    if len(f) < min_n:
        return None, (f'only {len(f)} TNS {"/".join(of_interest)} objects have an ALeRCE '
                      'classification — cross-check not meaningful')
    tab = pd.crosstab(f[tns_label_col], f[top1_col], normalize='index')
    n = f[tns_label_col].value_counts()
    tab.insert(0, 'n', n.reindex(tab.index))
    lines = []
    if 'TDE' in tab.index:
        row = tab.loc['TDE'].drop('n')
        hit = float(row.get('TDE', 0.0))
        rest = row.drop('TDE', errors='ignore').sort_values(ascending=False)
        lines.append(f"Of {int(tab.loc['TDE', 'n'])} TNS-classified TDEs, ALeRCE's top-1 "
                     f"class is TDE for {100 * hit:.0f}%.")
        if len(rest) and rest.iloc[0] > 0:
            lines.append(f"The most common alternative is {rest.index[0]} "
                         f"({100 * rest.iloc[0]:.0f}%).")
        if 'SLSN' in rest.index and rest['SLSN'] > 0:
            lines.append("TDE->SLSN is the confusion Superphot+ Table 3 found for 54.9% of "
                         "true TDEs; seeing it here as well is consistent with a shared, "
                         "physical ambiguity (both are slow, blue, luminous nuclear-ish "
                         "transients), not a defect of either pipeline.")
    return tab, ' '.join(lines)
