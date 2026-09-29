"""
Running v4's diagnostics — unchanged — on both label tracks.

v4's diagnostic cells are preserved verbatim. What changes is only what they are
pointed at:

  * **Real rows only.** v4's diagnostics draw their own random train/test splits
    (ablation, redshift check, learning curve, ...). Handed a table containing
    synthetic copies, those splits would put a copy in training and its parent in
    test, and every number would be inflated. They get real objects only.

  * **Both tracks, by rebinding, not copy-pasting.** The TNS pass runs the cells as
    ordinary notebook cells. The ALeRCE pass re-executes the SAME source (a test
    asserts it is byte-identical to the visible cells) with `feature_df_full`,
    `SN_SUBTYPES` and v4's modelling objects rebound to the ALeRCE-labelled track,
    and restored afterwards.

  * **Two cells are TNS-scheme-only.** v4's class-scheme comparison and class-imbalance
    test hardcode the 5-way vocabulary (SN_Ib, SN_Ic). ALeRCE never separates Ib from
    Ic, and its SN scheme already IS the "4-way Ia / Ib-c / II / SLSN" row of the
    class-scheme comparison — so those two run on the TNS track only, for that reason.

  * **v4 modelling objects via adapters.** The cascade and permutation-importance
    cells expect v4's `split` / `stage1` / `stage2a` dictionaries. `v4_adapters`
    presents v5's track output in that shape, rather than editing the cells.
"""

import numpy as np
import pandas as pd

from .modeling import impute

ALERCE_SN_CLASSES = ['SN_Ia', 'SN_Ibc', 'SN_II', 'SLSN']

# v4 notebook cell index -> why it cannot run on the ALeRCE track
TNS_SCHEME_ONLY = {
    35: ('class-scheme comparison: its merge maps are written for the 5-way TNS '
         'vocabulary, and ALeRCE\'s SN scheme is already its "4-way Ia / Ib-c" row'),
    37: ('class-imbalance test: its target proportions name SN_Ib and SN_Ic '
         'separately, which ALeRCE never emits'),
}

_MISSING = object()


def is_real(df):
    s = df['is_synthetic'] if 'is_synthetic' in df.columns else pd.Series(False, index=df.index)
    return ~s.fillna(False).astype(bool)


def tns_training_frame(df):
    """
    What the TNS track trains on: every row with a TNS label that clears v4's full,
    unrelaxed quality gate — real objects plus their synthetic copies (which the split
    keeps parent-aware). ALeRCE-native objects without a redshift are excluded: they
    only cleared a gate with the redshift requirement switched off.
    """
    ok = df['label'].notna()
    if 'passes_v4_gate' in df.columns:
        ok &= df['passes_v4_gate'].fillna(True).astype(bool)
    return df[ok].copy()


def tns_diagnostic_frame(df):
    """TNS track, real objects only — for v4's own random-split diagnostics."""
    t = tns_training_frame(df)
    return t[is_real(t)].copy()


def alerce_track_frame(df, real_only=False):
    """
    The ALeRCE-labelled view: ALeRCE-confident objects with `label` and `coarse_label`
    REPLACED by ALeRCE's crosswalked output, so v4's code (which reads `label`) sees
    ALeRCE's labels. The TNS label is kept alongside as `tns_label`.
    """
    col = 'in_alerce_pool' if 'in_alerce_pool' in df.columns else None
    f = df[df[col]] if col else df[df['alerce_coarse'].notna()]
    f = f.copy()
    f['tns_label'] = f['label']
    f['coarse_label'] = f['alerce_coarse']
    f['label'] = np.where(f['alerce_coarse'] == 'SNe', f['alerce_fine'], f['alerce_coarse'])
    f = f[f['label'].notna()]
    if real_only:
        f = f[is_real(f)]
    return f


def v4_adapters(track):
    """
    v5 track output -> v4's (split, stage1, stage2a) dictionaries.

    v4 fed RAW features into its scalers, which worked because v4 imputed the whole
    table up front. v5's table carries structural NaNs (host offset, native redshift),
    so `split['X']` here is imputed with Stage 1's TRAINING medians — the same values
    Stage 1 itself used — keeping v4's cascade code valid without editing it.
    """
    s1 = track['stage1']
    sp1 = s1['split']
    X_imp = impute(sp1['X'][sp1['idx_train']], sp1['X'])[1]
    split = {'feature_df': sp1['df'], 'X': X_imp,
             'idx_train': sp1['idx_train'], 'idx_test': sp1['idx_test']}

    def stage(st):
        if st is None or st.get('models') is None:
            return None
        sp = st['split']
        return {'models': st['models'], 'scaler': sp['scaler'], 'le': sp['le'],
                'X_test': sp['X_test'], 'y_test': sp['y_test'],
                'results': st['results'].rename(columns={'Macro-F1': 'Macro F1'})}

    return split, stage(s1), stage(track.get('stage2'))


def enough_per_class(frame, classes, min_per_class):
    counts = frame['label'].value_counts()
    short = {c: int(counts.get(c, 0)) for c in classes if counts.get(c, 0) < min_per_class}
    return (not short), short


def run_v4_cells(sources, namespace, bindings, label, min_per_class=8, verbose=True):
    """
    Execute v4 diagnostic cell sources with `bindings` temporarily in scope, then put
    every rebound name back exactly as it was (including removing names that did not
    exist before). Skips — loudly — when any SN class is too small to split.
    """
    frame = bindings.get('feature_df_full')
    classes = bindings.get('SN_SUBTYPES', [])
    if frame is not None and classes:
        ok, short = enough_per_class(frame, classes, min_per_class)
        if not ok:
            if verbose:
                print(f'[{label}] skipped: too few objects to split in {short} '
                      f'(need >= {min_per_class} per class)')
            return False
    saved = {k: namespace.get(k, _MISSING) for k in bindings}
    try:
        namespace.update(bindings)
        for i, src in sources:
            if verbose:
                print(f'\n----- [{label}] v4 diagnostic cell {i} -----')
            exec(compile(src, f'<v4 diagnostic {i}, {label}>', 'exec'), namespace)
    finally:
        for k, v in saved.items():
            if v is _MISSING:
                namespace.pop(k, None)
            else:
                namespace[k] = v
    return True


def v4_impute(frame, feature_cols):
    """
    Impute a frame the way v4's assemble_feature_df did, so v4's diagnostics see the
    fully imputed table they were written for.

    v4 filled every feature gap up front: gr_* and all other feature columns with a
    CLASS-INDEPENDENT median, peak_abs_mag with a 0.0 sentinel (far from any SN value,
    so a tree can split cleanly on "not extragalactic"). v5 adds columns after that
    step — host offset, redshift-less ALeRCE-native rows, synthetic copies — so a
    column can be entirely NaN (e.g. every host column, when astro-ghost cannot
    resolve hosts), and HistGradientBoosting's binner raises on an all-NaN column.
    A column with no values at all gets 0.0; it is then constant and carries nothing.

    Used only for v4's diagnostics. Training (run_track) imputes with TRAINING-fold
    medians inside each split instead, which is the leak-free choice.
    """
    out = frame.copy()
    for c in feature_cols:
        if c not in out.columns:
            continue
        col = pd.to_numeric(out[c], errors='coerce')
        if c == 'peak_abs_mag':
            out[c] = col.fillna(0.0)
            continue
        med = col.median()
        out[c] = col.fillna(med if np.isfinite(med) else 0.0)
    return out
