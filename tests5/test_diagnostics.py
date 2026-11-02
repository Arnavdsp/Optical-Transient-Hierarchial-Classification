"""Running v4's diagnostics on both tracks without copy-paste or leakage."""
import numpy as np
import pandas as pd
import pytest

from btp5 import diagnostics as D
from btp5.config import FEATURE_COLS
from btp5.modeling import run_track
import synthetic5


def _frame():
    df = synthetic5.make_feature_df()
    df['has_tns_label'] = True
    df['in_alerce_pool'] = df['alerce_coarse'].notna()
    return df


def test_diagnostic_frame_is_real_rows_only():
    df = _frame()
    kids = df.head(20).copy()
    kids['is_synthetic'], kids['parent_id'] = True, kids['id']
    kids['id'] = kids['id'] + '__aug0'
    full = pd.concat([df, kids], ignore_index=True)
    assert D.tns_training_frame(full)['is_synthetic'].sum() == 20, 'copies DO train'
    assert D.tns_diagnostic_frame(full)['is_synthetic'].sum() == 0, \
        'v4 random-split diagnostics must never see a copy'


def test_native_rows_without_redshift_stay_out_of_the_tns_track():
    df = _frame()
    df['passes_v4_gate'] = True
    df.loc[df.index[:5], 'passes_v4_gate'] = False
    assert len(D.tns_training_frame(df)) == len(df) - 5


def test_alerce_frame_puts_alerce_labels_where_v4_code_reads_them():
    f = D.alerce_track_frame(_frame())
    assert (f['coarse_label'] == f['alerce_coarse']).all()
    sn = f['alerce_coarse'] == 'SNe'
    assert (f.loc[sn, 'label'] == f.loc[sn, 'alerce_fine']).all()
    assert (f.loc[~sn, 'label'] == f.loc[~sn, 'alerce_coarse']).all()
    assert 'tns_label' in f.columns
    assert set(f.loc[sn, 'label']) <= set(D.ALERCE_SN_CLASSES)
    assert (f['survey'] == 'ZTF').all(), 'TESS objects cannot be in the ALeRCE track'


def test_adapters_reproduce_v4_shapes_and_run_v4_cascade_logic():
    df = _frame()
    t = run_track(df, FEATURE_COLS, 'coarse_label', 'label', 'TNS', verbose=False)
    split, s1, s2 = D.v4_adapters(t)
    assert not np.isnan(split['X']).any(), 'v4 cascade feeds X straight to the scaler'
    for st in (s1, s2):
        assert {'models', 'scaler', 'le', 'X_test', 'y_test', 'results'} <= set(st)
        assert 'Macro F1' in st['results'].columns
    # v4's cascade_predict, verbatim in spirit: stage1 -> route SNe -> stage2
    Xte = split['X'][split['idx_test']]
    for name in s1['models']:
        coarse = s1['le'].inverse_transform(s1['models'][name].predict(s1['scaler'].transform(Xte)))
        mask = coarse == 'SNe'
        fine = s2['le'].inverse_transform(s2['models'][name].predict(s2['scaler'].transform(Xte[mask])))
        assert len(fine) == mask.sum()


def test_runner_rebinds_then_restores_exactly():
    ns = {'feature_df_full': 'ORIGINAL', 'SN_SUBTYPES': ['SN_Ia']}
    frame = pd.DataFrame({'label': ['SN_Ia'] * 10 + ['SN_II'] * 10})
    src = [(1, "seen = (feature_df_full is not None, list(SN_SUBTYPES), NEW_NAME)")]
    ok = D.run_v4_cells(src, ns, {'feature_df_full': frame, 'SN_SUBTYPES': ['SN_Ia', 'SN_II'],
                                  'NEW_NAME': 42}, 'test', min_per_class=5, verbose=False)
    assert ok and ns['seen'] == (True, ['SN_Ia', 'SN_II'], 42)
    assert ns['feature_df_full'] == 'ORIGINAL' and ns['SN_SUBTYPES'] == ['SN_Ia']
    assert 'NEW_NAME' not in ns, 'a name that did not exist before must not linger'


def test_runner_restores_even_when_a_cell_raises():
    ns = {'SN_SUBTYPES': ['keep']}
    with pytest.raises(ZeroDivisionError):
        D.run_v4_cells([(0, '1/0')], ns, {'SN_SUBTYPES': ['x']}, 't', verbose=False)
    assert ns['SN_SUBTYPES'] == ['keep']


def test_runner_skips_loudly_when_a_class_is_too_small(capsys):
    frame = pd.DataFrame({'label': ['SN_Ia'] * 20 + ['SLSN'] * 3})
    ok = D.run_v4_cells([(0, 'raise RuntimeError("should not run")')], {},
                        {'feature_df_full': frame, 'SN_SUBTYPES': ['SN_Ia', 'SLSN']},
                        'ALeRCE', min_per_class=8)
    assert not ok and 'skipped' in capsys.readouterr().out


def test_v4_impute_leaves_no_gaps_and_is_class_independent():
    df = _frame()
    df['host_offset_arcsec'] = np.nan            # e.g. astro-ghost could not resolve anything
    df.loc[df.index[:30], 'redshift'] = np.nan    # e.g. ALeRCE-native rows
    df.loc[df.index[:10], 'peak_abs_mag'] = np.nan
    out = D.v4_impute(df, FEATURE_COLS)
    present = [c for c in FEATURE_COLS if c in out.columns]
    assert not out[present].isna().any().any()
    assert (out['host_offset_arcsec'] == 0.0).all(), 'an all-NaN column becomes a constant'
    assert (out.loc[df.index[:10], 'peak_abs_mag'] == 0.0).all(), "v4's sentinel for peak_abs_mag"
    # class-independent: every imputed redshift is the same global median
    assert out.loc[df.index[:30], 'redshift'].nunique() == 1
    assert df['redshift'].isna().sum() == 30, 'must not modify its input'


def test_hist_gradient_boosting_survives_the_imputed_frame():
    """The exact failure the validator hit: v4's redshift-matched control fits
    HistGradientBoosting on every column; an all-NaN column crashed its binner."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    df = _frame()
    df['host_offset_arcsec'] = np.nan
    present = [c for c in FEATURE_COLS if c in df.columns]
    out = D.v4_impute(df, present)
    HistGradientBoostingClassifier(max_iter=20).fit(out[present].values, out['label'].values)
