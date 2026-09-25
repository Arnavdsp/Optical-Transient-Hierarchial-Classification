"""Pools (3.3), augmentation (5), calibration (8), and the dual-track runner."""
import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use('Agg')

from btp5 import augment as AUG, calibration as CAL, pools as PL, hostoffset as HO
from btp5.config import FEATURE_COLS, NOISE_MODEL
from btp5.modeling import run_track, track_summary, prepare_split
import synthetic5


# ------------------------------------------------------------------ pools
def test_three_pools_stay_separate_and_overlap_is_the_intersection():
    df = synthetic5.make_feature_df()
    df['has_tns_label'] = True
    tagged, counts = PL.build_pools(df, verbose=False)

    n_tns = int(tagged['in_tns_pool'].sum())
    n_al = int(tagged['in_alerce_pool'].sum())
    n_ov = int(tagged['in_overlap_pool'].sum())
    assert n_ov <= min(n_tns, n_al), 'overlap cannot exceed either parent pool'
    assert n_ov == int((tagged['in_tns_pool'] & tagged['in_alerce_pool']).sum())
    assert set(counts['pool']) == {PL.POOL_TNS, PL.POOL_ALERCE, PL.POOL_OVERLAP}


def test_tess_flares_can_never_enter_an_alerce_pool():
    """Structural, not a bug: ALeRCE classifies ZTF alerts, TESS objects have none."""
    df = synthetic5.make_feature_df()
    df['has_tns_label'] = True
    tagged, _ = PL.build_pools(df, verbose=False)
    tess = tagged[tagged['survey'] == 'TESS']
    assert len(tess) > 0
    assert not tess['in_alerce_pool'].any()
    assert not tess['in_overlap_pool'].any()

    present = PL.stage1_class_set(tagged[tagged['in_overlap_pool']],
                                  'alerce_coarse', verbose=False)
    assert 'stellar_flare' not in present, \
        'the overlap pool cannot contain a class ALeRCE could never emit for TESS'
    assert 'get_pool' and PL.get_pool(tagged, PL.POOL_OVERLAP)['pool'].eq(
        PL.POOL_OVERLAP).all()


def test_attach_alerce_labels_leaves_unmatched_objects_as_nan():
    df = synthetic5.make_feature_df(with_alerce=False)
    rows = [{'id': df.iloc[0]['id'], 'alerce_class': 'SNIa', 'alerce_prob': 0.9,
             'alerce_coarse': 'SNe', 'alerce_fine': 'SN_Ia'}]
    out = PL.attach_alerce_labels(df, rows, verbose=False)
    assert int(out['alerce_coarse'].notna().sum()) == 1
    assert len(out) == len(df)


# ------------------------------------------------------------------ augmentation
def test_noise_model_uses_the_published_ztf_constants():
    assert NOISE_MODEL == {'e_b': 18.0, 'm': 0.04, 'c': 4.7, 'delta': -0.006}


def test_uncertainty_grows_as_flux_falls():
    rng = np.random.default_rng(0)
    bright = AUG.ztf_flux_uncertainty(np.full(4000, 5000.0), rng)
    faint = AUG.ztf_flux_uncertainty(np.full(4000, 100.0), rng)
    assert (faint / 100.0).mean() > (bright / 5000.0).mean(), \
        'fainter sources must have larger FRACTIONAL uncertainty'
    assert (bright > 0).all() and (faint > 0).all()
    assert np.isfinite(AUG.ztf_flux_uncertainty(np.array([0.0, -5.0]), rng)).all()


def test_redshifting_dims_and_dilates_but_never_jitters_time():
    t = np.linspace(0, 100, 60)
    f = 1000 * np.exp(-((t - 30) ** 2) / 200)
    t2, f2, k = AUG.rescale_to_redshift(t, f, z_true=0.05, z_sim=0.15)

    assert f2.max() < f.max(), 'moving an object further away must make it fainter'
    assert (t2.max() - t2.min()) > (t.max() - t.min()), 'time dilation must stretch'
    assert t2[0] == pytest.approx(t[0]), 'the anchor epoch should not move'
    assert np.all(np.diff(t2) > 0), 'ordering must be preserved'
    # Timestamp jitter is an evidenced negative result and must not be applied:
    # the transform has to be a deterministic monotone map of the input times.
    t2b, _, _ = AUG.rescale_to_redshift(t, f, z_true=0.05, z_sim=0.15)
    assert np.array_equal(t2, t2b), 'time axis must be deterministic, never jittered'
    assert k == pytest.approx(-2.5 * np.log10(1.15 / 1.05))


def test_augment_lightcurve_returns_errors_and_keeps_times_deterministic():
    rng = np.random.default_rng(3)
    t = np.linspace(0, 80, 40)
    f = 500 * np.exp(-((t - 25) ** 2) / 150) + 10
    a = AUG.augment_lightcurve(t, f, 0.03, 0.12, rng)
    b = AUG.augment_lightcurve(t, f, 0.03, 0.12, np.random.default_rng(99))
    assert np.array_equal(a['t'], b['t']), 'times identical across noise draws'
    assert not np.array_equal(a['flux'], b['flux']), 'flux should differ (noise)'
    assert (a['flux_err'] > 0).all() and len(a['flux_err']) == len(t)


def test_augmentation_plan_and_synthetic_flagging():
    counts = {'SLSN': 12, 'SN_Ib': 30, 'TDE': 0}
    plan = AUG.plan_augmentation(counts, 30, ['SLSN', 'SN_Ib', 'TDE'], verbose=False)
    p = plan.set_index('class')
    assert p.loc['SLSN', 'synthetic_needed'] == 18
    assert p.loc['SN_Ib', 'synthetic_needed'] == 0, 'a class at target is left alone'
    assert p.loc['TDE', 'synthetic_needed'] == 30
    assert not np.isfinite(p.loc['TDE', 'copies_per_real'])

    ok, msg = AUG.augmented_rows_are_flagged(
        pd.DataFrame({'is_synthetic': [True, False, False]}))
    assert ok and '1 synthetic' in msg
    bad, _ = AUG.augmented_rows_are_flagged(pd.DataFrame({'x': [1]}))
    assert not bad


def test_synthetic_rows_are_kept_out_of_the_test_set():
    """A synthetic copy shares its parent's light curve, so it in the test set is
    leakage wearing a hat."""
    df = synthetic5.make_feature_df()
    df.loc[df.index[:40], 'is_synthetic'] = True
    split = prepare_split(df, FEATURE_COLS, 'coarse_label', verbose=False)
    test_rows = split['df'].iloc[split['idx_test']]
    assert not test_rows['is_synthetic'].any(), 'synthetic rows leaked into test'
    train_rows = split['df'].iloc[split['idx_train']]
    assert train_rows['is_synthetic'].sum() > 0, 'synthetic rows should still train'


# ------------------------------------------------------------------ host offset
def test_host_offset_missing_data_follows_the_existing_convention():
    df = synthetic5.make_feature_df().head(12).copy()

    def resolver(ra, dec, name):
        if not np.isfinite(ra):
            return None
        return {'offset_arcsec': 0.3, 'host_radius_arcsec': 3.0}

    out = HO.add_host_features(df, resolver=resolver, verbose=False)
    assert {'host_offset_arcsec', 'host_offset_norm', 'has_host_match'} <= set(out.columns)
    ztf = out[out['survey'] == 'ZTF']
    assert (ztf['has_host_match'] == 1).all()
    assert ztf['host_offset_norm'].iloc[0] == pytest.approx(0.1)
    tess = out[out['survey'] == 'TESS']
    if len(tess):
        assert (tess['has_host_match'] == 0).all()
        assert tess['host_offset_arcsec'].isna().all()

    # A resolver that throws must degrade to the missing convention, not propagate.
    def boom(ra, dec, name):
        raise RuntimeError('catalogue down')
    out2 = HO.add_host_features(df, resolver=boom, verbose=False)
    assert (out2['has_host_match'] == 0).all()


# ------------------------------------------------------------------ calibration
def test_calibration_detects_overconfidence():
    conf = np.full(200, 0.95)
    correct = np.array([True] * 120 + [False] * 80)   # 60% right, 95% confident
    curve = CAL.calibration_curve(conf, correct, n_bins=10)
    assert len(curve) == 1
    assert CAL.overconfidence(curve) == pytest.approx(0.95 - 0.6, abs=1e-6)
    assert CAL.expected_calibration_error(curve) == pytest.approx(0.35, abs=1e-6)

    perfect = CAL.calibration_curve(np.full(100, 0.8), np.array([True] * 80 + [False] * 20))
    assert CAL.expected_calibration_error(perfect) == pytest.approx(0.0, abs=1e-6)
    assert CAL.calibration_curve(np.array([]), np.array([])).empty


def test_redshift_confidence_claim_is_answered_from_the_numbers():
    more_conf_same_acc = {'accuracy': 0.60, 'mean_confidence': 0.90, 'ece': 0.30}
    baseline = {'accuracy': 0.59, 'mean_confidence': 0.70, 'ece': 0.11}
    msg = CAL.compare_with_without_redshift(more_conf_same_acc, baseline, 'RF')
    assert 'without making it more correct' in msg and 'de Soto' in msg

    both_up = {'accuracy': 0.75, 'mean_confidence': 0.90, 'ece': 0.15}
    msg2 = CAL.compare_with_without_redshift(both_up, baseline, 'RF')
    assert 'does NOT reproduce' in msg2
    assert 'unavailable' in CAL.compare_with_without_redshift(None, baseline, 'RF')


# ------------------------------------------------------------------ dual track
def test_both_tracks_run_through_one_code_path():
    df = synthetic5.make_feature_df()
    df['alerce_fine_for_stage2'] = df['alerce_fine']

    tns = run_track(df, FEATURE_COLS, 'coarse_label', 'label', 'TNS', verbose=False)
    alerce = run_track(df[df['alerce_coarse'].notna()], FEATURE_COLS,
                       'alerce_coarse', 'alerce_fine', 'ALeRCE', verbose=False)

    for t in (tns, alerce):
        assert len(t['stage1']['results']) == 3, 'three headline models'
        assert list(t['stage1']['results']['Model']) == [
            'Logistic Regression', 'Random Forest', 'Bagging Classifier']
        for _, r in t['stage1']['results'].iterrows():
            assert r['CI low'] <= r['Test accuracy'] <= r['CI high']
            assert 0 <= r['Macro-F1'] <= 1

    # The ALeRCE track legitimately has fewer Stage 1 classes (no stellar_flare).
    assert len(alerce['stage1']['split']['le'].classes_) < \
           len(tns['stage1']['split']['le'].classes_)

    summary = track_summary([tns, alerce])
    assert set(summary['Track']) == {'TNS', 'ALeRCE'}
    assert 'Macro-F1' in summary.columns and '95% CI' in summary.columns


def test_stage2_is_nested_inside_stage1_training_partition():
    df = synthetic5.make_feature_df()
    t = run_track(df, FEATURE_COLS, 'coarse_label', 'label', 'TNS', verbose=False)
    s1, s2 = t['stage1'], t['stage2']
    assert s2 is not None
    s1_test_ids = set(s1['split']['df'].iloc[s1['split']['idx_test']]['id'])
    s2_train_ids = set(s2['split']['df'].iloc[s2['split']['idx_train']]['id'])
    assert not (s2_train_ids & s1_test_ids), \
        'Stage 2 trained on an object held out of Stage 1'
