"""
Augmentation, checked against Townsend et al. 2026 (arXiv:2602.13036) as the paper
actually describes it, and end to end through v4's real fitter and quality gate.
"""
import os

import numpy as np
import pandas as pd
import pytest
from astropy.cosmology import FlatLambdaCDM

from btp5 import augment as AUG
from btp5.config import NOISE_MODEL
from btp5.photometry import process_ztf_object
from btp5.modeling import prepare_split
from btp5.config import FEATURE_COLS
import synthetic5

COSMO = FlatLambdaCDM(H0=70, Om0=0.3)


def test_published_constants_and_microjansky_units():
    assert NOISE_MODEL == {'e_b': 18.0, 'm': 0.04, 'c': 4.7, 'delta': -0.006}
    assert AUG.UJY_ZP == 23.9
    f, fe = AUG.mag_to_ujy(np.array([18.0]), np.array([0.05]))
    # mag 18 must land inside Townsend's flux bins (<400 ... >1600), i.e. ~230 uJy.
    assert 200 < f[0] < 260
    mag, smag = AUG.ujy_to_mag(f, fe)
    assert mag[0] == pytest.approx(18.0) and smag[0] == pytest.approx(0.05)


def test_z_sim_range_and_volumetric_weighting():
    rng = np.random.default_rng(0)
    flat = [AUG.sample_z_sim(0.05, rng, 0.1, 0) for _ in range(4000)]
    vol = [AUG.sample_z_sim(0.05, rng, 0.1, 2) for _ in range(4000)]
    for zs in (flat, vol):
        assert min(zs) >= 0.05 and max(zs) <= 0.15 + 1e-12
    assert np.mean(vol) > np.mean(flat), 'z^2 weighting should favour higher z_sim'


def test_error_scale_factor_is_identity_without_a_redshift_change():
    f = np.array([50.0, 300.0, 1000.0, 5000.0])
    eps = np.array([0.02, 0.01, 0.005, 0.001])
    assert np.allclose(AUG.error_scale_factor(f, 1.0, eps), 1.0)


def test_fainter_copies_are_relatively_noisier():
    f = np.array([200.0, 800.0, 2000.0])
    eps = np.full(3, 0.01)
    D = 0.5
    fac = AUG.error_scale_factor(f, D, eps)
    frac_before = 1.0                         # sigma_t / sigma_t
    frac_after = fac / D                      # (sigma_z / D f) relative to (sigma_t / f)
    assert (frac_after > frac_before).all(), 'moving further away must raise fractional error'


def test_error_scale_factor_never_goes_negative_out_of_range():
    f = np.logspace(0, 6, 50)                 # up to 1e6 uJy, far past the fitted range
    fac = AUG.error_scale_factor(f, 0.3, np.zeros(50))
    assert np.isfinite(fac).all() and (fac > 0).all()


def test_observation_dates_are_never_touched(tmp_path):
    """The brief forbids timestamp perturbation, and Townsend keep the cadence.
    Every surviving epoch must be an original epoch, unchanged, in both bands."""
    det = synthetic5.make_lightcurve_csv(tmp_path / 'p.csv', seed=3)
    rng = np.random.default_rng(1)
    new, meta = AUG.augment_detections(det, 'SN_Ia', 0.03, 0.08, rng, COSMO,
                                       use_sncosmo=False)
    assert new is not None
    orig = set(zip(det['fid'], det['mjd'].round(9)))
    for fid, mjd in zip(new['fid'], new['mjd'].round(9)):
        assert (fid, mjd) in orig, 'an epoch was created or moved'
    # g-r relative timing preserved: the g- and r-band epochs are subsets of the originals
    for fid in (1, 2):
        assert set(new.loc[new.fid == fid, 'mjd']).issubset(set(det.loc[det.fid == fid, 'mjd']))


def test_copy_is_fainter_by_the_distance_modulus_plus_k(tmp_path):
    """Compare the SAME epochs before and after, restricted to points that were bright
    enough to survive with high SNR. Comparing the brightest points instead is biased:
    the maximum of a noisy set picks positive excursions."""
    det = synthetic5.make_lightcurve_csv(tmp_path / 'p.csv', seed=4, peak_mag=17.5)
    rng = np.random.default_rng(2)
    zt, zs = 0.03, 0.05                      # modest shift: SNR stays high
    new, meta = AUG.augment_detections(det, 'SLSN', zt, zs, rng, COSMO, use_sncosmo=False)
    d_mu = 5 * np.log10(COSMO.luminosity_distance(zs).value / COSMO.luminosity_distance(zt).value)
    k = AUG.hogg_k(zt, zs)
    m = det.merge(new, on=['fid', 'mjd'], suffixes=('_old', '_new'))
    bright = m[m['sigmapsf_old'] < 0.05]
    assert len(bright) >= 10
    assert np.median(bright['magpsf_new'] - bright['magpsf_old']) == pytest.approx(d_mu + k, abs=0.08)


def test_fitted_curve_peak_moves_by_distance_modulus_plus_k(tmp_path):
    """The physics check done on the quantity the fit pins down well: the peak of the
    fitted Villar curve (not v4's A parameter, which is ~7x noisier)."""
    from btp5.photometry import villar, fit_villar, mag_to_flux, clean_band, ZP

    def curve_peak_mag(df):
        s = df[df['fid'] == 2]
        f, fe = mag_to_flux(s['magpsf'].values, s['sigmapsf'].values)
        p, _ = fit_villar(*clean_band(s['mjd'].values, f, fe))
        tt = np.linspace(p['t0'] - 100, p['t0'] + 300, 4000)
        return ZP - 2.5 * np.log10(villar(tt, **p).max())

    zt, zs = 0.03, 0.05
    d_mu = 5 * np.log10(COSMO.luminosity_distance(zs).value / COSMO.luminosity_distance(zt).value)
    shifts = []
    for i in range(6):
        det = synthetic5.make_lightcurve_csv(tmp_path / f'p{i}.csv', seed=40 + i, peak_mag=17.5)
        new, _ = AUG.augment_detections(det, 'SLSN', zt, zs, np.random.default_rng(i),
                                        COSMO, use_sncosmo=False)
        shifts.append(curve_peak_mag(new) - curve_peak_mag(det))
    assert np.median(shifts) == pytest.approx(d_mu + AUG.hogg_k(zt, zs), abs=0.08)


def test_k_correction_methods():
    k, m = AUG.k_correction('SLSN', 2, 0.05, 0.15)
    assert m == 'hogg2002' and k == pytest.approx(-2.5 * np.log10(1.15 / 1.05))
    k, m = AUG.k_correction('TDE', 1, 0.05, 0.15)
    assert m == 'hogg2002'
    if AUG.sncosmo_available():
        k, m = AUG.k_correction('SN_Ia', 2, 0.05, 0.15)
        assert m == 'sncosmo:salt2'
        assert abs(k) < 0.3, 'K over dz=0.1 should be small (Townsend Fig. 5)'


def _parents(tmp_path, n=6, label='SLSN', z=0.04, peak_mag=17.5):
    rows = []
    for i in range(n):
        p = tmp_path / f'ZTFpar{i}.csv'
        synthetic5.make_lightcurve_csv(p, seed=100 + i, peak_mag=peak_mag)
        row = process_ztf_object(str(p), f'ZTFpar{i}', label, redshift=z)
        assert row is not None
        row.update({'survey': 'ZTF', 'lc_path': str(p), 'is_synthetic': False,
                    'coarse_label': 'SNe', 'has_tns_label': True,
                    'host_offset_arcsec': 1.5, 'host_offset_norm': 0.4, 'has_host_match': 1})
        rows.append(row)
    return pd.DataFrame(rows)


def test_generate_synthetic_rows_end_to_end_through_the_real_quality_gate(tmp_path):
    parents = _parents(tmp_path)
    rng = np.random.default_rng(7)
    syn = AUG.generate_synthetic_rows(parents, ['SLSN'], 2, process_ztf_object,
                                      str(tmp_path / 'syn'), rng, cosmo=COSMO,
                                      use_sncosmo=False, verbose=False)
    assert len(syn) > 0
    assert syn['is_synthetic'].all() and syn['parent_id'].notna().all()
    assert (syn['z_sim'] <= syn['z_true'] + 0.1 + 1e-9).all()
    assert (syn['z_sim'] >= syn['z_true']).all()
    assert syn['bts_peakabs'].isna().all() and (syn['from_bts'] == 0).all()
    # host offset: distance-free ratio preserved, arcsec offset shrinks with distance
    assert (syn['host_offset_norm'] == 0.4).all()
    assert (syn['host_offset_arcsec'] < 1.5).all()

    # Whole-chain consistency (rescale + noise + refit). Checked on the median, with a
    # tolerance set by v4's peak_abs_mag scatter (~0.25 mag per fit, because it is
    # derived from the degenerate Villar A parameter), not by the augmentation.
    small = AUG.generate_synthetic_rows(parents, ['SLSN'], 2, process_ztf_object,
                                        str(tmp_path / 'syn_small'), np.random.default_rng(3),
                                        cosmo=COSMO, dz_max=0.02, use_sncosmo=False,
                                        verbose=False)
    cons = AUG.augmentation_consistency(small, parents, verbose=False)
    assert cons['n'] >= 6
    assert abs(cons['median_residual_mag']) < 0.3


def test_copies_pushed_too_faint_are_rejected_and_tallied(tmp_path, capsys):
    parents = _parents(tmp_path, n=2, peak_mag=19.9, z=0.02)
    # z 0.02 -> up to 0.12: ~5.5 mag fainter, far past ZTF's alert depth
    syn = AUG.generate_synthetic_rows(parents, ['SLSN'], 2, process_ztf_object,
                                      str(tmp_path / 'syn'), np.random.default_rng(0),
                                      cosmo=COSMO, use_sncosmo=False, verbose=True)
    out = capsys.readouterr().out
    assert len(syn) < 4
    assert 'rejections' in out


def test_split_drops_children_whose_parent_is_in_test():
    df = synthetic5.make_feature_df()
    real_ids = df['id'].tolist()
    kids = df.sample(60, random_state=1).copy()
    kids['parent_id'] = kids['id']
    kids['id'] = kids['id'] + '__aug0'
    kids['is_synthetic'] = True
    df['parent_id'] = np.nan
    full = pd.concat([df, kids], ignore_index=True)

    split = prepare_split(full, FEATURE_COLS, 'coarse_label', verbose=False)
    d = split['df']
    test_ids = set(d['id'].iloc[split['idx_test']])
    train = d.iloc[split['idx_train']]
    leaked = train[train['is_synthetic'].astype(bool) & train['parent_id'].isin(test_ids)]
    assert leaked.empty, 'a synthetic copy of a test object was trained on'
    assert not d.iloc[split['idx_test']]['is_synthetic'].astype(bool).any()
    assert train['is_synthetic'].astype(bool).sum() > 0, 'children of train parents should stay'


def test_flagging_requires_a_parent_id():
    ok, _ = AUG.augmented_rows_are_flagged(pd.DataFrame(
        {'is_synthetic': [True, False], 'parent_id': ['a', None]}))
    assert ok
    bad, msg = AUG.augmented_rows_are_flagged(pd.DataFrame({'is_synthetic': [True]}))
    assert not bad and 'parent_id' in msg
