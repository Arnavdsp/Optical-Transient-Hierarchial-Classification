"""Coordinate recovery from saved light curves (the host-offset precondition)."""
import os

import numpy as np
import pandas as pd
import pytest

from btp5.lightcurves import attach_coordinates, index_lightcurves, position_from_csv
from btp5.hostoffset import add_host_features
import synthetic5


def test_feature_rows_really_have_no_coordinates():
    """Guard the fixture itself: v4 feature rows carry no ra/dec, so a test that
    supplies them would hide exactly the bug this module fixes."""
    df = synthetic5.make_feature_df()
    assert 'ra' not in df.columns and 'dec' not in df.columns


def test_positions_recovered_from_the_saved_csvs(tmp_path):
    base = tmp_path / 'raw'
    (base / 'sn_ia').mkdir(parents=True)
    (base / 'tns_catalog').mkdir()
    synthetic5.make_lightcurve_csv(base / 'sn_ia' / 'ZTFa.csv', ra=150.0, dec=20.0, seed=1)
    synthetic5.make_lightcurve_csv(base / 'sn_ia' / 'ZTFb.csv', ra=10.0, dec=-5.0, seed=2)
    # a catalogue file must never be mistaken for a light curve
    pd.DataFrame({'ra': [1.0], 'dec': [1.0]}).to_csv(base / 'tns_catalog' / 'ZTFa.csv')

    idx = index_lightcurves(str(base))
    assert set(idx) == {'ZTFa', 'ZTFb'}
    assert 'tns_catalog' not in idx['ZTFa']

    df = pd.DataFrame({'id': ['ZTFa', 'ZTFb', 'AD_Leo', 'ZTFmissing'],
                       'survey': ['ZTF', 'ZTF', 'TESS', 'ZTF']})
    out = attach_coordinates(df, idx, verbose=False)
    a = out.set_index('id')
    assert a.loc['ZTFa', 'ra'] == pytest.approx(150.0, abs=1e-3)
    assert a.loc['ZTFb', 'dec'] == pytest.approx(-5.0, abs=1e-3)
    assert np.isnan(a.loc['AD_Leo', 'ra']), 'TESS flares have no host: must stay NaN'
    assert np.isnan(a.loc['ZTFmissing', 'ra'])
    assert a.loc['ZTFa', 'lc_path'].endswith('ZTFa.csv')


def test_ra_wraparound_does_not_average_to_180(tmp_path):
    p = tmp_path / 'wrap.csv'
    pd.DataFrame({'ra': [359.9999, 0.0001, 359.9998, 0.0002, 0.0],
                  'dec': [1.0] * 5}).to_csv(p, index=False)
    ra, dec = position_from_csv(str(p))
    assert min(ra, 360 - ra) < 0.01, f'wraparound mishandled: ra={ra}'


def test_host_offset_is_populated_end_to_end_without_row_coordinates(tmp_path):
    """The regression that motivated this module: with v4-shaped rows (no ra/dec),
    host matching must still succeed once coordinates come from the light curves."""
    base = tmp_path / 'raw' / 'sn'
    base.mkdir(parents=True)
    df = synthetic5.make_feature_df().head(20).copy()
    for oid in df.loc[df['survey'] == 'ZTF', 'id']:
        synthetic5.make_lightcurve_csv(base / f'{oid}.csv', seed=hash(oid) % 999)

    def resolver(ra, dec, name):
        return {'offset_arcsec': 0.4, 'host_radius_arcsec': 2.0}

    without = add_host_features(df, resolver=resolver, verbose=False)
    assert without['has_host_match'].sum() == 0, \
        'sanity: with no coordinates nothing can match — this was the live-run bug'

    with_coords = attach_coordinates(df, index_lightcurves(str(tmp_path / 'raw')),
                                     verbose=False)
    out = add_host_features(with_coords, resolver=resolver, verbose=False)
    ztf = out['survey'] == 'ZTF'
    assert (out.loc[ztf, 'has_host_match'] == 1).all()
    assert (out.loc[~ztf, 'has_host_match'] == 0).all()
