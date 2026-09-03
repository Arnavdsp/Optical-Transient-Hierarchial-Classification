"""
Retry-until-target acquisition, exercised entirely against fakes.

The point of these tests is to prove the control flow tops up to EXACTLY the
target in the presence of realistic attrition, before any real API quota is
spent — and that it reports a shortfall honestly when the pool genuinely runs out.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from btp_pipeline.acquisition import (  # noqa: E402
    assemble_feature_df, build_flares_to_target, build_tns_class_to_target, verify_counts)
import synthetic  # noqa: E402


def _fakes(fail_every=3, no_oid_every=5, thin_every=7):
    """Injectable stand-ins for resolve_oid / query_detections / process_ztf_object,
    with attrition at three different stages — exactly where the real pipeline loses
    objects (no ZTF cross-match, too few detections, GP/cleaning failure)."""
    calls = {'resolve': 0, 'detections': 0, 'process': 0}

    def resolve_oid_fn(row):
        calls['resolve'] += 1
        n = int(str(row['name']).split('_')[-1])
        return None if n % no_oid_every == 0 else f'ZTF{n:06d}'

    def query_detections_fn(oid):
        calls['detections'] += 1
        n = int(oid[3:9])
        # Too few detections -> rejected upstream of feature extraction.
        return synthetic.make_detections(n_points=2 if n % thin_every == 0 else 40, seed=n)

    def process_fn(csv_path, oid, label):
        calls['process'] += 1
        n = int(oid[3:9])
        if n % fail_every == 0:
            return None  # GP / cleaning failure
        row = {'id': oid, 'label': label, 'survey': 'ZTF',
               'peak_val': 1.0 + n % 5, 'rise_time': 10.0 + n % 7,
               'decay_time': 40.0 + n % 11, 'amplitude': 1.0 + n % 3,
               'color_g_r': 0.1}
        return row

    return resolve_oid_fn, query_detections_fn, process_fn, calls


def test_tns_retry_hits_exact_target_despite_attrition(tmp_path):
    base, feat = str(tmp_path / 'raw'), str(tmp_path / 'features')
    resolve, detect, process, calls = _fakes()
    pool = synthetic.make_tns_pool('AGN', 400, seed=1)

    manifest, features = build_tns_class_to_target(
        'AGN', pool, target=150, base_dir=base, feature_dir=feat,
        resolve_oid_fn=resolve, query_detections_fn=detect, process_fn=process,
        sleep=0, verbose=False)

    assert len(features) == 150, f'expected exactly 150 survivors, got {len(features)}'
    assert len(manifest) == 150
    # Attrition was real: we had to attempt far more candidates than we kept.
    assert calls['resolve'] > 150, 'attrition never triggered — the fake is not exercising the retry path'
    assert all(f['label'] == 'AGN' for f in features)
    assert len({m['oid'] for m in manifest}) == 150, 'duplicate objects in manifest'


def test_tns_retry_reports_shortfall_when_pool_exhausted(tmp_path, capsys):
    base, feat = str(tmp_path / 'raw'), str(tmp_path / 'features')
    resolve, detect, process, _ = _fakes()
    pool = synthetic.make_tns_pool('TDE', 40, seed=2)  # far too small to yield 150

    manifest, features = build_tns_class_to_target(
        'TDE', pool, target=150, base_dir=base, feature_dir=feat,
        resolve_oid_fn=resolve, query_detections_fn=detect, process_fn=process,
        sleep=0, verbose=True)

    assert len(features) < 150
    assert 'SHORTFALL' in capsys.readouterr().out, 'a shortfall must be reported loudly, not swallowed'


def test_tns_retry_resumes_from_checkpoint(tmp_path):
    base, feat = str(tmp_path / 'raw'), str(tmp_path / 'features')
    resolve, detect, process, calls = _fakes()
    pool = synthetic.make_tns_pool('SN_Ia', 300, seed=3)

    kw = dict(base_dir=base, feature_dir=feat, resolve_oid_fn=resolve,
              query_detections_fn=detect, process_fn=process, sleep=0, verbose=False)

    _, first = build_tns_class_to_target('SN_Ia', pool, target=20, **kw)
    assert len(first) == 20
    after_first = dict(calls)

    # Second call at the same target must do no API work at all.
    _, again = build_tns_class_to_target('SN_Ia', pool, target=20, **kw)
    assert len(again) == 20
    assert calls['resolve'] == after_first['resolve'], 'resume re-hit the API instead of using the checkpoint'

    # Raising the target tops up rather than starting over.
    _, more = build_tns_class_to_target('SN_Ia', pool, target=30, **kw)
    assert len(more) == 30
    assert calls['resolve'] > after_first['resolve']
    assert os.path.exists(os.path.join(base, 'sn_ia_failed_candidates.csv'))


def test_flare_retry_counts_survivors_not_downloads(tmp_path):
    """The bug this guards: the old loop counted downloads, so sectors that later
    failed feature extraction silently shrank the class."""
    base, feat = str(tmp_path / 'raw'), str(tmp_path / 'features')
    stars = [f'Star {i}' for i in range(60)]
    seen = {'downloads': 0, 'processed': 0}

    def search_fn(name):
        return [f'{name}::sector{i}' for i in range(6)]

    def download_fn(entry):
        seen['downloads'] += 1
        return synthetic.make_tess_frame(n_points=300, seed=seen['downloads'])

    def process_fn(csv_path, star_name, label):
        seen['processed'] += 1
        if seen['processed'] % 4 == 0:
            return None  # every 4th sector fails cleaning/GP
        return {'id': star_name, 'label': label, 'survey': 'TESS',
                'peak_val': 3.0, 'rise_time': 0.4, 'decay_time': 1.2,
                'amplitude': 2.5, 'color_g_r': float('nan')}

    manifest, features = build_flares_to_target(
        stars, target=150, base_dir=base, feature_dir=feat,
        search_fn=search_fn, download_fn=download_fn, process_fn=process_fn, verbose=False)

    assert len(features) == 150, f'expected exactly 150 surviving flare light curves, got {len(features)}'
    assert seen['downloads'] > 150, 'failures never occurred — the retry path is untested'
    # Per-sector ids must stay unique; 6 sectors of one star are 6 distinct light curves.
    assert len({f['id'] for f in features}) == 150, 'flare rows collapsed onto duplicate ids'


def test_flare_retry_reports_shortfall(tmp_path, capsys):
    base, feat = str(tmp_path / 'raw'), str(tmp_path / 'features')

    def search_fn(name):
        return ['only-one-sector']

    def download_fn(entry):
        return synthetic.make_tess_frame(n_points=200)

    def process_fn(csv_path, star_name, label):
        return {'id': star_name, 'label': label, 'survey': 'TESS', 'peak_val': 3.0,
                'rise_time': 0.4, 'decay_time': 1.2, 'amplitude': 2.5, 'color_g_r': float('nan')}

    _, features = build_flares_to_target(
        [f'Star {i}' for i in range(10)], target=150, base_dir=base, feature_dir=feat,
        search_fn=search_fn, download_fn=download_fn, process_fn=process_fn, verbose=True)

    assert len(features) == 10
    assert 'SHORTFALL' in capsys.readouterr().out


def test_assemble_and_verify_counts(capsys):
    df = synthetic.make_feature_df()
    # assemble_feature_df expects raw rows with a possibly-NaN colour column
    raw = df.drop(columns=[c for c in ['has_color'] if c in df.columns])
    combined = assemble_feature_df([raw.to_dict('records')], verbose=False)

    assert combined['color_g_r'].notna().all(), 'colour imputation left NaNs behind'
    assert combined.loc[combined['label'] == 'stellar_flare', 'has_color'].eq(0).all()
    assert combined.loc[combined['label'] == 'AGN', 'has_color'].eq(1).all()

    ok, report = verify_counts(combined, 150, 30, synthetic.SN_SUBTYPE_LABELS, verbose=False)
    assert ok, f'balanced synthetic set should verify clean:\n{report}'
    assert report['shortfall'].sum() == 0

    short = synthetic.make_feature_df(counts={'SN_Ia': 30, 'SN_Ib': 11, 'SN_Ic': 30,
                                              'SN_II': 30, 'SLSN': 30, 'AGN': 150,
                                              'TDE': 150, 'stellar_flare': 150})
    short = assemble_feature_df([short.to_dict('records')], verbose=False)
    ok2, report2 = verify_counts(short, 150, 30, synthetic.SN_SUBTYPE_LABELS, verbose=False)
    assert not ok2, 'an under-filled class must fail verification'
    assert int(report2.loc[report2['class'] == 'SN_Ib', 'shortfall'].iloc[0]) == 19
