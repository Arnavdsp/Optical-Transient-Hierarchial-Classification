"""
The ALeRCE label track. The bugs guarded here are ones the live API actually exhibits.
"""
import numpy as np
import pandas as pd
import pytest

from btp5 import alerce_labels as AL
import synthetic5

LIVE_BHRF_CLASSES = [  # verified against query_classes on 2026-09-25
    'AGN', 'Blazar', 'CEP', 'CV/Nova', 'DSCT', 'EA', 'EB/EW', 'LPV', 'Microlensing',
    'Periodic-Other', 'QSO', 'RRLab', 'RRLc', 'RSCVn', 'SESN', 'SLSN', 'SNII',
    'SNIIn', 'SNIa', 'TDE', 'YSO']


def test_top_class_is_keyed_on_version_not_just_name():
    """query_probabilities stacks every version it has run. Grouping by classifier
    name alone crosses between complete probability distributions and returns a
    top-1 that belongs to no single classifier."""
    p = synthetic5.make_probabilities_frame(multi_version=True)
    assert p['classifier_version'].nunique() == 2
    assert p['probability'].sum() == pytest.approx(2.0, abs=1e-6), \
        'fixture should stack two distributions, as the real API does'

    top = AL.top_class(p, 'lc_classifier_BHRF_forced_phot', '2.1.0')
    assert top['prob_sum'] == pytest.approx(1.0, abs=1e-6), \
        'version-keyed selection must see exactly one distribution'
    assert top['class_name'] == 'SNIa'
    assert top['classifier_version'] == '2.1.0'


def test_top_class_without_version_picks_one_distribution_deterministically():
    p = synthetic5.make_probabilities_frame(multi_version=True)
    top = AL.top_class(p, 'lc_classifier_BHRF_forced_phot', None)
    assert top['prob_sum'] == pytest.approx(1.0, abs=1e-6), \
        'must never mix versions even when none is specified'


def test_crosswalk_covers_the_live_taxonomy_and_excludes_the_rest():
    table, unmapped = AL.build_crosswalk(LIVE_BHRF_CLASSES, verbose=False)
    assert len(table) == len(LIVE_BHRF_CLASSES)

    m = dict(zip(table['alerce_class'], table['our_coarse_class']))
    # AGN family
    for c in ['AGN', 'QSO', 'Blazar']:
        assert m[c] == 'AGN'
    # every SN-ish class, including BHRF's SESN and SNIIn spellings
    for c in ['SNIa', 'SESN', 'SNII', 'SNIIn', 'SLSN']:
        assert m[c] == 'SNe'
    assert m['TDE'] == 'TDE'

    # The brief warns specifically against assuming "stochastic" == AGN. The BHRF
    # stochastic branch also holds YSO and CV/Nova, which are Galactic, not AGN.
    assert m['YSO'] != 'AGN'
    assert m['CV/Nova'] != 'AGN'

    # Periodic classes and Microlensing have no counterpart and must be excluded,
    # never reassigned into one of our four.
    for c in ['LPV', 'EA', 'EB/EW', 'RRLab', 'RRLc', 'CEP', 'DSCT', 'RSCVn',
              'Periodic-Other', 'Microlensing']:
        assert c in unmapped, f'{c} should be excluded, not mapped'


def test_fine_crosswalk_merges_iin_symmetrically_and_keeps_ibc_merged():
    table, _ = AL.build_crosswalk(LIVE_BHRF_CLASSES, verbose=False)
    f = dict(zip(table['alerce_class'], table['our_fine_class']))
    # BHRF 2.1.0 HAS a separate SNIIn; we merge it into SN_II on both sides so the
    # agreement matrices compare like with like.
    assert f['SNIIn'] == 'SN_II'
    assert f['SNII'] == 'SN_II'
    # ALeRCE never separates Ib from Ic — SESN is the merged stripped-envelope class.
    assert f['SESN'] == 'SN_Ibc'
    # ...and our TNS-side Ib/Ic collapse onto it only for comparison, not in the
    # TNS track's own Stage 2.
    assert AL.collapse_tns_fine('SN_Ib') == 'SN_Ibc'
    assert AL.collapse_tns_fine('SN_Ic') == 'SN_Ibc'
    assert AL.collapse_tns_fine('SN_Ia') == 'SN_Ia'


def test_confidence_threshold_and_unmapped_exclusion():
    p = synthetic5.make_probabilities_frame(multi_version=False)
    kw = dict(classifier_name='lc_classifier_BHRF_forced_phot',
              classifier_version='2.1.0')

    lab = AL.label_from_probabilities(p, threshold=0.5, **kw)
    assert lab is not None and lab['alerce_coarse'] == 'SNe'
    assert lab['alerce_fine'] == 'SN_Ia'

    # Above the object's own top-1 probability -> rejected, not forced through.
    assert AL.label_from_probabilities(p, threshold=0.95, **kw) is None

    # An object whose top-1 is an unmapped class is dropped entirely.
    q = pd.DataFrame([{'classifier_name': 'lc_classifier_BHRF_forced_phot',
                       'classifier_version': '2.1.0', 'class_name': 'RRLab',
                       'probability': 0.99, 'ranking': 1}])
    assert AL.label_from_probabilities(q, threshold=0.5, **kw) is None


def test_survey_kwargs_tracks_client_major_version():
    """v2.x requires survey='ztf'; v1.x does not accept it."""
    assert AL._survey_kwargs('2.3.1') == {'survey': 'ztf'}
    assert AL._survey_kwargs('1.2.0') == {}


def test_empty_and_missing_inputs_do_not_raise():
    kw = dict(classifier_name='lc_classifier_BHRF_forced_phot', classifier_version='2.1.0')
    assert AL.top_class(None, **kw) is None
    assert AL.top_class(pd.DataFrame(columns=['classifier_name', 'classifier_version',
                                              'class_name', 'probability']), **kw) is None
    assert AL.label_from_probabilities(None, **kw) is None
