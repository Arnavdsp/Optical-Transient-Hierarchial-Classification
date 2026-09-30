"""Section 6 agreement framework, on constructed cases with known answers."""
import numpy as np
import pandas as pd
import pytest

from btp5 import agreement as AG

CLASSES = ['AGN', 'SNe', 'TDE']


def test_purity_and_completeness_are_the_two_normalisations():
    y_true = ['AGN'] * 10 + ['SNe'] * 10
    y_pred = ['AGN'] * 8 + ['SNe'] * 2 + ['SNe'] * 9 + ['AGN'] * 1

    P = AG.purity_matrix(y_true, y_pred, CLASSES)
    C = AG.completeness_matrix(y_true, y_pred, CLASSES)

    # purity: columns sum to 1 (over predicted), completeness: rows sum to 1 (over true)
    assert P['AGN'].sum() == pytest.approx(1.0)
    assert C.loc['AGN'].sum() == pytest.approx(1.0)
    # of the 9 things called AGN, 8 really were AGN
    assert P.loc['AGN', 'AGN'] == pytest.approx(8 / 9)
    # of the 10 real AGN, 8 were called AGN
    assert C.loc['AGN', 'AGN'] == pytest.approx(0.8)
    # empty classes give zeros rather than NaN, so the matrix product stays defined
    assert np.isfinite(P.values).all() and np.isfinite(C.values).all()


def test_perfect_classifiers_give_identity_expected_agreement():
    y = ['AGN'] * 5 + ['SNe'] * 5 + ['TDE'] * 5
    P = AG.purity_matrix(y, y, CLASSES)
    C = AG.completeness_matrix(y, y, CLASSES)
    A = AG.expected_agreement(P, C)
    assert np.allclose(A.values, np.eye(3)), \
        'two perfect pipelines should be expected to agree perfectly'


def test_expected_agreement_rejects_mismatched_class_order():
    y = ['AGN'] * 4 + ['SNe'] * 4 + ['TDE'] * 4
    P = AG.purity_matrix(y, y, CLASSES)
    C = AG.completeness_matrix(y, y, list(reversed(CLASSES)))
    with pytest.raises(AssertionError):
        AG.expected_agreement(P, C)


def test_actual_agreement_and_rate():
    ours = ['AGN', 'AGN', 'SNe', 'TDE']
    theirs = ['AGN', 'SNe', 'SNe', 'TDE']
    assert AG.agreement_rate(ours, theirs) == pytest.approx(0.75)
    A = AG.actual_agreement(ours, theirs, CLASSES, normalize='all')
    assert A.values.sum() == pytest.approx(1.0)
    assert A.loc['AGN', 'SNe'] == pytest.approx(0.25)


def test_shared_signal_is_reported_when_actual_exceeds_expected():
    n = 200
    rng = np.random.default_rng(0)
    truth = rng.choice(CLASSES, n)
    # Both pipelines make the SAME mistakes -> agreement far above independent-error
    shared = np.where(rng.random(n) < 0.3, rng.choice(CLASSES, n), truth)
    P = AG.purity_matrix(truth, shared, CLASSES)
    C = AG.completeness_matrix(truth, shared, CLASSES)
    res = AG.compare_expected_actual(shared, shared, P, C, CLASSES)

    assert res['actual_rate'] == pytest.approx(1.0)
    assert res['actual_rate'] > res['expected_rate_weighted']
    text = AG.interpret_agreement(res, 'Random Forest', 'overlap')
    assert 'not* independent verification' in text or 'not independent verification' in text
    assert 'shared' in text.lower()
    assert 'Random Forest' in text and 'overlap' in text


def test_independent_errors_are_reported_as_such():
    rng = np.random.default_rng(1)
    n = 400
    truth = rng.choice(CLASSES, n)
    a = np.where(rng.random(n) < 0.75, truth, rng.choice(CLASSES, n))
    b = np.where(rng.random(n) < 0.75, truth, rng.choice(CLASSES, n))
    P = AG.purity_matrix(truth, a, CLASSES)
    C = AG.completeness_matrix(truth, b, CLASSES)
    res = AG.compare_expected_actual(a, b, P, C, CLASSES)
    assert 0.0 <= res['actual_rate'] <= 1.0
    assert 0.0 <= res['expected_rate_weighted'] <= 1.0
    text = AG.interpret_agreement(res, 'LogReg', 'overlap')
    assert isinstance(text, str) and len(text) > 80


def test_tde_slsn_confusion_cites_the_external_corroboration():
    classes = ['SLSN', 'TDE']
    ours = ['TDE'] * 20
    theirs = ['SLSN'] * 18 + ['TDE'] * 2     # the documented failure direction
    truth = ['TDE'] * 20
    P = AG.purity_matrix(truth, ours, classes)
    C = AG.completeness_matrix(truth, theirs, classes)
    res = AG.compare_expected_actual(ours, theirs, P, C, classes)
    text = AG.interpret_agreement(res, 'RF', 'overlap')
    assert 'Superphot+' in text and '54.9%' in text


def test_external_table_and_caveats_are_present_and_sourced():
    t = AG.EXTERNAL_RESULTS
    assert len(t) == 6
    assert {'source', 'task', 'result', 'N'} <= set(t.columns)
    joined = ' '.join(t['source'])
    for cite in ['de Soto', 'Villar', 'Aleo', 'Townsend']:
        assert cite in joined
    assert '2403.07975' in joined and '2602.13036' in joined
    assert 'three independent pipelines' in AG.TDE_SLSN_STATEMENT
    assert 'SNID' in AG.TNS_LABEL_NOISE_CAVEAT
    assert 'recorded' in AG.TNS_LABEL_NOISE_CAVEAT
