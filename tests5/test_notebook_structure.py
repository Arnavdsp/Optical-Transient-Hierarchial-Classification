"""
Structural guarantees the v5 notebook's own markdown makes, enforced by test rather
than hoped for.
"""
import ast
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = os.path.join(ROOT, 'notebooks', 'BTP_TNS_Hierarchical_v5.ipynb')


def _cells():
    nb = json.load(open(NB))
    return [(c['cell_type'], ''.join(c['source'])) for c in nb['cells']]


def _index(pred):
    return next(i for i, (t, s) in enumerate(_cells()) if pred(t, s))


def test_alerce_pass_reexecutes_exactly_the_visible_diagnostic_cells():
    cells = _cells()
    emb = next(s for t, s in cells if '_V4_DIAGNOSTIC_SOURCES = ' in s)
    line = next(l for l in emb.split('\n') if l.startswith('_V4_DIAGNOSTIC_SOURCES = '))
    embedded = ast.literal_eval(line.split(' = ', 1)[1])
    visible = {s for t, s in cells if t == 'code'}
    assert len(embedded) == 8
    for i, src in embedded:
        assert src in visible, f'ALeRCE pass runs a different cell {i} than the TNS pass shows'
    assert {i for i, _ in embedded}.isdisjoint({35, 37}), 'TNS-scheme-only cells leaked in'


def test_iin_justification_sits_directly_above_classify_sn_subtype():
    cells = _cells()
    k = _index(lambda t, s: t == 'code' and 'def classify_sn_subtype' in s)
    assert cells[k - 1][0] == 'markdown' and 'SN IIn' in cells[k - 1][1]
    assert 'sample size' in cells[k - 1][1]


def test_crosswalk_is_printed_before_any_alerce_label_is_used():
    cw = _index(lambda t, s: t == 'code' and 'build_crosswalk(live_classes)' in s)
    use = _index(lambda t, s: t == 'code' and 'describe_probabilities(probs' in s)
    native = _index(lambda t, s: t == 'code' and 'query_native_candidates(' in s
                    and 'def ' not in s)
    assert cw < native < use


def test_coarse_label_exists_before_the_first_training_call():
    """The first v5 build omitted this and would have crashed on a live run."""
    made = _index(lambda t, s: t == 'code' and "feature_df['coarse_label']" in s
                  and 'SN_SUBTYPES' in s)
    train = _index(lambda t, s: t == 'code' and 'tns_track = run_track(' in s)
    assert made < train


def test_coordinates_are_recovered_before_host_offset_and_augmentation():
    cells = _cells()
    coords = _index(lambda t, s: t == 'code' and 'attach_coordinates(feature_df' in s)
    host = _index(lambda t, s: t == 'code' and 'add_host_features(feature_df' in s)
    aug = _index(lambda t, s: t == 'code' and 'generate_synthetic_rows(' in s and 'def ' not in s)
    assert coords <= host < aug
    if coords == host:   # same cell: check the order of the two calls inside it
        body = cells[coords][1]
        assert body.index('attach_coordinates(') < body.index('add_host_features(')


def test_no_literal_credentials_and_helper_is_used():
    cred = next(s for t, s in _cells()
                if t == 'code' and 'TNS_API_KEY' in s and 'def _get_secret' in s)
    assert "TNS_API_KEY  = _get_secret('TNS_API_KEY')" in cred
    assert not re.search(r'TNS_API_KEY\s*=\s*["\']', cred)
    assert not re.search(r'"tns_id"\s*:\s*\d', cred)


def test_every_inlined_module_matches_its_source_file():
    for t, s in _cells():
        m = re.search(r'Inlined verbatim from btp5/(\w+)\.py', s)
        if not m:
            continue
        src = open(os.path.join(ROOT, 'btp5', m.group(1) + '.py')).read()
        expect = '\n'.join(l for l in src.split('\n') if not l.startswith('from .'))
        # the builder stores cells without the file's trailing newline
        assert s.rstrip('\n').endswith(expect.rstrip('\n')), \
            f'{m.group(1)}.py drifted from the notebook cell'


def test_all_four_deferred_items_and_required_caveats_are_in_the_summary_cell():
    summ = next(s for t, s in _cells() if "summary_md = '\\n'.join(L)" in s)
    for needle in ['AllWISE', 'Colour variability', 'ParSNIP', 'SNANA', '6,061', '6,123',
                   'TDE_SLSN_STATEMENT', 'TNS_LABEL_NOISE_CAVEAT', 'EXTERNAL_RESULTS',
                   'unverified']:
        assert needle in summ, f'summary cell is missing {needle!r}'
