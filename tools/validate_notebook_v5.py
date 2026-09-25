"""
Execute the v5 notebook's OWN cells from STEP 5 onward against synthetic data.

Steps 1-4 are skipped: they install packages, mount Drive and talk to
TNS/ALeRCE/MAST. Everything after acquisition is pure computation over a feature
table, and that is what is checked here — that the cells are in a runnable order,
that every name a cell uses is defined by an earlier one, and that the inlined module
code works in notebook scope.
"""
import json, os, sys, tempfile, warnings
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'tests5'))
import synthetic5

NB = os.path.join(ROOT, 'notebooks', 'BTP_TNS_Hierarchical_v5.ipynb')
INJECT_BEFORE = 'STEP 5 — ALeRCE-labelled ground truth'

nb = json.load(open(NB))
sources = [''.join(c['source']) for c in nb['cells']]
kinds = [c['cell_type'] for c in nb['cells']]
inject_at = next(i for i, s in enumerate(sources) if INJECT_BEFORE in s)
print(f'{len(sources)} cells; module cells run from the top, network cells skipped, '
      f'synthetic feature_df injected at #{inject_at}.\n')

# Cells that cannot run offline: installs, Drive mount, credentials, and anything
# that talks to TNS / ALeRCE / MAST. Everything else — including every inlined
# module — is executed for real.
# Only skip cells that genuinely perform a network / Colab action. An earlier,
# looser rule matched the substring 'bts' and silently skipped the inlined module
# cells too, which made the validation far weaker than it looked.
SKIP_CALL_MARKERS = [
    '!pip install', 'drive.mount(', '_get_secret(', 'requests.post(',
    '= build_class_to_target(', 'build_flares_to_target(FLARE_STAR_NAMES',
    'fetch_taxonomy(alerce)', 'fetch_classes(alerce',
    'add_host_features(feature_df',
    'plot_gp_sanity_check(', 'tns_full[', 'tns_full =', 'tns_full.',
    'assemble_feature_df(\n', 'verify_counts(feature_df',
    'fetch_bts(', 'BTS_TDE_INLINE', 'sn_pools', 'agn_pool', 'tde_pool',
    'all_features', 'all_manifests',
    'Bright Transient Survey Sample Explorer',
]
MODULE_CELL_MARKER = 'Inlined verbatim from btp5/'


def should_skip(src):
    # Never skip an inlined module: those are exactly what we want to execute.
    if MODULE_CELL_MARKER in src:
        return None
    for mk in SKIP_CALL_MARKERS:
        if mk in src:
            return mk
    return None

tmp = tempfile.mkdtemp(prefix='v5val_')
captured = []

# Stand in for what steps 1-4 produce.
feature_df = synthetic5.make_feature_df(with_alerce=False)
LIVE = ['AGN','Blazar','CEP','CV/Nova','DSCT','EA','EB/EW','LPV','Microlensing',
        'Periodic-Other','QSO','RRLab','RRLc','RSCVn','SESN','SLSN','SNII','SNIIn',
        'SNIa','TDE','YSO']


class FakeAlerce:
    """Offline stand-in for the ALeRCE client, matching the real return shapes."""
    def query_probabilities(self, oid, format='pandas', **kw):
        seed = abs(hash(oid)) % 10000
        rng = np.random.default_rng(seed)
        classes = ['SNIa', 'SESN', 'SNII', 'SNIIn', 'SLSN', 'TDE', 'AGN', 'QSO',
                   'Blazar', 'CV/Nova', 'YSO', 'LPV', 'RRLab']
        p = rng.dirichlet(np.ones(len(classes)) * 0.4)
        rows = [{'classifier_name': 'lc_classifier_BHRF_forced_phot',
                 'classifier_version': '2.1.0', 'class_name': c,
                 'probability': float(x), 'ranking': 0} for c, x in zip(classes, p)]
        # a second stacked version, exactly as the real API returns
        p2 = rng.dirichlet(np.ones(len(classes)) * 2.0)
        rows += [{'classifier_name': 'lc_classifier_BHRF_forced_phot',
                  'classifier_version': '1.0.7', 'class_name': c,
                  'probability': float(x), 'ranking': 0} for c, x in zip(classes, p2)]
        return pd.DataFrame(rows)


def _tqdm(it, **kw):
    return it


import gc, io, time, warnings as _warnings, zipfile
try:
    import requests
except Exception:
    requests = None

ns = {
    '__name__': '__main__', 'os': os, 'np': np, 'pd': pd, 'plt': plt, 'json': json,
    'gc': gc, 'io': io, 'time': time, 'zipfile': zipfile, 'warnings': _warnings,
    'requests': requests,
    'display': lambda *a: captured.append(a[0] if a else None),
    'tqdm': _tqdm, 'alerce': FakeAlerce(),
    'feature_df': feature_df, 'FEATURE_DIR': tmp, 'BASE_DIR': tmp,
    'live_classes': LIVE,
}

executed, skipped = 0, []
injected = False
for i in range(len(sources)):
    if i >= inject_at and not injected:
        # Stand in for what steps 1-4 would have produced.
        ns['feature_df'] = feature_df
        injected = True
    if kinds[i] != 'code':
        continue
    src = sources[i]
    why = should_skip(src)
    if why:
        skipped.append((i, f'needs network / Colab ({why})')); continue
    label = next((l for l in src.strip().splitlines() if l.strip()), '')[:70]
    # Cells that only talk to the live API are replaced by their offline equivalent.
    # Guarded on MODULE_CELL_MARKER: the inlined module *defines*
    # `def fetch_classes(alerce, ...)`, whose text matches the call-site pattern, so
    # without this the handler fires on the definition and the module never runs.
    if MODULE_CELL_MARKER not in src:
        if 'fetch_taxonomy(alerce)' in src:
            skipped.append((i, 'live taxonomy fetch')); continue
        if 'fetch_classes(alerce' in src:
            ns['crosswalk_table'], ns['unmapped_classes'] = ns['build_crosswalk'](LIVE)
            skipped.append((i, 'live class fetch -> offline crosswalk')); continue
    try:
        exec(compile(src, f'<cell {i}>', 'exec'), ns)
    except Exception:
        print(f'\n*** CELL {i} FAILED: {label}')
        import traceback; traceback.print_exc()
        sys.exit(1)
    executed += 1
    print(f'  cell {i:>3} OK   {label}')
    plt.close('all')

print(f'\nExecuted {executed} notebook code cells with no errors.')
for i, why in skipped:
    print(f'  (cell {i} skipped: {why})')

out = os.path.join(tmp, 'v5_results_summary.md')
assert os.path.exists(out), 'the notebook did not write its results summary'
text = open(out).read()
assert len(text) > 2000, f'summary suspiciously short ({len(text)} chars)'
for required in ['Deferred items', 'ParSNIP', 'AllWISE', 'SNANA',
                 'External corroboration', 'Townsend', 'de Soto', 'SNID',
                 'three independent pipelines', 'recorded']:
    assert required in text, f'summary is missing required content: {required}'
for bad in ['TODO', 'FIXME', 'nan%']:
    assert bad not in text, f'placeholder {bad!r} in summary'
print(f'\nResults summary: {len(text)} chars, all four deferred items present, '
      f'external table and TNS caveat present, no placeholders.')
print('\nV5 NOTEBOOK VALIDATION PASSED (steps 1-4 not exercised: they need the network).')
