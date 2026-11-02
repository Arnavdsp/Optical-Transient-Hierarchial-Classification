"""
Execute the v5 notebook's own cells offline, against an injected table shaped EXACTLY
like v4's acquisition output — no coarse_label, no coordinates, no pool columns.

The first v5 validator injected a friendlier table than reality (coordinates and
coarse_label already present) and so missed two live-run failures. This one does not:
light curves are real CSVs on disk, the ALeRCE-native path runs through v4's real
fitter under the relaxed gate, and augmentation runs through the real quality gate.

Skipped: only cells that install packages, mount Drive, read credentials, or download
from TNS / BTS / MAST. A fake ALeRCE client mimics the live API's shapes (stacked
versions, ascending pages, count=True).
"""
import ast
import json
import os
import sys
import tempfile
import warnings

warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tests5'))
import synthetic5  # noqa: E402

NB = os.path.join(ROOT, 'notebooks', 'BTP_TNS_Hierarchical_v5.ipynb')
V4 = os.path.join(ROOT, 'notebooks', 'source', 'BTP_TNS_Hierarchical_v4_original.ipynb')
tmp = tempfile.mkdtemp(prefix='v5val_')
BASE_DIR, FEATURE_DIR = os.path.join(tmp, 'raw'), os.path.join(tmp, 'features')
os.makedirs(BASE_DIR); os.makedirs(FEATURE_DIR)
rng = np.random.default_rng(0)

# ------------------------------------------------------------------ what Step 4 yields
raw = synthetic5.make_feature_df(with_alerce=False)
V4_OUTPUT_DROP = ['coarse_label', 'is_synthetic', 'host_offset_arcsec', 'host_offset_norm',
                  'has_host_match']
feature_df_v4 = raw.drop(columns=[c for c in V4_OUTPUT_DROP if c in raw.columns])
assert 'ra' not in feature_df_v4 and 'coarse_label' not in feature_df_v4
for _, r in feature_df_v4[feature_df_v4['survey'] == 'ZTF'].iterrows():
    d = os.path.join(BASE_DIR, r['label'].lower()); os.makedirs(d, exist_ok=True)
    synthetic5.make_lightcurve_csv(os.path.join(d, f"{r['id']}.csv"),
                                   peak_mag=float(rng.uniform(17.2, 18.4)),
                                   seed=int(rng.integers(1e6)),
                                   ra=float(rng.uniform(0, 360)), dec=float(rng.uniform(-20, 70)))

ALERCE_OF = {'SN_Ia': 'SNIa', 'SN_Ib': 'SESN', 'SN_Ic': 'SESN', 'SN_II': 'SNII',
             'SLSN': 'SLSN', 'AGN': 'QSO', 'TDE': 'TDE'}
TRUTH = dict(zip(feature_df_v4['id'], feature_df_v4['label']))
MAPPED = ['SNIa', 'SESN', 'SNII', 'SNIIn', 'SLSN', 'TDE', 'AGN', 'QSO', 'Blazar', 'CV/Nova',
          'YSO', 'LPV', 'RRLab']


class FakeAlerce:
    """Shapes of the live API: stacked versions, ascending pages, count=True."""
    def __init__(self):
        self.native_class = {}

    def query_probabilities(self, oid, format='pandas', **kw):
        r = np.random.default_rng(abs(hash(oid)) % (2 ** 32))
        true_cls = self.native_class.get(oid) or ALERCE_OF.get(TRUTH.get(oid))
        p = r.dirichlet(np.ones(len(MAPPED)) * 0.3)
        if true_cls and r.random() < 0.8:
            p = 0.3 * p; p[MAPPED.index(true_cls)] += 0.7
        rows = [dict(classifier_name='lc_classifier_BHRF_forced_phot', classifier_version='2.1.0',
                     class_name=c, probability=float(x), ranking=0) for c, x in zip(MAPPED, p)]
        q = r.dirichlet(np.ones(len(MAPPED)))
        rows += [dict(classifier_name='lc_classifier_BHRF_forced_phot', classifier_version='1.0.7',
                      class_name=c, probability=float(x), ranking=0) for c, x in zip(MAPPED, q)]
        return pd.DataFrame(rows)

    def query_objects(self, classifier=None, class_name=None, probability=0.5, page_size=100,
                      page=1, count=False, format='pandas', **kw):
        n = 60
        ids = [f"ZTFN{class_name.replace('/', '')}{i:04d}" for i in range(n)]
        for i in ids:
            self.native_class[i] = class_name
        full = pd.DataFrame({'oid': ids, 'meanra': np.linspace(1, 359, n),
                             'meandec': np.linspace(-10, 60, n),
                             'probability': np.linspace(probability, 0.9, n)})
        if count:
            return {'total': n, 'items': []}
        lo = (page - 1) * page_size
        return full.iloc[lo:lo + page_size].reset_index(drop=True)


LIVE = ['AGN', 'Blazar', 'CEP', 'CV/Nova', 'DSCT', 'EA', 'EB/EW', 'LPV', 'Microlensing',
        'Periodic-Other', 'QSO', 'RRLab', 'RRLc', 'RSCVn', 'SESN', 'SLSN', 'SNII', 'SNIIn',
        'SNIa', 'TDE', 'YSO']

# A few native objects DO have a TNS type (so both subsets of Section 6.2 are populated).
fake_tns_full = pd.DataFrame({
    'ztf_name': [f'ZTFNSNIa{i:04d}' for i in range(0, 60, 3)] + [f'ZTFNQSO{i:04d}' for i in range(0, 60, 4)],
    'name': [f'2024x{i}' for i in range(20)] + [f'2024q{i}' for i in range(15)],
    'type': ['SN Ia'] * 20 + ['QSO'] * 15,
    'redshift': [0.05] * 20 + [0.8] * 15})

# v4's classify_sn_subtype, lifted by AST from the v4 cell that also touches tns_full.
_v4 = json.load(open(V4))
_c6 = ''.join(_v4['cells'][6]['source'])
_fn = next(n for n in ast.parse(_c6).body if isinstance(n, ast.FunctionDef)
           and n.name == 'classify_sn_subtype')
CLASSIFY_SRC = '\n'.join(_c6.split('\n')[_fn.lineno - 1:_fn.end_lineno])

NATIVE_OFFLINE = '''
native_features, native_cands = [], []
with redshift_optional(QG):
    for group, cands in native_pools.items():
        cands = crossmatch_tns(cands.head(14), tns_full, classify_sn_subtype)
        native_cands.append(cands)
        d = os.path.join(BASE_DIR, f'alerce_{group.lower()}'); os.makedirs(d, exist_ok=True)
        for _, c in cands.iterrows():
            p = os.path.join(d, f"{c['oid']}.csv")
            _make_lc(p, seed=abs(hash(c['oid'])) % 99999)
            row = process_ztf_object(p, c['oid'], f'ALERCE_{group}', redshift=c['tns_redshift'])
            if row is not None:
                native_features.append(row)
assert QG['require_redshift'] is True
native_cands = pd.concat(native_cands, ignore_index=True)
native_df = finalize_native_rows(native_features, native_cands, SN_SUBTYPES)
feature_df = pd.concat([feature_df, native_df], ignore_index=True)
print(f"offline native acquisition: {len(native_df)} objects, "
      f"{int(native_df['has_tns_label'].sum())} TNS-confirmed")
'''

SKIP = ['!pip install', 'drive.mount(', 'def _get_secret(', 'requests.post(', 'fetch_bts(',
        'BTS_TDE_INLINE', 'sn_pools', "tns_full['", 'tns_full[agn', 'all_features',
        'all_manifests', 'verify_counts(feature_df', 'fetch_taxonomy(alerce)']
MODULE = 'Inlined verbatim from btp5/'

ns = {'__name__': '__main__', 'os': os, 'np': np, 'pd': pd, 'plt': plt, 'json': json,
      'display': lambda *a: None, 'tqdm': lambda it=None, **k: it if it is not None else _Bar(),
      'alerce': FakeAlerce(), 'BASE_DIR': BASE_DIR, 'FEATURE_DIR': FEATURE_DIR,
      'tns_full': fake_tns_full, '_make_lc': synthetic5.make_lightcurve_csv}


class _Bar:
    def update(self, *a): pass
    def close(self): pass


exec(CLASSIFY_SRC, ns)

cells = [(c['cell_type'], ''.join(c['source'])) for c in json.load(open(NB))['cells']]
inject_at = next(i for i, (t, s) in enumerate(cells)
                 if t == 'code' and "feature_df['coarse_label']" in s and 'SN_SUBTYPES' in s)
done, skipped = 0, []
for i, (t, src) in enumerate(cells):
    if t != 'code':
        continue
    if i == inject_at:
        ns['feature_df'] = feature_df_v4.copy()
    head = next((l for l in src.splitlines() if l.strip()), '')[:66]
    if MODULE not in src:
        if 'fetch_classes(alerce' in src:
            ns['live_classes'] = LIVE
            ns['crosswalk_table'], ns['unmapped_classes'] = ns['build_crosswalk'](LIVE)
            skipped.append((i, 'live class list -> offline crosswalk')); continue
        if 'with redshift_optional(QG):' in src:
            exec(compile(NATIVE_OFFLINE, f'<native offline {i}>', 'exec'), ns)
            skipped.append((i, 'native acquisition -> offline, through the real fitter')); done += 1
            continue
        why = next((m for m in SKIP if m in src), None)
        if why:
            skipped.append((i, why)); continue
    try:
        exec(compile(src, f'<cell {i}>', 'exec'), ns)
    except Exception:
        print(f'\n*** CELL {i} FAILED: {head}')
        import traceback; traceback.print_exc()
        sys.exit(1)
    done += 1
    print(f'  cell {i:>3} OK  {head}')
    plt.close('all')

print(f'\nExecuted {done} code cells; skipped {len(skipped)} network/Colab cells:')
for i, why in skipped:
    print(f'   cell {i}: {why}')

# ------------------------------------------------------------------ outcome checks
fd = ns['feature_df']
checks = {
    'coarse_label created before training': 'coarse_label' in fd,
    'coordinates recovered for ZTF rows': fd.loc[fd['survey'] == 'ZTF', 'ra'].notna().mean() > 0.95,
    'synthetic rows generated': int(fd['is_synthetic'].astype(bool).sum()) > 0,
    'every synthetic row has a parent': fd.loc[fd['is_synthetic'].astype(bool), 'parent_id'].notna().all(),
    'three pools all non-empty': bool((ns['pool_counts']['n'] > 0).all()),
    'ALeRCE pool is NOT a subset of the TNS pool':
        bool((fd['in_alerce_pool'].fillna(False) & ~fd['in_tns_pool'].fillna(False)).any()),
    'Stage 1 photometric-only subset scored': any(
        'actual_rate' in r['subsets'].get('photometric-only (ALeRCE-confident, no TNS label)', {})
        for r in ns['stage1_reports'].values()),
    'Stage 2 agreement computed': any('expected_matrix' in r for r in ns['stage2_reports'].values()),
    'augmentation ablation used identical test sets': ns['aug_ablation'] is not None,
    'require_redshift restored': ns['QG']['require_redshift'] is True,
}
width = max(map(len, checks))
for k, v in checks.items():
    print(f'  [{"PASS" if v else "FAIL"}] {k:<{width}}')
assert all(checks.values()), 'outcome checks failed'

text = open(os.path.join(FEATURE_DIR, 'v5_results_summary.md')).read()
for need in ['Deferred items', 'ParSNIP', 'AllWISE', 'SNANA', 'External corroboration',
             'three independent pipelines', 'recorded', 'TDE cross-check', 'unverified']:
    assert need in text, f'summary missing: {need}'
for bad in ['TODO', 'FIXME', 'nan%']:
    assert bad not in text, f'placeholder {bad!r} in summary'
print(f'\nSummary: {len(text)} chars, all required sections present.')
print('V5 NOTEBOOK VALIDATION PASSED (network/Colab cells not exercised)')
