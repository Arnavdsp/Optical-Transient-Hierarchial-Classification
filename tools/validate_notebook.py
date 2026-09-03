"""
Execute the notebook's OWN Phase 3 + Phase 4 + summary cells against a synthetic
600-object feature table.

The module tests prove `btp_pipeline/` is correct. This proves the notebook is
correct: that the cells are in a runnable order, that every name a cell uses is
already defined by an earlier cell, and that the inlined module code actually
works in notebook scope. Phase 2's cells are the only ones skipped, because they
are the ones that talk to TNS/ALeRCE/MAST.
"""

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
sys.path.insert(0, os.path.join(ROOT, 'tests'))
import synthetic  # noqa: E402
from btp_pipeline.acquisition import assemble_feature_df, verify_counts  # noqa: E402

NB = os.path.join(ROOT, 'notebooks', 'BTP_TNS_Hierarchical_Balanced.ipynb')
START_MARKER = 'Phase 3 — split, models, Stage 1 / Stage 2'

with open(NB) as f:
    nb = json.load(f)

sources = [''.join(c['source']) for c in nb['cells']]
kinds = [c['cell_type'] for c in nb['cells']]
start = next(i for i, s in enumerate(sources) if START_MARKER in s)
print(f'Notebook has {len(sources)} cells; executing code cells from #{start} onward.\n')

tmp = tempfile.mkdtemp(prefix='nbvalidate_')

# Phase 2's product, faked. Same schema, same balanced counts.
feature_df = assemble_feature_df([synthetic.make_feature_df().to_dict('records')], verbose=False)
counts_ok, count_report = verify_counts(feature_df, 150, 30, synthetic.SN_SUBTYPE_LABELS,
                                        verbose=False)

captured = []


def _display(*args):
    captured.append(args[0] if args else None)


ns = {
    '__name__': '__main__',
    'os': os, 'np': np, 'pd': pd, 'plt': plt, 'json': json,
    'display': _display,
    'feature_df': feature_df,
    'count_report': count_report,
    'counts_ok': counts_ok,
    'FEATURE_DIR': tmp,
    'BASE_DIR': tmp,
    'SN_SUBTYPE_LABELS_LIST': synthetic.SN_SUBTYPE_LABELS,
    'FEATURES': ['peak_val', 'rise_time', 'decay_time', 'amplitude', 'color_g_r'],
}
ns['X_COLS'] = ns['FEATURES'] + ['has_color']

executed = 0
for i in range(start, len(sources)):
    if kinds[i] != 'code':
        continue
    src = sources[i]
    label = src.strip().splitlines()[0][:72]
    try:
        exec(compile(src, f'<cell {i}>', 'exec'), ns)
    except Exception as e:
        print(f'\n*** CELL {i} FAILED: {label}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
    executed += 1
    print(f'  cell {i:>3} OK   {label}')
    plt.close('all')

print(f'\nExecuted {executed} notebook code cells with no errors.')

# The notebook must have produced its deliverables on disk.
expected = ['results_summary.md', 'option_ab_comparison.csv',
            'stage1_confusion.png', 'stage1_perm_importance.png',
            'stage1_decision_boundary.png', 'stage2a_confusion.png',
            'stage2b_confusion.png', 'stage2_perm_importance.png',
            'stage2_decision_boundary.png', 'stage1_physics_summary.csv',
            'stage2_physics_summary.csv']
missing = [f for f in expected if not os.path.exists(os.path.join(tmp, f))]
assert not missing, f'notebook did not produce: {missing}'
print(f'All {len(expected)} expected output artefacts were written.')

summary = open(os.path.join(tmp, 'results_summary.md')).read()
assert 'Results summary' in summary and len(summary) > 3000
for bad in ['TODO', 'FIXME', 'nan%']:
    assert bad not in summary, f'placeholder {bad!r} in generated summary'
print(f'Generated results summary: {len(summary)} chars, no placeholders.')
print('\nNOTEBOOK VALIDATION PASSED (Phase 2 network cells not exercised).')
