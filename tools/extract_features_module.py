"""
Regenerate btp_pipeline/features.py from the vendored original notebook.

Functions are lifted by AST line span, never retyped, so `features.py` is a
verbatim copy of the code the brief said to reuse unchanged. Re-run this if the
vendored source notebook is ever updated.
"""
import ast
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'notebooks', 'source', 'BTP_TNS_Hierarchical_2_original.ipynb')
OUT = os.path.join(ROOT, 'btp_pipeline', 'features.py')

# (cell index, names to lift) — order here is the order in the generated module.
PLAN = [
    (5,  ['classify_sn_subtype']),
    (6,  ['extract_ztf_name', 'resolve_oid']),
    (13, ['clean_lightcurve']),
    (14, ['MAX_GP_POINTS', 'downsample_for_gp', 'gp_interpolate']),
    (15, ['extract_shape_features']),
    (16, ['mag_to_relflux', 'process_ztf_object']),
    (17, ['process_tess_object']),
]

HEADER = '''"""
Light-curve preprocessing and feature extraction.

Every function below is copied VERBATIM out of the previous notebook
(`notebooks/source/BTP_TNS_Hierarchical_2_original.ipynb`) by
`tools/extract_features_module.py`, which lifts them by AST span rather than
retyping them. The brief said to reuse these unchanged, and they are unchanged.

They live in a module rather than only in notebook cells so that Phase 2 can be
run headless (outside Colab) and so the retry loops can be tested against them.
`tests/test_features_parity.py` asserts this module and the generated notebook
still carry byte-identical definitions.
"""

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C

# `resolve_oid` calls a module-level `alerce` client. When this module is inlined
# into the notebook, cell 2 has already created one — so do not clobber it. When
# imported headless, the caller assigns it (see tools/run_phase2.py).
if 'alerce' not in globals():
    alerce = None
'''


def grab(cell_source, names):
    lines = cell_source.split('\n')
    found = []
    for node in ast.parse(cell_source).body:
        nm = None
        if isinstance(node, ast.FunctionDef):
            nm = node.name
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            nm = node.targets[0].id
        if nm in names:
            seg = '\n'.join(lines[node.lineno - 1:node.end_lineno])
            j = node.lineno - 2
            if j >= 0 and lines[j].strip().startswith('#'):
                seg = lines[j] + '\n' + seg
            found.append((names.index(nm), seg))
    missing = set(names) - {names[i] for i, _ in found}
    assert not missing, f'not found in cell: {missing}'
    return [s for _, s in sorted(found)]


def main():
    nb = json.load(open(SRC))
    blocks = []
    for idx, names in PLAN:
        blocks += grab(''.join(nb['cells'][idx]['source']), names)
    with open(OUT, 'w') as f:
        f.write(HEADER + '\n\n' + '\n\n\n'.join(blocks) + '\n')
    print(f'Wrote {OUT} with {sum(len(n) for _, n in PLAN)} verbatim definitions')


if __name__ == '__main__':
    main()
