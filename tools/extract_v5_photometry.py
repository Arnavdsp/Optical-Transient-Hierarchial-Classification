"""
Lift v4's photometry / Villar-fit / quality-gate / feature code into btp5/photometry.py.

Section 1 of the v5 brief says preserve v4 unless explicitly changed. These functions
are therefore taken VERBATIM by AST line span rather than retyped, so they cannot
drift. Re-run this if the vendored v4 notebook is ever updated.
"""
import ast
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'notebooks', 'source', 'BTP_TNS_Hierarchical_v4_original.ipynb')
OUT = os.path.join(ROOT, 'btp5', 'photometry.py')

PLAN = [
    (13, ['ZP', 'mag_to_flux', 'clean_band']),
    (14, ['villar', 'fit_villar', '_log10']),
    (15, ['quality_gate']),
    (16, ['COSMO', 'distance_modulus', 'process_ztf_object']),
    (17, ['process_tess_object']),
]

HEADER = '''"""
Photometry, the Villar (2019) parametric fit, the quality gate, and feature extraction.

Every definition below is VERBATIM from the v4 notebook
(`notebooks/source/BTP_TNS_Hierarchical_v4_original.ipynb`), lifted by AST span by
`tools/extract_v5_photometry.py`. The v5 brief (Section 1) says to preserve v4's
pipeline except where explicitly changed, and none of these were.

They live in a module so the v5 additions can be tested against them offline and so
Phase 2/3 can run headless. `tools/build_notebook_v5.py` inlines this module back into
the notebook, and `tests5/test_photometry_parity.py` asserts the two stay identical.
"""

import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
from scipy.optimize import least_squares

from .config import QG

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
            while j >= 0 and lines[j].strip().startswith('#'):
                seg = lines[j] + '\n' + seg
                j -= 1
            found.append((names.index(nm), seg))
    missing = set(names) - {names[i] for i, _ in found}
    assert not missing, f'not found: {missing}'
    return [s for _, s in sorted(found)]


def main():
    nb = json.load(open(SRC))
    blocks = []
    for idx, names in PLAN:
        blocks += grab(''.join(nb['cells'][idx]['source']), names)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        f.write(HEADER + '\n' + '\n\n\n'.join(blocks) + '\n')
    print(f'Wrote {OUT} with {sum(len(n) for _, n in PLAN)} verbatim definitions')


if __name__ == '__main__':
    main()
