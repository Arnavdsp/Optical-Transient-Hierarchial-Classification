"""
The headless module and the notebook must carry the same reused code.

Phase 2 can be run either from the notebook (Colab) or headless via
tools/run_phase2.py. Those two paths must not be able to diverge: a fix applied
in one that never reaches the other would mean the pipeline behaves differently
depending on how it was launched.
"""

import ast
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = os.path.join(ROOT, 'notebooks', 'BTP_TNS_Hierarchical_Balanced.ipynb')
MODULE = os.path.join(ROOT, 'btp_pipeline', 'features.py')

REUSED = ['classify_sn_subtype', 'extract_ztf_name', 'resolve_oid', 'clean_lightcurve',
          'downsample_for_gp', 'gp_interpolate', 'extract_shape_features',
          'mag_to_relflux', 'process_ztf_object', 'process_tess_object']


def _defs(source):
    """name -> normalised source, for every function defined at top level."""
    out = {}
    lines = source.split('\n')
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef):
            body = '\n'.join(lines[node.lineno - 1:node.end_lineno])
            out[node.name] = ast.dump(ast.parse(body))
    return out


def test_module_and_notebook_define_the_same_reused_functions():
    module_defs = _defs(open(MODULE).read())

    nb = json.load(open(NB))
    notebook_defs = {}
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        src = ''.join(cell['source'])
        try:
            notebook_defs.update(_defs(src))
        except SyntaxError:
            continue  # cells with Colab magics

    for name in REUSED:
        assert name in module_defs, f'{name} missing from btp_pipeline/features.py'
        assert name in notebook_defs, f'{name} missing from the generated notebook'
        assert module_defs[name] == notebook_defs[name], (
            f'{name} differs between btp_pipeline/features.py and the notebook — '
            f're-run tools/extract_features_module.py and tools/build_notebook.py')


def test_module_does_not_clobber_a_notebook_alerce_client():
    """Inlined into the notebook, this module runs AFTER the cell that creates the
    ALeRCE client, so an unconditional `alerce = None` would silently break
    resolve_oid."""
    src = open(MODULE).read()
    assert "if 'alerce' not in globals():" in src
    assert '\nalerce = None' not in src, 'unconditional alerce=None would clobber the client'
