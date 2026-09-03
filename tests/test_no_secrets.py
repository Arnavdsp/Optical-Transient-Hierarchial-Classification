"""
Guard against committing credentials.

The previous notebook hard-coded a live TNS bot API key. Because that notebook is
the provenance for the verbatim-reused cells, the key travelled into the generated
notebook and into a public repository. It has been removed and must be rotated —
this test makes sure neither comes back.
"""

import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# TNS API keys are a 24-hex-char id, a dot, then an 8-digit suffix.
TNS_KEY_RE = re.compile(r'\b[0-9a-f]{24}\.\d{8}\b')
# A tns_marker with a literal numeric bot id baked in, rather than a variable.
TNS_MARKER_LITERAL_RE = re.compile(r'tns_marker\{"tns_id"\s*:\s*\d+')

SCAN_EXTS = {'.py', '.ipynb', '.md', '.yml', '.yaml', '.txt', '.cfg', '.toml'}
SKIP_DIRS = {'.git', '__pycache__', '.pytest_cache', 'outputs', '.ipynb_checkpoints'}


def _files():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if os.path.splitext(fn)[1] in SCAN_EXTS:
                yield os.path.join(dirpath, fn)


def test_no_tns_api_key_anywhere_in_the_repo():
    offenders = []
    for path in _files():
        if os.path.basename(path) == os.path.basename(__file__):
            continue  # this file carries the patterns by necessity
        with open(path, encoding='utf-8', errors='replace') as f:
            text = f.read()
        if TNS_KEY_RE.search(text):
            offenders.append((os.path.relpath(path, ROOT), 'TNS API key'))
        if TNS_MARKER_LITERAL_RE.search(text):
            offenders.append((os.path.relpath(path, ROOT), 'tns_marker with literal bot id'))
    assert not offenders, f'credential-shaped strings found: {offenders}'


def test_generated_notebook_reads_credentials_at_runtime():
    nb_path = os.path.join(ROOT, 'notebooks', 'BTP_TNS_Hierarchical_Balanced.ipynb')
    with open(nb_path) as f:
        nb = json.load(f)
    cred_cells = [''.join(c['source']) for c in nb['cells']
                  if 'TNS_API_KEY' in ''.join(c['source'])]
    assert cred_cells, 'no credential cell found in the notebook'

    src = cred_cells[0]
    # Sourced, not assigned a literal.
    assert '_get_secret' in src
    assert 'userdata' in src or 'environ' in src
    assert not re.search(r'TNS_API_KEY\s*=\s*["\'][0-9a-f]', src), \
        'TNS_API_KEY is being assigned a literal again'
