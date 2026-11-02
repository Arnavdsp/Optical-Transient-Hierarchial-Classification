"""Headless v6 SN acquisition: TNS + OSC (+ BTS if reachable) pools, parallel ALeRCE pull.

Usage: python tools/run_v6_acquire.py <data_dir> [tns_csv] [osc_json]
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from btp6.osc import load_osc_catalog, osc_frame, union_pool
from btp6.acquire import acquire_to_target

data = sys.argv[1]
tns_csv = sys.argv[2] if len(sys.argv) > 2 else os.path.join(data, 'tns_public_objects.csv')
osc_json = sys.argv[3] if len(sys.argv) > 3 else os.path.join(data, 'catalog.min.json')

# the notebook's own TNS rule, copied verbatim from cell 12 of the v5 notebook
def classify_sn_subtype(t):
    t = str(t)
    if 'SLSN' in t: return 'SLSN'
    if not t.startswith('SN '): return None
    rest = t[3:]
    if rest.startswith('Ib/c') or rest.startswith('Ibc'): return None
    if rest.startswith('Ia'): return 'SN_Ia'
    if rest.startswith('Ib'): return 'SN_Ib'
    if rest.startswith('Ic'): return 'SN_Ic'
    if rest.startswith('II'): return 'SN_II'
    return None

def extract_ztf_name(s):
    if not isinstance(s, str): return None
    for tok in s.split(','):
        tok = tok.strip()
        if tok.startswith('ZTF'): return tok
    return None

tns = pd.read_csv(tns_csv, skiprows=1, low_memory=False)
tns['sn_subtype'] = tns['type'].apply(classify_sn_subtype)
tns['ztf_name'] = tns['internal_names'].apply(extract_ztf_name)
tns = tns[tns['sn_subtype'].notna()]
osc = osc_frame(load_osc_catalog(osc_json), classify_sn_subtype)
print('OSC usable rows:', osc['sn_subtype'].value_counts().to_dict())

from alerce.core import Alerce
alerce = Alerce()
fetch = lambda oid: alerce.query_detections(oid, format='pandas', survey='ztf')

TARGETS = {'SN_Ia': int(os.environ.get('N_IA', 400)), 'SN_II': int(os.environ.get('N_II', 400)),
           'SN_Ic': 5000, 'SN_Ib': 5000, 'SLSN': 5000}
out = []
for st, tgt in TARGETS.items():
    pool, stats = union_pool(st, [('TNS', tns), ('OSC', osc)])
    print(st, stats, flush=True)
    t0 = time.time()
    f, m = acquire_to_target(st, pool, tgt, base_dir=data, fetch=fetch, out_subdir=st.lower(), workers=8)
    print(f'   {time.time()-t0:.0f}s', flush=True)
    out.append(f)
df = pd.concat(out, ignore_index=True)
df.to_csv(os.path.join(data, 'sn_features_v6.csv'), index=False)
print(df['label'].value_counts().to_string())
