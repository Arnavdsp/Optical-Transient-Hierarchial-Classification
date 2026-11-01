"""Extra v6 acquisition: cone-matched minority-class TNS objects, and the SN IIb set.

    python tools/run_v6_extra.py <base_dir> <tns_csv>
"""
import os, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from alerce.core import Alerce
from btp6.osc import resolve_ztf_ids, union_pool
from btp6.acquire import acquire_to_target

base, tns_csv = sys.argv[1:3]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run_v6_acquire.py')).read()
exec(src.split('# the notebook')[1].split('\n', 1)[1].split('tns = pd.read_csv')[0])

tns = pd.read_csv(tns_csv, skiprows=1, low_memory=False)
tns['sn_subtype'] = tns['type'].apply(classify_sn_subtype)
tns['ztf_name'] = tns['internal_names'].apply(extract_ztf_name)
tns = tns[tns['sn_subtype'].notna()]
alerce = Alerce()
fetch = lambda oid: alerce.query_detections(oid, format='pandas', survey='ztf')

cand = tns[tns['ztf_name'].isna() & tns['redshift'].notna() & (tns['declination'] > -30)
           & (pd.to_datetime(tns['discoverydate'], errors='coerce') >= '2018-03-01')
           & (tns['sn_subtype'].isin(['SN_Ib', 'SN_Ic', 'SLSN']) | tns['type'].astype(str).str.startswith('SN IIb'))]
cone = resolve_ztf_ids(cand, alerce, survey_kwargs={'survey': 'ztf'},
                       cache_path=os.path.join(base, 'cone_matches.csv'))
print('cone-matched:', cone[cone['ztf_from_cone']].groupby('sn_subtype').size().to_dict())
tns_all = pd.concat([tns[tns['ztf_name'].notna()], cone[cone['ztf_name'].notna()]], ignore_index=True)

for st in ['SN_Ib', 'SN_Ic', 'SLSN']:
    pool, _ = union_pool(st, [('TNS', tns_all)])
    acquire_to_target(st, pool, 5000, base_dir=base, fetch=fetch, out_subdir=st.lower(), workers=8)

iib = tns_all[tns_all['type'].astype(str).str.startswith('SN IIb')]
pool, _ = union_pool('SN_II', [('TNS', iib)])
acquire_to_target('SN_II', pool, 5000, base_dir=base, fetch=fetch, out_subdir='sn_iib_extra', workers=8)
