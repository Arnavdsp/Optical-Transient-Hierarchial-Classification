"""Execute PART B of the v6 notebook outside Colab, for real (live ALeRCE + OSC).

PART A's globals are supplied from the tested btp5 modules and a local TNS CSV, and
BASE_DIR points at a local folder (light curves already there are re-used, exactly as
on Drive). Matplotlib runs headless; figures are saved next to the data.

    python tools/validate_notebook_v6.py <notebook.ipynb> <data_dir> <tns_csv>
"""
import json, os, sys, warnings
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')

nb_path, data_dir, tns_csv = sys.argv[1:4]
nb = json.load(open(nb_path))
cells = nb['cells']
start = next(i for i, c in enumerate(cells) if c['cell_type'] == 'markdown'
             and ''.join(c['source']).lstrip().startswith('# PART B'))

from btp5 import config, photometry, alerce_labels
from alerce.core import Alerce
ns = {}
for mod in (config, photometry):
    ns.update({k: v for k, v in vars(mod).items() if not k.startswith('__')})
ns['_survey_kwargs'] = alerce_labels._survey_kwargs
ns['alerce'] = Alerce()
ns['BASE_DIR'] = data_dir

# PART A cell 12, verbatim rule
src12 = [c for c in cells if 'def classify_sn_subtype' in ''.join(c['source'])][0]
code12 = ''.join(src12['source']).split("tns_full['sn_subtype']")[0]
exec(code12, ns)
tns = pd.read_csv(tns_csv, skiprows=1, low_memory=False)
tns['sn_subtype'] = tns['type'].apply(ns['classify_sn_subtype'])
tns['ztf_name'] = tns['internal_names'].apply(ns['extract_ztf_name'])
ns['tns_full'] = tns
from IPython.display import display
ns['display'] = lambda x: print(x.to_string() if hasattr(x, 'to_string') else getattr(x, 'data', x))
fig_n = [0]
def _show(*a, **k):
    fig_n[0] += 1
    plt.savefig(os.path.join(data_dir, f'v6_fig{fig_n[0]}.png'), dpi=110, bbox_inches='tight')
    plt.close('all')
plt.show = _show

for i, c in enumerate(cells[start:], start):
    if c['cell_type'] != 'code':
        continue
    src = ''.join(c['source'])
    print(f'\n===== cell {i} =====', flush=True)
    exec(compile(src, f'cell{i}', 'exec'), ns)
print('\nPART B executed end to end.')
