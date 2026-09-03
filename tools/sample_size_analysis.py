"""
How many SN objects per subtype does Stage 2 actually need?

The rebuild's whole purpose is to tell whether Stage 2 improved on its previous
~47-53%. That is only answerable if the held-out set can resolve a difference of
that size. This measures the resolution directly: for each candidate
N_PER_SN_SUBTYPE, run Stage 2 Option A over many random splits of the SAME
synthetic population and record how much the held-out accuracy moves from split
luck alone.

The underlying class structure is identical across all rows of the table, so any
spread in held-out accuracy at a given N is pure measurement noise. Read the
'spread' columns, not the mean.

Synthetic data cannot predict the real accuracy — only the precision. That is the
question being asked here.
"""

import os
import sys
import warnings

warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tests'))

from btp_pipeline.acquisition import assemble_feature_df  # noqa: E402
from btp_pipeline.config import SN_SUBTYPE_LABELS, X_COLS  # noqa: E402
from btp_pipeline.modeling import (  # noqa: E402
    add_coarse_label, make_global_split, run_stage2_option_a, single_split_resolution)
import synthetic  # noqa: E402

N_GRID = [30, 50, 75, 100]
N_SEEDS = 15
MODEL = 'Random Forest'

rows = []
for n_sub in N_GRID:
    counts = {s: n_sub for s in SN_SUBTYPE_LABELS}
    counts.update({'AGN': 150, 'TDE': 150, 'stellar_flare': 150})

    accs = []
    for seed in range(N_SEEDS):
        # Same population every time; only the split seed changes.
        df = assemble_feature_df(
            [synthetic.make_feature_df(counts=counts, seed=1234).to_dict('records')],
            verbose=False)
        df = add_coarse_label(df, SN_SUBTYPE_LABELS)
        split = make_global_split(df, X_COLS, random_state=seed, verbose=False)
        s2a = run_stage2_option_a(split, SN_SUBTYPE_LABELS, random_state=42, verbose=False)
        accs.append(float(s2a['results'].set_index('Model').loc[MODEL, 'Test accuracy']))
        n_test = len(s2a['y_test'])

    accs = np.array(accs)
    res = single_split_resolution(n_test, 5)
    rows.append({
        'N per subtype': n_sub,
        'total SNe': 5 * n_sub,
        'Stage 2 test n': n_test,
        'test objects/subtype': round(n_test / 5, 1),
        'mean acc': accs.mean(),
        'spread: std across splits': accs.std(),
        'spread: min-max range': accs.max() - accs.min(),
        'theoretical worst-case SE': res['worst_case_std_error'],
        'per-subtype recall step': res['per_class_recall_step'],
    })
    print(f"N={n_sub:>3}: test n={n_test:>3}, acc {accs.mean():.3f}, "
          f"std across splits {accs.std():.3f}, range {accs.max() - accs.min():.3f}")

table = pd.DataFrame(rows)
print('\n' + '=' * 100)
print(table.round(3).to_string(index=False))
print('=' * 100)

out = os.path.join(ROOT, 'outputs', 'sample_size_analysis.csv')
os.makedirs(os.path.dirname(out), exist_ok=True)
table.to_csv(out, index=False)

print(f"""
Reading this table
------------------
The class structure is identical in every row, so the 'spread' columns are pure
measurement noise from split luck. At N=30 the held-out accuracy moves by
{table.iloc[0]['spread: min-max range']:.2f} across splits of the same data -- larger than the
~0.10 improvement the rebuild is trying to detect. Detecting that improvement
reliably needs the spread comfortably below it.

Note this is the cost per subtype: N=50 means 250 SNe rather than 150, so the
acquisition loop pulls (and the GP fits) proportionally more.

Written to {out}
""")
