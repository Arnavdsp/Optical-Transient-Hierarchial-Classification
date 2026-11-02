"""Template features for every object in a v6 feature table, in parallel, resumable.

    python tools/compute_template_features.py <base_dir> <features.csv> <out.csv> [workers]
"""
import os, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from btp6.templates import TemplateBank, template_features

_bank = None


def _one(args):
    global _bank
    if _bank is None:
        _bank = TemplateBank()
    base, oid, label, z, ebv = args
    try:
        d = pd.read_csv(os.path.join(base, label.lower(), f'{oid}.csv'))
        o = template_features(d, z, _bank, ebv)
    except Exception:
        o = {}
    o['id'] = oid
    return o


if __name__ == '__main__':
    base, feats, out = sys.argv[1:4]
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    df = pd.read_csv(feats)
    done = pd.read_csv(out) if os.path.exists(out) else pd.DataFrame(columns=['id'])
    todo = df[~df['id'].isin(done['id'])]
    rows = []
    with ProcessPoolExecutor(workers) as ex:
        for i, o in enumerate(ex.map(_one, [(base, r.id, r.label, r.redshift, r.ebv) for r in todo.itertuples()],
                                     chunksize=8)):
            rows.append(o)
            if len(rows) % 100 == 0:
                done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True); rows = []
                done.to_csv(out, index=False); print(len(done), flush=True)
    done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
    done.to_csv(out, index=False); print('done', len(done))
