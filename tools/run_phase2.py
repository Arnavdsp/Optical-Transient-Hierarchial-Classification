"""
Phase 2, headless: the real balanced acquisition against TNS / ALeRCE / MAST.

Same code the notebook runs — btp_pipeline.features holds the reused extraction
functions verbatim, btp_pipeline.acquisition holds the retry loops — just driven
from the command line instead of Colab, so it can run unattended and resume.

Credentials come from the environment (or .env.local); nothing is hard-coded.

  python tools/run_phase2.py            # full run, resumes automatically
  python tools/run_phase2.py --classes AGN,TDE
"""

import argparse
import io
import os
import sys
import zipfile

import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from btp_pipeline import features as F  # noqa: E402
from btp_pipeline.acquisition import (  # noqa: E402
    assemble_feature_df, build_flares_to_target, build_tns_class_to_target, verify_counts)
from btp_pipeline.config import N_PER_CLASS, N_PER_SN_SUBTYPE, SN_SUBTYPE_LABELS  # noqa: E402

DATA = os.path.join(ROOT, 'data')
BASE_DIR = os.path.join(DATA, 'raw')
FEATURE_DIR = os.path.join(DATA, 'features')
TNS_CSV_URL = "https://www.wis-tns.org/system/files/tns_public_objects/tns_public_objects.csv.zip"
KEEP_COLS = ['name', 'ra', 'declination', 'redshift', 'type', 'discoverydate', 'internal_names']
CANDIDATE_POOL_MULTIPLIER = 8

# The same curated flare-star list as the notebook, extended so the retry loop has
# somewhere to go if the leading stars run dry.
FLARE_STAR_NAMES = [
    "UV Ceti", "AD Leo", "EV Lac", "YZ CMi", "AU Mic", "Proxima Centauri",
    "GJ 1243", "Ross 154", "Ross 128", "Wolf 359", "Lalande 21185", "Kruger 60",
    "TZ Ari", "GJ 1111", "V1216 Sgr", "EQ Peg", "DO Cep", "V1005 Ori",
    "V371 Ori", "WX UMa", "Luyten 726-8", "GJ 896A", "GJ 1156", "GJ 1245A",
    "GJ 3236", "GJ 3338", "GJ 3737", "GJ 3685A", "GJ 424", "GJ 1002",
    "TRAPPIST-1", "LHS 1140", "Teegarden's Star", "GJ 1061", "YZ Cet",
    "Luyten's Star", "Lacaille 8760", "Lacaille 9352", "Gliese 1", "Gliese 876",
    "Gliese 682", "Gliese 832", "Gliese 667 C", "Kepler-411",

    # --- Extra active stars, added when the list above ran out at 148/150 ---
    "AT Mic", "DX Cnc", "FL Vir", "BY Dra", "CM Dra", "II Peg", "AB Dor",
    "LQ Hya", "V1054 Oph", "GJ 3622", "LP 944-20", "GJ 65", "EQ Peg B",
    "GJ 9520", "V1352 Ori", "GJ 4053", "GJ 2036A", "HK Aqr", "BO Mic",
]


def load_env():
    path = os.path.join(ROOT, '.env.local')
    if os.path.exists(path):
        for line in open(path):
            if '=' in line and not line.strip().startswith('#'):
                k, v = line.strip().split('=', 1)
                os.environ.setdefault(k, v)
    missing = [k for k in ('TNS_BOT_ID', 'TNS_BOT_NAME', 'TNS_API_KEY') if not os.environ.get(k)]
    assert not missing, f'missing credentials: {missing}'


def tns_headers():
    return {'user-agent': ('tns_marker{"tns_id":' + os.environ['TNS_BOT_ID'] +
                           ',"type": "bot", "name":"' + os.environ['TNS_BOT_NAME'] + ' "}')}


def load_tns_catalog():
    """Download the TNS bulk public-objects CSV once and cache it locally."""
    cache = os.path.join(BASE_DIR, 'tns_catalog', 'tns_public_objects.csv')
    if os.path.exists(cache):
        print(f'Using cached TNS catalogue: {cache}')
        return pd.read_csv(cache, low_memory=False)

    os.makedirs(os.path.dirname(cache), exist_ok=True)
    print('Downloading the TNS bulk public-objects CSV (this is the one big download)...')
    resp = requests.post(TNS_CSV_URL, headers=tns_headers(),
                         data={'api_key': os.environ['TNS_API_KEY']}, timeout=600)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        name = z.namelist()[0]
        with z.open(name) as fh:
            # TNS ships a one-line preamble above the real header.
            df = pd.read_csv(fh, skiprows=1, low_memory=False)
    df.to_csv(cache, index=False)
    print(f'TNS catalogue: {len(df):,} rows -> {cache}')
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--classes', default='all',
                    help='comma-separated subset, e.g. SN_Ia,AGN,stellar_flare')
    args = ap.parse_args()

    load_env()
    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(FEATURE_DIR, exist_ok=True)

    from alerce.core import Alerce
    F.alerce = Alerce()          # the client resolve_oid reaches for
    from tqdm.auto import tqdm

    tns_full = load_tns_catalog()
    tns_full['sn_subtype'] = tns_full['type'].apply(F.classify_sn_subtype)
    # AGN family. The original mask matched only the literal string 'AGN', which
    # misses TNS's other AGN type strings: QSOs and blazars ARE active galactic
    # nuclei (a quasar is a luminous AGN; a blazar is one with its jet toward us),
    # so excluding them discarded real class members from an already-scarce class.
    types = tns_full['type'].astype(str)
    agn_mask = types.str.contains('AGN|QSO|Blazar', case=False, na=False, regex=True)
    # 'TDE' already matches TNS's TDE subtypes (TDE-H, TDE-He, TDE-H-He,
    # TDE-featureless) via substring, so this one needs no widening.
    tde_mask = types.str.contains('TDE', case=False, na=False)

    print('\nTNS availability before any download:')
    for st in SN_SUBTYPE_LABELS:
        print(f'  {st:<16} {int((tns_full["sn_subtype"] == st).sum()):>7,} rows')
    print(f'  {"AGN":<16} {int(agn_mask.sum()):>7,} rows')
    print(f'  {"TDE":<16} {int(tde_mask.sum()):>7,} rows')

    def pool_for(mask, target, seed=42):
        pool = tns_full[mask][[c for c in KEEP_COLS if c in tns_full.columns]].copy()
        n = min(len(pool), target * CANDIDATE_POOL_MULTIPLIER)
        return pool.sample(n, random_state=seed).reset_index(drop=True)

    wanted = None if args.classes == 'all' else set(args.classes.split(','))

    def want(name):
        return wanted is None or name in wanted

    tns_features, tns_manifests = {}, {}
    jobs = ([(st, tns_full['sn_subtype'] == st, N_PER_SN_SUBTYPE) for st in SN_SUBTYPE_LABELS]
            + [('AGN', agn_mask, N_PER_CLASS), ('TDE', tde_mask, N_PER_CLASS)])

    for label, mask, target in jobs:
        if not want(label):
            continue
        print(f'\n=== {label} ===')
        tns_manifests[label], tns_features[label] = build_tns_class_to_target(
            label, pool_for(mask, target), target,
            base_dir=BASE_DIR, feature_dir=FEATURE_DIR,
            resolve_oid_fn=F.resolve_oid,
            query_detections_fn=lambda oid: F.alerce.query_detections(
                oid, format='pandas', sort='mjd'),
            process_fn=F.process_ztf_object,
            progress=tqdm)

    flare_features = []
    if want('stellar_flare'):
        print('\n=== stellar_flare (TESS) ===')
        import lightkurve as lk

        def search_fn(name):
            return lk.search_lightcurve(name, mission='TESS', author='SPOC', exptime=120)

        def download_fn(entry):
            lc = entry.download().remove_nans()
            df = lc.to_pandas().reset_index()[['time', 'flux', 'flux_err']]
            return df.rename(columns={'time': 'bjd'})

        _, flare_features = build_flares_to_target(
            FLARE_STAR_NAMES, N_PER_CLASS,
            base_dir=BASE_DIR, feature_dir=FEATURE_DIR,
            search_fn=search_fn, download_fn=download_fn,
            process_fn=F.process_tess_object, progress=tqdm,
            max_sectors_per_star=25)

    if wanted is not None:
        print('\nPartial run complete (subset requested); skipping assembly.')
        return

    print('\n=== assembling ===')
    feature_df = assemble_feature_df(
        [tns_features[s] for s in SN_SUBTYPE_LABELS]
        + [tns_features['AGN'], tns_features['TDE'], flare_features])
    out = os.path.join(FEATURE_DIR, 'phase3_features_balanced.csv')
    feature_df.to_csv(out, index=False)
    print(f'Wrote {out}')

    ok, report = verify_counts(feature_df, N_PER_CLASS, N_PER_SN_SUBTYPE, SN_SUBTYPE_LABELS)
    report.to_csv(os.path.join(FEATURE_DIR, 'count_report.csv'), index=False)
    print('\nPHASE 2 COMPLETE' if ok else '\nPHASE 2 COMPLETE WITH SHORTFALLS (see report)')


if __name__ == '__main__':
    main()
