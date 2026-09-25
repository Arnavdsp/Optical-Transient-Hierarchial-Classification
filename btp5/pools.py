"""
Section 3.3 — the three pools, kept separate throughout.

  1. TNS-confirmed    — objects with a TNS spectroscopic label (v4 behaviour).
  2. ALeRCE-confident — objects ALeRCE itself classifies above the confidence cut,
                        independent of whether they also carry a TNS label.
  3. Overlap          — in both. Much smaller than either parent, and where the
                        sharpest comparison (Section 6.3) happens.

Every table and plot downstream must state which pool it was computed on; the helpers
here carry a `pool` column so that cannot be forgotten.

-------------------------------------------------------------------------------
A structural asymmetry the brief did not anticipate, found while building this
-------------------------------------------------------------------------------
Our stellar-flare class comes from **TESS**, via lightkurve. ALeRCE classifies **ZTF**
alerts. A TESS M-dwarf has no ZTF object id and therefore no ALeRCE probability
vector at all — not a low-confidence one, none.

Consequences, which are reported rather than hidden:
  * The ALeRCE-confident pool cannot contain our TESS flares.
  * The overlap pool therefore has **no stellar_flare class at all**, so the
    ALeRCE-labelled Stage 1 is a 3-way problem (AGN / SNe / TDE) on the overlap,
    against 4-way for the TNS-labelled track.
  * ALeRCE's nearest flare-like classes, CV/Nova and YSO, are ZTF-detected Galactic
    variables — cataclysmic binaries and young stellar objects. They are not the same
    population as UV-Ceti-type M-dwarf flares, so mapping them onto our stellar_flare
    class is an approximation, flagged here and in the summary, not a clean match.

`stage1_class_set` below derives the class list a track can actually support instead
of assuming all four, so the comparison never silently scores a class one side could
not have predicted.
"""

import numpy as np
import pandas as pd

POOL_TNS = 'TNS-confirmed'
POOL_ALERCE = 'ALeRCE-confident'
POOL_OVERLAP = 'overlap'

# ALeRCE classes that stand in for our TESS flare sample, with the caveat above.
FLARE_ANALOGUE_CLASSES = ['CV/Nova', 'YSO']


def attach_alerce_labels(feature_df, alerce_label_rows, verbose=True):
    """
    Join per-object ALeRCE labels onto the feature table.

    `alerce_label_rows` is a list of dicts from
    `alerce_labels.label_from_probabilities`, each carrying an 'id'.
    Objects with no ALeRCE entry get NaN — which for TESS rows is the expected,
    structural outcome described above, not a failure.
    """
    df = feature_df.copy()
    lab = pd.DataFrame(alerce_label_rows) if alerce_label_rows else pd.DataFrame(
        columns=['id', 'alerce_class', 'alerce_prob', 'alerce_coarse', 'alerce_fine'])
    for c in ['alerce_class', 'alerce_prob', 'alerce_coarse', 'alerce_fine']:
        if c not in lab.columns:
            lab[c] = np.nan
    df = df.merge(lab[['id', 'alerce_class', 'alerce_prob', 'alerce_coarse', 'alerce_fine']],
                  on='id', how='left')
    if verbose:
        n = int(df['alerce_coarse'].notna().sum())
        print(f'ALeRCE labels attached to {n}/{len(df)} objects')
        if 'survey' in df.columns:
            by = df.groupby('survey')['alerce_coarse'].apply(lambda s: s.notna().sum())
            for sv, k in by.items():
                tot = int((df['survey'] == sv).sum())
                note = ('  <- structural: ALeRCE classifies ZTF, not TESS'
                        if sv == 'TESS' and k == 0 else '')
                print(f'   {sv}: {k}/{tot}{note}')
    return df


def _as_plain_str(series):
    """numpy string scalars render as np.str_('AGN') in printed class lists and in
    any table built from them. Coerce to builtin str so output stays readable."""
    return series.map(lambda v: str(v) if pd.notna(v) else v)


def build_pools(df, has_tns_col='has_tns_label', verbose=True):
    """
    Tag every row with which pools it belongs to. Returns the frame plus a counts
    table. Nothing is dropped here — membership is marked, so a later step can never
    silently collapse the three pools into one.
    """
    out = df.copy()
    for c in ('alerce_coarse', 'alerce_fine', 'alerce_class'):
        if c in out.columns:
            out[c] = _as_plain_str(out[c])
    if has_tns_col not in out.columns:
        # v4's acquisition is TNS-driven, so every acquired row has a TNS label
        # unless something explicitly says otherwise.
        out[has_tns_col] = out['label'].notna()
    out['in_tns_pool'] = out[has_tns_col].astype(bool)
    out['in_alerce_pool'] = out['alerce_coarse'].notna()
    out['in_overlap_pool'] = out['in_tns_pool'] & out['in_alerce_pool']

    counts = pd.DataFrame([
        {'pool': POOL_TNS, 'n': int(out['in_tns_pool'].sum())},
        {'pool': POOL_ALERCE, 'n': int(out['in_alerce_pool'].sum())},
        {'pool': POOL_OVERLAP, 'n': int(out['in_overlap_pool'].sum())},
    ])
    if verbose:
        print(counts.to_string(index=False))
        print()
        print('Class composition per pool (coarse):')
        for name, mask in [(POOL_TNS, out['in_tns_pool']),
                           (POOL_ALERCE, out['in_alerce_pool']),
                           (POOL_OVERLAP, out['in_overlap_pool'])]:
            sub = out[mask]
            col = 'coarse_label' if name == POOL_TNS else 'alerce_coarse'
            if col not in sub.columns or sub.empty:
                print(f'  {name}: empty')
                continue
            comp = sub[col].value_counts().to_dict()
            print(f'  {name} (by {col}): {comp}')
    return out, counts


def get_pool(df, pool):
    """One pool as its own frame, tagged so downstream output can name it."""
    key = {POOL_TNS: 'in_tns_pool', POOL_ALERCE: 'in_alerce_pool',
           POOL_OVERLAP: 'in_overlap_pool'}[pool]
    sub = df[df[key]].copy()
    sub['pool'] = pool
    return sub


def stage1_class_set(df, label_col, verbose=True):
    """
    Which Stage 1 classes this track can actually support, derived from the data
    rather than assumed.

    Guards the asymmetry above: scoring a 4-class confusion matrix on a pool whose
    ALeRCE side can never contain stellar_flare would report a row of structural
    zeros as if it were a model failure.
    """
    present = sorted(pd.unique(df[label_col].dropna()))
    if verbose:
        print(f'{label_col}: {len(present)} classes present -> {present}')
        missing = set(['AGN', 'SNe', 'TDE', 'stellar_flare']) - set(present)
        if missing:
            print(f'  absent from this pool: {sorted(missing)}')
            if 'stellar_flare' in missing:
                print('  (stellar_flare absent from an ALeRCE-labelled pool is '
                      'expected: TESS objects carry no ZTF alerts to classify)')
    return present
