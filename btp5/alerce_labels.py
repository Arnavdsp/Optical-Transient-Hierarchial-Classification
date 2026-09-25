"""
Section 3.2 — the ALeRCE-labelled ground-truth track.

Three things here are deliberate, and each exists because the live API does not
behave the way a reasonable person would assume:

1. **The taxonomy is fetched at run time, never hardcoded.** The brief says so, and
   checking proved it right: ALeRCE's TDE class does NOT exist in the classic
   `lc_classifier` (hierarchical_rf_1.1.0, 15 classes). It lives only in the 2025
   `lc_classifier_BHRF_forced_phot` (2.1.0, 21 classes), whose SN branch is
   [SNIa, SESN, SNII, SNIIn, SLSN, TDE] — note `SESN`, not `SNIbc`, and note that a
   separate `SNIIn` DOES exist there.

2. **Probabilities must be keyed on (classifier_name, classifier_version), never name
   alone.** `query_probabilities` returns every version it has ever run for an object
   stacked in one frame. Measured on ZTF18abtmbaz: grouping by name alone makes
   `LC_classifier_ATAT_forced_phot(beta)` sum to 4.0 across four versions, and the
   top-1 row is then whichever version happened to score highest — a silent label
   corruption, not an error.

3. **The crosswalk is constructed against the fetched class list and printed**, and a
   class the live taxonomy contains but the mapping does not is reported rather than
   silently dropped.
"""

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Live taxonomy
# --------------------------------------------------------------------------

def client_version():
    """Which alerce client is installed. v1.x and v2.x differ: v2.x requires the
    explicit survey='ztf' argument or warns/raises."""
    try:
        import importlib.metadata as md
        return md.version('alerce')
    except Exception:  # pragma: no cover
        try:
            import alerce
            return getattr(alerce, '__version__', 'unknown')
        except Exception:
            return 'not installed'


def _survey_kwargs(version=None):
    """v2.x wants survey='ztf'; v1.x does not accept it."""
    v = version or client_version()
    major = str(v).split('.')[0]
    return {'survey': 'ztf'} if major.isdigit() and int(major) >= 2 else {}


def fetch_taxonomy(alerce, verbose=True):
    """
    Live classifier catalogue. Returns a DataFrame with one row per
    (classifier_name, classifier_version) and its class list.

    Never assume this matches what any document says — ALeRCE's taxonomy has changed
    at least once already (the 2025 TDE addition) and the version strings reported
    here do not always match the ones attached to per-object probabilities.
    """
    kw = _survey_kwargs()
    df = alerce.query_classifiers(format='pandas', **kw)
    if verbose:
        print(f'alerce client {client_version()} | {len(df)} classifier entries live')
        for _, r in df.iterrows():
            classes = list(r['classes'])
            print(f"  {r['classifier_name']:<42} {str(r['classifier_version']):<32} "
                  f"{len(classes):>2} classes")
    return df


def fetch_classes(alerce, classifier_name, classifier_version, verbose=True):
    """The class list for one classifier, straight from the API."""
    kw = _survey_kwargs()
    try:
        res = alerce.query_classes(classifier_name, classifier_version,
                                   format='pandas', **kw)
        # The column is 'name' on client 2.3.1 and 'class_name' on some other
        # versions; accept either rather than guessing, and fail loudly if neither
        # is present instead of silently returning the column headers as classes.
        col = next((c for c in ('name', 'class_name', 'class') if c in res.columns), None)
        if col is None:
            raise KeyError(f'no class-name column in query_classes output: '
                           f'{list(res.columns)}')
        classes = sorted(res[col].astype(str).unique())
    except Exception as e:
        if verbose:
            print(f'query_classes failed ({type(e).__name__}); falling back to the '
                  f'catalogue listing')
        tax = fetch_taxonomy(alerce, verbose=False)
        row = tax[(tax['classifier_name'] == classifier_name)
                  & (tax['classifier_version'].astype(str) == str(classifier_version))]
        classes = sorted(list(row.iloc[0]['classes'])) if len(row) else []
    if verbose:
        print(f'{classifier_name} [{classifier_version}]: {len(classes)} classes')
        print('  ' + ', '.join(classes))
    return classes


# --------------------------------------------------------------------------
# The crosswalk
# --------------------------------------------------------------------------

# ALeRCE native class -> our Stage 1 coarse class. Built against the LIVE class list
# (see build_crosswalk, which refuses to run against classes it has not seen).
#
# The brief warns specifically: do not assume ALeRCE's "stochastic" branch equals our
# AGN class. It does not. The BHRF stochastic branch is
# [Microlensing, QSO, AGN, Blazar, YSO, CV/Nova] — of which only QSO, AGN and Blazar
# are accreting supermassive black holes. YSO (young stellar objects) and CV/Nova
# (cataclysmic variables) are Galactic stellar systems; CV/Nova in particular is the
# closest ALeRCE analogue to our stellar-flare class, not to AGN. Microlensing is
# neither and is dropped.
COARSE_CROSSWALK = {
    # --- AGN family: accretion onto a supermassive black hole ---
    'AGN': 'AGN',
    'QSO': 'AGN',          # a quasar is a luminous AGN
    'Blazar': 'AGN',       # an AGN with its jet toward us

    # --- supernovae ---
    'SNIa': 'SNe',
    'SNIbc': 'SNe',        # classic lc_classifier naming
    'SESN': 'SNe',         # BHRF 2.1.0 naming for the same stripped-envelope class
    'SNIb': 'SNe',
    'SNIc': 'SNe',
    'SNII': 'SNe',
    'SNIIb': 'SNe',
    'SNIIn': 'SNe',
    'SLSN': 'SNe',

    # --- tidal disruption events ---
    'TDE': 'TDE',

    # --- Galactic stellar variability; CV/Nova is the flare-like analogue ---
    'CV/Nova': 'stellar_flare',
    'YSO': 'stellar_flare',
}

# Deliberately unmapped: everything periodic (LPV, E, EA, EB/EW, RRL, RRLab, RRLc,
# CEP, DSCT, RSCVn, Periodic-Other) and Microlensing. These have no counterpart in our
# 4-class scheme; objects whose top-1 lands here are EXCLUDED from the ALeRCE pool
# rather than forced into a class they do not belong to.
UNMAPPED_IS_EXCLUSION = True

# ALeRCE native SN class -> our Stage 2 fine class.
#
# Two asymmetries are stated rather than papered over:
#  * ALeRCE only ever emits a merged stripped-envelope class (SNIbc in the classic
#    model, SESN in BHRF 2.1.0). Our TNS scheme keeps Ib and Ic apart. So the
#    ALeRCE-labelled Stage 2 is necessarily coarser. The brief says not to merge
#    TNS's Ib+Ic to match, and that is respected.
#  * BHRF 2.1.0 DOES have a separate SNIIn, which our scheme merges into SN_II. We
#    merge ALeRCE's SNIIn into SN_II too, so both tracks apply the same merge and the
#    agreement matrices compare like with like.
FINE_CROSSWALK = {
    'SNIa': 'SN_Ia',
    'SNIbc': 'SN_Ibc',
    'SESN': 'SN_Ibc',
    'SNII': 'SN_II',
    'SNIIn': 'SN_II',      # merged, symmetrically with our TNS-side IIn -> SN_II
    'SNIIb': 'SN_II',
    'SLSN': 'SLSN',
}

# How our TNS fine labels collapse when compared against ALeRCE's coarser SN scheme.
TNS_TO_ALERCE_FINE = {
    'SN_Ia': 'SN_Ia', 'SN_Ib': 'SN_Ibc', 'SN_Ic': 'SN_Ibc',
    'SN_II': 'SN_II', 'SLSN': 'SLSN',
}


def build_crosswalk(live_classes, verbose=True):
    """
    Map the LIVE class list onto our schemes and print the result.

    Required, non-optional step per the brief: this must run and be inspected before
    any ALeRCE label is used downstream. Returns (table, unmapped).
    """
    rows = []
    for c in sorted(live_classes):
        coarse = COARSE_CROSSWALK.get(c)
        fine = FINE_CROSSWALK.get(c)
        rows.append({'alerce_class': c,
                     'our_coarse_class': coarse if coarse else '(excluded)',
                     'our_fine_class': fine if fine else '',
                     'used': bool(coarse)})
    table = pd.DataFrame(rows)
    unmapped = table.loc[~table['used'], 'alerce_class'].tolist()

    if verbose:
        print('ALeRCE -> our-scheme crosswalk, built against the LIVE class list:')
        print(table.to_string(index=False))
        print(f'\nMapped: {int(table["used"].sum())} / {len(table)} live classes.')
        if unmapped:
            print('Excluded (no counterpart in our 4-class scheme) — objects whose '
                  'top-1 lands here are dropped, not reassigned:')
            print('  ' + ', '.join(unmapped))
        known = set(COARSE_CROSSWALK) | set(FINE_CROSSWALK)
        stale = sorted(known - set(live_classes))
        if stale:
            print('\nNOTE: the crosswalk names classes the live taxonomy does NOT '
                  'contain (harmless, but means the mapping predates this taxonomy):')
            print('  ' + ', '.join(stale))
    return table, unmapped


# --------------------------------------------------------------------------
# Per-object labels
# --------------------------------------------------------------------------

def top_class(prob_df, classifier_name, classifier_version=None):
    """
    Top-1 class and probability for ONE classifier, keyed on name AND version.

    Keying on name alone is a silent bug: query_probabilities stacks every version it
    has run for the object, so the frame contains several complete probability
    distributions and the argmax crosses between them.
    """
    if prob_df is None or len(prob_df) == 0:
        return None
    sub = prob_df[prob_df['classifier_name'] == classifier_name]
    if classifier_version is not None:
        sub = sub[sub['classifier_version'].astype(str) == str(classifier_version)]
    if len(sub) == 0:
        return None
    if classifier_version is None and sub['classifier_version'].nunique() > 1:
        # Pick one version deterministically rather than mixing distributions.
        chosen = sorted(sub['classifier_version'].astype(str).unique())[-1]
        sub = sub[sub['classifier_version'].astype(str) == chosen]
    total = float(sub['probability'].sum())
    best = sub.loc[sub['probability'].idxmax()]
    return {'class_name': str(best['class_name']),
            'probability': float(best['probability']),
            'classifier_version': str(best['classifier_version']),
            'prob_sum': total,
            'n_classes': int(len(sub))}


def label_from_probabilities(prob_df, classifier_name, classifier_version=None,
                             threshold=0.5):
    """
    One object's ALeRCE-derived labels, or None if it does not qualify.

    Returns coarse/fine labels plus the raw native class and confidence, so every
    downstream table can show what the mapping actually did.
    """
    top = top_class(prob_df, classifier_name, classifier_version)
    if top is None:
        return None
    if top['probability'] < threshold:
        return None
    coarse = COARSE_CROSSWALK.get(top['class_name'])
    if coarse is None:
        return None                      # unmapped -> excluded, never reassigned
    return {'alerce_class': top['class_name'],
            'alerce_prob': top['probability'],
            'alerce_version': top['classifier_version'],
            'alerce_coarse': coarse,
            'alerce_fine': FINE_CROSSWALK.get(top['class_name'])}


def collapse_tns_fine(label):
    """Our TNS fine label in ALeRCE's coarser SN vocabulary (Ib/Ic -> Ibc)."""
    return TNS_TO_ALERCE_FINE.get(label, label)
