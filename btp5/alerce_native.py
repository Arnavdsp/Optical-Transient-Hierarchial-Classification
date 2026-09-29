"""
Section 3.3, pool 2, built the way the brief defines it: objects ALeRCE itself
classifies with confidence, sourced FROM ALeRCE — "independent of whether they also
have a TNS label".

The first v5 build did not do this. It attached ALeRCE labels only to objects already
acquired from TNS, which made the ALeRCE-confident pool a strict subset of the
TNS-confirmed pool: the three-pool structure collapsed to two, and Section 6.2's
spectroscopic vs photometric-only comparison could not be computed at all.

Three details below are deliberate, each because the live API does something a
reasonable person would not assume:

1. **Sampling within the confidence cut.** `query_objects` returns matches in
   ASCENDING probability by default: the first objects back for SNIa at p>=0.5 are
   0.500, 0.500, 0.501... Taking "the first N" would build the pool from exactly the
   objects whose ALeRCE labels are least reliable. The full set above the cut is
   fetched and sampled uniformly at random instead — an unbiased draw from
   top-1 >= 0.5, which is the selection Superphot+ used.

2. **Redshift.** v4's quality gate requires one, for peak absolute magnitude. An
   object that ALeRCE classified but TNS never observed spectroscopically has no
   spectroscopic redshift, so under v4's gate every one of them would be rejected and
   the photometric-only subset would be empty by construction. For these objects the
   redshift requirement alone is relaxed (every other gate threshold stands), and
   their redshift / peak_abs_mag are NaN, imputed downstream like any other missing
   feature. That mirrors de Soto et al.'s photometric-only subset, which likewise has
   no spectroscopic redshifts — and it is one reason to EXPECT lower agreement there.

3. **TNS cross-match.** A native object may still have a TNS spectroscopic type. It is
   matched on ZTF object id against v4's `tns_full['ztf_name']`, labelled with v4's
   own rules (classify_sn_subtype, and the same AGN|QSO|Quasar / TDE masks), and
   marked has_tns_label accordingly. TNS types outside our scheme (CV, Varstar, ...)
   are recorded but do not count as a label in our scheme.
"""

from contextlib import contextmanager

import numpy as np
import pandas as pd

from .alerce_labels import COARSE_CROSSWALK, FINE_CROSSWALK, _survey_kwargs

SOURCE_NATIVE = 'ALeRCE-native'


def native_query_groups(live_classes):
    """
    {our group -> [ALeRCE classes to query]}, derived from the crosswalk against the
    LIVE class list. SN classes group by our fine label (so Stage 2 gets every SN
    type); other classes group by our coarse label.
    """
    groups = {}
    for c in sorted(live_classes):
        coarse = COARSE_CROSSWALK.get(c)
        if coarse is None:
            continue
        key = FINE_CROSSWALK.get(c) if coarse == 'SNe' else coarse
        if key:
            groups.setdefault(key, []).append(c)
    return groups


def class_total(alerce, classifier, cls, threshold):
    """How many objects ALeRCE classifies as `cls` at top-1 >= threshold (count=True)."""
    try:
        r = alerce.query_objects(classifier=classifier, class_name=cls, probability=threshold,
                                 page_size=1, count=True, format='json', **_survey_kwargs())
        return int(r.get('total') or 0) if isinstance(r, dict) else None
    except Exception:
        return None


def uniform_sample_class(alerce, classifier, cls, threshold, n, rng, page_size=1000):
    """
    n objects drawn uniformly at random from EVERY object of this class above the cut.

    The API returns matches in ascending probability, so one big page is not a uniform
    sample once a class outgrows it: it is the least confident `page_size` objects.
    (Measured: AGN has 10,744 objects at p>=0.5, so a 5,000-row page is its bottom
    half.) Instead, draw uniformly random ranks over the whole ordered set and fetch
    only the pages that contain them.
    """
    kw = _survey_kwargs()
    total = class_total(alerce, classifier, cls, threshold)
    if total is None:
        # count unavailable: fall back to one page and say so if it was truncated
        r = alerce.query_objects(classifier=classifier, class_name=cls, probability=threshold,
                                 page_size=page_size, format='pandas', **kw)
        if r is None or len(r) == 0:
            return pd.DataFrame(), 0
        if len(r) >= page_size:
            print(f'  WARNING: {cls}: total count unavailable and the page is full — this '
                  'sample is biased toward borderline-confidence objects')
        take = min(n, len(r))
        return r.iloc[np.sort(rng.choice(len(r), size=take, replace=False))], len(r)
    if total == 0:
        return pd.DataFrame(), 0

    take = min(n, total)
    ranks = np.sort(rng.choice(total, size=take, replace=False))
    parts = []
    for page in np.unique(ranks // page_size):
        r = alerce.query_objects(classifier=classifier, class_name=cls, probability=threshold,
                                 page_size=page_size, page=int(page) + 1,
                                 format='pandas', **kw)
        if r is None or len(r) == 0:
            continue
        offs = ranks[(ranks // page_size) == page] % page_size
        parts.append(r.iloc[[o for o in offs if o < len(r)]])
    got = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return got, total


def query_native_candidates(alerce, classifier, live_classes, per_group, threshold, rng,
                            oversample=8, page_size=1000, exclude_oids=(), verbose=True):
    """
    Candidate pools of ALeRCE-classified objects, one per group, each a uniform random
    sample of everything ALeRCE classifies into that group at top-1 >= threshold.
    Within a multi-class group (AGN = AGN + QSO + Blazar) the draw is allocated in
    proportion to each class's total, so it is uniform over the union.
    """
    exclude = set(map(str, exclude_oids))
    pools, rows = {}, []
    for group, classes in native_query_groups(live_classes).items():
        want = int(per_group * oversample)
        totals = {c: class_total(alerce, classifier, c, threshold) for c in classes}
        known = {c: t for c, t in totals.items() if t}
        if known:
            T = sum(known.values())
            alloc = {c: int(np.floor(want * t / T)) for c, t in known.items()}
            for c in sorted(known, key=lambda c: -(want * known[c] / T - alloc[c])):
                if sum(alloc.values()) >= want:
                    break
                alloc[c] += 1
        else:
            alloc = {c: want for c in classes}
        frames, n_avail = [], 0
        for c in classes:
            if alloc.get(c, 0) <= 0:
                continue
            # Already-acquired objects are removed after sampling; the 8x oversample
            # leaves ample headroom for them.
            got, tot = uniform_sample_class(alerce, classifier, c, threshold,
                                            alloc[c], rng, page_size)
            n_avail += tot or 0
            if len(got):
                got = got.copy()
                got['alerce_query_class'] = c
                frames.append(got)
        if not frames:
            rows.append({'group': group, 'ALeRCE classes': ', '.join(classes),
                         f'available (p>={threshold})': n_avail, 'pooled': 0})
            continue
        pool = pd.concat(frames, ignore_index=True)
        pool['oid'] = pool['oid'].astype(str)
        pool = pool.drop_duplicates('oid')
        pool = pool[~pool['oid'].isin(exclude)]
        pool = pool.sample(frac=1.0, random_state=int(rng.integers(1e9))).reset_index(drop=True)
        pool['native_group'] = group
        pools[group] = pool
        rows.append({'group': group, 'ALeRCE classes': ', '.join(classes),
                     f'available (p>={threshold})': n_avail, 'pooled': len(pool),
                     'median p (pooled)': round(float(pool['probability'].median()), 3)})
    summary = pd.DataFrame(rows)
    if verbose:
        print(summary.to_string(index=False))
        print('\nEach pool is a uniform random sample of ALL objects above the cut — ALeRCE '
              'returns the least confident first, so "the first N" would be the worst-'
              'labelled objects.')
    return pools, summary


def tns_label_from_type(t, sn_subtype_fn):
    """v4's labelling rules, applied to one TNS type string."""
    if not isinstance(t, str) or not t or t == 'nan':
        return None
    st = sn_subtype_fn(t)
    if st:
        return st
    tl = t.lower()
    if any(k in tl for k in ('agn', 'qso', 'quasar')):
        return 'AGN'
    if 'tde' in tl:
        return 'TDE'
    return None


def crossmatch_tns(cands, tns_full, sn_subtype_fn):
    """Attach TNS name / type / redshift / our-scheme label by ZTF object id."""
    out = cands.copy()
    if tns_full is None or 'ztf_name' not in tns_full.columns:
        out['tns_name'] = out['tns_type'] = None
        out['tns_redshift'] = np.nan
        out['tns_label'] = None
        out['has_tns_label'] = False
        return out
    ref = (tns_full.dropna(subset=['ztf_name']).drop_duplicates('ztf_name')
           .set_index('ztf_name'))
    oid = out['oid'].astype(str)
    out['tns_name'] = oid.map(ref['name']) if 'name' in ref else None
    out['tns_type'] = oid.map(ref['type']) if 'type' in ref else None
    out['tns_redshift'] = (pd.to_numeric(oid.map(ref['redshift']), errors='coerce')
                           if 'redshift' in ref else np.nan)
    out['tns_label'] = out['tns_type'].apply(lambda t: tns_label_from_type(t, sn_subtype_fn))
    out['has_tns_label'] = out['tns_label'].notna()
    return out


def to_acquisition_pool(cands):
    """
    Reshape into the row format v4's build_class_to_target consumes. `ztf_name` is the
    ALeRCE oid, so v4's resolve_oid returns it directly and never cone-searches.
    """
    z = cands['tns_redshift'] if 'tns_redshift' in cands else np.nan
    return pd.DataFrame({
        'name': cands['oid'].astype(str).values,
        'ztf_name': cands['oid'].astype(str).values,
        'internal_names': cands['oid'].astype(str).values,
        'ra': cands.get('meanra', np.nan), 'declination': cands.get('meandec', np.nan),
        'redshift': z, 'type': cands.get('tns_type'), 'source': SOURCE_NATIVE,
    }).reset_index(drop=True)


@contextmanager
def redshift_optional(qg):
    """
    Relax ONLY v4's require_redshift, for the duration of the native acquisition, and
    restore it afterwards even if acquisition raises. Mutating the dict in place is
    what reaches v4's quality_gate, whose default argument is this same object.
    """
    saved = qg.get('require_redshift')
    qg['require_redshift'] = False
    try:
        yield qg
    finally:
        qg['require_redshift'] = saved


def finalize_native_rows(features, cands, sn_subtypes):
    """
    Feature rows from the native acquisition, labelled honestly: `label` is the TNS
    label if one exists and NaN otherwise — never ALeRCE's opinion, which lives in the
    alerce_* columns. `passes_v4_gate` is True only when a real redshift exists, i.e.
    when the full, unrelaxed v4 gate would also have accepted the object.
    """
    if not features:
        return pd.DataFrame()
    df = pd.DataFrame(features)
    ref = cands.drop_duplicates('oid').set_index('oid')
    oid = df['id'].astype(str)
    df['label'] = oid.map(ref['tns_label']) if 'tns_label' in ref else np.nan
    df['has_tns_label'] = df['label'].notna()
    df['coarse_label'] = df['label'].apply(
        lambda l: ('SNe' if l in sn_subtypes else l) if isinstance(l, str) else np.nan)
    df['source'] = SOURCE_NATIVE
    df['native_group'] = oid.map(ref['native_group']) if 'native_group' in ref else np.nan
    df['is_synthetic'] = False
    df['from_bts'] = 0
    z = pd.to_numeric(df.get('redshift', np.nan), errors='coerce')
    df['passes_v4_gate'] = np.isfinite(z) & (z > 0)
    return df
