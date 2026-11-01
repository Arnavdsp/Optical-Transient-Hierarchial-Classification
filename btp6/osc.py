"""
Open Supernova Catalog (sne.space, Guillochon et al. 2017, Harvard CfA) as a third
label source for the SN subtype pools.

Why it helps: the OSC aggregates spectroscopic classifications from TNS, ATel, the
literature and survey releases, and it records the ZTF internal name as an alias. On
2026-10-31 the live catalogue held 336 SN Ic, 233 SN Ib and 201 SLSNe that carry a
ZTF alias and a redshift, against the 30 per subtype the v5 run stopped at.

Only the label and the ZTF id are taken from the OSC. The photometry still comes from
ALeRCE, so every object is measured by the same instrument and pipeline as the rest of
the sample: mixing in the OSC's own heterogeneous photometry (many telescopes, many
filter systems) would teach the classifier the survey, not the supernova.

Label hygiene. An OSC event can carry several `claimedtype` values from different
sources. An object is kept only when every claimed type maps to the same subtype
bucket; conflicting claims are dropped, not voted on. 'Ib/c' maps to nothing, exactly
as `classify_sn_subtype` does for TNS.
"""

import json
import re

import numpy as np
import pandas as pd

OSC_CATALOG_URL = 'https://sne.space/astrocats/astrocats/supernovae/output/catalog.min.json'


def load_osc_catalog(path=None, url=OSC_CATALOG_URL, timeout=600):
    """The full OSC catalogue as a list of event dicts (~110k events, ~53 MB)."""
    if path is not None:
        try:
            with open(path) as fh:
                return json.load(fh)
        except (OSError, ValueError):
            pass
    import requests
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    if path is not None:
        with open(path, 'wb') as fh:
            fh.write(r.content)
    return r.json()


def _values(event, key):
    return [str(v.get('value', '')).strip() for v in (event.get(key) or [])
            if str(v.get('value', '')).strip()]


def _first_float(values):
    for v in values:
        try:
            x = float(v)
        except ValueError:
            continue
        if np.isfinite(x):
            return x
    return np.nan


def osc_subtype(claimed, classify):
    """OSC claimedtype string -> subtype bucket via the notebook's own TNS rule.

    OSC writes 'Ic-BL' where TNS writes 'SN Ic-BL', so the TNS prefix is added and the
    same `classify_sn_subtype` decides. Anything that rule rejects stays rejected.
    """
    t = str(claimed).strip()
    if not t:
        return None
    if 'SLSN' in t:
        return 'SLSN'
    t = re.sub(r'^SN\s*', '', t)
    return classify('SN ' + t)


def osc_frame(catalog, classify):
    """One row per OSC event with a ZTF alias and an unambiguous subtype."""
    rows = []
    for e in catalog:
        ztf = [a for a in _values(e, 'alias') if a.startswith('ZTF')]
        if not ztf:
            continue
        claims = _values(e, 'claimedtype')
        buckets = {osc_subtype(c, classify) for c in claims}
        buckets.discard(None)
        if len(buckets) != 1:
            continue                      # no usable claim, or claims that disagree
        rows.append({
            'name': e.get('name'),
            'ztf_name': ztf[0],
            'sn_subtype': buckets.pop(),
            'type': claims[0],
            'n_claims': len(claims),
            'redshift': _first_float(_values(e, 'redshift')),
            'ebv': _first_float(_values(e, 'ebv')),
            'ra': (_values(e, 'ra') or [None])[0],
            'declination': (_values(e, 'dec') or [None])[0],
        })
    df = pd.DataFrame(rows)
    if len(df):
        df = df.drop_duplicates('ztf_name', keep='first').reset_index(drop=True)
    return df


def union_pool(subtype, sources, require_redshift=True, random_state=42):
    """Merge per-source candidate frames for one subtype, de-duplicated on ZTF id.

    `sources` is an ordered list of (source_name, frame). Each frame carries at least
    `ztf_name`, `sn_subtype`, `redshift`. Earlier sources win for every column they
    fill; later sources only fill gaps (a missing redshift, a missing E(B-V)).

    Cross-source label conflicts are removed: if any source assigns the same ZTF id a
    different subtype, that id is dropped from every pool. That costs a few objects and
    removes the label noise that hurts the minority classes most.
    """
    label_sets = {}
    for _, fr in sources:
        for z, s in zip(fr['ztf_name'], fr['sn_subtype']):
            if isinstance(z, str) and isinstance(s, str):
                label_sets.setdefault(z, set()).add(s)
    conflicted = {z for z, s in label_sets.items() if len(s) > 1}

    merged = {}
    order = []
    for src, fr in sources:
        fr = fr[(fr['sn_subtype'] == subtype) & fr['ztf_name'].notna()]
        for rec in fr.to_dict('records'):
            z = rec['ztf_name']
            if z in conflicted:
                continue
            if z not in merged:
                rec = dict(rec)
                rec['source'] = src
                merged[z] = rec
                order.append(z)
            else:
                cur = merged[z]
                for k, v in rec.items():
                    if k == 'source':
                        continue
                    cv = cur.get(k)
                    if cv is None or (isinstance(cv, float) and not np.isfinite(cv)):
                        cur[k] = v
                cur['source'] = cur['source'] + '+' + src if src not in cur['source'] else cur['source']
    pool = pd.DataFrame([merged[z] for z in order])
    if not len(pool):
        return pool, {'conflicted_ids': len(conflicted)}
    pool['redshift'] = pd.to_numeric(pool['redshift'], errors='coerce')
    if require_redshift:
        pool = pool[pool['redshift'].notna() & (pool['redshift'] > 0)]
    pool = pool.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    stats = {'conflicted_ids': len(conflicted), 'pool': len(pool)}
    for src, _ in sources:
        stats[src] = int(pool['source'].str.startswith(src).sum())
    return pool, stats


def resolve_ztf_ids(frame, alerce, *, radius_arcsec=2.0, workers=8, survey_kwargs=None,
                    cache_path=None):
    """Fill `ztf_name` for rows that lack one, by ALeRCE cone search on (ra, declination).

    TNS records the ZTF internal name only when ZTF (or a broker) reported the object;
    many SNe classified from other surveys' discoveries are still in ZTF's footprint and
    have a ZTF light curve. A 2" match is the radius v4's `resolve_oid` used. Results
    (including misses) are cached so a re-run does not repeat the queries.
    """
    from concurrent.futures import ThreadPoolExecutor
    survey_kwargs = survey_kwargs or {}
    cache = {}
    if cache_path:
        try:
            c = pd.read_csv(cache_path)
            cache = dict(zip(c['name'], c['ztf_name']))
        except (OSError, ValueError, KeyError):
            cache = {}

    def one(row):
        name = row['name']
        if name in cache:
            return name, cache[name]
        try:
            o = alerce.query_objects(ra=float(row['ra']), dec=float(row['declination']),
                                     radius=radius_arcsec, format='pandas', **survey_kwargs)
            z = o['oid'].iloc[0] if len(o) and 'oid' in o else None
        except Exception:
            z = None
        return name, z

    need = frame[frame['ztf_name'].isna()]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        res = dict(ex.map(one, [r for _, r in need.iterrows()]))
    cache.update(res)
    if cache_path:
        pd.DataFrame({'name': list(cache), 'ztf_name': list(cache.values())}).to_csv(cache_path, index=False)
    out = frame.copy()
    miss = out['ztf_name'].isna()
    out.loc[miss, 'ztf_name'] = out.loc[miss, 'name'].map(res)
    out['ztf_from_cone'] = miss & out['ztf_name'].notna()
    return out
