"""
Section 4.1 — host-galaxy offset via astro-ghost (Gagliano et al. 2021).

Why this feature and not the other two candidates: ALeRCE's own 2025 TDE update added
exactly this feature type, because nuclear offset directly targets the
TDE-vs-AGN-vs-nuclear-transient boundary — which v4's diagnostics identified as Stage
1's weakest link. A TDE is nuclear by definition (offset ~0); a core-collapse SN is
displaced into its host's disc; an AGN sits at the nucleus but varies stochastically.

Missing data follows v4's existing convention exactly — NaN plus a presence flag,
the same pattern as color_g_r/has_color. No new missing-data convention is invented.
"""

import numpy as np
import pandas as pd

from .config import HOST_COLS


def ghost_available():
    """astro-ghost is an optional dependency; the pipeline degrades to NaN without it."""
    try:
        import astro_ghost  # noqa: F401
        return True
    except Exception:
        return False


def _empty_host_row():
    return {'host_offset_arcsec': np.nan, 'host_offset_norm': np.nan,
            'has_host_match': 0}


def host_offset_for(ra, dec, snName=None, resolver=None, verbose=False):
    """
    Offset between a transient and its associated host.

    `resolver(ra, dec, name) -> dict|None` is injectable so the control flow can be
    tested without network access; the default uses astro-ghost.

    Returns the three HOST_COLS. A failed association is NaN + has_host_match=0 —
    expected to happen regularly (hostless transients, crowded fields, catalogue
    gaps), not an error.
    """
    if resolver is None:
        resolver = _ghost_resolver
    try:
        res = resolver(ra, dec, snName)
    except Exception as e:
        if verbose:
            print(f'  host lookup failed for ({ra}, {dec}): {type(e).__name__}')
        return _empty_host_row()
    if not res:
        return _empty_host_row()

    offset = res.get('offset_arcsec')
    radius = res.get('host_radius_arcsec')
    if offset is None or not np.isfinite(offset):
        return _empty_host_row()

    row = {'host_offset_arcsec': float(offset),
           # Normalised by host size: 0.1 host-radii is nuclear whatever the distance,
           # while 2 arcsec means very different things for a nearby vs distant galaxy.
           'host_offset_norm': (float(offset) / float(radius)
                                if radius and np.isfinite(radius) and radius > 0
                                else np.nan),
           'has_host_match': 1}
    return row


def _ghost_resolver(ra, dec, snName=None):  # pragma: no cover - needs network
    """Default resolver: astro-ghost's transient-host association."""
    from astro_ghost.ghostHelperFunctions import getTransientHosts
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    coord = SkyCoord(ra=float(ra) * u.deg, dec=float(dec) * u.deg)
    hosts = getTransientHosts(snCoord=[coord], snName=[snName or 'transient'],
                              verbose=False, starcut='gentle', ascentMatch=False)
    if hosts is None or len(hosts) == 0:
        return None
    h = hosts.iloc[0]
    host_ra = h.get('raMean', np.nan)
    host_dec = h.get('decMean', np.nan)
    if not (np.isfinite(host_ra) and np.isfinite(host_dec)):
        return None
    hc = SkyCoord(ra=float(host_ra) * u.deg, dec=float(host_dec) * u.deg)
    offset = float(coord.separation(hc).arcsec)
    # Kron radius in the r band is astro-ghost's most reliably populated size column.
    radius = h.get('rKronRad', np.nan)
    return {'offset_arcsec': offset,
            'host_radius_arcsec': float(radius) if np.isfinite(radius) else np.nan}


def add_host_features(df, ra_col='ra', dec_col='dec', name_col='id',
                      resolver=None, progress=None, verbose=True):
    """
    Add the three HOST_COLS to a feature table.

    Objects without coordinates (our TESS flares) get NaN + has_host_match=0, which
    is correct rather than missing: a Galactic flare star has no host galaxy at all.
    """
    out = df.copy()
    if resolver is None and not ghost_available():
        if verbose:
            print('astro-ghost not installed — host columns filled with NaN and '
                  'has_host_match=0. Install it (pip install astro-ghost) to enable '
                  'this feature; the pipeline runs either way.')
        for c in HOST_COLS:
            out[c] = np.nan if c != 'has_host_match' else 0
        return out

    rows = []
    it = out.iterrows()
    if progress is not None:
        it = progress(it, total=len(out), desc='host offsets')
    for _, r in it:
        ra, dec = r.get(ra_col, np.nan), r.get(dec_col, np.nan)
        if not (np.isfinite(ra) and np.isfinite(dec)):
            rows.append(_empty_host_row())
            continue
        rows.append(host_offset_for(ra, dec, r.get(name_col), resolver=resolver))
    host = pd.DataFrame(rows, index=out.index)
    for c in HOST_COLS:
        out[c] = host[c]

    if verbose:
        n = int(out['has_host_match'].sum())
        print(f'host association succeeded for {n}/{len(out)} objects')
        if 'survey' in out.columns:
            tess = int(((out['survey'] == 'TESS') & (out['has_host_match'] == 0)).sum())
            if tess:
                print(f'  ({tess} of the misses are TESS flare stars, which are '
                      'Galactic and correctly have no host galaxy)')
    return out
