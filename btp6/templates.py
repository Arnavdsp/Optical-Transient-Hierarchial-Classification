"""
Template-matching features: how well does each SN subclass's real template set fit?

The classic photometric-typing idea (PSNID, Sako et al. 2011; Poznanski et al. 2007),
used here as features rather than as the classifier. For every object, every
template is fitted to the g and r detections at the object's known redshift, and the
best reduced chi^2 per subclass becomes a feature.

Templates:
  * Core collapse: Vincenzi et al. (2019, MNRAS 489, 5802; arXiv:1908.05228), 66
    spectrophotometric templates of real, well-observed SNe (II 23, IIn 6, IIb 12,
    Ib 12, Ic 7, Ic-BL 6), built largely from the CfA and Carnegie Supernova Project
    photometry and spectra. The host-extinction-corrected (`-corr`) versions are used,
    with host reddening put back as a free parameter (below).
  * SN Ia: SALT2 (Guy et al. 2007) on a small (x1, c) grid.
  * SLSN: no template set is used; their luminosity and timescale features already
    separate them, and the 3-way/4-way confusion is not there.

Why this helps where the light-curve features plateau: the remaining 4-way errors are
SN II vs stripped-envelope. A single Villar or GP fit describes shape; a template fit
asks whether the joint g and r shape and colour evolution of THIS object matches the
detailed multi-band evolution of real SNe II or of real Ib/c, with the colour locked
by one shared amplitude across both bands.

Fit per (object, template, host E(B-V)):
  * time of template phase zero: grid search, 1 observer-frame day steps;
  * amplitude: analytic weighted least squares, one value shared by g and r, > 0;
  * host E(B-V) in {0, 0.15, 0.3, 0.5}, CCM89 with R_V = 3.1 in the rest frame;
  * error model: photometric error plus a 10 % flux floor for template mismatch;
  * points outside the template's phase range are excluded, and a fit must model
    at least 60 % of the points and both bands to count.

Template light curves are precomputed once per (template, extinction, redshift bin of
0.005), so the per-object cost is a vectorised interpolation.

Leakage: none of the templates is an object in this sample (they are 1987-2016 SNe,
before the ZTF alerts this sample comes from).
"""

import numpy as np

try:
    from btp5.photometry import ZP, mag_to_flux
except ImportError:                     # pragma: no cover - notebook path
    pass

TEMPLATE_CLASSES = ['Ia', 'II', 'IIn', 'IIb', 'Ib', 'Ic', 'IcBL']
SESN_CLASSES = ['IIb', 'Ib', 'Ic', 'IcBL']
HOST_EBV = (0.0, 0.15, 0.3, 0.5)
Z_BIN = 0.005
SALT2_GRID = [(-2.0, -0.1), (0.0, 0.0), (2.0, 0.0), (0.0, 0.2), (-1.0, 0.1)]
FLUX_FLOOR = 0.10
MIN_FRACTION_MODELLED = 0.6

TEMPLATE_FEATURES = (
    [f'tmpl_lchi2_{c}' for c in TEMPLATE_CLASSES] +
    ['tmpl_d_II_SESN', 'tmpl_d_Ia_CC', 'tmpl_d_IIn_II', 'tmpl_d_Ib_Ic',
     'tmpl_best_ebv_II', 'tmpl_best_ebv_SESN']
)

_V19_TYPE = {'SN II': 'II', 'SN IIn': 'IIn', 'SN IIb': 'IIb', 'SN Ib': 'Ib',
             'SN Ic': 'Ic', 'SN Ic-BL': 'IcBL'}


def template_catalogue():
    """[(name, version, class, params)] for every template used."""
    import sncosmo
    reg = sncosmo.registry._get_registry(sncosmo.Source)
    out = []
    for m in reg.get_loaders_metadata():
        if m['name'].startswith('v19') and m['name'].endswith('-corr') and m.get('type') in _V19_TYPE:
            out.append((m['name'], m['version'], _V19_TYPE[m['type']], {}))
    for x1, c in SALT2_GRID:
        out.append(('salt2', None, 'Ia', {'x1': x1, 'c': c}))
    return out


class TemplateBank:
    """Caches template light curves in ZTF g and r per (template, E(B-V), z bin)."""

    def __init__(self, catalogue=None, phase_step=1.0):
        import sncosmo
        self.sncosmo = sncosmo
        self.catalogue = catalogue or template_catalogue()
        self.phase_step = phase_step
        self._sources = {}
        self._cache = {}

    def _model(self, name, version, params, ebv):
        import sncosmo
        key = (name, version)
        if key not in self._sources:
            self._sources[key] = sncosmo.get_source(name, version=version) if version else sncosmo.get_source(name)
        m = sncosmo.Model(source=self._sources[key], effects=[sncosmo.CCM89Dust()],
                          effect_names=['host'], effect_frames=['rest'])
        m.set(hostebv=ebv, hostr_v=3.1, **params)
        return m

    def curves(self, idx, ebv, z):
        """(observer-frame phase grid, g flux, r flux) at the bin centre, or None."""
        zb = round(max(z, Z_BIN) / Z_BIN) * Z_BIN
        key = (idx, ebv, zb)
        if key in self._cache:
            return self._cache[key]
        name, version, _, params = self.catalogue[idx]
        try:
            m = self._model(name, version, params, ebv)
            m.set(z=zb, t0=0.0)
            if name == 'salt2':
                m.set(x0=1.0)
            ph = np.arange(m.mintime(), m.maxtime(), self.phase_step * (1 + zb))
            g = m.bandflux('ztfg', ph, zp=ZP, zpsys='ab')
            r = m.bandflux('ztfr', ph, zp=ZP, zpsys='ab')
            res = (ph, np.asarray(g, float), np.asarray(r, float))
            if not (np.isfinite(res[1]).all() and np.isfinite(res[2]).all()) or res[2].max() <= 0:
                res = None
        except Exception:
            res = None
        self._cache[key] = res
        return res


def _fit_one(t, f, s2, band_is_g, curve, min_frac=MIN_FRACTION_MODELLED):
    """Best reduced chi^2 over a grid of phase-zero times. Vectorised over the grid."""
    ph, mg, mr = curve
    span_lo, span_hi = ph[0], ph[-1]
    t0s = np.arange(t.min() - span_hi, t.max() - span_lo, 1.0)
    if not len(t0s):
        return np.inf
    P = t[None, :] - t0s[:, None]                      # (n_t0, n_obs)
    inside = (P >= span_lo) & (P <= span_hi)
    M = np.where(band_is_g[None, :], np.interp(P, ph, mg), np.interp(P, ph, mr))
    M = np.where(inside, M, 0.0)
    W = np.where(inside, 1.0 / s2[None, :], 0.0)
    num = (W * M * f[None, :]).sum(1)
    den = (W * M * M).sum(1)
    A = np.where(den > 0, num / np.where(den > 0, den, 1), 0.0)
    A = np.clip(A, 0, None)
    # the 10 % floor is relative to the model, so a template cannot buy a good chi^2
    # by inflating its own error bars on points it does not reach
    var = s2[None, :] + (FLUX_FLOOR * A[:, None] * M) ** 2
    chi = np.where(inside, (f[None, :] - A[:, None] * M) ** 2 / var, 0.0).sum(1)
    n_in = inside.sum(1)
    both = (inside & band_is_g[None, :]).any(1) & (inside & ~band_is_g[None, :]).any(1)
    ok = (n_in >= min_frac * len(t)) & both & (A > 0)
    if not ok.any():
        return np.inf
    red = chi / np.maximum(n_in - 2, 1)
    return float(red[ok].min())


def template_features(df, redshift, bank, ebv_mw=0.0):
    """TEMPLATE_FEATURES for one object; NaN when no template could be fitted."""
    out = {k: np.nan for k in TEMPLATE_FEATURES}
    if redshift is None or not np.isfinite(redshift) or redshift <= 0:
        return out
    d = df[df['fid'].isin([1, 2]) & np.isfinite(df['magpsf'])]
    if len(d) < 6 or d['fid'].nunique() < 2:
        return out
    sig = d['sigmapsf'].values if 'sigmapsf' in d else np.full(len(d), 0.1)
    # undo Milky Way extinction on the data, so templates only need host dust
    a = np.where(d['fid'].values == 1, 3.172, 2.271) * (ebv_mw if np.isfinite(ebv_mw) else 0.0)
    f, fe = mag_to_flux(d['magpsf'].values - a, sig)
    t = d['mjd'].values.astype(float)
    g = d['fid'].values == 1
    s2 = fe ** 2
    best = {c: (np.inf, np.nan) for c in TEMPLATE_CLASSES}
    for i, (_, _, cls, _) in enumerate(bank.catalogue):
        for ebv in HOST_EBV:
            if cls == 'Ia' and ebv > 0:
                continue                       # SALT2's own colour term covers it
            cur = bank.curves(i, ebv, float(redshift))
            if cur is None:
                continue
            c2 = _fit_one(t, f, s2, g, cur)
            if c2 < best[cls][0]:
                best[cls] = (c2, ebv)
    lc = {c: (np.log10(v[0]) if np.isfinite(v[0]) else np.nan) for c, v in best.items()}
    for c in TEMPLATE_CLASSES:
        out[f'tmpl_lchi2_{c}'] = lc[c]
    sesn = np.nanmin([lc[c] for c in SESN_CLASSES]) if any(np.isfinite(lc[c]) for c in SESN_CLASSES) else np.nan
    cc = np.nanmin([lc[c] for c in ['II', 'IIn'] + SESN_CLASSES]) if any(
        np.isfinite(lc[c]) for c in ['II', 'IIn'] + SESN_CLASSES) else np.nan
    out['tmpl_d_II_SESN'] = lc['II'] - sesn
    out['tmpl_d_Ia_CC'] = lc['Ia'] - cc
    out['tmpl_d_IIn_II'] = lc['IIn'] - lc['II']
    out['tmpl_d_Ib_Ic'] = lc['Ib'] - min(lc['Ic'], lc['IcBL']) if np.isfinite(lc['Ib']) else np.nan
    out['tmpl_best_ebv_II'] = best['II'][1]
    sesn_best = min(SESN_CLASSES, key=lambda c: best[c][0])
    out['tmpl_best_ebv_SESN'] = best[sesn_best][1]
    return out
