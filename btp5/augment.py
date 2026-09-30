"""
Section 5 — noise-model augmentation for the small classes (SLSN, Ib, Ic, TDE).

Implements the procedure of Townsend et al. 2026 (NoiZTF, arXiv:2602.13036, Sects.
3.1-3.3) as the paper describes it — checked against the paper text, not paraphrased
from memory. Each point below is what the paper does, and where this pipeline departs
from it the departure is stated.

1. **Units.** The error model's constants (e_b=18.0, m=0.04, c=4.7, delta=-0.006) are
   fitted to fluxes "expressed in janskys and scaled to be consistent with the AB
   magnitude system" (their footnote 5), with flux bins at f<400, 800-1200, >1600.
   Those bins only make sense in microjanskys, i.e. AB zero point 23.9 (mag 18 ->
   ~230 uJy). The paper does not print the zero point, so this is an INFERENCE, but
   the alternatives are off by orders of magnitude: at v4's internal ZP of 27.5 the
   bias floor e_b would sit at mag ~24.4, far below anything ZTF detects. v4's ZP is
   untouched for its own photometry; conversion happens only inside this module.

2. **Redshift.** z_sim is drawn from [z_true, z_true + 0.1] with p(z) ~ z^z_scale
   (z_scale in {0, 2}; 2 follows a constant volumetric rate, dV ~ z^2 dz).

3. **Flux.** f_z = D*f_t + N(0, sigma_z^2), with D = D_L(z_true)^2 / D_L(z_sim)^2
   (their Eq. 3), and sigma_z the ORIGINAL error scaled by their Eq. 4:
       sigma_z/sigma_t = [sqrt(D f + e_b^2) + D f (eps+delta)]
                       / [sqrt(f + e_b^2)   + f (eps+delta)],   eps ~ Exp(m f + c).

4. **K-correction** as an additive magnitude offset: sncosmo templates where one
   exists, else the Hogg et al. (2002) form K = -2.5 log10[(1+z_sim)/(1+z_true)],
   which Townsend use for SLSNe. TDEs are outside Townsend's (SN-only) scope; the
   Hogg form is applied to them too and that is an extrapolation, flagged here.

5. **Observation dates are left exactly as observed.** Townsend do not apply time
   dilation ("without altering the cadence"; <~10% for dz<=0.1), and the brief
   forbids timestamp perturbation because 0.1-day jitter measurably degraded SN Ia
   precision. An earlier version of this module stretched the time axis by
   (1+z_sim)/(1+z_true) and anchored each band separately, which shifted the g-band
   and r-band epochs relative to one another — exactly the colour-evolution
   corruption the brief warns about. That is gone.

6. **Quality cuts.** Townsend require a peak detection with SNR>5 and >=5 detections
   with SNR>5. Departure: this pipeline trains on the ZTF *alert stream*, which only
   ever contains >=5-sigma detections, so points that fall below SNR 5 at the new
   redshift are dropped rather than kept as low-SNR forced photometry. The copy then
   also has to pass v4's own quality gate via process_ztf_object, so a copy pushed
   too faint to be a useful training object is rejected, as a real one would be.

7. **Split before augmentation.** Townsend split train/test *before* augmenting "to
   ensure that no copies of the same object appear in both sets". Here every
   synthetic row carries parent_id, and modeling.prepare_split drops any synthetic
   row whose parent landed in a test partition — the same guarantee.

**Deferred: full SNANA simulation** (Aleo et al. 2023 / YSE-DR1, ~60,000 objects per
class). Mature templates exist for Ia/II/Ibc but not obviously for SLSN or TDE —
exactly the classes that need help — so the cost is badly matched to the bottleneck.
"""

import os

import numpy as np
import pandas as pd

from .config import NOISE_MODEL

UJY_ZP = 23.9          # AB zero point for flux density in microjansky
LN10 = np.log(10.0)

# sncosmo source per class. Townsend's Table 7 (their exact template list) was not
# verified here; these are the standard sncosmo templates for each type, with SALT2
# fixed at their stated [x1, c] = [1.0, 0.2].
SNCOSMO_TEMPLATES = {
    'SN_Ia': ('salt2', {'x1': 1.0, 'c': 0.2}),
    'SN_Ib': ('nugent-sn1bc', {}),
    'SN_Ic': ('nugent-sn1bc', {}),
    'SN_II': ('nugent-sn2p', {}),
    # SLSN and TDE: no clean template -> Hogg fallback (Townsend do this for SLSN).
}
ZTF_BANDS = {1: 'ztfg', 2: 'ztfr'}


# --------------------------------------------------------------------------
# Units
# --------------------------------------------------------------------------

def mag_to_ujy(mag, sigma_mag):
    f = 10.0 ** (-0.4 * (np.asarray(mag, float) - UJY_ZP))
    return f, f * 0.4 * LN10 * np.asarray(sigma_mag, float)


def ujy_to_mag(f, fe):
    f = np.asarray(f, float)
    with np.errstate(divide='ignore', invalid='ignore'):
        mag = UJY_ZP - 2.5 * np.log10(f)
        smag = (2.5 / LN10) * np.asarray(fe, float) / f
    return mag, smag


# --------------------------------------------------------------------------
# The Townsend et al. model
# --------------------------------------------------------------------------

def sample_z_sim(z_true, rng, dz_max=0.1, z_scale=2):
    """Draw z_sim in [z_true, z_true + dz_max] with p(z) ~ z^z_scale (inverse CDF)."""
    a, b = float(z_true), float(z_true) + float(dz_max)
    k = float(z_scale) + 1.0
    u = rng.uniform()
    return float((a ** k + u * (b ** k - a ** k)) ** (1.0 / k))


def distance_factor(z_true, z_sim, cosmo):
    """Their Eq. 3: D = D_L(z_true)^2 / D_L(z_sim)^2."""
    dt = float(cosmo.luminosity_distance(max(z_true, 1e-4)).value)
    ds = float(cosmo.luminosity_distance(max(z_sim, 1e-4)).value)
    return (dt / ds) ** 2


def error_scale_factor(f_t, D, eps, e_b=None, delta=None):
    """
    Their Eq. 4. For very bright points (f >~ 2.8e4 uJy, mag <~ 12.8) the negative
    delta can drive the denominator through zero — far outside the flux range the
    model was fitted on (their highest bin is f>1600). There the pure Poisson+bias
    scaling of their Eq. 1 is used instead, so the factor never goes negative.
    """
    p = NOISE_MODEL
    e_b = p['e_b'] if e_b is None else e_b
    delta = p['delta'] if delta is None else delta
    f = np.clip(np.asarray(f_t, float), 0.0, None)
    num = np.sqrt(D * f + e_b ** 2) + D * f * (eps + delta)
    den = np.sqrt(f + e_b ** 2) + f * (eps + delta)
    poisson = np.sqrt(D * f + e_b ** 2) / np.sqrt(f + e_b ** 2)
    ok = (num > 0) & (den > 0)
    return np.where(ok, num / np.where(ok, den, 1.0), poisson)


def draw_epsilon(f_t, rng, m=None, c=None):
    """eps ~ Exp(lambda_f), lambda_f = m f + c is a RATE; numpy takes scale = 1/rate."""
    p = NOISE_MODEL
    m = p['m'] if m is None else m
    c = p['c'] if c is None else c
    lam = m * np.clip(np.asarray(f_t, float), 0.0, None) + c
    return rng.exponential(1.0 / lam)


# --------------------------------------------------------------------------
# K-correction
# --------------------------------------------------------------------------

_SNCOSMO_CACHE = {}


def sncosmo_available():
    try:
        import sncosmo  # noqa: F401
        return True
    except Exception:
        return False


def hogg_k(z_true, z_sim):
    """Hogg et al. (2002) constant form, used by Townsend for SLSNe."""
    return float(-2.5 * np.log10((1.0 + z_sim) / (1.0 + z_true)))


def k_correction(label, fid, z_true, z_sim, use_sncosmo=True):
    """
    Additive magnitude offset between z_true and z_sim, at peak. Returns (K, method).
    Template K is the difference of the model's observed peak magnitude at the two
    redshifts with its amplitude held fixed, so distance dimming — already in D — is
    not counted twice.
    """
    tmpl = SNCOSMO_TEMPLATES.get(label)
    band = ZTF_BANDS.get(int(fid))
    if use_sncosmo and tmpl and band and sncosmo_available():
        key = (label, band, round(z_true, 4), round(z_sim, 4))
        if key in _SNCOSMO_CACHE:
            return _SNCOSMO_CACHE[key], 'sncosmo:' + tmpl[0]
        try:
            import sncosmo
            model = sncosmo.Model(source=tmpl[0])
            if tmpl[1]:
                model.set(**tmpl[1])
            phase = model.source.peakphase('bessellb')
            mags = []
            for z in (z_true, z_sim):
                model.set(z=z, t0=0.0)
                mags.append(float(model.bandmag(band, 'ab', phase * (1.0 + z))))
            k = mags[1] - mags[0]
            if np.isfinite(k):
                _SNCOSMO_CACHE[key] = k
                return k, 'sncosmo:' + tmpl[0]
        except Exception:
            pass
    return hogg_k(z_true, z_sim), 'hogg2002'


# --------------------------------------------------------------------------
# One synthetic light curve
# --------------------------------------------------------------------------

def augment_detections(det, label, z_true, z_sim, rng, cosmo, snr_min=5.0,
                       subsampling_rate=1.0, use_sncosmo=True):
    """
    Move one object's alert detections from z_true to z_sim.

    `det` has v4's CSV columns (mjd, fid, magpsf, sigmapsf[, ra, dec]). Returns
    (new_det, meta) or (None, reason). mjd is returned untouched, row for row.
    """
    D = distance_factor(z_true, z_sim, cosmo)
    parts, ks = [], {}
    for fid, g in det.groupby('fid'):
        f, fe = mag_to_ujy(g['magpsf'].values, g['sigmapsf'].values)
        k, method = k_correction(label, fid, z_true, z_sim, use_sncosmo=use_sncosmo)
        ks[int(fid)] = (k, method)
        eps = draw_epsilon(f, rng)
        fe_z = fe * error_scale_factor(f, D, eps)
        f_z = D * f * 10 ** (-0.4 * k) + rng.normal(0.0, fe_z)
        out = g.copy()
        mag, smag = ujy_to_mag(f_z, fe_z)
        out['magpsf'], out['sigmapsf'] = mag, smag
        out['_snr'] = f_z / fe_z
        parts.append(out)
    new = pd.concat(parts).sort_values('mjd')

    # alert stream: only >=5-sigma detections exist
    new = new[np.isfinite(new['_snr']) & (new['_snr'] >= snr_min)]
    if subsampling_rate < 1.0 and new.groupby('fid').size().max() > 10:
        new = new[rng.uniform(size=len(new)) < subsampling_rate]
    if len(new) < 5:
        return None, 'fewer than 5 detections above SNR 5 at z_sim'
    new = new.drop(columns='_snr')
    return new, {'D': D, 'k': ks}


# --------------------------------------------------------------------------
# Many synthetic rows
# --------------------------------------------------------------------------

def generate_synthetic_rows(df, classes, copies_per_parent, process_fn, out_dir, rng,
                            cosmo=None, dz_max=0.1, z_scale=2, max_attempts=3,
                            subsampling_rate=1.0, use_sncosmo=True, verbose=True):
    """
    Synthetic training rows for `classes`, `copies_per_parent` per real parent.

    Each copy is written as its own CSV and pushed through `process_fn` — v4's
    process_ztf_object — so it faces exactly the fit and quality gate a real object
    does. Copies that fail are retried with a fresh z_sim up to `max_attempts`, then
    abandoned; the tally says why.
    """
    if cosmo is None:
        from astropy.cosmology import FlatLambdaCDM
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)   # v4's cosmology, so peak_abs_mag agrees
    os.makedirs(out_dir, exist_ok=True)

    real = df[~df.get('is_synthetic', pd.Series(False, index=df.index)).astype(bool)]
    parents = real[real['label'].isin(classes) & real.get('survey', 'ZTF').eq('ZTF')
                   & real['lc_path'].apply(lambda p: isinstance(p, str) and os.path.exists(p))
                   & np.isfinite(real['redshift'].astype(float)) & (real['redshift'] > 0)]

    carry = ['coarse_label', 'has_tns_label', 'ra', 'dec', 'has_host_match',
             'host_offset_norm']
    rows, tally = [], {}
    for _, p in parents.iterrows():
        det = pd.read_csv(p['lc_path'])
        for k in range(copies_per_parent):
            made = False
            for attempt in range(max_attempts):
                z_sim = sample_z_sim(float(p['redshift']), rng, dz_max, z_scale)
                new, meta = augment_detections(det, p['label'], float(p['redshift']), z_sim,
                                               rng, cosmo, subsampling_rate=subsampling_rate,
                                               use_sncosmo=use_sncosmo)
                if new is None:
                    tally[meta] = tally.get(meta, 0) + 1
                    continue
                sid = f"{p['id']}__aug{k}"
                path = os.path.join(out_dir, f'{sid}.csv')
                new.to_csv(path, index=False)
                row = process_fn(path, sid, p['label'], redshift=z_sim)
                if row is None:
                    tally['v4 fit / quality gate'] = tally.get('v4 fit / quality gate', 0) + 1
                    continue
                row = dict(row)
                for c in carry:
                    if c in p.index:
                        row[c] = p[c]
                # Offset in host-radii is distance-free; the arcsec offset shrinks with
                # angular-diameter distance as the same galaxy moves further away.
                if 'host_offset_arcsec' in p.index and np.isfinite(p['host_offset_arcsec']):
                    da_t = float(cosmo.angular_diameter_distance(p['redshift']).value)
                    da_s = float(cosmo.angular_diameter_distance(z_sim).value)
                    row['host_offset_arcsec'] = float(p['host_offset_arcsec']) * da_t / da_s
                else:
                    row['host_offset_arcsec'] = np.nan
                for c in ['bts_peakabs', 'bts_rise', 'bts_fade', 'bts_duration']:
                    row[c] = np.nan      # BTS measured the PARENT; nothing for the copy
                row.update({'from_bts': 0, 'is_synthetic': True, 'parent_id': p['id'],
                            'z_true': float(p['redshift']), 'z_sim': z_sim,
                            'k_method': sorted({m for _, m in meta['k'].values()})[0],
                            'lc_path': path})
                rows.append(row)
                made = True
                break
            if not made:
                tally['abandoned after retries'] = tally.get('abandoned after retries', 0) + 1

    out = pd.DataFrame(rows)
    if verbose:
        print(f'{len(parents)} eligible parents in {list(classes)}; '
              f'{len(out)} synthetic rows kept')
        if len(out):
            print(out.groupby('label').size().rename('synthetic rows').to_string())
            print('K-correction method used:', out['k_method'].value_counts().to_dict())
        if tally:
            print('rejections:', tally)
    return out


def plan_augmentation(counts, copies_per_parent, classes_to_augment, verbose=True):
    """What augmentation will attempt, per class, before anything is generated."""
    rows = [{'class': c, 'real': int(counts.get(c, 0)),
             'copies_per_parent': copies_per_parent,
             'max_synthetic': int(counts.get(c, 0)) * copies_per_parent}
            for c in classes_to_augment]
    plan = pd.DataFrame(rows)
    if verbose:
        print(plan.to_string(index=False))
        starved = plan[plan['real'] == 0]
        if len(starved):
            print(f"WARNING: no real objects to augment for {list(starved['class'])}")
        if copies_per_parent > 3:
            print('NOTE: >3 copies per parent. Copies of one parent are correlated; past '
                  '~3x this inflates apparent training size without independent information.')
    return plan


def augmented_rows_are_flagged(df, flag_col='is_synthetic'):
    """Synthetic rows must be identifiable and traceable to a parent."""
    if flag_col not in df.columns:
        return False, f'no {flag_col} column — synthetic rows are untraceable'
    n = int(df[flag_col].astype(bool).sum())
    if n and ('parent_id' not in df.columns
              or df.loc[df[flag_col].astype(bool), 'parent_id'].isna().any()):
        return False, 'synthetic rows without a parent_id cannot be kept out of test sets'
    return True, f'{n} synthetic / {len(df)} total rows, flagged by {flag_col} with parent_id'


def augmentation_consistency(syn, parents, verbose=True):
    """
    Does a synthetic copy's fitted absolute magnitude agree with its parent's?

    After redshift rescaling it should differ only by the K-correction. Reported
    rather than asserted, because two effects are real and worth seeing:

      * Truncation bias. Copies pushed near ZTF's alert depth keep only the points
        that scattered bright past SNR 5, so their fitted peak is biased bright.
        Measured on synthetic parents: median residual -0.12 mag at dz<=0.03,
        -0.30 mag at dz<=0.1. Faint REAL objects suffer the same bias, so this is
        realism, not error, but it means copies are not unbiased clones.
      * v4's peak_abs_mag comes from the Villar A parameter, which is degenerate
        with t0 and tau_rise. Under an identical noise realization its scatter was
        0.255 mag against 0.035 mag for the fitted curve's actual peak — about 7x
        noisier. That is a v4 property, preserved here, and flagged as the single
        cheapest improvement to v4's feature set.
    """
    if syn is None or len(syn) == 0:
        return None
    pm = parents.set_index('id')['peak_abs_mag']
    k = [hogg_k(zt, zs) for zt, zs in zip(syn['z_true'], syn['z_sim'])]
    resid = syn['peak_abs_mag'].values - syn['parent_id'].map(pm).values - np.array(k)
    out = {'n': int(len(syn)), 'median_residual_mag': float(np.nanmedian(resid)),
           'residual_spread_mag': float(np.nanstd(resid)),
           'median_dz': float(np.median(syn['z_sim'] - syn['z_true']))}
    if verbose:
        print(f"{out['n']} copies: fitted peak_abs_mag minus (parent + K) has median "
              f"{out['median_residual_mag']:+.3f} mag, spread {out['residual_spread_mag']:.3f} "
              f"(median dz {out['median_dz']:.3f}).")
        print('  Negative = copies fit brighter than their parent predicts: the '
              'truncation bias of near-limit detection. Spread is dominated by v4 '
              'deriving peak_abs_mag from the Villar A parameter.')
    return out
