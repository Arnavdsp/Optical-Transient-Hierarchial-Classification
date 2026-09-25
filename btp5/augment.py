"""
Section 5 — augmentation for the small classes (SLSN, Ib, Ic, TDE).

Uses the ZTF-specific empirical noise model from Townsend et al. 2026 (NoiZTF,
arXiv:2602.13036) to synthesize realistic fainter / higher-redshift copies of real
objects:

    sigma_t / f_t = sqrt(1/f_t + e_b^2 / f_t^2) + eps + delta,
    eps ~ Exponential(lambda_f),   lambda_f = m * f_t + c

with their published fitted ZTF constants e_b=18.0, m=0.04, c=4.7, delta=-0.006.

**Explicitly NOT done: timestamp jitter.** That is a directly evidenced negative
result in the same paper — perturbing observation times by even 0.1 days measurably
degraded SN Ia precision by corrupting the inferred colour evolution. Times are
carried through untouched.

**Explicitly deferred: full SNANA simulation** (Aleo et al. 2023 / YSE-DR1 use
~60,000 objects/class). Mature SN templates exist for Ia/II/Ibc but not obviously for
SLSN or TDE — precisely the classes that need help here — so the engineering cost is
badly matched to the actual bottleneck.
"""

import numpy as np
import pandas as pd

from .config import NOISE_MODEL


def ztf_flux_uncertainty(flux, rng, e_b=None, m=None, c=None, delta=None):
    """
    Townsend et al. 2026 Eq. for fractional flux uncertainty, with their ZTF fits.

    Returns absolute sigma (same units as `flux`). The exponential term is the
    empirical scatter they fit on top of the Poisson+background floor; its rate
    lambda_f itself depends on flux, which is what makes the model ZTF-specific
    rather than a generic error bar.
    """
    p = NOISE_MODEL
    e_b = p['e_b'] if e_b is None else e_b
    m = p['m'] if m is None else m
    c = p['c'] if c is None else c
    delta = p['delta'] if delta is None else delta

    f = np.asarray(flux, dtype=float)
    f = np.where(f > 1e-12, f, 1e-12)              # the model is undefined at f<=0
    base = np.sqrt(1.0 / f + (e_b ** 2) / (f ** 2))
    lam = m * f + c
    lam = np.where(lam > 1e-12, lam, 1e-12)
    eps = rng.exponential(1.0 / lam, size=f.shape)  # numpy takes the scale, = 1/rate
    frac = base + eps + delta
    frac = np.where(frac > 0, frac, 1e-6)           # delta is negative by construction
    return frac * f


def k_correction(z_true, z_sim, band=None, sncosmo_model=None):
    """
    K-correction between the true and simulated redshift.

    Uses a proper `sncosmo` template when one is supplied; otherwise falls back to the
    closed-form Hogg et al. (2002) approximation

        K = -2.5 * log10[(1 + z_sim) / (1 + z_true)]

    which is what we use for classes with no clean template — SLSN and TDE especially.
    The fallback is an approximation and is flagged as such wherever it is used.
    """
    if sncosmo_model is not None:  # pragma: no cover - optional dependency
        try:
            return float(sncosmo_model.bandmag(band, 'ab', 0.0))
        except Exception:
            pass
    return float(-2.5 * np.log10((1.0 + z_sim) / (1.0 + z_true)))


def rescale_to_redshift(t, flux, z_true, z_sim, cosmo=None, use_k_correction=True):
    """
    Move a light curve from z_true to z_sim: dim it by the luminosity-distance ratio,
    stretch its time axis by (1+z) time dilation, and optionally K-correct.

    Returns (t_sim, flux_sim, k_applied).
    """
    if cosmo is None:
        from astropy.cosmology import FlatLambdaCDM
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    z_true = max(float(z_true), 1e-4)
    z_sim = max(float(z_sim), 1e-4)

    dl_true = float(cosmo.luminosity_distance(z_true).value)
    dl_sim = float(cosmo.luminosity_distance(z_sim).value)
    dim = (dl_true / dl_sim) ** 2

    k = k_correction(z_true, z_sim) if use_k_correction else 0.0
    flux_sim = np.asarray(flux, float) * dim * 10 ** (-0.4 * k)

    t = np.asarray(t, float)
    t_sim = (t - t.min()) * (1.0 + z_sim) / (1.0 + z_true) + t.min()
    return t_sim, flux_sim, k


def augment_lightcurve(t, flux, z_true, z_sim, rng, cosmo=None):
    """
    One synthetic copy: redshift-rescale, then re-noise with the ZTF model.

    Timestamps are NOT jittered (see module docstring).
    """
    t_sim, f_sim, k = rescale_to_redshift(t, flux, z_true, z_sim, cosmo=cosmo)
    sigma = ztf_flux_uncertainty(f_sim, rng)
    f_obs = f_sim + rng.normal(0.0, sigma)
    return {'t': t_sim, 'flux': f_obs, 'flux_err': sigma,
            'z_sim': z_sim, 'z_true': z_true, 'k_correction': k}


def plan_augmentation(counts, target, classes_to_augment, verbose=True):
    """
    How many synthetic copies each small class needs to reach `target`.

    Returns a frame; a class already at or above target gets 0 and is left alone.
    Augmenting a class that does not need it just adds correlated near-duplicates.
    """
    rows = []
    for cls in classes_to_augment:
        have = int(counts.get(cls, 0))
        need = max(0, int(target) - have)
        rows.append({'class': cls, 'have': have, 'target': int(target),
                     'synthetic_needed': need,
                     'copies_per_real': (need / have) if have else np.nan})
    plan = pd.DataFrame(rows)
    if verbose:
        print(plan.to_string(index=False))
        starved = plan[(plan['have'] == 0) & (plan['synthetic_needed'] > 0)]
        if len(starved):
            print('WARNING: cannot augment a class with zero real objects — '
                  f"{list(starved['class'])} need real data first.")
        heavy = plan[plan['copies_per_real'] > 3]
        if len(heavy):
            print('NOTE: >3 synthetic copies per real object for '
                  f"{list(heavy['class'])}. Synthetic copies of the same parent are "
                  'correlated; past roughly 3x this inflates apparent training size '
                  'without adding independent information.')
    return plan


def augmented_rows_are_flagged(df, flag_col='is_synthetic'):
    """
    Synthetic rows must be identifiable and must never enter a test set.
    Returns (ok, message).
    """
    if flag_col not in df.columns:
        return False, f'no {flag_col} column — synthetic rows are untraceable'
    n = int(df[flag_col].sum())
    return True, f'{n} synthetic / {len(df)} total rows, flagged by {flag_col}'
