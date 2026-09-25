"""
Constants for v5. v4's values are carried over unchanged; v5 additions are marked.
"""

# ---------------------------------------------------------------- v4, unchanged
N_PER_CLASS = 150          # AGN / stellar flares
N_TDE_TARGET = 90          # TNS lists ~153 TDEs total; ZTF BTS ~78 with ZTF ids.
                           # Asking for what the universe can supply, not 150.
N_HEADLINE_SN_SUBTYPE = 30
N_ACQUIRE_SN_SUBTYPE = 30  # raise to 150 for the recommended run
N_PER_SN_SUBTYPE = N_ACQUIRE_SN_SUBTYPE

QG = dict(
    min_det_total=15,
    min_det_best_band=8,
    min_peak_snr=8.0,
    max_chisq_dof=20.0,
    require_redshift=True,
    require_bracket=True,
    min_days_prepeak=5.0,
    min_days_postpeak=30.0,
    min_obs_span=60.0,
    require_two_bands=True,
)

POOL_OVERSAMPLE = 40
RANDOM_STATE = 42

SN_SUBTYPES = ['SN_Ia', 'SN_Ib', 'SN_Ic', 'SN_II', 'SLSN']
COARSE_CLASSES = ['AGN', 'SNe', 'TDE', 'stellar_flare']

LIGHTCURVE_COLS = ['log_A', 'beta', 'log_gamma', 'log_tau_rise', 'log_tau_fall', 'log_chisq',
                   'gr_log_A', 'gr_dt0', 'gr_log_tau_rise', 'gr_log_tau_fall', 'gr_dbeta',
                   'gr_log_gamma', 'has_color',
                   'duration_half', 'rise_half', 'fade_half', 'n_det', 'max_snr', 'obs_span']
PHYSICAL_COLS = ['redshift', 'peak_abs_mag']
CATALOGUE_COLS = ['bts_peakabs', 'bts_rise', 'bts_fade', 'bts_duration', 'from_bts']

# ---------------------------------------------------------------- v5 additions
# Section 4.1: host offset via astro-ghost. Same missing-data convention as
# color_g_r/has_color in v4 — NaN plus an explicit presence flag.
HOST_COLS = ['host_offset_arcsec', 'host_offset_norm', 'has_host_match']

FEATURE_COLS = LIGHTCURVE_COLS + PHYSICAL_COLS + CATALOGUE_COLS + HOST_COLS

# Section 7: three models in the headline comparison. HistGradientBoosting and
# Bagging(SVM) stay available for diagnostics only — v4 measured all five converging
# to within ~1 point at Stage 1, so the extra two add noise to the headline table.
HEADLINE_MODELS = ['Logistic Regression', 'Random Forest', 'Bagging Classifier']
DIAGNOSTIC_ONLY_MODELS = ['Bagging (SVM)', 'HistGradientBoosting']

# Section 3.2: ALeRCE top-1 confidence cut. 0.5 matches the threshold Superphot+
# (de Soto et al. 2024) used when building their own ALeRCE-labelled comparison set.
ALERCE_CONF_THRESHOLD = 0.5

# Section 3.2/3.1: which ALeRCE classifier supplies the labels.
#
# Verified live against query_classifiers() on 2026-09-25 — NOT hardcoded from
# memory, and re-verified at run time by alerce_labels.fetch_taxonomy():
#
#   lc_classifier (hierarchical_rf_1.1.0)      15 classes, NO TDE
#   lc_classifier_BHRF_forced_phot (2.1.0)     21 classes, INCLUDING TDE
#
# The brief's Section 3.1 wants ALeRCE's 2025 TDE output cross-checked, which only
# the BHRF family provides, so that is the primary. The classic lc_classifier is kept
# as a sensitivity check because it is the longer-established model.
ALERCE_PRIMARY_CLASSIFIER = 'lc_classifier_BHRF_forced_phot'
ALERCE_PRIMARY_VERSION = '2.1.0'
ALERCE_SECONDARY_CLASSIFIER = 'lc_classifier'

# Section 5: ZTF empirical noise model, Townsend et al. 2026 (arXiv:2602.13036),
# their published fitted constants.
NOISE_MODEL = dict(e_b=18.0, m=0.04, c=4.7, delta=-0.006)
AUGMENT_CLASSES = ['SLSN', 'SN_Ib', 'SN_Ic', 'TDE']
