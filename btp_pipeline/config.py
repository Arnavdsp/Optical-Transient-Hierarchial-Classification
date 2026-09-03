"""Shared constants. Kept in one place so the notebook and the tests agree."""

# AGN / TDE / Stellar Flares target this many objects each
N_PER_CLASS = 150
# Each SN subtype (Ia/Ib/Ic/II/SLSN) targets this many objects; 5 x 30 = 150 SNe,
# so the coarse Stage 1 problem is balanced 150/150/150/150.
N_PER_SN_SUBTYPE = 30

SN_SUBTYPE_LABELS = ['SN_Ia', 'SN_Ib', 'SN_Ic', 'SN_II', 'SLSN']
COARSE_LABELS = ['SNe', 'AGN', 'TDE', 'stellar_flare']

FEATURES = ['peak_val', 'rise_time', 'decay_time', 'amplitude', 'color_g_r']
X_COLS = FEATURES + ['has_color']

# Display names — used verbatim in every table and plot.
MODEL_NAMES = ['Random Forest', 'Logistic Regression', 'Bagging (Trees)', 'Bagging (SVM)']

RANDOM_STATE = 42
TEST_SIZE = 0.2
