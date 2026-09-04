# Results summary

## Phase 2 — final sample

| level | class | count | target | shortfall |
| --- | --- | --- | --- | --- |
| coarse | AGN | 125 | 150 | 25 |
| coarse | SNe | 150 | 150 | 0 |
| coarse | TDE | 124 | 150 | 26 |
| coarse | stellar_flare | 150 | 150 | 0 |
| subtype | SN_Ia | 30 | 30 | 0 |
| subtype | SN_Ib | 30 | 30 | 0 |
| subtype | SN_Ic | 30 | 30 | 0 |
| subtype | SN_II | 30 | 30 | 0 |
| subtype | SLSN | 30 | 30 | 0 |

**Shortfalls remain:**

- `AGN`: 125/150 (short by 25)
- `TDE`: 124/150 (short by 26)

A shortfall here means the candidate pool was genuinely exhausted, not that objects were silently dropped: raise the pool size / star list and re-run, and the loop resumes from its checkpoint.


**Split.** One stratified 80/20 split over all 549 rows (439 train / 110 test), stratified on the coarse label. Stage 2's rows are the SN-labelled subsets of those same two partitions, so a Stage 1 test object cannot appear in Stage 2 training. This is asserted in code, not assumed.


## Phase 3, Stage 1 — coarse classification

| Model | CV folds | CV accuracy | CV std | Test accuracy |
| --- | --- | --- | --- | --- |
| Random Forest | 5 | 0.804 | 0.070 | 0.791 |
| Logistic Regression | 5 | 0.765 | 0.079 | 0.745 |
| Bagging (Trees) | 5 | 0.806 | 0.081 | 0.791 |
| Bagging (SVM) | 5 | 0.779 | 0.065 | 0.745 |


Best: **Random Forest** at 79.1%; weakest: Logistic Regression at 74.5%. All four sit well above the 25.0% chance rate for four balanced classes.


Measurement resolution: the Stage 1 test set holds 110 objects, so accuracy moves in steps of 0.9% and carries a worst-case standard error of about 4.8%. Differences between the four models smaller than that are not real differences.


**Per-class recall (Random Forest):**

| true | recall | n_test |
| --- | --- | --- |
| TDE | 0.600 | 25 |
| AGN | 0.720 | 25 |
| SNe | 0.833 | 30 |
| stellar_flare | 0.967 | 30 |


**Dominant confusions:**

| true | pred | n | share_of_errors |
| --- | --- | --- | --- |
| AGN | TDE | 5 | 0.217 |
| TDE | AGN | 5 | 0.217 |
| TDE | SNe | 3 | 0.130 |
| SNe | TDE | 3 | 0.130 |
| SNe | AGN | 2 | 0.087 |
| AGN | SNe | 2 | 0.087 |
| TDE | stellar_flare | 2 | 0.087 |
| stellar_flare | SNe | 1 | 0.043 |


The single largest error mode is AGN -> TDE (5 objects, 21.7% of all Stage 1 errors).
 Both live in galactic nuclei and both can show slow, long-timescale variability; TDE searches in practice suffer exactly this contamination.


## Phase 3, Stage 2 — SN subtype classification

### Option A — ground-truth routing (the ceiling)

| Model | CV folds | CV accuracy | CV std | Test accuracy |
| --- | --- | --- | --- | --- |
| Random Forest | 5 | 0.583 | 0.091 | 0.633 |
| Logistic Regression | 5 | 0.417 | 0.075 | 0.367 |
| Bagging (Trees) | 5 | 0.583 | 0.070 | 0.600 |
| Bagging (SVM) | 5 | 0.358 | 0.117 | 0.400 |


### Option B — Stage-1-predicted SNe (the realistic path)

| Model | End-to-end accuracy (all test objects) | Conditional accuracy (correctly-routed SNe) | Option-A-comparable accuracy (true SNe) | Non-SNe mis-routed into Stage 2 | True SNe lost before Stage 2 |
| --- | --- | --- | --- | --- | --- |
| Random Forest | 0.727 | 0.720 | 0.600 | 6 | 5 |
| Logistic Regression | 0.636 | 0.333 | 0.200 | 5 | 12 |
| Bagging (Trees) | 0.709 | 0.640 | 0.533 | 5 | 5 |
| Bagging (SVM) | 0.636 | 0.368 | 0.233 | 5 | 11 |


### Option A vs Option B

| Model | Option A (ground-truth routing) | Option B (comparable: true SNe, mis-route = wrong) | Option B (conditional: correctly-routed only) | Option B (end-to-end, all 8 classes) | Routing cost (A - B comparable) |
| --- | --- | --- | --- | --- | --- |
| Random Forest | 0.633 | 0.600 | 0.720 | 0.727 | 0.033 |
| Logistic Regression | 0.367 | 0.200 | 0.333 | 0.636 | 0.167 |
| Bagging (Trees) | 0.600 | 0.533 | 0.640 | 0.709 | 0.067 |
| Bagging (SVM) | 0.400 | 0.233 | 0.368 | 0.636 | 0.167 |


The honest comparison is the middle column: Option A and 'Option B comparable' score the *same* population (true SN test objects) the same way, the only difference being that Option B has to survive Stage 1's routing first. That routing costs between 3.3% and 16.7% of accuracy depending on the model — this gap is the price of the hierarchy, and it is why the end-to-end column must never be compared directly against Option A: end-to-end is computed over all 110 test objects including the easy non-SN classes, which inflates it.


Option A tops out at 63.3% (Random Forest). Subtyping is decisively harder than the coarse problem (79.1%), and the reason is physical rather than statistical — see the misclassification analysis below.


**Caveat — and this one is load-bearing.** The Stage 2 held-out set is only 30 objects, about 6 per subtype. A single accuracy on that many draws has a worst-case standard error of roughly 9.1%, and a per-subtype recall moves in steps of 16.7%. A subtype recall of 0.00 or 1.00 at this sample size is not evidence of anything. Balancing the dataset fixed the *bias* from uneven attrition; it could not fix the variance, because 30 objects per subtype is simply a small sample.


Repeated stratified CV over the Stage 2 training partition — many more fits, no contact with Stage 1's test partition — gives a better-resolved estimate:

| Model | Repeated-CV mean | Repeated-CV std | CI 2.5% | CI 97.5% | n_fits |
| --- | --- | --- | --- | --- | --- |
| Random Forest | 0.607 | 0.091 | 0.458 | 0.782 | 50 |
| Logistic Regression | 0.425 | 0.089 | 0.259 | 0.583 | 50 |
| Bagging (Trees) | 0.596 | 0.098 | 0.458 | 0.782 | 50 |
| Bagging (SVM) | 0.393 | 0.100 | 0.218 | 0.671 | 50 |


Use the held-out number as the untouched evaluation and this one to judge whether two models actually differ.


**Per-class recall, Option A (Random Forest):**

| true | recall | n_test |
| --- | --- | --- |
| SN_II | 0.333 | 6 |
| SN_Ib | 0.429 | 7 |
| SN_Ia | 0.714 | 7 |
| SN_Ic | 0.833 | 6 |
| SLSN | 1.000 | 4 |


**Dominant subtype confusions:**

| true | pred | n | share_of_errors |
| --- | --- | --- | --- |
| SN_Ib | SN_II | 3 | 0.273 |
| SN_II | SLSN | 2 | 0.182 |
| SN_II | SN_Ia | 2 | 0.182 |
| SN_Ia | SN_II | 2 | 0.182 |
| SN_Ib | SN_Ic | 1 | 0.091 |
| SN_Ic | SN_Ia | 1 | 0.091 |


- **SN_II -> SLSN** (2 objects): SLSNe are separable mainly by being far more luminous and far slower; on baseline-relative flux, without an absolute-magnitude correction, that luminosity advantage is partly thrown away and only the timescale survives.

- **SN_II -> SN_Ia** (2 objects): SNe II retain a hydrogen envelope and many show a plateau, which should make them the most separable subtype from Ia on decay shape; residual confusion usually comes from IIb/IIn sub-subtypes folded into the same bucket.



## Phase 4 — feature importance and physical interpretation


### Stage 1 — most discriminative features

| feature | mean permutation importance |
| --- | --- |
| color_g_r | 0.106 |
| rise_time | 0.101 |
| has_color | 0.090 |
| decay_time | 0.077 |
| peak_val | 0.057 |
| amplitude | 0.038 |


Permutation importance is computed on the held-out test set, identically for all four models — including the bagged RBF SVM, which has no native importance of any kind. That is the whole reason for using it: Gini importance and logistic-regression coefficients are not comparable to each other, let alone to a kernel machine.


The top feature is **color_g_r** — g-r colour at peak, i.e. photospheric temperature. Hot young SN photospheres and the very hot, near-constant-temperature TDE continuum are blue; AGN are redder and their colour barely changes; SNe redden steadily as the ejecta cool and expand.


Followed by **rise_time** — time from first detection to peak. Set by the photon diffusion time through the ejecta, so it scales with ejecta mass and inversely with expansion velocity: stripped-envelope SNe (Ib/Ic) rise in ~10-20 d, SLSNe take weeks-to-months because of their much larger ejecta masses and additional central-engine input, and stellar flares rise in minutes because the energy is released impulsively by magnetic reconnection rather than diffusing outward.


### Stage 2 — most discriminative features

| feature | mean permutation importance |
| --- | --- |
| color_g_r | 0.188 |
| rise_time | 0.083 |
| decay_time | 0.031 |
| peak_val | 0.029 |
| amplitude | 0.020 |
| has_color | -0.001 |


Permutation importance is computed on the held-out test set, identically for all four models — including the bagged RBF SVM, which has no native importance of any kind. That is the whole reason for using it: Gini importance and logistic-regression coefficients are not comparable to each other, let alone to a kernel machine.


The top feature is **color_g_r** — g-r colour at peak, i.e. photospheric temperature. Hot young SN photospheres and the very hot, near-constant-temperature TDE continuum are blue; AGN are redder and their colour barely changes; SNe redden steadily as the ejecta cool and expand.


Followed by **rise_time** — time from first detection to peak. Set by the photon diffusion time through the ejecta, so it scales with ejecta mass and inversely with expansion velocity: stripped-envelope SNe (Ib/Ic) rise in ~10-20 d, SLSNe take weeks-to-months because of their much larger ejecta masses and additional central-engine input, and stellar flares rise in minutes because the energy is released impulsively by magnetic reconnection rather than diffusing outward.


Contributing essentially nothing at this stage: `has_color`. A permutation importance at or below zero means shuffling the column did not hurt accuracy — the model was not using it.


## Feature-construction issues found in the real data

**Two features are one feature.** Measured on the real table:

- On ZTF objects (n=399): `peak_val = amplitude + 1`, Pearson r = 1.000000.

This is structural, not a coincidence: `process_ztf_object` uses the faintest magnitude as the flux baseline, so relative flux bottoms out at exactly 1.0 and `amplitude = peak - min` reduces to `peak_val - 1`. The ZTF half of the dataset therefore carries five independent features, not six. Nothing here is *wrong* — the models are not harmed by a duplicated column — but any statement of the form 'peak brightness and amplitude both matter' is counting one quantity twice, and their permutation importances should be read as a single shared contribution.

**`rise_time` and `decay_time` partly measure the observing baseline, not the transient.**

- Median interpolated span: 60 days; longest: 2908 days.
- 25.1% of objects span more than a year, 17.9% more than three years.
- Median span by class: AGN 1754 d, SLSN 116 d, TDE 80 d, SN_II 70 d, SN_Ic 64 d, SN_Ia 46 d, SN_Ib 34 d, stellar_flare 26 d

A supernova rises in roughly 10-30 days, so a span of years is archival coverage of the position, not the event. `extract_shape_features` measures across the whole interpolated curve, so for objects with long ZTF histories these columns encode how long the field has been monitored. That correlates with class through observing strategy rather than physics: persistently variable AGN accumulate the longest histories, while TESS flares are capped at the ~27-day sector length. Some of Stage 1's separability on these features is therefore survey signature.

**The fix**, if you want these features to mean what they are named: window the light curve around the detected peak (say peak minus 50 days to peak plus 150 days) before fitting the GP, and measure rise and decay inside that window. That is a change to `extract_shape_features` / `process_ztf_object`, which this rebuild deliberately left untouched, so it is flagged rather than applied.


## Honest caveats

- **Feature set is deliberately minimal.** Four shape scalars plus a colour and a flag. The brief asks for peak magnitude, rise/decay times, amplitude and colour, and that is what is here — but subtype separation is known to need spectroscopic or richer photometric information (secondary maxima, plateau detection, absolute magnitude via redshift), so Stage 2's ceiling is set by the features, not the models.

- **Flux is baseline-relative, not absolute.** `peak_val` and `amplitude` are measured against each object's own baseline, so intrinsic luminosity — the thing that most cleanly separates SLSNe from normal SNe — is largely divided out. TNS redshifts are already downloaded and would enable an absolute-magnitude feature; that is the single highest-value addition available.

- **Flares come from a different instrument.** TESS flares are single-band, high-cadence, and drawn from a curated list of known active stars, whereas the other three classes are ZTF discoveries from TNS. Some of Stage 1's flare performance is survey signature rather than astrophysics.

- **Sub-subtypes are folded into parent buckets** (Ia-91bg, Iax -> SN_Ia; Ic-BL, Icn -> SN_Ic; IIb, IIn, IIP, IIL -> SN_II). This raises within-class variance, especially for SN_II, and some residual confusion is a direct consequence of that choice.

- **Small test sets**, as quantified above. Every accuracy in this notebook should be read with its error bar attached.
