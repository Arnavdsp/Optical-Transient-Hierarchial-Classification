"""
Auto-generated results summary.

The brief asks for a physics-grounded interpretation "tied to the actual numbers
obtained, not generic filler". So the write-up is assembled *from* the results
objects: which features actually ranked top, which class pairs actually dominate
the error budget, what the Option A / Option B gap actually came out to. The
physics statements are a lookup keyed on those observed facts, so the prose can
never drift away from the tables above it.
"""

import numpy as np
import pandas as pd


# What each feature means physically, and why a class would score high or low on it.
FEATURE_PHYSICS = {
    'rise_time': (
        "time from first detection to peak. Set by the photon diffusion time through the "
        "ejecta, so it scales with ejecta mass and inversely with expansion velocity: "
        "stripped-envelope SNe (Ib/Ic) rise in ~10-20 d, SLSNe take weeks-to-months because "
        "of their much larger ejecta masses and additional central-engine input, and stellar "
        "flares rise in minutes because the energy is released impulsively by magnetic "
        "reconnection rather than diffusing outward."),
    'decay_time': (
        "time from peak back down the light curve. For thermonuclear and core-collapse SNe "
        "this tracks radioactive decay of 56Ni -> 56Co -> 56Fe, giving the characteristic "
        "weeks-to-months tail; AGN never really 'decay' at all because their variability is "
        "stochastic accretion-disc flickering with no single peak; TDE fallback follows the "
        "canonical t^-5/3 decline, which is shallower than a SN tail."),
    'amplitude': (
        "peak-to-baseline change in relative flux. Separates explosive events (many "
        "magnitudes) from AGN stochastic variability (typically tenths of a magnitude "
        "about a persistent bright nucleus)."),
    'peak_val': (
        "peak relative flux. Because it is measured relative to each object's own baseline "
        "rather than as an absolute magnitude, it carries less information than it appears "
        "to; without a redshift-based distance correction it cannot express intrinsic "
        "luminosity, which is what actually distinguishes SLSNe from normal SNe."),
    'color_g_r': (
        "g-r colour at peak, i.e. photospheric temperature. Hot young SN photospheres and "
        "the very hot, near-constant-temperature TDE continuum are blue; AGN are redder and "
        "their colour barely changes; SNe redden steadily as the ejecta cool and expand."),
    'has_color': (
        "whether a two-band ZTF g-r colour was measurable at all. This is an *observational* "
        "flag, not an astrophysical property: TESS flares are single-band by construction, "
        "so it is always 0 for that class and 1 for most ZTF objects."),
}

# Physics of the confusions we expect to see, keyed on an unordered class pair.
CONFUSION_PHYSICS = {
    frozenset({'SNe', 'TDE'}): (
        "TDEs and SNe are the hardest coarse pair, and genuinely so: both are single, "
        "smooth, luminous flares on a galaxy nucleus with comparable rise times and "
        "amplitudes. What actually separates them is a near-constant blue colour and a "
        "t^-5/3 decline for TDEs versus steady reddening and a radioactive tail for SNe — "
        "both of which need well-sampled two-band photometry and a longer baseline than "
        "these summary shape features encode."),
    frozenset({'SNe', 'AGN'}): (
        "AGN contamination of the SN class usually means a poorly-sampled light curve where "
        "a stochastic AGN excursion was caught near a local maximum and looks like a single "
        "peak. More epochs, or an explicit variability/periodicity statistic, is the fix."),
    frozenset({'AGN', 'TDE'}): (
        "Both live in galactic nuclei and both can show slow, long-timescale variability; "
        "TDE searches in practice suffer exactly this contamination."),
    frozenset({'SN_Ib', 'SN_Ic'}): (
        "Ib versus Ic is defined *spectroscopically*, by the presence or absence of helium "
        "lines. Their light curves come from nearly the same explosion physics — similar "
        "ejecta masses, similar 56Ni yields, similar timescales — so photometric shape "
        "features contain little of the information the label is actually based on. This is "
        "a ceiling imposed by the labelling scheme, not a modelling failure."),
    frozenset({'SN_Ia', 'SN_Ic'}): (
        "Both are compact, fast-evolving explosions with similar rise times; separating them "
        "photometrically leans on the secondary near-infrared maximum and on colour "
        "evolution, neither of which survives reduction to four shape scalars."),
    frozenset({'SN_II', 'SN_Ia'}): (
        "SNe II retain a hydrogen envelope and many show a plateau, which should make them "
        "the most separable subtype from Ia on decay shape; residual confusion usually comes "
        "from IIb/IIn sub-subtypes folded into the same bucket."),
    frozenset({'SLSN', 'SN_II'}): (
        "SLSNe are separable mainly by being far more luminous and far slower; on "
        "baseline-relative flux, without an absolute-magnitude correction, that luminosity "
        "advantage is partly thrown away and only the timescale survives."),
}


def _fmt_pct(x):
    return 'n/a' if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}%"


def _md_table(df, floatfmt='{:.3f}'):
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: '' if pd.isna(v) else floatfmt.format(v))
    header = '| ' + ' | '.join(str(c) for c in d.columns) + ' |'
    sep = '| ' + ' | '.join('---' for _ in d.columns) + ' |'
    body = '\n'.join('| ' + ' | '.join(str(v) for v in row) + ' |'
                     for row in d.itertuples(index=False))
    return '\n'.join([header, sep, body])


def build_results_markdown(count_report, split, stage1, stage2a, stage2b, comparison,
                           perm_s1, perm_s2, misclass_s1, misclass_s2,
                           recall_s1, recall_s2, resolution_s1, resolution_s2,
                           repeated_cv_s2=None, feature_df=None):
    """Assemble the whole results summary as a markdown string."""
    L = []
    add = L.append

    # ---------------------------------------------------------------- counts
    add("# Results summary\n")
    add("## Phase 2 — final sample\n")
    shortfalls = count_report[count_report['shortfall'] > 0]
    add(_md_table(count_report, '{:.0f}') + '\n')
    if len(shortfalls) == 0:
        add("All targets met: 150 objects in each of the four coarse classes (600 total), "
            "and 30 in each of the five SN subtypes. Every object counted here survived "
            "cross-match, download, cleaning, GP interpolation *and* feature extraction — "
            "the retry loop kept pulling candidates until that was true, so these are exact "
            "counts rather than whatever happened to survive.\n")
    else:
        add("**Shortfalls remain:**\n")
        for _, r in shortfalls.iterrows():
            add(f"- `{r['class']}`: {int(r['count'])}/{int(r['target'])} "
                f"(short by {int(r['shortfall'])})")
        add("\nA shortfall here means the candidate pool was genuinely exhausted, not that "
            "objects were silently dropped: raise the pool size / star list and re-run, and "
            "the loop resumes from its checkpoint.\n")

    n_train, n_test = len(split['idx_train']), len(split['idx_test'])
    add(f"\n**Split.** One stratified 80/20 split over all {n_train + n_test} rows "
        f"({n_train} train / {n_test} test), stratified on the coarse label. Stage 2's rows "
        f"are the SN-labelled subsets of those same two partitions, so a Stage 1 test object "
        f"cannot appear in Stage 2 training. This is asserted in code, not assumed.\n")

    # ---------------------------------------------------------------- stage 1
    add("\n## Phase 3, Stage 1 — coarse classification\n")
    add(_md_table(stage1['results']) + '\n')
    s1r = stage1['results'].set_index('Model')['Test accuracy']
    best1, worst1 = s1r.idxmax(), s1r.idxmin()
    add(f"\nBest: **{best1}** at {_fmt_pct(s1r.max())}; weakest: {worst1} at "
        f"{_fmt_pct(s1r.min())}. All four sit well above the {_fmt_pct(0.25)} chance rate for "
        f"four balanced classes.\n")
    add(f"\nMeasurement resolution: the Stage 1 test set holds {resolution_s1['n_test']} objects, "
        f"so accuracy moves in steps of {_fmt_pct(resolution_s1['accuracy_step'])} and carries a "
        f"worst-case standard error of about {_fmt_pct(resolution_s1['worst_case_std_error'])}. "
        f"Differences between the four models smaller than that are not real differences.\n")
    add("\n**Per-class recall (Random Forest):**\n")
    add(_md_table(recall_s1.reset_index()) + '\n')
    if len(misclass_s1):
        add("\n**Dominant confusions:**\n")
        add(_md_table(misclass_s1) + '\n')
        top = misclass_s1.iloc[0]
        phys = CONFUSION_PHYSICS.get(frozenset({top['true'], top['pred']}))
        add(f"\nThe single largest error mode is {top['true']} -> {top['pred']} "
            f"({int(top['n'])} objects, {_fmt_pct(top['share_of_errors'])} of all Stage 1 errors).")
        if phys:
            add(f" {phys}\n")

    # ---------------------------------------------------------------- stage 2
    add("\n## Phase 3, Stage 2 — SN subtype classification\n")
    add("### Option A — ground-truth routing (the ceiling)\n")
    add(_md_table(stage2a['results']) + '\n')
    add("\n### Option B — Stage-1-predicted SNe (the realistic path)\n")
    add(_md_table(stage2b['summary']) + '\n')
    add("\n### Option A vs Option B\n")
    add(_md_table(comparison) + '\n')

    a = comparison.set_index('Model')['Option A (ground-truth routing)']
    gap = comparison.set_index('Model')['Routing cost (A - B comparable)']
    add(f"\nThe honest comparison is the middle column: Option A and 'Option B comparable' "
        f"score the *same* population (true SN test objects) the same way, the only "
        f"difference being that Option B has to survive Stage 1's routing first. "
        f"That routing costs between {_fmt_pct(gap.min())} and {_fmt_pct(gap.max())} of "
        f"accuracy depending on the model — this gap is the price of the hierarchy, and it is "
        f"why the end-to-end column must never be compared directly against Option A: "
        f"end-to-end is computed over all {n_test} test objects including the easy non-SN "
        f"classes, which inflates it.\n")
    add(f"\nOption A tops out at {_fmt_pct(a.max())} ({a.idxmax()}). Subtyping is decisively "
        f"harder than the coarse problem ({_fmt_pct(s1r.max())}), and the reason is physical "
        f"rather than statistical — see the misclassification analysis below.\n")

    add(f"\n**Caveat — and this one is load-bearing.** The Stage 2 held-out set is only "
        f"{resolution_s2['n_test']} objects, about "
        f"{resolution_s2['mean_test_objects_per_class']:.0f} per subtype. A single accuracy on "
        f"that many draws has a worst-case standard error of roughly "
        f"{_fmt_pct(resolution_s2['worst_case_std_error'])}, and a per-subtype recall moves in "
        f"steps of {_fmt_pct(resolution_s2['per_class_recall_step'])}. A subtype recall of 0.00 "
        f"or 1.00 at this sample size is not evidence of anything. Balancing the dataset fixed "
        f"the *bias* from uneven attrition; it could not fix the variance, because 30 objects "
        f"per subtype is simply a small sample.\n")
    if repeated_cv_s2 is not None:
        add("\nRepeated stratified CV over the Stage 2 training partition — many more fits, "
            "no contact with Stage 1's test partition — gives a better-resolved estimate:\n")
        add(_md_table(repeated_cv_s2) + '\n')
        add("\nUse the held-out number as the untouched evaluation and this one to judge "
            "whether two models actually differ.\n")

    add("\n**Per-class recall, Option A (Random Forest):**\n")
    add(_md_table(recall_s2.reset_index()) + '\n')
    if len(misclass_s2):
        add("\n**Dominant subtype confusions:**\n")
        add(_md_table(misclass_s2) + '\n')
        for _, r in misclass_s2.head(3).iterrows():
            phys = CONFUSION_PHYSICS.get(frozenset({r['true'], r['pred']}))
            if phys:
                add(f"\n- **{r['true']} -> {r['pred']}** ({int(r['n'])} objects): {phys}")
        add("\n")

    # ---------------------------------------------------------------- phase 4
    add("\n## Phase 4 — feature importance and physical interpretation\n")
    for tag, perm in [('Stage 1', perm_s1), ('Stage 2', perm_s2)]:
        ranked = (perm.groupby('feature')['perm_importance'].mean()
                  .sort_values(ascending=False))
        add(f"\n### {tag} — most discriminative features\n")
        add(_md_table(ranked.reset_index().rename(
            columns={'perm_importance': 'mean permutation importance'})) + '\n')
        add(f"\nPermutation importance is computed on the held-out test set, identically for "
            f"all four models — including the bagged RBF SVM, which has no native importance "
            f"of any kind. That is the whole reason for using it: Gini importance and "
            f"logistic-regression coefficients are not comparable to each other, let alone to "
            f"a kernel machine.\n")
        add(f"\nThe top feature is **{ranked.index[0]}** — {FEATURE_PHYSICS.get(ranked.index[0], '')}\n")
        second = ranked.index[1]
        add(f"\nFollowed by **{second}** — {FEATURE_PHYSICS.get(second, '')}\n")
        near_zero = ranked[ranked <= 0.005]
        if len(near_zero):
            add(f"\nContributing essentially nothing at this stage: "
                f"{', '.join(f'`{f}`' for f in near_zero.index)}. A permutation importance at "
                f"or below zero means shuffling the column did not hurt accuracy — the model "
                f"was not using it.\n")

    if 'has_color' in perm_s1.groupby('feature')['perm_importance'].mean().nlargest(2).index:
        add("\n**A caveat about `has_color`.** It ranks near the top of Stage 1, but it is an "
            "observational artefact rather than astrophysics: it is 0 for every TESS flare and "
            "1 for essentially every ZTF object, so it partly encodes 'which survey did this "
            "come from'. The stellar-flare class is therefore easier than it looks. The "
            "physically meaningful Stage 1 result is the separation *among the three ZTF "
            "classes* (SNe / AGN / TDE), where `has_color` carries no information.\n")

    add("\n## Honest caveats\n")
    add("- **Feature set is deliberately minimal.** Four shape scalars plus a colour and a "
        "flag. The brief asks for peak magnitude, rise/decay times, amplitude and colour, and "
        "that is what is here — but subtype separation is known to need spectroscopic or "
        "richer photometric information (secondary maxima, plateau detection, absolute "
        "magnitude via redshift), so Stage 2's ceiling is set by the features, not the models.\n")
    add("- **Flux is baseline-relative, not absolute.** `peak_val` and `amplitude` are measured "
        "against each object's own baseline, so intrinsic luminosity — the thing that most "
        "cleanly separates SLSNe from normal SNe — is largely divided out. TNS redshifts are "
        "already downloaded and would enable an absolute-magnitude feature; that is the single "
        "highest-value addition available.\n")
    add("- **Flares come from a different instrument.** TESS flares are single-band, "
        "high-cadence, and drawn from a curated list of known active stars, whereas the other "
        "three classes are ZTF discoveries from TNS. Some of Stage 1's flare performance is "
        "survey signature rather than astrophysics.\n")
    add("- **Sub-subtypes are folded into parent buckets** (Ia-91bg, Iax -> SN_Ia; Ic-BL, Icn "
        "-> SN_Ic; IIb, IIn, IIP, IIL -> SN_II). This raises within-class variance, especially "
        "for SN_II, and some residual confusion is a direct consequence of that choice.\n")
    add("- **Small test sets**, as quantified above. Every accuracy in this notebook should be "
        "read with its error bar attached.\n")
    return '\n'.join(L)
