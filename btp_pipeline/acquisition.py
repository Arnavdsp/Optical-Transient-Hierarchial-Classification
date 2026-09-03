"""
Phase 2 — balanced acquisition.

Generalises the `build_subtype_to_target` pattern already proven on the 5 SN
subtypes to every class in the dataset:

  * `build_tns_class_to_target`  — TNS + ALeRCE classes (5 SN subtypes, AGN, TDE)
  * `build_flares_to_target`     — TESS/lightkurve stellar flares

The shared idea in both: a candidate that fails cross-match, download, cleaning
or GP fitting is *not* absorbed as a loss. We move on to the next candidate and
keep pulling until `target` objects have actually survived feature extraction.
That is what makes the final per-class counts exact instead of "whatever
happened to make it through", which is what previously gutted the rare classes.

Every external-API call site is injected as a callable so the control flow can
be exercised against synthetic fakes without touching TNS/ALeRCE/MAST.
"""

import gc
import os
import time

import pandas as pd


def _load_checkpoint(manifest_path, features_path, failed_path):
    """Resume from a partial pull, if one is on disk."""
    manifest, features, failed = [], [], set()
    if os.path.exists(manifest_path) and os.path.exists(features_path):
        manifest = pd.read_csv(manifest_path).to_dict('records')
        features = pd.read_csv(features_path).to_dict('records')
    if os.path.exists(failed_path):
        failed = set(pd.read_csv(failed_path)['candidate'].astype(str))
    return manifest, features, failed


def _save_checkpoint(manifest, features, failed, manifest_path, features_path, failed_path):
    pd.DataFrame(manifest).to_csv(manifest_path, index=False)
    pd.DataFrame(features).to_csv(features_path, index=False)
    pd.DataFrame({'candidate': sorted(failed)}).to_csv(failed_path, index=False)


def build_tns_class_to_target(
    label,
    pool,
    target,
    *,
    base_dir,
    feature_dir,
    resolve_oid_fn,
    query_detections_fn,
    process_fn,
    out_subdir=None,
    min_detections=5,
    sleep=0.15,
    checkpoint_every=10,
    progress=None,
    verbose=True,
):
    """
    Pull candidates from `pool` until exactly `target` of them survive all the
    way through feature extraction.

    Parameters
    ----------
    label : str
        Class label written into the manifest and feature rows (e.g. 'SN_Ia', 'AGN').
    pool : pandas.DataFrame
        Candidate TNS rows, already filtered to this class and already shuffled.
        Must contain at least 'name'; 'ra'/'declination'/'internal_names' are used
        by `resolve_oid_fn`.
    target : int
        Number of *surviving* objects required.
    resolve_oid_fn(row) -> str|None
        TNS row -> ZTF object id.
    query_detections_fn(oid) -> DataFrame|None
        ALeRCE detections for an object id.
    process_fn(csv_path, oid, label) -> dict|None
        Feature extractor; None means this object failed and we should try the next.

    Returns
    -------
    (manifest, features) : (list[dict], list[dict])
        Both truncated to exactly `target` when the pool was deep enough.
    """
    out_subdir = out_subdir or label.lower()
    out_dir = os.path.join(base_dir, out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(feature_dir, exist_ok=True)

    manifest_path = os.path.join(base_dir, f'{out_subdir}_manifest_topup.csv')
    features_path = os.path.join(feature_dir, f'{out_subdir}_features_topup.csv')
    failed_path = os.path.join(base_dir, f'{out_subdir}_failed_candidates.csv')

    manifest, features, failed = _load_checkpoint(manifest_path, features_path, failed_path)
    done_names = {str(m['tns_name']) for m in manifest}
    if features and verbose:
        print(f"Resuming {label}: {len(features)}/{target} already survived, "
              f"{len(failed)} known failures will be skipped")
    if len(features) >= target:
        return manifest[:target], features[:target]

    if verbose:
        print(f"{label}: candidate pool = {len(pool)} rows, need {target} survivors "
              f"({len(features)} already in hand)")

    bar = progress(total=target, initial=len(features), desc=f'{label} (survivors)') if progress else None
    attempted = 0
    try:
        for count, (_, row) in enumerate(pool.iterrows()):
            if len(features) >= target:
                break
            name = str(row['name'])
            if name in done_names or name in failed:
                continue
            attempted += 1

            oid = resolve_oid_fn(row)
            if oid is None:
                failed.add(name)
                continue
            try:
                det = query_detections_fn(oid)
            except Exception:
                failed.add(name)
                continue
            if det is None or len(det) < min_detections:
                failed.add(name)
                continue

            keep = [c for c in ['mjd', 'fid', 'magpsf', 'sigmapsf', 'ra', 'dec'] if c in det.columns]
            det = det[keep]
            csv_path = os.path.join(out_dir, f"{oid}.csv")
            det.to_csv(csv_path, index=False)

            feat = process_fn(csv_path, oid, label)
            if feat is None:
                # GP / cleaning failed -> try the NEXT candidate rather than eat the loss.
                failed.add(name)
                continue

            manifest.append({'tns_name': name, 'oid': oid, 'label': label,
                             'n_points': len(det), 'redshift': row.get('redshift')})
            features.append(feat)
            done_names.add(name)
            if bar is not None:
                bar.update(1)
            if sleep:
                time.sleep(sleep)

            if count % checkpoint_every == 0:
                _save_checkpoint(manifest, features, failed,
                                 manifest_path, features_path, failed_path)
                gc.collect()
    finally:
        if bar is not None:
            bar.close()
        _save_checkpoint(manifest, features, failed, manifest_path, features_path, failed_path)

    if verbose:
        if len(features) < target:
            print(f"  SHORTFALL: {label} reached only {len(features)}/{target} after exhausting a pool "
                  f"of {len(pool)} ({attempted} attempted this run, {len(failed)} cumulative failures). "
                  f"Genuine scarcity or attrition — raise the pool size and re-run; it resumes.")
        else:
            print(f"  {label}: reached target {target}/{target} "
                  f"({attempted} candidates attempted this run, {len(failed)} cumulative failures).")
    return manifest[:target], features[:target]


def build_flares_to_target(
    star_names,
    target,
    *,
    base_dir,
    feature_dir,
    search_fn,
    download_fn,
    process_fn,
    out_subdir='stellar_flares',
    max_sectors_per_star=14,
    checkpoint_every=5,
    progress=None,
    verbose=True,
):
    """
    Same "keep trying until N survive" contract as `build_tns_class_to_target`,
    adapted to how flares are actually sourced: TESS light curves via lightkurve,
    keyed by star name + sector rather than by a TNS row / ZTF oid.

    The important difference from the original `fetch_tess_flares` is that a
    sector only counts once `process_fn` has produced a feature row from it. The
    old loop counted *downloads*, so sectors that later failed cleaning or GP
    fitting silently shrank the class below its target.

    search_fn(star_name) -> sequence of downloadable entries (may be empty)
    download_fn(entry)   -> DataFrame with 'bjd','flux'(,'flux_err') or None
    process_fn(csv_path, star_name, label) -> dict|None
    """
    out_dir = os.path.join(base_dir, out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(feature_dir, exist_ok=True)

    manifest_path = os.path.join(base_dir, f'{out_subdir}_manifest_topup.csv')
    features_path = os.path.join(feature_dir, f'{out_subdir}_features_topup.csv')
    failed_path = os.path.join(base_dir, f'{out_subdir}_failed_candidates.csv')

    manifest, features, failed = _load_checkpoint(manifest_path, features_path, failed_path)
    done_files = {str(m['file']) for m in manifest}
    if features and verbose:
        print(f"Resuming stellar_flare: {len(features)}/{target} already survived, "
              f"{len(failed)} known failures will be skipped")
    if len(features) >= target:
        return manifest[:target], features[:target]

    bar = progress(total=target, initial=len(features), desc='stellar_flare (survivors)') if progress else None
    stars_used = 0
    try:
        for name in star_names:
            if len(features) >= target:
                break
            stars_used += 1
            try:
                search = search_fn(name)
            except Exception:
                if verbose:
                    print(f"  search failed for {name}, moving on")
                continue
            if search is None or len(search) == 0:
                continue

            safe_name = str(name).replace(' ', '_')
            for i, entry in enumerate(list(search)[:max_sectors_per_star]):
                if len(features) >= target:
                    break
                fname = f"{safe_name}_sector{i}.csv"
                if fname in done_files or fname in failed:
                    continue
                try:
                    df = download_fn(entry)
                except Exception:
                    failed.add(fname)
                    continue
                if df is None or len(df) == 0:
                    failed.add(fname)
                    continue

                csv_path = os.path.join(out_dir, fname)
                df.to_csv(csv_path, index=False)

                feat = process_fn(csv_path, name, 'stellar_flare')
                if feat is None:
                    # Cleaning/GP failed -> next sector, then next star. Same retry contract.
                    failed.add(fname)
                    continue

                feat = dict(feat)
                # Keep a per-sector unique id so 14 sectors of AD Leo aren't 14 rows all
                # called "AD Leo" — they are distinct light curves and must stay distinguishable.
                feat['id'] = f"{safe_name}_sector{i}"
                manifest.append({'star_name': name, 'file': fname, 'label': 'stellar_flare',
                                 'n_points': len(df)})
                features.append(feat)
                done_files.add(fname)
                if bar is not None:
                    bar.update(1)

            if stars_used % checkpoint_every == 0:
                _save_checkpoint(manifest, features, failed,
                                 manifest_path, features_path, failed_path)
                gc.collect()
    finally:
        if bar is not None:
            bar.close()
        _save_checkpoint(manifest, features, failed, manifest_path, features_path, failed_path)

    if verbose:
        if len(features) < target:
            print(f"  SHORTFALL: stellar_flare reached only {len(features)}/{target} after working "
                  f"through {stars_used} stars ({len(failed)} cumulative failures). Add more names to "
                  f"FLARE_STAR_NAMES and/or raise max_sectors_per_star, then re-run; it resumes.")
        else:
            print(f"  stellar_flare: reached target {target}/{target} "
                  f"(used {stars_used} stars, {len(failed)} cumulative failures).")
    return manifest[:target], features[:target]


def assemble_feature_df(feature_lists, verbose=True):
    """
    Combine per-class feature rows into one frame, flag rows that genuinely had a
    two-band colour measurement, and impute the rest.

    `has_color` is kept as a feature precisely because "colour was measurable at
    all" is itself informative: TESS flares never have a ZTF g-r colour, so the
    flag is a real observational signal, not just an imputation artefact.
    """
    rows = []
    for lst in feature_lists:
        rows.extend(lst)
    df = pd.DataFrame(rows)
    df['has_color'] = df['color_g_r'].notna().astype(int)
    global_median = df.loc[df['has_color'] == 1, 'color_g_r'].median()
    class_medians = df.groupby('label')['color_g_r'].transform('median')
    df['color_g_r'] = df['color_g_r'].fillna(class_medians).fillna(global_median)
    if verbose:
        print(f"Combined feature table: {df.shape[0]} rows x {df.shape[1]} cols")
    return df.reset_index(drop=True)


def verify_counts(feature_df, n_per_class, n_per_subtype, sn_subtype_labels, verbose=True):
    """
    Definition-of-done check: 150 per coarse class, 30 per SN subtype.
    Returns (ok, report_df). Never raises — a shortfall is reported, not hidden.
    """
    coarse = feature_df['label'].apply(lambda l: 'SNe' if l in sn_subtype_labels else l)
    rows = []
    ok = True
    for cls in sorted(coarse.unique()):
        n = int((coarse == cls).sum())
        rows.append({'level': 'coarse', 'class': cls, 'count': n,
                     'target': n_per_class, 'shortfall': max(0, n_per_class - n)})
        ok = ok and n == n_per_class
    for st in sn_subtype_labels:
        n = int((feature_df['label'] == st).sum())
        rows.append({'level': 'subtype', 'class': st, 'count': n,
                     'target': n_per_subtype, 'shortfall': max(0, n_per_subtype - n)})
        ok = ok and n == n_per_subtype
    report = pd.DataFrame(rows)
    if verbose:
        print(report.to_string(index=False))
        print("All targets met." if ok else "NOT all targets met — see 'shortfall' column above.")
    return ok, report
