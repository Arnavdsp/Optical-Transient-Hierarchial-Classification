"""
v5: hierarchical optical-transient classification with a dual ground-truth
comparison (TNS spectroscopic labels vs ALeRCE's own classifier output).

Built on v4's acquisition/Villar/quality-gate pipeline, which is preserved verbatim
in `photometry.py`. The v5 additions are the ALeRCE label track (`alerce_labels`),
the three-pool construction (`pools`), host offset (`hostoffset`), noise-model
augmentation (`augment`), the dual-track model runner (`modeling`), the agreement
framework (`agreement`) and calibration (`calibration`).
"""
