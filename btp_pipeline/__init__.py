"""
BTP: hierarchical classification of optical transients.

The modules here are the single source of truth for every piece of logic that is
not a raw API call. `tools/build_notebook.py` inlines them verbatim into the
Colab notebook, so the notebook stays self-contained (no repo clone needed at
run time) while the same code is unit-tested offline against synthetic data.
"""
