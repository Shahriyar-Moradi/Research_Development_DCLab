"""DCLab notebook copilot: evidence-backed review comments beside every notebook cell.

The copilot reads a Jupyter notebook statically (it never executes code), runs a
set of methodology detectors in cell order, and attaches to each finding:

* a concrete fix,
* the DCLab rules it violates (curated IDs, not fuzzy matches), and
* a measured precedent from the R&D campaigns ("show proof"), retrieved with
  type and dataset filters from ``dclab_rnd.evidence_index``.

Entry points: ``review_notebook`` (Python) and ``python -m dclab_rnd.copilot`` (CLI).
"""

from .analyzer import Finding, review_notebook, review_source

__all__ = ["Finding", "review_notebook", "review_source"]
