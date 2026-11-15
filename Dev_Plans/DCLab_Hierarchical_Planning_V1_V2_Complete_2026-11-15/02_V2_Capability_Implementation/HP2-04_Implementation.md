<!-- Extracted from DCLab Hierarchical Planning V2 Additive Capability Pack (2026-11-05). -->
<!-- Mandatory: V1 acceptance gates first; do not replace or bypass deterministic scientific and safety guards. -->

### HP2-04 | Evidence-backed experiment memory

**Objective:** Retain usable lessons without treating unverified agent text as truth.

**Memory entry:** case signature; proposal; hypothesis; tool/code/dataset fingerprints; test conditions; metric and statistical uncertainty; verifier outcome; applicability scope; failure modes; lesson status `candidate/verified/refuted/stale`.

**Policies:** Save signed-off lessons only after verifier approval; source-data changes invalidate applicability; recency/confidence scoring affects retrieval but does not bypass registry. Never retrieve benchmark answer keys.

**Acceptance:** A previously proven failure mode can be avoided on a new but compatible case; a conflicting lesson is surfaced as uncertain; revoked lesson stops being used; retrieval never exposes held-out labels.
