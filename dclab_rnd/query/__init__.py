"""The query engine Home answers from (package H6; `dclab_rnd.query`, beside the evidence module `analysis.py`): a typed plan, its validator, its executor, the row floor and the
exposure log.

Every analysis, report and business answer runs on this engine, never on free code:

- ``plan``: the versioned schema of a query plan (filters, at most three groups, a period with a grain, measures,
  sort, a limit of at most 500). A plan always aggregates: there is no plan that lists rows.
- ``validate``: refuses unknown columns, free-text and identifier columns in groups or filters, a sum or a mean on a
  column that is not numeric, a limit over 500, and any plan without a measure.
- ``execute``: runs a valid plan with pandas on the draft's cleaned table, opened only when it still matches the
  fingerprint recorded when it was written (``table``).
- the row floor (``execute.floor``) removes groups with fewer rows than ``DCLAB_ROW_FLOOR`` (default 5, never below 3) and counts them.
- the exposure log (``execute.exposure_entry``) records which columns each executed plan related, on the draft (``draft["exposure"]``), so a later
  prediction project can tell which relations were explored before its split (H8).

A model may write a plan (purpose ``home_query``: column summaries and the request only); code validates and runs it.
"""

from .execute import execute, run  # noqa: F401
from .plan import SCHEMA_VERSION, plan_hash  # noqa: F401
from .validate import PlanError, validate  # noqa: F401
