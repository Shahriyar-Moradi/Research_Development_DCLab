"""Datasets: the catalog of the training and evaluation sets on the Admin page (package 19.13).

    GET /api/datasets                  every set `make datasets` built: builder, name, version, consent, rows, splits, size
    GET /api/datasets/lineage?path=    the sessions one set came from
    GET /api/datasets/failures         the failure candidates (sessions a detector flagged) with each one's review
    POST /api/datasets/failures/{id}/review   {"verdict": "accept" | "reject"}: a reviewer's verdict, read at the next build

The owner's (``accounts.guard``: ``admin``): the sets hold redacted text from consented projects, never cells.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from fastapi import Body, HTTPException

from ..api_models import Page, Router

PATH = re.compile(r"[a-z_]{3,20}(/[A-Za-z0-9_.]{1,120})?")  # as the catalog lists them: decisions/column.is_identifier, development


def register(app, ctx) -> None:
    router = Router()

    def home() -> Any:
        return ctx.store.home

    @router.get("/api/datasets", response_model=Page)
    async def datasets_catalog():
        """Every dataset of this workspace, newest first, with where it came from in counts."""
        from ...experience import catalog

        items = await asyncio.to_thread(catalog.catalog, home())
        return {"items": items, "total": len(items), "rows": sum(i.get("rows") or 0 for i in items)}

    @router.get("/api/datasets/lineage", response_model=Page)
    async def datasets_lineage(path: str):
        """The sessions (visits) one dataset's sources came from."""
        from ...experience import catalog

        if not PATH.fullmatch(path):
            raise HTTPException(422, "path is a dataset's builder/name, as the catalog lists it")
        try:
            return await asyncio.to_thread(catalog.lineage, home(), path, getattr(ctx, "experience", None), getattr(ctx, "ledger", None))
        except KeyError:
            raise HTTPException(404, "No such dataset in this workspace") from None

    def failures_root() -> Any:
        from pathlib import Path

        return Path(home()) / "datasets" / "failures"

    def reviews() -> dict[str, Any]:
        import json

        try:
            doc = json.loads((failures_root() / "reviews.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return doc if isinstance(doc, dict) else {}

    @router.get("/api/datasets/failures", response_model=Page)
    async def failure_candidates():
        """19.12: the failure candidates the last build kept, each with the review it has now (a verdict given since the
        build is shown at once; the accepted ones are written for the next benchmark version at the next build)."""
        import json

        rows_path = failures_root() / "benchmark_candidates" / "rows.jsonl"
        try:
            rows = [json.loads(x) for x in rows_path.read_text(encoding="utf-8").split("\n") if x.strip()]
        except OSError:
            rows = []
        given = reviews()
        items = [{"id": r.get("id"), "detectors": r.get("detectors") or [], "flags": r.get("flags") or [], "problem": r.get("problem"),
                  "family": r.get("family"), "outcome": r.get("outcome"),
                  "review": (given.get(str(r.get("id"))) or {}).get("verdict") or r.get("review") or "pending",
                  "reviewed_by": (given.get(str(r.get("id"))) or {}).get("by") or r.get("reviewed_by")} for r in rows]
        return {"items": items, "total": len(items), "pending": sum(1 for i in items if i["review"] == "pending")}

    @router.post("/api/datasets/failures/{candidate_id}/review", response_model=Page)
    async def review_failure(candidate_id: str, payload: Any = Body(None)):
        """19.12: a reviewer accepts a failure candidate for the next benchmark version, or rejects it."""
        import json
        import os
        import tempfile
        from datetime import datetime, timezone

        from ...accounts.principal import current

        verdict = str((payload or {}).get("verdict") or "") if isinstance(payload, dict) else ""
        if verdict not in ("accept", "reject"):
            raise HTTPException(422, "verdict is accept or reject")
        known = {i["id"] for i in (await failure_candidates())["items"]}
        if candidate_id not in known:
            raise HTTPException(404, "No such failure candidate in the last build")
        who = current()
        by = who.user_id if who is not None and who.user_id else "owner"
        doc = {**reviews(), candidate_id: {"verdict": verdict, "by": by, "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}}
        root = failures_root()
        root.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=root, prefix=".reviews-", suffix=".part")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=1, ensure_ascii=False)
        os.replace(tmp, root / "reviews.json")
        try:
            from ... import audit

            audit.record(getattr(ctx, "audit", None), "datasets", "human", "Owner", move="review_failure_candidate", status="allowed",
                         args={"candidate": candidate_id, "verdict": verdict, "by": by}, message="")
        except Exception:  # noqa: BLE001 - the verdict stands; its audit line is best effort
            pass
        return {"id": candidate_id, "review": verdict, "reviewed_by": by}

    app.include_router(router)
