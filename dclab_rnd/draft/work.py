"""The background work of a draft, outside the routes (package 10.3): the Home agent, the data pipeline of one asset,
and synthetic data. The routes queue jobs; a worker (in the server or ``python -m dclab_rnd.worker``) runs them here.

``jobs`` is the app's ``Services`` (or None in a test that runs the work inline): when the Home agent asks for
synthetic data in the middle of its turn, the request becomes a job the turn runs at once, so it shows on the
Compute page and can be stopped, while the chat keeps its order (the data is ready before the agent's next message).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import pipeline
from .chat import HomeAgent
from .store import DraftStore


class DraftWork:
    def __init__(self, drafts: DraftStore, models: Any, traces: Any = None, jobs: Any = None):
        self.drafts, self.models, self.traces, self.jobs = drafts, models, traces, jobs

    def agent(self, draft_id: str | None = None) -> HomeAgent:
        return HomeAgent(self.drafts, self.models.client("home_agent", draft_id=draft_id), on_request=self.requests, traces=self.traces)

    def requests(self, draft_id: str, what: str, args: dict[str, Any]) -> None:
        """The agent asked for something that runs in the background (today: simulate data)."""
        if what != "simulate":
            return
        payload = {"draft_id": draft_id, "prompt": args.get("prompt", ""), "rows": int(args.get("rows") or 5000), "template": None, "by": "agent"}
        if self.jobs is None:
            self.simulate(draft_id, payload["prompt"], payload["rows"], actor="agent")
            return
        from .. import limits
        from ..jobs import ActiveJob

        try:
            job = self.jobs.queue_job("synthetic", f"{draft_id}:synthetic", payload, by="agent", here=True)
        except ActiveJob:  # the Synthetic tab is generating a table for this draft already
            return
        except limits.LimitExceeded as over:  # at the job limit: say so plainly, in the chat
            self.drafts.emit(draft_id, "status", {"note": f"No synthetic data now: {over.body['message']}"})
            return
        self.jobs.worker.run_here(job)

    def process(self, draft_id: str, asset_id: str) -> dict[str, Any]:
        return pipeline.run(self.drafts, draft_id, asset_id, self.agent(draft_id), self.models.client("parse_pattern", draft_id=draft_id))

    def simulate(self, draft_id: str, prompt: str, rows: int, template: str | None = None, actor: str = "human") -> dict[str, Any]:
        from ..jobs import TEXT, Stopped, checkpoint
        from . import synthetic

        drafts, models = self.drafts, self.models
        draft = drafts.get(draft_id)
        if template:  # the user picked a built-in template in the Synthetic tab
            drafts.emit(draft_id, "pipeline", {"step": "designing", "text": "Generating synthetic data from the template you chose"})
            spec, info = synthetic.from_template(template, rows)
        else:
            context = {"problem": draft["problem"], "understanding": draft.get("understanding") or {}, "pack": (draft.get("pack") or {}).get("key")}
            drafts.emit(draft_id, "pipeline", {"step": "designing", "text": "Designing a synthetic dataset from the conversation"})
            spec, info = synthetic.spec_from_model(models.client("synthetic_schema", draft_id=draft_id), prompt or draft["problem"], context, rows)
        checkpoint()
        note = synthetic.template_note(spec, info)
        if note:  # never let template rows pass for a table designed from the user's description
            drafts.emit(draft_id, "status", {"note": note})
        frame = synthetic.generate(spec)
        saved = synthetic.save(frame, spec, drafts.data_dir(draft_id))
        asset = pipeline.new_asset(drafts, draft_id, "synthetic", f"Synthetic · {spec.name}", Path(saved["path"]).name, actor=actor,
                                   synthetic=True, spec_source=info.get("source"), spec_file=Path(saved["spec_path"]).name,
                                   **({"suggestion": {"target": spec.target.name}} if spec.target else {}),  # the generator knows its outcome column
                                   **({"template": info["template"], "template_note": note} if info.get("template") else {}))
        try:
            pipeline.run(drafts, draft_id, asset["id"], self.agent(draft_id), models.client("parse_pattern", draft_id=draft_id))
        except Stopped as stop:  # a stop between the pipeline's steps: the new asset is failed, not left "queued"
            current = next((a for a in drafts.get(draft_id).get("assets") or [] if a["id"] == asset["id"]), {})
            if current.get("status") not in ("ready", "failed"):
                pipeline.set_asset(drafts, draft_id, asset["id"], status="failed", error=TEXT[stop.reason])
                drafts.emit(draft_id, "pipeline", {"asset": asset["id"], "step": "failed", "text": TEXT[stop.reason]})
            raise
        return asset
