"""One case under one arm configuration, played on a fresh workspace through the real team runtime (B4, 2026-11-24).

The person (and the Sales and Marketing members) are a script: each scripted message fires once, the first whose trigger
holds, after every delivery has settled. Everything else is the product: Home hands the request to the Lead (``lead.ask``)
or the person writes in the Business thread (``agents.from_thread``), the outbox delivers each row (``delivery.run``), the
team agents act through their guard, and their brains are the case's arm (``DCLAB_TEAM_BRAIN``: the roles under test on
the arm, every other role on the rules).

Every model request goes through the play's own gateway (``dclab_rnd.models.Gateway``, its usage log and every exchange,
with its thinking, under the play's folder, which lives under ``agent_runs/live_tests/<run>/`` for a live run), seen
through :class:`ArmGateway`: every client is served by the run's model, capped by the run's euros, and refused when its
model or endpoint is Claude's. The rules arm gets no gateway at all: it can ask no model.

What a play leaves: ``play.json`` (the case, the arm, the model, the script's messages as sent, the ids, seconds, euros,
how it stopped) beside the workspace folder. :mod:`score` reads only those records.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterator, Mapping, Optional

from . import cases as C

ARMS = ("R", "H", "A")
ARM_NAMES = {"R": "rules", "H": "helper", "A": "agent"}
FORBIDDEN = ("claude", "anthropic", "fable")  # the owner (2026-11-15): never a Claude model, at any provider
MAX_ROUNDS = 12  # scripted messages a case may send, plus the opening: a script that keeps firing is cut here
MAX_DELIVERIES = 400
PLAY = "play.json"
# B4.1 (the first dev run, 2026-11-24): the provider refused every request for want of credit (HTTP 402) and the model arms
# quietly fell back to the rules, their plays marked final. A model request's outcome, as the gateway records it
# (``models/client.failure``'s words), says whose failure it was:
PROVIDER_REFUSED = ("the model account has no credit left", "the provider refused the key")  # the account: the run stops at once
PROVIDER_FAILED = ("the provider is limiting requests for now", "the provider had an error",  # 429 after the retries, a 5xx,
                   "the model endpoint could not be reached")  # the network: the play counts for no arm and is played again
REPLAYABLE = ("cap", "provider_refused", "provider_failed")  # a play stopped so is played again by the next sitting (--resume)


class Refused(ValueError):
    """A run or a play that may not start; the reason says why."""


def model_refusal(model: Optional[str]) -> Optional[str]:
    """Why a model id may not serve an arm or judge, or None."""
    name = str(model or "").lower()
    if not name:
        return "a model arm needs a model id"
    hit = next((w for w in FORBIDDEN if w in name), None)
    return f"{model} is refused: no Claude model serves any arm or judge here (the owner, 2026-11-15)" if hit else None


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-") or "x"


def provider_reason(text: Any) -> Optional[str]:
    """``refused`` (the provider refused the account: no credit, a refused key), ``failed`` (a rate limit after the retries,
    a provider error, an unreachable endpoint) or None (the request's own failure, or no failure) for a failed request's
    words: an exception's text or a usage row's outcome."""
    said = str(text or "")
    if any(reason in said for reason in PROVIDER_REFUSED):
        return "refused"
    if any(reason in said for reason in PROVIDER_FAILED):
        return "failed"
    return None


def provider_failures(folder: Path) -> dict[str, int]:
    """The play's model requests that failed for the provider's reason, from its gateway's usage log: ``refused`` and ``failed``."""
    out = {"refused": 0, "failed": 0}
    for path in Path(folder).glob("**/usage.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                outcome = json.loads(line).get("outcome")
            except (ValueError, AttributeError):
                continue
            kind = provider_reason(outcome) if outcome != "ok" else None
            if kind:
                out[kind] += 1
    return out


# ---------------------------------------------------------------------------------------------------------- the euros
class RunBudget:
    """One euro cap over every client of a run, across its plays. A client's own limit (an agent turn's euros) is kept: the
    request is refused at whichever is lower, the turn's or what the run has left.

    The review of B4 (F3): the gateway refuses a request whose upper-bound cost passes what is left, so a run just below its
    cap would silently turn every model arm into the rules. :meth:`reached` therefore holds once less than one request's
    upper bound (``margin``) is left, and :attr:`exhausted` once the gateway refused a request for the run's money: a play
    during which that happened is cut (``stopped: cap``), never scored as complete."""

    def __init__(self, cap_eur: float, margin_eur: float = 0.0):
        self.cap_eur, self.margin_eur = float(cap_eur), float(margin_eur)
        self.clients: list[tuple[str, Any]] = []
        self.exhausted = False
        self.provider_refused: Optional[str] = None  # B4.1: the provider refused the account: the run stops at once

    def spent_eur(self, key: Optional[str] = None) -> float:
        return sum(float(getattr(c, "spent_eur", 0.0) or 0.0) for k, c in self.clients if key is None or k == key)

    def left(self) -> float:
        return max(0.0, self.cap_eur - self.spent_eur())

    def reached(self) -> bool:
        return self.exhausted or self.spent_eur() >= self.cap_eur or self.left() <= self.margin_eur

    def wrap(self, key: str, client: Any) -> "_Capped":
        self.clients.append((key, client))
        return _Capped(client, self)


class _Capped:
    def __init__(self, inner: Any, budget: RunBudget):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_budget", budget)
        object.__setattr__(self, "_own", None)

    def _limit(self) -> bool:
        """Set the request's limit; True when the run's money (not the client's own limit) is what binds."""
        inner = self._inner
        run = float(getattr(inner, "spent_eur", 0.0) or 0.0) + self._budget.left()
        own = self._own
        binding = own is None or own <= 0 or run <= own
        inner.run_limit_eur = run if binding else own
        return binding

    def _call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        binding = self._limit()
        try:
            return getattr(self._inner, name)(*args, **kwargs)
        except Exception as exc:
            if binding and str(exc).startswith("BudgetExceeded"):
                self._budget.exhausted = True  # refused for the run's money: the play is cut, the run stops after it
            if provider_reason(str(exc)) == "refused":  # B4.1: no credit or a refused key: no arm can be measured any more
                self._budget.provider_refused = self._budget.provider_refused or str(exc).split(": ", 1)[-1][:200]
            raise

    def complete(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("complete", *args, **kwargs)

    def stream(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("stream", *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "run_limit_eur":  # the caller's own limit (an agent turn's euros): kept, and never above the run's
            object.__setattr__(self, "_own", None if value is None else float(value))
        setattr(self._inner, name, value)


TEAM_PURPOSES = ("team_helper", "team_agent")  # what the arms differ in; every other purpose gets no model, as on the rules arm


class ArmGateway:
    """The play's gateway as the team agents see it: the team purposes served by the run's model, capped by the run, filed
    under the play's key; every other purpose gets no model (the rules arm has none at all: same everything but the
    brain). A client whose model or endpoint is Claude's is never handed out (the agent then falls back to the rules,
    recorded); :attr:`refused` keeps why."""

    def __init__(self, gateway: Any, budget: RunBudget, model: str, key: str):
        self.gateway, self.budget, self.model, self.key = gateway, budget, model, key
        self.refused: list[str] = []

    def client(self, purpose: str, *args: Any, project_id: Optional[str] = None, draft_id: Optional[str] = None,
               model: Optional[str] = None, **kwargs: Any) -> Any:
        if purpose not in TEAM_PURPOSES:
            self.refused.append(f"{purpose}: not a team purpose (no model, as on the rules arm)")
            return None
        found = self.gateway.client(purpose, project_id=project_id, draft_id=draft_id or self.key, model=self.model)
        if found is None:
            return None
        served = str(getattr(found, "model", "") or "")
        endpoint = str((getattr(found, "tier", None) or {}).get("base_url") or "")
        why = model_refusal(served) or (f"{endpoint} is Claude's endpoint" if any(w in endpoint.lower() for w in FORBIDDEN) else None)
        if why:
            self.refused.append(f"{purpose}: {why}")
            return None
        return self.budget.wrap(self.key, found)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.gateway, name)


# ---------------------------------------------------------------------------------------------------------- the workspace
@contextlib.contextmanager
def environment(values: Mapping[str, Optional[str]]) -> Iterator[None]:
    """Set (or, for None, remove) environment values for the play, and put back what was there."""
    before = {k: os.environ.get(k) for k in values}
    try:
        for k, v in values.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in before.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def open_services(home: Path) -> SimpleNamespace:
    """The workspace's stores, as the team runtime reads them (files, or PostgreSQL when DCLAB_DATABASE_URL is set)."""
    from dclab_rnd import experience
    from dclab_rnd.jobs import open_jobs
    from dclab_rnd.ledger import open_ledger
    from dclab_rnd.storage import db, open_stores, open_team_work, open_teams

    projects, drafts, _ = open_stores(home)
    wid = db.workspace_for(home) if db.database_url(required=False) else None
    services = SimpleNamespace(store=SimpleNamespace(home=home), projects=projects, drafts=drafts, workspace_id=wid,
                               experience=experience.open_log(home, wid), job_store=open_jobs(home),
                               teamwork=open_team_work(home, projects=projects), teams=open_teams(home), gateway=None)
    services.ledger = open_ledger(home, wid)
    return services


def _member(services: Any, name: str) -> str:
    """A scripted person of this workspace: their name on files, an account on PostgreSQL (a team's member must be one)."""
    from dclab_rnd.storage import db

    if not db.database_url(required=False):
        return name
    from dclab_rnd.accounts.store import Accounts

    accounts = Accounts()
    user = accounts.create_user(f"team-arms-{name}-{os.urandom(4).hex()}@example.com", name.title(), password=os.urandom(12).hex())["id"]
    accounts.set_member(services.teamwork.workspace, user, "data_scientist")
    return user


def override(roles: list[str], arm: str) -> str:
    """``DCLAB_TEAM_BRAIN`` for the play: the roles under test on the arm, every other role on the rules."""
    return ",".join(f"{r}={ARM_NAMES[arm] if r in roles else 'rules'}" for r in C.ROLES)


# ---------------------------------------------------------------------------------------------------------- one play
class Player:
    """One case, one arm, one model, one fresh workspace in ``folder``."""

    def __init__(self, case: C.Case, arm: str, model: Optional[str], folder: Path, *, gateway: Any = None, budget: Optional[RunBudget] = None,
                 set_name: str = "quick"):
        if arm not in ARMS:
            raise Refused(f"no arm {arm!r}: one of {', '.join(ARMS)}")
        if arm != "R":
            why = model_refusal(model)
            if why:
                raise Refused(why)
            if gateway is None or budget is None:
                raise Refused("a model arm plays through the run's gateway and budget")
        self.case, self.arm, self.model, self.folder, self.set_name = case, arm, (model if arm != "R" else None), Path(folder), set_name
        self.raw_gateway, self.budget = gateway, budget
        # a fresh workspace every time: on PostgreSQL a folder's path names its workspace, so a play run again (a resumed run's
        # cut play, in a folder of the same name) must never reach the earlier attempt's records
        self.home = self.folder / f"workspace-{os.urandom(4).hex()}"
        self.record: dict[str, Any] = {}

    # -- the plumbing
    def _pump(self) -> None:
        from dclab_rnd.teamspace import brains, delivery, seats

        work = self.services.teamwork
        for _ in range(MAX_DELIVERIES):
            heads = work.pending_heads()
            if not heads:
                brains.drain(150)  # B1/B2: a model's after-turn request on its thread (its record) ends before anything is read
                seats.drain(150)
                if not work.pending_heads():
                    return
                continue
            head = heads[0]
            for row in delivery.run(self.services, {"project_id": head["project_id"], "task_id": delivery.lane_of(work, head)}):
                if row.get("status") != "delivered":
                    self.record["delivery_problems"].append({"kind": row.get("kind"), "to": row.get("to_team"), "status": row.get("status"),
                                                             "reason": str(row.get("reason") or "")[:300]})
        self.record["delivery_problems"].append({"status": "never_settled"})

    def _thread(self, kind: str) -> Mapping[str, Any]:
        from dclab_rnd.teamspace.threads import team_key, threads_for

        key = team_key(self.services.teams, kind)[0]
        return threads_for(self.services).open_thread(self.pid, self.prob, key, label=kind.replace("_", " ").title())[0]

    def _say(self, kind: str, person: str, text: str) -> None:
        """The person's words saved in their thread, as the route saves a message before the team's agent takes it."""
        from dclab_rnd.teamspace.threads import threads_for

        thread = self._thread(kind)
        threads_for(self.services).post(thread["id"], {"kind": "person", "id": person, "team": kind}, text)

    def _tree(self) -> list[Any]:
        work, out, todo = self.services.teamwork, [], [self.top] if self.top else []
        while todo:
            tid = todo.pop(0)
            view = work.task_view(self.pid, tid)
            if view is None:
                continue
            out.append(view["task"])
            todo += [c.id for c in view["children"]]
        return out

    def _requester_team(self) -> str:
        return "home" if self.case.entry == "home" else "business"

    def open_question(self) -> Optional[tuple[str, str]]:
        """The question the person is asked now and still open: (task id, its words), or None."""
        work = self.services.teamwork
        rows = work.deliveries(project_id=self.pid)
        to = self._requester_team()
        for i in range(len(rows) - 1, -1, -1):
            row = rows[i]
            if row["kind"] != "clarification" or row["to_team"] != to:
                continue
            task = work.task(self.pid, row["task_id"])
            if task is None or task.status != "waiting_for_person":
                continue
            later = rows[i + 1:]
            if any(r["kind"] == "answer" and r["task_id"] == task.id and r["from_team"] == to for r in later):
                continue
            return task.id, str(row.get("text") or "")
        return None

    def open_team_question(self, team: str) -> Optional[tuple[str, str]]:
        from dclab_rnd.teamspace import commercial

        if team not in self.case.data.members:
            return None
        thread = self._thread(team)
        for task in commercial.open_asks(self.services, thread, team):
            ask = commercial.awaiting(self.services.teamwork.task_view(self.pid, task.id)) or {}
            return task.id, str(ask.get("question") or "")
        return None

    def top_reports(self) -> list[Any]:
        if not self.top:
            return []
        view = self.services.teamwork.task_view(self.pid, self.top)
        return list(view["reports"]) if view else []

    @staticmethod
    def _holds(text: str, words: list[str]) -> bool:
        said = " ".join(str(text or "").lower().split())
        return not words or any(" ".join(w.lower().split()) in said for w in words)

    def _due(self, reply: C.Reply) -> Optional[Any]:
        when = reply.when
        if when.on == "question":
            found = self.open_question()
            return found if found and self._holds(found[1], when.contains_any) else None
        if when.on == "team_question":
            found = self.open_team_question(reply.who)
            return found if found and self._holds(found[1], when.contains_any) else None
        reports = self.top_reports()
        if len(reports) >= when.nth:
            text = "\n".join([reports[when.nth - 1].answer, *reports[when.nth - 1].limits])
            return (self.top, text) if self._holds(text, when.contains_any) else None
        return None

    def _task_of(self, which: Optional[str]) -> Optional[str]:
        if which in (None, "top"):
            return self.top
        found = [t.id for t in self._tree() if t.owner_team == which]
        return found[0] if found else None

    def _fire(self, reply: C.Reply, due: Any, n: int) -> dict[str, Any]:
        from dclab_rnd.teamspace import agents, lead

        person = self.people[reply.who if reply.who != "person" else "person"]
        rid = f"{self.case.id}-{reply.id}"
        sent: dict[str, Any] = {"id": reply.id, "who": reply.who, "person": person, "kind": reply.kind, "text": reply.text,
                                "minutes": reply.minutes, "response_id": rid, "round": n, "task": due[0] if due else None, "refused": []}
        initiator = self.initiator if reply.who == "person" else {**self.initiator, "user_id": person}
        if reply.who != "person":
            self._say(reply.who, person, reply.text or "")
            out = agents.from_thread(self.services, self._thread(reply.who), person=person, text=reply.text or "", initiator=initiator,
                                     team_kind=reply.who, task_id=due[0], response_id=rid, message_id=rid) or {}
        elif reply.kind == "answer" and reply.when.on == "question":
            if self.case.entry == "home":
                out = lead.answer(self.services, self.pid, due[0], person=person, text=reply.text or "", initiator=initiator, response_id=rid)
            else:
                self._say("business", person, reply.text or "")
                out = agents.from_thread(self.services, self._thread("business"), person=person, text=reply.text or "", initiator=initiator,
                                         team_kind="business", task_id=due[0], response_id=rid, message_id=rid) or {}
        elif reply.kind == "answer":  # after a report: a new request, through the case's own way in
            out = self._open(reply.text or "", rid, initiator)
        elif reply.kind == "correct_outcome":
            task = self._task_of(reply.task)
            out = agents.confirm_outcome(self.services, self.pid, task, person=person, text=reply.text or "", initiator=initiator,
                                         response_id=rid) if task else {"refused": [f"no task of {reply.task}"]}
        else:
            task = self._task_of(reply.task)
            current = [r for r in (self.services.teamwork.task_view(self.pid, task)["reports"] if task else []) if r.status != "superseded"]
            base = max((r.revision for r in current), default=None)
            out = agents.correct(self.services, self.pid, task, kitchen_minutes=int(reply.minutes or 0), base_revision=base, correction_id=rid,
                                 initiator=initiator) if task else {"refused": [f"no task of {reply.task}"]}
        sent["refused"] = [str(x)[:300] for x in (out or {}).get("refused") or []]
        return sent

    def _open(self, text: str, rid: str, initiator: Mapping[str, Any]) -> dict[str, Any]:
        """A request in the person's own words, through the case's way in. ``last_request_refused`` says whether the person's
        latest request was refused at the door (the re-review's N6: the end is read on the last request)."""
        out = self._open_request(text, rid, initiator)
        self.record["last_request_refused"] = out.get("entry_refused")
        return out

    def _open_request(self, text: str, rid: str, initiator: Mapping[str, Any]) -> dict[str, Any]:
        from dclab_rnd.teamspace import agents, lead

        person = self.people["person"]
        if self.case.entry == "home":
            try:
                out = lead.ask(self.services, self.pid, self.prob, question=text, track=self.case.track or "", person=person,
                               draft_id=self.draft, initiator=initiator, inputs=tuple(self.case.data.inputs))
            except lead.TeamsRefused as exc:
                return {"refused": [str(exc)], "entry_refused": str(exc)}
            self.top = self.top or out.get("task_id")
            if out.get("task_id") and out["task_id"] not in self.record["tops"]:
                self.record["tops"].append(out["task_id"])
            return out
        self._say("business", person, text)
        out = agents.from_thread(self.services, self._thread("business"), person=person, text=text, initiator=initiator, team_kind="business",
                                 ask=self.case.ask, message_id=rid) or {}
        if out.get("did") == "refused" or not out.get("task_id") or self.services.teamwork.task(self.pid, out["task_id"]) is None:
            return {**out, "entry_refused": "; ".join(str(x) for x in out.get("refused") or []) or "no task was opened"}
        self.top = self.top or out.get("task_id")
        if out["task_id"] not in self.record["tops"]:
            self.record["tops"].append(out["task_id"])
        return out

    # -- the play
    def play(self) -> dict[str, Any]:
        """Play the case once and write ``play.json``. Never raises for the case's own failures: they are recorded."""
        from dclab_rnd import context
        from dclab_rnd.memory.scope import create_problem
        from dclab_rnd.teamspace import delivery

        self.home.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        case = self.case
        self.record = {"case": case.id, "family": case.family, "roles": list(case.roles), "entry": case.entry, "arm": self.arm,
                       "brain": ARM_NAMES[self.arm], "model": self.model, "set": self.set_name, "workspace": self.home.name, "fired": [],
                       "delivery_problems": [], "entry_refused": None, "stopped": None, "error": None, "top_task": None, "tops": [], "rounds": 0,
                       "last_request_refused": None,
                       "private_notes": [{"team": n.team, "marker": n.marker} for n in case.data.private_notes]}
        sealed = self.set_name == "sealed"
        env = {"DCLAB_TEAM_BRAIN": override(list(case.roles), self.arm), "DCLAB_AUTH": "", "DCLAB_TEAM_DELIVERY_BACKOFF_SECONDS": "0"}
        if not sealed:
            env["DCLAB_LOG_PROMPTS"] = "1"  # rule 9: prompts, replies and thinking kept whole (a sealed run keeps them in its vault)
        target, checker = delivery.set_target(None), delivery.set_membership_checker(None)
        token = None
        try:
            with environment(env):
                self.services = open_services(self.home)
                if self.arm != "R":
                    self.services.gateway = ArmGateway(self.raw_gateway, self.budget, self.model or "", f"team-arms:{case.id}:{self.arm}")
                token = context.set_services(self.services)
                self._setup(create_problem)
                self.top = None
                out = self._open(case.opening, f"{case.id}-open", self.initiator)
                self.record["entry_refused"] = out.get("entry_refused")
                self.record["top_task"] = self.top
                fired: set[str] = set()
                for n in range(1, MAX_ROUNDS + 1):
                    if self.top:
                        self._pump()
                    if self.budget is not None and self.budget.exhausted:
                        self.record["stopped"] = "cap"
                        break
                    if self.budget is not None and self.budget.provider_refused:  # B4.1: at once, never on to the next message
                        self.record["stopped"] = "provider_refused"
                        break
                    if not self.top:
                        break
                    due = next(((r, d) for r in case.replies if r.id not in fired for d in [self._due(r)] if d is not None), None)
                    if due is None:
                        break
                    fired.add(due[0].id)
                    self.record["rounds"] = n
                    self.record["fired"].append(self._fire(due[0], due[1], n))
                    self.record["top_task"] = self.top
                if self.top:
                    self._pump()
        except Exception as exc:  # noqa: BLE001 - a crash of the runtime is the case's record, never the run's end
            self.record["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
            self.record["traceback"] = traceback.format_exc()[-4000:]
        finally:
            if token is not None:
                context.reset_services(token)
            delivery.set_target(target)
            delivery.set_membership_checker(checker)
        self.record["seconds"] = round(time.monotonic() - started, 3)
        self.record["euros"] = round(usage_euros(self.folder), 6)
        if self.budget is not None and (self.budget.exhausted or self.budget.spent_eur() >= self.budget.cap_eur) and not self.record["stopped"]:
            self.record["stopped"] = "cap"  # a request was refused for the run's money, or the run spent its cap during this play
        # B4.1: a request the provider refused or failed (402, a refused key; 429 after the retries, a 5xx, the network) turned
        # that turn into the rules': the play measures no arm. It is played again, never scored (a client the harness does not
        # wrap is read from the usage log too)
        failures = provider_failures(self.folder) if self.arm != "R" else {"refused": 0, "failed": 0}
        if any(failures.values()):
            self.record["provider_failures"] = failures
        if self.budget is not None and self.budget.provider_refused and not failures["refused"]:
            failures["refused"] = 1
        if not self.record["stopped"] and failures["refused"]:
            self.record["stopped"] = "provider_refused"
        elif not self.record["stopped"] and failures["failed"]:
            self.record["stopped"] = "provider_failed"
        gw = self.services.gateway if getattr(self, "services", None) is not None else None
        if isinstance(gw, ArmGateway) and gw.refused:
            self.record["model_refused"] = gw.refused[:20]
        (self.folder / PLAY).write_text(json.dumps(self.record, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
        return self.record

    def _setup(self, create_problem: Callable[..., Any]) -> None:
        case, services = self.case, self.services
        self.people = {"person": _member(services, C.PERSON)}
        for team in case.data.members:
            self.people[team] = _member(services, C.MEMBERS[team])
        self.pid = services.projects.create(f"Team arms {case.id}", goal=case.title[:200])["id"]
        services.projects.update(self.pid, team_inputs=list(case.data.inputs))
        self.prob = create_problem(services.projects, self.pid, case.title[:80], workspace_id=services.teamwork.workspace)["problem_id"]
        self.draft = services.drafts.create(case.title[:80])["id"]
        self.initiator = {"user_id": self.people["person"], "role": "data_scientist", "workspace_id": services.teamwork.workspace}
        business = services.teams.create_team("Business", "business")
        services.teams.add_member(business["id"], self.people["person"])
        for team in case.data.members:
            made = services.teams.create_team(team.title(), team)
            services.teams.add_member(made["id"], self.people[team])
        from dclab_rnd.teamspace.threads import threads_for

        for note in case.data.private_notes:
            author = self.people.get(note.team) or (self.people["person"] if note.team == "business" else f"{note.team}-member")
            thread = self._thread(note.team)
            threads_for(services).post(thread["id"], {"kind": "person", "id": author, "team": note.team}, note.text)
        self.record.update(project_id=self.pid, problem_id=self.prob, draft_id=self.draft, people=dict(self.people))


def usage_euros(folder: Path) -> float:
    """What the play's model requests cost, from its gateway's usage log."""
    total = 0.0
    for path in Path(folder).glob("**/usage.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                total += float(json.loads(line).get("cost_eur") or 0.0)
            except (ValueError, TypeError):
                continue
    return total


def play(case: C.Case, arm: str, model: Optional[str], folder: Path, **kw: Any) -> dict[str, Any]:
    return Player(case, arm, model, folder, **kw).play()


__all__ = ["ARMS", "ARM_NAMES", "FORBIDDEN", "PLAY", "PROVIDER_FAILED", "PROVIDER_REFUSED", "REPLAYABLE", "ArmGateway", "Player", "Refused", "RunBudget", "environment", "model_refusal",
           "open_services", "override", "play", "provider_failures", "provider_reason", "slug", "usage_euros"]
