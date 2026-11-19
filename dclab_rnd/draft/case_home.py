"""Home, conversation first (packages 37.9 and 37.11, docs/guides/PHASE13_CLAUDE_FIRST_PLAN.md): Home's draft conversation is
the lead agent's (``agents.case_agent``) by default wherever a model serves it; ``DCLAB_CASE_AGENT=0`` or no model keeps the
rules-first flow.

:class:`CaseHome` is Home with its conversation replaced: :meth:`start` reads the person's first sentence, :meth:`reply`
their next message (a pick of an option is their choice), :meth:`data_ready` records a table that became ready. Each turn
writes the agent's words as Home's messages, its options as one ``case_options`` message (cards on the page), the case file
on the draft (``draft["case"]``, the side panel the person can read) and the whole state (``draft["case_state"]``) so the
next turn, a restart or another server instance resumes it. A route the agent or the person starts becomes the draft's
kind of help, so the rest of DCLab (briefs, queries, the build) follows. Everything else of Home is unchanged, and with the
switch off nothing here runs: the old flow stays the fallback and the open edition's path.
"""

from __future__ import annotations

import re
import secrets
from pathlib import Path
from typing import Any, Optional

from dclab_core import case as C

from ..agents import case_agent as A
from ..agents import case_session
from . import intent as intents
from .chat import HomeAgent, turn
from .store import now


def table_of(store: Any, draft: dict[str, Any]) -> tuple[Optional[dict[str, Any]], Any]:
    """The draft's ready table as the agent sees it (column view and row count, never a value) and the table itself."""
    import pandas as pd

    from . import understand as U

    asset = next((a for a in draft.get("assets") or [] if a["id"] == draft.get("active_asset") and a.get("status") == "ready"), None)
    if asset is None:
        return None, None
    try:
        frame = pd.read_parquet(store.data_dir(draft["id"]) / asset.get("clean_file", "clean.parquet"))
    except Exception:  # noqa: BLE001 - a table that cannot be read is no table for the agent
        return None, None
    return A.table_of(str(asset.get("name") or "table"), U.columns_view(draft), len(frame)), frame


class CaseHome(HomeAgent):
    """Home whose conversation is the lead agent's (``DCLAB_CASE_AGENT``)."""

    def __init__(self, store: Any, models: Any, **kwargs: Any):
        super().__init__(store, None, on_request=kwargs.pop("on_request", lambda *a: None), **kwargs)
        self.models = models

    def _state(self, draft: dict[str, Any]) -> A.CaseState:
        saved = draft.get("case_state")
        if saved:
            state = A.CaseState(case=C.CaseFile.read(saved["case"]), table=saved.get("table"), computed=list(saved.get("computed") or []),
                                turns=list(saved.get("turns") or []), asking=bool(saved.get("asking")), build=saved.get("build"))
            state.spent_eur, state.cap_eur = float(saved.get("spent_eur") or 0.0), float(saved.get("cap_eur") or case_session.cap_eur())
            state.waiting_on_budget, state.consulted = bool(saved.get("waiting_on_budget")), list(saved.get("consulted") or [])
            case_session.restore(state, saved)  # names seen, exposure and the tracer's place (the review of 37.8)
        else:
            state = A.CaseState(case=C.new(f"draft-{draft['id']}"))
        table, frame = table_of(self.store, draft)
        if table is not None:
            state.table, state.frame = table, frame
        from .. import home_query
        from . import brief as briefs

        intent = draft.get("intent") or {}
        if (state.case.route is None and intent.get("source") == "user" and intent.get("track") in C.TRACKS
                and intent["track"] != state.chip_intent):  # 2026-11-04: a chip pressed (first or again) before any work started;
            state.chip, state.chip_checked, state.chip_intent = intent["track"], False, intent["track"]  # a change of mind in words stays
        elif state.case.route is None and intent.get("source") != "user":
            state.chip, state.chip_checked, state.chip_intent = None, False, None
        state.personal = {c["name"] for c in briefs.columns_of(draft) if str(c.get("sensitivity") or "none") in home_query.SENSITIVE}
        state.models = self.models
        state.draft_id = draft["id"]
        state.workspace_home = getattr(self.store, "files_home", None) or (Path(self.store.home).parent if getattr(self.store, "home", None) else None)
        state.project_id = draft.get("project_id") or getattr(state, "project_id", None)
        state.problem_id = draft.get("problem_id") or getattr(state, "problem_id", None)
        if state.project_id and state.problem_id and state.workspace_home is not None:
            try:
                from .. import context
                from ..accounts.principal import current as current_principal
                from ..memory import scope as mem_scope, summary as mem_summary

                ws = mem_scope.workspace_of(self.store)  # the one workspace id the memory routes and the build use too
                projects = getattr(context.services(), "projects", None)
                if projects is not None:  # the host checks the person against the project and its bound problem (a read)
                    state.authorized_scope = mem_scope.resolve_scope(
                        workspace_id=mem_scope.workspace_of(projects), projects=projects,
                        project_id=str(state.project_id), problem_id=str(state.problem_id), draft_id=str(draft["id"]),
                        require_problem=True, auto_create_legacy_problem=False,
                    )
                else:
                    who = current_principal()
                    state.authorized_scope = mem_scope.AuthorizedScope(
                        workspace_id=ws, principal_id=str(getattr(who, "user_id", None) or "owner"),
                        role=str(getattr(who, "role", None) or "owner"),
                        project_id=str(state.project_id), problem_id=str(state.problem_id),
                        draft_id=str(draft["id"]), auth_mode="authenticated" if mem_scope.auth_enabled() else "local",
                    )
                doc = mem_summary.read_summary(Path(state.workspace_home), state.authorized_scope, auto_rebuild=True, include_personal_overlay=True)
                state.problem_summary = doc
                state.problem_revision = int(doc.get("problem_revision") or 0)
            except Exception:  # noqa: BLE001 - draft still works if summary is unavailable
                pass
        state.no_data_solution = draft.get("no_data_solution")
        state.specialist_client = self.models.client("case_specialist", draft_id=draft["id"]) if self.models is not None else None
        # the plan's step 1: Home gets the cheap route check the exam measured (before, only the exam's arm had it)
        state.route_client = self.models.client("route_check", draft_id=draft["id"], model=A.route_model()) if self.models is not None else None
        return state

    def choose_intent(self, draft_id: str, track: str, *, by_person: bool = True) -> dict[str, Any]:
        """A chip pressed in an open chat (2026-11-04 review): with the lead agent, the choice is kept as the person's and read as
        the chip on their next message; the rules-first track is never opened inside the lead agent's chat. Once work has
        started the press changes nothing (the route changes only through the person's words)."""
        if self._client(draft_id) is None:
            return super().choose_intent(draft_id, track, by_person=by_person)
        with self.store.turn(draft_id):
            draft = self.store.get(draft_id)
            before = draft.get("intent") or {}
            if ((draft.get("case_state") or {}).get("case") or {}).get("route"):
                return before
            value = intents.chosen(track, before)
            if by_person:
                from ..ledger import capture

                capture.intent_chosen(before.get("record"), track)
            def put(d: dict[str, Any]) -> None:
                d["intent"] = value
                if isinstance(d.get("case_state"), dict):
                    d["case_state"]["chip_intent"] = None  # a press is always read, even the same chip again
            self.store.update(draft_id, put)
            return value

    def clear_intent(self, draft_id: str) -> dict[str, Any]:
        """A chip deselected in Home: clear user intent and chip selection."""
        if self._client(draft_id) is None:
            super().clear_intent(draft_id)
            return {}
        with self.store.turn(draft_id):
            draft = self.store.get(draft_id)
            before = draft.get("intent") or {}
            if ((draft.get("case_state") or {}).get("case") or {}).get("route"):
                return before
            def put(d: dict[str, Any]) -> None:
                d["intent"] = None
                if isinstance(d.get("case_state"), dict):
                    d["case_state"]["chip"] = None
                    d["case_state"]["chip_intent"] = None
                    d["case_state"]["chip_checked"] = False
            self.store.update(draft_id, put)
            return {}

    def _client(self, draft_id: str) -> Any:
        return self.models.client(A.PURPOSE, draft_id=draft_id) if self.models is not None else None

    def _hear(self, draft_id: str, text: str) -> None:
        """The person's words join the conversation as the rules-first Home's do: saved, and sent to the page. The page draws the
        chat only from what is sent, so before this a person's message never showed under the lead agent (the owner, 2026-11-02:
        no prompt in the chat) and the buttons of the question they answered stayed live."""
        msg = {"id": "u" + secrets.token_hex(5), "role": "user", "kind": "text", "text": text.strip()[:2000], "at": now()}
        self.store.update(draft_id, lambda d: d["messages"].append(msg))
        self.store.emit(draft_id, "chat", msg)

    def _save(self, draft_id: str, state: A.CaseState, out: dict[str, Any], *, emit_text: bool = True) -> None:
        snapshot = case_session.snapshot(state)
        view = {**C.views(state.case), "findings": [f.model_dump(mode="json") for f in state.case.findings],
                "route": state.case.route.model_dump(mode="json") if state.case.route else None}

        draw = False  # a prediction route has the solution workflow drawn, once, as the rules-first Home draws it

        def put(d: dict[str, Any]) -> None:
            nonlocal draw
            d["case_state"], d["case"] = snapshot, view
            route = state.case.route
            if route is not None and (d.get("intent") or {}).get("track") != route.track:  # the route is the draft's kind of help
                d["intent"] = {**(d.get("intent") or {}), "track": route.track, "status": "chosen" if route.by == "person" else "read",
                               "source": "user" if route.by == "person" else "case_agent",  # never a person's choice the agent made
                               "by": "case_agent", "tool": route.tool}
            draw = route is not None and route.track == "prediction" and not d.get("workflow")
            if route is None or route.tool not in A.BUILDS:  # the review of 37.11: a build question left open after the route moved on
                d.pop("build_waiting", None)
                d.pop("moment_said", None)
            u = d.setdefault("understanding", {})
            if route is not None and route.tool == "segmentation_project":  # the build reads no outcome for groups (as take_answer records)
                u.update(family="segmentation", family_by="answer")
            elif route is not None and u.get("family") == "segmentation" and u.get("family_by") == "answer":
                u.pop("family", None)  # the review: groups chosen, then a prediction: the build asks for the outcome again
                u.pop("family_by", None)
        self.store.update(draft_id, put)
        if draw:
            self.refresh_workflow(draft_id)  # emits "workflow": without it the page's Solution workflow card stays empty
        for card in out.get("cards") or []:  # the answers and dashboards of this turn, drawn before the words that read them
            self._card(draft_id, card)
        if emit_text and out.get("text"):
            self.say(draft_id, out["text"], kind="text", by="case_agent")
        if out.get("options"):
            self.say(draft_id, "", kind="case_options", options=[o.model_dump(mode="json") for o in out["options"]])

    def _card(self, draft_id: str, card: dict[str, Any]) -> None:
        """One answer or dashboard the lead agent asked for, run and drawn by Home's own renderers (``give_answer``, ``publish_board``):
        the engine runs the plan again on the draft's table, so what the person sees is what code computed, never the agent's text."""
        from dclab_core import dashboard as D
        from dclab_core import query as Q

        from .. import home_query

        try:
            if card.get("kind") == "no_data_solution" and isinstance(card.get("solution"), dict):
                sol = dict(card["solution"])
                self.store.update(draft_id, lambda d: d.update(no_data_solution=sol))
                self.say(draft_id, "", kind="no_data_solution", solution=sol, actions=["ask_simulate", "copy_spec"])
                return
            from . import brief as briefs

            draft = self.store.get(draft_id)
            plans = [Q.QueryPlan.model_validate(card["plan"])] if card.get("kind") == "answer" else [Q.QueryPlan.model_validate(p) for p in card.get("plans") or []]
            marked = sorted({c for p in plans for c in home_query.sensitive_columns(p, briefs.columns_of(draft))})
            if marked:  # the agent's own check came first; this is the page's (the review of 37.11j)
                self.say(draft_id, f"That uses {', '.join(marked)}, which holds personal data; I don't draw it. Ask about another column.", kind="text")
                return
            if card.get("kind") == "answer":
                shown = self.give_answer(draft_id, plans[0], "case_agent")
                if shown.get("kind") != "answer":  # the same question on the same table was answered already: said, never a silent turn
                    self.say(draft_id, "This chart is already above: the same question on the same table.", kind="text")
                return
            kept = ((draft.get("dashboard") or {}).get("board") or {}).get("kpis") or []
            if [Q.plan_hash(k_plan) for k_plan in (Q.QueryPlan.model_validate(k["plan"]) for k in kept)] == [Q.plan_hash(p) for p in plans]:
                return  # the same dashboard is on the page already
            frame = home_query.table(self.store, draft)
            draft = self.store.get(draft_id)  # the table's hash, filled in by the read when it was missing
            active = draft.get("active_asset")
            version = next((a.get("clean_sha256") for a in draft.get("assets") or [] if a.get("id") == active), None)
            board = D.Dashboard(title=str(card.get("title") or "Your dashboard"), refresh="on_new_data", floor=home_query.floor(),
                                kpis=[D.Kpi(name=home_query.describe(p)[:120], plan=p) for p in plans])
            self.publish_board(draft_id, board, frame, version, [], False)
        except Exception as error:  # noqa: BLE001 - a card that cannot be drawn is said in words, never a failed turn
            self.say(draft_id, f"I could not draw that here: {str(error)[:200]}", kind="text")

    def _run(self, draft_id: str, words: str, pick: Optional[str] = None) -> None:
        client = self._client(draft_id)
        draft = self.store.get(draft_id)
        state = self._state(draft)
        state.build = None  # the review of 37.11: a yes counts in the turn it is said (a second yes after a refusal builds again)
        if state.table is not None:
            A._read_table(state)
        try:
            out = A.pick(state, client, pick) if pick else A.turn(state, client, words)
        except RuntimeError as exc:
            self.store.emit(draft_id, "status", {"thinking": False})
            A.record_changes(state)
            self._save(draft_id, state, {}, emit_text=False)
            reason = str(exc).split(":", 1)[-1].strip() if ":" in str(exc) else str(exc)
            if state.table is not None:
                cols = ", ".join(str(c.get("name")) for c in (state.table.get("columns") or [])[:8])
                self.say(draft_id, f"Your table {state.table.get('table', 'data')} ({state.table.get('rows', 0):,} rows, "
                                   f"{len(state.table.get('columns') or [])} columns{': ' + cols if cols else ''}) is attached and ready, "
                                   f"but the AI model could not be reached ({reason}). Check your model API key in .env or switch providers.",
                         kind="text", by="case_agent")
            else:
                self.say(draft_id, f"The AI model could not be reached ({reason}). Check your model API key in .env or switch providers.",
                         kind="text", by="case_agent")
            return
        building = bool(state.build and state.build.get("tool") in A.BUILDS)
        self._save(draft_id, state, out, emit_text=not building)
        if building:
            before = len(self.store.get(draft_id).get("messages") or [])
            self._build(draft_id, outcome=state.build.get("outcome"))  # the person's yes starts the build, as "build it" does on the rules-first Home
            if out.get("text") and len(self.store.get(draft_id).get("messages") or []) == before:
                self.say(draft_id, out["text"], kind="text", by="case_agent")

    BUILD_NEEDS = ("outcome column", "prediction moment")  # "confirm moment" follows the second  # the build's own refusals whose answer code reads (draft/api.py quick_build)

    def _outcome_candidates(self, draft: dict[str, Any]) -> list[str]:
        """The two-valued columns the person's words (or an assumption they confirmed) name: what the model may learn, never a guess."""
        columns = [str(c["name"]) for c in (draft.get("analysis") or {}).get("columns") or [] if c.get("name") and c.get("unique") == 2]
        saved = (draft.get("case_state") or {}).get("case")
        return C.grounded_columns(C.CaseFile.read(saved), columns) if saved and columns else []

    def _stated_moment(self, draft: dict[str, Any]) -> str:
        """The prediction moment the person stated in their own words, or "". Their newest message first, then the
        case's kept words, then the draft's first sentence. Only a sentence that opens by saying when the prediction
        (or the decision) is made, names a time and is not a question: never a guess from other words."""
        saved = (draft.get("case_state") or {}).get("case")
        texts = [str(m.get("text") or "") for m in reversed(draft.get("messages") or []) if m.get("role") == "user"]
        texts += list(reversed(C.CaseFile.read(saved).words)) if saved else []
        texts.append(str(draft.get("problem") or ""))
        for text in texts:
            for sentence in self.SENTENCES.split(text):
                s = sentence.strip()
                found = self.MOMENT_STATED.match(s)
                if not found or "?" in s:
                    continue
                moment = s[found.end():].strip().rstrip(".!").strip()
                if (len(moment.split()) >= 3 and self.MOMENT_OPENS.match(moment) and self.TIME_WORDS.search(moment)
                        and not self.VAGUE_MOMENT.match(moment)):
                    return moment[:400]
        return ""

    def _build(self, draft_id: str, outcome: Optional[str] = None) -> None:
        """The same build step as the button and the rules-first Home's "build it". What it still needs is asked in the chat:
        a column check as the rules-first Home asks it (``built``), the outcome column or the prediction moment as the build's
        own question, whose answer code reads on the next message (``_build_answer``). The outcome the person already named or
        confirmed (the agent's ``outcome``, or the one two-valued column their words name) is written first, so it is not asked again."""
        draft = self.store.get(draft_id)
        if not (draft.get("understanding") or {}).get("target"):
            candidates = self._outcome_candidates(draft)
            column = outcome or (candidates[0] if len(candidates) == 1 else None)
            if column:
                self.store.update(draft_id, lambda d: d.setdefault("understanding", {}).update(target=column))
        draft = self.store.get(draft_id)
        if not (draft.get("understanding") or {}).get("prediction_moment") and not draft.get("build_waiting"):
            # the owner, 2026-11-17 ("both"): a moment the person stated is used at once AND said back, so they can correct it
            moment = self._stated_moment(draft)
            if moment:
                self.store.update(draft_id, lambda d: d.setdefault("understanding", {}).update(prediction_moment=moment))
                self.say(draft_id, f"I am using the prediction moment you gave: {moment}. Tell me if that is wrong.", kind="text", by="case_agent")
        done = self.on_request(draft_id, "build", {})
        if done is None:
            return
        rule = done.get("rule")
        self.store.update(draft_id, lambda d: d.update(build_waiting=rule) if rule in self.BUILD_NEEDS else d.pop("build_waiting", None))
        if rule == "outcome column":  # the build's fixed example is replaced by the table's own candidates
            names = [str(c["name"]) for c in (self.store.get(draft_id).get("analysis") or {}).get("columns") or [] if c.get("name") and c.get("unique") == 2][:3]
            done = {**done, "refused": "Which column is the outcome the model should learn? "
                    + (f"{', '.join(names)} look like yes or no outcomes. " if names else "") + "Name it (for example “the outcome is "
                    + (names[0] if names else "churned") + "”) and I will go on."}
        self.built(draft_id, done)

    # The answer to the said-back moment (37.11b; the moment is the prediction contract, so it is built only on a plain yes):
    # - a hold on the build builds nothing and the agent reads the words;
    # - every word plain approval ("Yes, that is correct, go ahead", "No problem", "Yes, no other changes"): it builds;
    # - anything else is never built: words naming a time ("Yes, it is at checkout, not at signup") are said back as the new
    #   moment, and any other words get the question again. (The first fix built any yes, so a yes naming another moment
    #   built the old one; before it, any yes over four words was said back as a moment and a person answering in
    #   sentences looped.)
    HOLD_BUILD = re.compile(r"\bhold (?:off|on)\b|\b(?:don'?t|do not|not yet|wait|later|never|cancel)\b[^.!?]{0,40}\b(?:build|start|go ahead)|"
                            r"\b(?:build|start)\w*\b[^.!?]{0,20}\b(?:later|tomorrow|next week|yet)\b", re.IGNORECASE)
    # "but" is approval only before a go-ahead: a trailing "Yes, but..." is a correction still being typed (the re-review)
    APPROVAL = re.compile(r"\b(?:but (?:go ahead|build it|do it|please)|at once|right away|right now|no problem|no other changes?|no changes?|"
                          r"nothing else(?: to add)?|go ahead|build it|do it|let'?s go|"
                          r"yes|yeah|yep|yup|sure|ok|okay|correct|right|exactly|perfect|great|good|fine|true|that'?s|that|this|it'?s|it|is|"
                          r"sounds|looks|seems|all|to|me|please|now|then|so|and|actually|build|start|proceed|with)\b", re.IGNORECASE)
    ASKS = re.compile(r"^\W*(?:is|are|was|does|do|did|should|shall|can|could|would|will|what|why|how|which)\b", re.IGNORECASE)
    YES_WORD = re.compile(r"\b(?:no problem|yes|yeah|yep|yup|sure|ok|okay|correct|right|exactly|perfect|fine|true|go ahead|build|proceed|do it|let'?s go)\b",
                          re.IGNORECASE)
    TIME_WORDS = re.compile(r"(?<!look )(?<!looking )\bat\b(?! once\b| all\b| the moment\b(?! of))|\b(?:when|after|before|once|upon|during|moment|time|day|days|week|weeks|month|months|hour|hours)\b",
                            re.IGNORECASE)
    VAGUE_MOMENT = re.compile(
        r"^\W*(?:(?:yes|yeah|yep|yup|ok(?:ay)?|sure)\b\W*)*"
        r"(?:at\s+(?:the|this)\s+moment(?:\s+of)?|right\s+now|now|currently|today|at\s+present|for\s+now|the\s+moment)\W*$",
        re.IGNORECASE,
    )
    LEADING_ANSWER = re.compile(r"^\W*(?:(?:yes|yeah|yep|ok(?:ay)?|sure|no|nope|not(?: quite)?|wait|actually|sorry|rather|instead)\b\W*)*"
                                r"(?:but\b\W*)?", re.IGNORECASE)
    # A sentence in which the person themselves states when the prediction is made ("The prediction is made at the
    # monthly review", "We predict when the order is placed"). Live, 2026-11-16: the person said it in the same message
    # as "Build it", nothing carried it to the build, and Home asked for it again.
    # The review of 2026-11-18: "is" alone was enough, so "The score is higher when tenure is long" was read as a moment.
    # The verb must now say the prediction is MADE (or happens, or runs).
    MOMENT_STATED = re.compile(
        r"^\W*(?:and\s+|so\s+|also\s+)?"
        r"(?:(?:the\s+)?(?:predictions?|forecasts?|scores?|scoring|decisions?)\s+"
        r"(?:(?:is|are|gets|get|will\s+be|should\s+be)\s+(?:made|done|run|taken|produced|computed)|happens?|runs?)"
        r"|we\s+(?:predict|score|forecast|decide))\s*",
        re.IGNORECASE)
    # What follows must open with a time ("at ...", "when ...", "every ..."). The review of 2026-11-18: "We predict churn
    # for customers before renewal" names what is predicted, and it was built on at once as the moment.
    MOMENT_OPENS = re.compile(r"^(?:at|when|whenever|before|after|once|on|every|each|during|upon|right\s+(?:before|after)|as\s+soon\s+as|by\s+the\s+time)\b",
                              re.IGNORECASE)
    SENTENCES = re.compile(r"(?<=[.?!;])\s+|\n+")

    def _moment_of(self, text: str) -> str:
        """The moment in an answer, without the sentences that are an order to build or plain approval, and without an
        opening "The prediction is made". Live, 2026-11-16: "…on that day. Build it now." was stored whole, so the
        project's prediction moment ended with "Build it now."."""
        from .chat import build_order

        kept = []
        for sentence in self.SENTENCES.split(str(text or "").strip()):
            s = sentence.strip()
            if not s or build_order(s) or not re.sub(r"[\W_]+", " ", self.APPROVAL.sub(" ", s)).strip():
                continue  # empty, an order to build, or only words of approval
            kept.append(self.MOMENT_STATED.sub("", s, count=1).strip())
        return " ".join(k for k in kept if k).strip()

    def _say_moment_back(self, draft_id: str, moment: str) -> None:
        moment = moment.strip()[:400]
        self.store.update(draft_id, lambda d: d.update(build_waiting="confirm moment", moment_said=moment))
        # the person's own "using only what is known ..." is not said a second time
        tail = "" if re.search(r"\bknown\b", moment, re.IGNORECASE) else ", using only what is known then"
        self.say(draft_id, f"So the prediction is made {moment.rstrip('.')}{tail}. Shall I build with that? "
                           "Say yes, or tell me the moment again.", kind="text", by="case_agent")
    OUTCOME_SAID = re.compile(r"^\s*(?:(?:the\s+)?(?:outcome|target|label|answer)(?:\s+column)?\s*(?:is|=|:)\s*)?[`'\"]?([\w.\- ]+?)[`'\"]?\s*[.!]?\s*$",
                              re.IGNORECASE)

    def _build_answer(self, draft_id: str, text: str) -> bool:
        """The person's answer to the build's question, read by code only when it plainly is one (the review of 37.11), else
        False and the agent reads the words:

        - the outcome: the whole message is a column's name, or "the outcome is <column>", with exactly one column and no
          question in it;
        - the prediction moment: a sentence with no question in it is said back, and the build runs only on the person's
          yes to it (a moment is the prediction contract: never taken from words that only look like one)."""
        draft = self.store.get(draft_id)
        need, said = draft.get("build_waiting"), text.strip().replace("’", "'").replace("‘", "'")  # phone keyboards' apostrophes
        if "?" in said or self.ASKS.match(said):  # "Is that right" is a question without its mark: the agent reads it
            return False
        if need == "outcome column":
            columns = {str(c.get("name")).casefold(): str(c.get("name")) for c in (draft.get("analysis") or {}).get("columns") or [] if c.get("name")}
            m = self.OUTCOME_SAID.match(said)
            column = columns.get(m.group(1).strip().casefold()) if m else None
            if column is None and A.DELEGATE.search(said) and not (A.HOLD.search(said) or A.REFUSE_BUILD.search(said) or A.NOT_NOW.search(said)):
                # "you decide": the one column their own words name, never a guess (and never in a message that holds the build back)
                grounded = self._outcome_candidates(draft)
                if len(grounded) == 1:
                    column = grounded[0]
                    self.say(draft_id, f"I will use {column} as the outcome: it is the column your own words name.", kind="text", by="case_agent")
                elif len(grounded) > 1:
                    self.say(draft_id, f"Your words name more than one yes-or-no column ({', '.join(grounded[:3])}). Which one is the outcome?", kind="text", by="case_agent")
                    return True
            if column is None:
                return False
            self.store.update(draft_id, lambda d: (d.setdefault("understanding", {}).update(target=column), d.pop("build_waiting", None)))
            self._build(draft_id)
            return True
        if need == "prediction moment":
            moment = self._moment_of(said)
            if len(moment.split()) < 3 or self.VAGUE_MOMENT.match(moment):
                return False
            self._say_moment_back(draft_id, moment)
            return True
        if need == "confirm moment":
            if self.HOLD_BUILD.search(said):
                self.store.update(draft_id, lambda d: (d.pop("build_waiting", None), d.pop("moment_said", None)))
                return False
            left = re.sub(r"[\W_]+", " ", self.APPROVAL.sub(" ", said)).strip()
            if left or not self.YES_WORD.search(said):
                # not a plain yes: never built. A time named is the new moment, said back ("Yes, but it is at checkout");
                # other words get the question again, with the moment kept
                moment = self._moment_of(self.LEADING_ANSWER.sub("", said, count=1))
                if self.TIME_WORDS.search(moment) and not self.VAGUE_MOMENT.match(moment) and len(moment.split()) >= 2:
                    self._say_moment_back(draft_id, moment)
                elif not self.YES_WORD.search(said):  # no yes and no moment ("let us look at groups instead"): the agent reads it
                    self.store.update(draft_id, lambda d: (d.pop("build_waiting", None), d.pop("moment_said", None)))
                    return False
                else:
                    self.say(draft_id, f"I am not sure that is a yes. Shall I build with the prediction made "
                                       f"{str(draft.get('moment_said') or '').rstrip('.')}? Say yes, or tell me the moment again.",
                             kind="text", by="case_agent")
                return True
            moment = str(draft.get("moment_said") or "")
            self.store.update(draft_id, lambda d: (d.setdefault("understanding", {}).update(prediction_moment=moment),
                                                   d.pop("build_waiting", None), d.pop("moment_said", None)))
            self._build(draft_id)
            return True
        return False

    def _emit_analysis_and_sync_table(self, draft_id: str, asset: dict[str, Any]) -> tuple[A.CaseState, bool]:
        """Ensure the ready table is recorded on the case file and its analysis card is shown once. Returns (state, already_read)."""
        draft = self.store.get(draft_id)
        state = self._state(draft)
        already_read = bool(
            state.table is not None
            and state.case.data is not None
            and state.case.data.status == "computed"
            and state.case.data.value.startswith(f"{state.table['table']}:")
            and state.turns
        )
        if state.table is not None:
            A._read_table(state)
            A.record_changes(state)
            snapshot = case_session.snapshot(state)
            view = {**C.views(state.case), "findings": [f.model_dump(mode="json") for f in state.case.findings],
                    "route": state.case.route.model_dump(mode="json") if state.case.route else None}
            self.store.update(draft_id, lambda d: d.update(case_state=snapshot, case=view))
        aid = asset.get("id")
        has_card = any(m.get("kind") == "analysis" and m.get("asset") == aid for m in (self.store.get(draft_id).get("messages") or []))
        if not has_card:
            analysis = draft.get("analysis") or {}
            s = analysis.get("summary") or {}
            steps = len(draft.get("cleaning_log") or [])
            label = "synthetic " if asset.get("synthetic") else ""
            cols = [str(c.get("name")) for c in (analysis.get("columns") or []) if c.get("name")]
            col_preview = (", ".join(cols[:8]) + (f" and {len(cols) - 8} more" if len(cols) > 8 else "")) if cols else ""
            text = (f"Your {label}table {asset.get('name', 'data')} is ready: {s.get('rows', 0):,} rows and {s.get('columns', 0)} columns"
                    + (f", after {steps} cleaning step{'s' if steps != 1 else ''}" if steps else "")
                    + (f" ({col_preview})." if col_preview else ".")
                    + " Here is a first look.")
            self.say(draft_id, text, kind="analysis", asset=aid, by="case_agent")
        return state, already_read

    @turn  # the review of 37.11: one turn at a time per draft, seen by every instance (the rules-first Home's own lock)
    def start(self, draft_id: str) -> None:
        client = self._client(draft_id)
        if client is None:
            return super().start(draft_id)
        draft = self.store.get(draft_id)
        problem = str(draft.get("problem") or "")
        if not hasattr(client, "replies") and (
            problem.startswith("Help me understand and analyze ")
            or any(a.get("status") in ("queued", "structuring", "cleaning", "analysing") for a in (draft.get("assets") or []))
        ):
            import time

            deadline = time.monotonic() + 8.0
            while time.monotonic() < deadline:
                cur = self.store.get(draft_id)
                assets = cur.get("assets") or []
                if any(a.get("status") in ("ready", "failed") for a in assets):
                    break
                if not assets and time.monotonic() > deadline - 7.2:
                    break
                time.sleep(0.05)
            cur = self.store.get(draft_id)
            ready_asset = next((a for a in (cur.get("assets") or []) if a.get("id") == cur.get("active_asset") and a.get("status") == "ready"), None)
            if ready_asset is not None:
                self._emit_analysis_and_sync_table(draft_id, ready_asset)
        self._run(draft_id, problem)

    @turn
    def reply(self, draft_id: str, text: str) -> None:
        if self._client(draft_id) is None:  # no model: the rules-first Home answers, as with the switch off
            return super().reply(draft_id, text)
        from . import column_check as checks

        draft = self.store.get(draft_id)
        open_question = self.pending(draft)
        if open_question is not None and open_question.get("field") == checks.FIELD:
            # the review of 37.11: the build's column question is answered by code (known, written later, or DCLab's call),
            # then the build goes on, as on the rules-first Home; the agent never reads it
            self._hear(draft_id, text)
            return self.take_answer(draft_id, open_question, text)
        if draft.get("build_waiting"):
            self._hear(draft_id, text)
            if self._build_answer(draft_id, text):
                return
            return self._run(draft_id, text)
        offered = [o.get("title") for o in ((draft.get("case_state") or {}).get("case") or {}).get("options") or []]
        if not (draft.get("case_state") or {}).get("case", {}).get("route") and text in offered:
            self._hear(draft_id, text)
            return self._run(draft_id, text, pick=text)
        self._hear(draft_id, text)
        self._run(draft_id, text)

    @turn
    def data_ready(self, draft_id: str, asset: dict[str, Any]) -> None:
        """A table is ready: record it on the case file, emit the analysis card, and let the lead agent read and explain the uploaded data."""
        self.store.update(draft_id, lambda d: (d.pop("build_waiting", None), d.pop("moment_said", None)))
        state, already_read = self._emit_analysis_and_sync_table(draft_id, asset)
        client = self._client(draft_id)
        if client is not None and state.table is not None and not already_read and not (hasattr(client, "replies") and not client.replies):
            self._run(draft_id, "I uploaded the table.")


__all__ = ["CaseHome", "table_of"]
