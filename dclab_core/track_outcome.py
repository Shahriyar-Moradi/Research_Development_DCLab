"""What a person did with what Home produced (package 36.2 of the training-data plan).

A kind of help that is not a prediction ends with something a person can judge: an answer, a prepared table, a brief, a
dashboard. ``read`` turns the ordered names of what was produced and what the person then did into one ``TrackOutcome``:

- **kept**: they saved the answer, took the table or the notebook, said yes to a guarded step, or gave the reply a thumb up;
- **changed**: they edited the answer's controls or changed the recipe;
- **refused**: they said no to a step, took a step back, let a saved answer go, or gave the reply a thumb down;
- **asked_again**: they asked for another result without judging the one before (read by the rules, not said by them);
- **open**: something was produced and nothing was done with it yet (no verdict: nothing to learn from).

The last decisive action wins, and a kept result stays kept while the next one is pending. A verdict from an action of the
person's is ``by="person"``; one the rules inferred (open, asked_again) is ``by="rule"``. Only names and counts are kept,
never what an answer showed. The rule was written and tested by the Home session; it lives here so Home, the ledger's
capture and the store read one rule.

One verdict per conversation: the last decisive one. Earlier results in the same conversation lose theirs (a limit of v1).
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field, model_validator

from dclab_core.contracts import CONTRACTS, Versioned, _version

Verdict = Literal["kept", "changed", "refused", "asked_again", "open"]
VERDICTS = ("kept", "changed", "refused", "asked_again")  # the verdicts that are an outcome (open is not one yet)
CONFIRMING = ("kept", "changed")  # the verdicts that confirm the routing taken (a person's label on the exit and the reading)
BY_PERSON = ("kept", "changed", "refused")  # decided by an act of the person's; asked_again and open are the rules' reading

#: what Home produced for the person to judge
PRODUCED = ("answer", "table", "brief", "dashboard")
#: what the person did with it, and the verdict each action gives
DECISIVE: dict[str, str] = {
    "save": "kept", "export": "kept", "notebook": "kept", "step_yes": "kept", "thumb_up": "kept",
    "edit": "changed", "change": "changed",
    "unsave": "refused", "step_no": "refused", "undo": "refused", "thumb_down": "refused",
}  # a dashboard refresh only recomputes its tiles: no verdict (a person may refresh because the numbers look wrong)


class TrackOutcome(Versioned):
    schema_version: Literal["dclab_core.track_outcome.v1"] = _version("track_outcome")
    track: str = Field(max_length=40, description="the kind of help (analysis, preparation, reporting, business, ...)")
    produced: list[str] = Field(default_factory=list, max_length=len(PRODUCED), description="the kinds of result produced, in order")
    verdict: Verdict
    by: Literal["person", "rule"] = Field(description="person: an action of theirs decided it; rule: inferred from the order of requests")
    actions: int = Field(ge=0, description="how many known actions were counted")
    evidence: list[str] = Field(default_factory=list, max_length=5, description="the last actions' names, never their content")

    @model_validator(mode="after")
    def _who_decided(self) -> "TrackOutcome":  # a rule never keeps, changes or refuses: only the person's own act does
        if (self.by == "person") != (self.verdict in BY_PERSON):
            raise ValueError(f"a {self.verdict} verdict is decided by {'the person' if self.verdict in BY_PERSON else 'the rules'}, not by {self.by}")
        return self


CONTRACTS["track_outcome"] = TrackOutcome


def read(track: str, actions: list[str]) -> Optional[TrackOutcome]:
    """The outcome of a conversation from its ordered actions; None while nothing was produced (a verdict on nothing is no
    outcome). Names it does not know are ignored."""
    produced: list[str] = []
    verdict, by, pending, known = "open", "rule", False, []
    for a in actions:
        if a in PRODUCED:
            if pending:  # another result asked for before the last one was judged
                verdict, by = "asked_again", "rule"
            if a not in produced:
                produced.append(a)
            pending = True
            known.append(a)
        elif a in DECISIVE and produced:  # a verdict on a result that exists
            verdict, by, pending = DECISIVE[a], "person", False
            known.append(a)
    if not produced:
        return None
    return TrackOutcome(track=track[:40], produced=produced, verdict=verdict, by=by, actions=len(known), evidence=known[-5:])


__all__ = ["TrackOutcome", "Verdict", "VERDICTS", "CONFIRMING", "BY_PERSON", "PRODUCED", "DECISIVE", "read"]
