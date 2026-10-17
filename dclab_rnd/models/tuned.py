"""Serve DCLab's own trained model where it measured better (package 27.8), role by role, through the existing switches.

The owner's trained model (the decision model of 26.4 or the solver of 27.7) is served behind the gateway's ``tuned`` tier
(``DCLAB_TIER_TUNED_BASE_URL`` and ``DCLAB_TIER_TUNED_MODEL``: vLLM, llama.cpp's server or Ollama on this machine). Each role
goes the same way, and every step is the audited switch that already exists (``models.routing.switch``), never a shortcut:

1. ``shadow(purpose)``: the tuned model answers beside the role's model on every request, logged, never used (only a model on
   this machine may shadow: its requests cost nothing a cap does not see);
2. ``plan(purpose)``: what the role still needs before the tuned model may serve it: the tier configured, its shadow's
   agreement on real requests, the harness suite passed with it (21.5), and, for a role that reads what the person means, a
   measurement on the same conversations showing it better than the model serving now (21.4);
3. ``promote(purpose, measured, reason)``: refused while ``plan`` lists anything missing, then the switch to serve under the
   routing's own rules;
4. ``retire(purpose, reason)``: the model that served before the tuned one serves again (its approval still holding), else
   the role's own tier.

    python -m dclab_rnd.models.tuned plan|shadow|promote|retire PURPOSE [--reason ...] [--measured model=score,...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

from . import settings
from .routing import RoutingError, approved, open_routing, route, switch
from .shadow import agreement, open_shadows


def plan(purpose: str, home: Path) -> dict[str, Any]:
    """Where a role stands on the way to being served by the tuned model, and what is still missing before ``promote``.
    ``ready``: nothing is missing but, for a role that reads what the person means, the measurement given to promote
    (``needs_measurement``)."""
    from ..agent_eval.regression import passed_with
    from .routing import UNDERSTANDING

    settings.purpose(purpose)  # an unknown purpose is an error
    tier = settings.public(settings.TUNED)
    model = str(tier.get("model") or "")
    current = route(open_routing(home).get(), purpose)
    shadows = [a for a in agreement([r for r in open_shadows(home).rows() if r.get("purpose") == purpose])
               if model and a["shadow_model"] == model]
    seen = shadows[0] if shadows else None
    if current.get("serve") == settings.TUNED and approved(current):  # the gateway serves it: the approval still holds
        return {"purpose": purpose, "tuned_model": model, "routing": current, "shadow_agreement": seen, "harness_suite": None,
                "missing": [], "needs_measurement": False, "ready": True, "serving": True}
    missing = []
    if current.get("serve") == settings.TUNED:  # routed, but to another model than the one configured now: not served
        missing.append(f"the approval was for {current.get('serve_model')}, not {model or 'an unset tuned tier'}: the role's own model serves it")
    if not model:
        missing.append("the tuned tier is not configured (DCLAB_TIER_TUNED_BASE_URL, DCLAB_TIER_TUNED_MODEL)")
    elif not tier.get("local"):
        missing.append("the tuned tier is not on this machine: only a local model may shadow a role first")
    if current.get("shadow") != settings.TUNED:
        missing.append("the tuned model does not shadow this role (python -m dclab_rnd.models.tuned shadow PURPOSE)")
    if seen is None or not seen["requests"]:
        missing.append("no shadowed request yet: its agreement with the serving model is unknown")
    elif seen["errors"] >= seen["requests"]:
        missing.append(f"every one of its {seen['requests']} shadowed requests failed")
    suite = passed_with(purpose, model) if model else None
    if suite is None:
        missing.append("the harness suite has not passed with it serving this role (make harness-eval with the tuned tier serving it)")
    return {"purpose": purpose, "tuned_model": model or None, "endpoint": tier.get("endpoint"), "routing": current,
            "shadow_agreement": seen, "harness_suite": suite["id"] if suite else None, "missing": missing,
            "needs_measurement": purpose in UNDERSTANDING, "ready": not missing, "serving": False}


def shadow(purpose: str, home: Path, audit_log: Any = None, role: str = "owner", on: bool = True) -> dict[str, Any]:
    return switch(open_routing(home), audit_log, purpose, "shadow", settings.TUNED if on else None, role,
                  "27.8: the owner's tuned model shadows this role" if on else "27.8: the tuned model's shadow ends")


def promote(purpose: str, home: Path, reason: str, measured: Optional[dict[str, float]] = None, audit_log: Any = None,
            role: str = "owner") -> dict[str, Any]:
    """The tuned model serves the role, only after it shadowed the role on this machine (``plan`` with nothing missing),
    and then by the routing's own rules: the suite passed with it and, for an understanding role, ``measured`` showing
    it better on the same conversations."""
    p = plan(purpose, home)
    if p["serving"]:
        return open_routing(home).get()
    if p["missing"]:
        raise RoutingError("the tuned model may not serve " + purpose + " yet: " + "; ".join(p["missing"]))
    return switch(open_routing(home), audit_log, purpose, "serve", settings.TUNED, role, reason, measured=measured)


def retire(purpose: str, home: Path, reason: str, audit_log: Any = None, role: str = "owner") -> dict[str, Any]:
    """The earlier model serves again: the tier the role was served by before the tuned model, while that approval still
    holds and the routing's rules still allow it; otherwise the role's own tier."""
    store = open_routing(home)
    doc = store.get()
    if route(doc, purpose).get("serve") != settings.TUNED:
        return doc
    before = next((h for h in reversed(doc.get("history") or []) if h.get("purpose") == purpose and h.get("setting") == "serve"
                   and h.get("to") == settings.TUNED), None)
    earlier = (before or {}).get("from")
    if earlier and earlier != settings.TUNED:
        earlier_approval = next((h for h in reversed(doc.get("history") or []) if h.get("purpose") == purpose and h.get("setting") == "serve"
                                 and h.get("to") == earlier and h.get("model")), None)
        if earlier_approval and approved({"serve": earlier, "serve_model": earlier_approval["model"], "serve_endpoint": earlier_approval.get("endpoint")}):
            measured = ((earlier_approval.get("evidence") or {}).get("measured")) or None
            try:  # its measurement did not include the tuned model: putting it back is the owner's override, audited as one
                return switch(store, audit_log, purpose, "serve", earlier, role, reason, measured=measured, override=role == "owner")
            except (RoutingError, PermissionError):
                pass  # the earlier tier no longer passes the rules: the role's own tier, which needs none
    return switch(store, audit_log, purpose, "serve", None, role, reason)


def _measured(text: str) -> Optional[dict[str, float]]:
    out = {}
    for pair in (x.strip() for x in text.split(",") if x.strip()):
        name, sep, value = pair.partition("=")
        try:
            out[name.strip()] = float(value)
        except ValueError:
            raise RoutingError(f"--measured is model=score,...: {pair!r}") from None
        if not sep or not name.strip():
            raise RoutingError(f"--measured is model=score,...: {pair!r}")
    return out or None


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    from .. import audit
    from ..settings import Settings

    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.models.tuned", description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=("plan", "shadow", "unshadow", "promote", "retire"))
    parser.add_argument("purpose")
    parser.add_argument("--reason", default="")
    parser.add_argument("--measured", default="", help="model=score,... on the same conversations (an understanding role)")
    args = parser.parse_args(argv)
    home = Path(Settings().agent_home)
    log = audit.open_audit(home)
    try:
        if args.command == "plan":
            print(json.dumps(plan(args.purpose, home), indent=1, default=str))
        elif args.command in ("shadow", "unshadow"):
            shadow(args.purpose, home, log, on=args.command == "shadow")
            print(json.dumps(route(open_routing(home).get(), args.purpose), indent=1))
        elif args.command == "promote":
            promote(args.purpose, home, args.reason, _measured(args.measured), log)
            print(json.dumps(route(open_routing(home).get(), args.purpose), indent=1))
        else:
            retire(args.purpose, home, args.reason or "27.8: back to the role's own model", log)
            print(json.dumps(route(open_routing(home).get(), args.purpose), indent=1))
    except (RoutingError, PermissionError) as e:
        print(f"refused: {e}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
