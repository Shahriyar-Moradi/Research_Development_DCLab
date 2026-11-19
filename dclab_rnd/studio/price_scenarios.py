"""Price studies on explicitly attached historical offers, event histories and matured outcomes."""

from __future__ import annotations

import hashlib
import json

from dclab_core.price_scenarios import fit_study, score_study

from .data import load_tables

TABLES = ["price_offers", "price_events", "price_outcomes"]


def sources(project):
    return {name: (project.get("tables", {}).get(name) or {}).get("sha256") for name in TABLES}


def prepare(store, project_id, spec):
    project = store.get(project_id)
    hashes = sources(project)
    if not all(hashes.values()):
        raise ValueError("Attach price_offers, price_events and price_outcomes first.")
    tables = load_tables(store, project, TABLES)
    result = fit_study(*(tables[name] for name in TABLES), spec)
    identity = {"format": 1, "tables": hashes, "spec": spec.model_dump()}
    result["version"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    result["sources"] = hashes

    with store.turn(project_id):
        if sources(store.get(project_id)) != hashes:
            raise ValueError("A source table changed while preparing the study. Prepare it again.")
        store.update(project_id, price_study=result)
    store.log(project_id, "price_study_prepared", {"version": result["version"], "source": result["source"]})
    return {"version": result["version"], "source": result["source"], "audience": result["audience"]}


def read(project, target, independent):
    study = project.get("price_study")
    if not study or study["sources"] != sources(project):
        raise LookupError("Prepare the price study for the current table versions first.")
    return score_study(study, target, independent)
