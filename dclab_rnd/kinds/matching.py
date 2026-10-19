"""Matching (package 30.16, the owner's pattern 4): which records are the same thing; a copied bill and the bill it copies,
one product in two catalogues, one paper in two libraries. One row of the problem is a pair of records.

The known matches are the project's related table ``matches`` (two id columns: the duplicate and its original, or a left
record and a right one); a second table to link to is attached by name (``right_table``), else the duplicates are within
the main table. The engine runs it as its own machine (``MatchingMachine``): the split by entity is locked in the data
stage (a record and every record it matches on one side), the compared fields and the blocking are the person's, the
models are chosen on the validation records, and the holdout's matches are read once, in the final stage. The library is
``dclab_core.matching``."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from .base import ProblemKind, _get
from .machine import Machine, claim

ROOT = Path(__file__).resolve().parents[2]
REAL = ROOT / "data" / "downloads" / "entity_resolution"  # the owner's download (2026-10-19): the Leipzig benchmarks, gitignored


class Compared(BaseModel):
    model_config = ConfigDict(extra="forbid")
    column: str
    kind: Literal["text", "number", "time", "category", "flag"] = "text"  # flag: a 0/1 fact of one record (a bill its seller returned)
    right: Optional[str] = None  # the right table's column, when it is named otherwise
    key: Optional[bool] = None   # part of the exact rule (every key field equal); by default a text field is


class MatchingFields(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id_column: str
    fields: list[Compared] = Field(min_length=1, max_length=20)
    matches_table: str = "matches"   # the known matches, attached by name: two id columns
    right_table: Optional[str] = None  # a second table to link to; none: duplicates within the main table
    right_id_column: Optional[str] = None
    block_equal: list[str] = Field(default_factory=list, max_length=5)
    window_column: Optional[str] = None
    window_minutes: int = Field(default=120, ge=1, le=60 * 24 * 365)
    block_text: Optional[str] = None
    top_k: int = Field(default=10, ge=1, le=100)


class Matching(ProblemKind):
    key = "matching"
    title = "Matching records"
    description = "Which records are the same thing: a copied bill and its original, one product or person under two entries."
    base_task = "binary"
    task_type = "matching"
    specificity = 15
    packs = ()
    # the review: "the export has duplicate rows" or "deduplicate the catalogue first" is cleaning, never this kind's problem
    words = (r"\b(copied (?:bills|invoices|receipts)|record linkage|entity resolution|match(?:ing)? (?:records|customers|products) across"
             r"|same (?:customer|product|person|record) (?:under two|in two (?:systems|catalogues|lists)))\b")
    questions = ("Which column names each record, and which fields say two records are the same (a name, an amount, a time)?",
                 "Which pairs are known to match (a table of the duplicate and its original, or of the two ids)?",
                 "Which records may be compared at all (the same store, the same day, a similar name)?")
    metrics = (("pair_f1", "Pair F1"), ("pair_precision", "Pair precision"), ("pair_recall", "Pair recall"),
               ("average_precision", "Average precision over the candidate pairs"))
    baselines = ("exact", "single_field")
    checks = ("PIT-005",)
    reliability = ("blocking's recall: the true pairs it keeps", "the exact rule and the single best field beside the model",
                   "a record and its matches on one side of the split")
    evidence_track = "record-linkage"
    aliases = ("record linkage", "entity resolution", "deduplication", "duplicate detection", "near-duplicate matching")
    stratify = False
    runs_in_engine = True
    needs_target = False  # the outcome is whether a pair matches, read from the matches table
    export = {"notebook": False, "service": False}
    Fields = MatchingFields

    def own_machine(self) -> Any:
        return MatchingMachine

    def columns_of(self, solution: Any) -> list[str]:
        f = self.fields(_get(solution, "kind_fields"))
        main = [f["id_column"], *(x["column"] for x in f["fields"]), *f["block_equal"], *([f["window_column"]] if f["window_column"] else []),
                *([f["block_text"]] if f["block_text"] else [])]
        return list(dict.fromkeys(main))

    def tables_of(self, solution: Any) -> list[str]:
        f = self.fields(_get(solution, "kind_fields"))
        return [f["matches_table"], *([f["right_table"]] if f["right_table"] else [])]

    def table_columns_of(self, solution: Any) -> dict[str, list[str]]:
        f = self.fields(_get(solution, "kind_fields"))
        if not f["right_table"]:
            return {}
        compared = {x["column"]: (x["right"] or x["column"]) for x in f["fields"]}
        blocks = [*f["block_equal"], *([f["window_column"]] if f["window_column"] else []), *([f["block_text"]] if f["block_text"] else [])]
        return {f["right_table"]: list(dict.fromkeys([f["right_id_column"] or f["id_column"], *compared.values(), *(compared.get(c, c) for c in blocks)]))}

    def check_solution(self, solution: Any) -> None:
        f = self.fields(solution.kind_fields)
        if f["id_column"] in {x["column"] for x in f["fields"]}:
            raise ValueError("The record's id names a record; it is never a field compared")
        used = {x["column"] for x in f["fields"]} | set(f["block_equal"]) | {c for c in (f["window_column"], f["block_text"]) if c}
        refused = used & ({x.column for x in solution.forbidden} | set(solution.identifiers or []))
        if refused:  # rule 1: what the solution forbids, or names an identifier, never reaches a pair's features or its blocking
            raise ValueError(f"{', '.join(sorted(refused))} is forbidden or an identifier in the solution: it is never compared or blocked on")
        if not (f["block_equal"] or f["window_column"] or f["block_text"]):
            raise ValueError("Say which records may be compared (block columns, a time window or a text to search by): every pair of a table is too many")
        if solution.metric and solution.metric not in self.metric_keys():
            raise ValueError(f"The matching machine is judged by {', '.join(self.metric_keys())}, not {solution.metric}")

    def default_metric(self, solution: Any, positive_rate: float | None, pack: dict[str, Any] | None = None) -> str:
        return "pair_f1"

    def applies(self, solution: Any) -> bool:
        return _get(solution, "kind") == self.key

    # ------------------------------------------------------------------ acceptance: the copied bills, and the real benchmarks
    # the exact rule is the plan's link: the same items, and a bill its seller returned (the replica edits a copy's amount, so
    # the amount is a similarity, never a key)
    BILL_FIELDS = [{"column": "amount", "kind": "number", "key": False}, {"column": "items", "kind": "category", "key": True},
                   {"column": "store_id", "kind": "category", "key": False}, {"column": "sold_at", "kind": "time", "key": False},
                   {"column": "cashier_id", "kind": "category", "key": False}, {"column": "payment", "kind": "category", "key": False},
                   {"column": "returned_by_seller", "kind": "flag", "key": True}]

    def case(self, seed: int = 0) -> Any:
        """The retail replica's sales: copied bills (a new bill minutes after a real one, the amount edited a little,
        sometimes in another store) among honest repeat purchases; the known matches are the copies and their originals.
        Each bill carries the plan's exact link, whether it was returned by the cashier who sold it (from the returns
        table, which the business holds when it checks a return): a copy always is, an honest bill rarely."""
        from dclab_core.replicas import retail_returns

        from .acceptance import Case

        replica = retail_returns.generate(seed)
        truth = replica.truth
        copies = truth[truth["copy_of"].astype(str) != ""][["bill_id", "copy_of"]].drop_duplicates().reset_index(drop=True)
        sales = replica.sales[[c for c in replica.sales.columns if not c.startswith("_")]].copy()
        returned = sales[["bill_id", "cashier_id"]].merge(replica.returns[["bill_id", "cashier_id"]].rename(columns={"cashier_id": "returned_by"}),
                                                           on="bill_id", how="left")
        by_seller = (returned["returned_by"] == returned["cashier_id"]).groupby(returned["bill_id"]).max()
        sales["returned_by_seller"] = sales["bill_id"].map(by_seller).fillna(False).astype(int)  # never the label: the return, not the verdict
        fields = {"id_column": "bill_id", "fields": self.BILL_FIELDS, "block_equal": ["items"], "window_column": "sold_at", "window_minutes": 120}
        return Case(sales, None, "binary", "At the end of a review period: each bill against the bills sold around it, with every return filed against it.",
                    solution={"kind": "matching", "kind_fields": fields}, tables={"matches": copies},
                    facts={"seed": seed, "copies": int(len(copies)), "repeat_share": 0.03, "source": "retail_returns", "family": "rare_event", "labels": "file"})

    def real_cases(self) -> list[Any]:
        """The real benchmarks the owner downloaded (DBLP-ACM, Abt-Buy), when they are on this machine."""
        from .acceptance import Case

        out = []
        for name, left, right, pairs, fields, block in (
                ("DBLP-ACM", "DBLP2.csv", "ACM.csv", "DBLP-ACM_perfectMapping.csv",
                 [{"column": "title"}, {"column": "authors", "key": False}, {"column": "venue", "key": False}, {"column": "year", "kind": "number", "key": False}], "title"),
                ("Abt-Buy", "Abt.csv", "Buy.csv", "abt_buy_perfectMapping.csv",
                 [{"column": "name"}, {"column": "description", "key": False}, {"column": "price", "kind": "number", "key": False}], "name")):
            folder = REAL / name / "files"
            if not (folder / left).is_file():
                continue
            read = lambda p: _read_csv(folder / p)  # noqa: E731
            fieldset = {"id_column": "id", "fields": fields, "right_table": "right", "right_id_column": "id", "block_text": block, "top_k": 10}
            out.append(Case(read(left), None, "binary", "When two catalogues are joined, record by record.",
                            solution={"kind": "matching", "kind_fields": fieldset}, tables={"right": read(right), "matches": read(pairs)},
                            facts={"source": name, "real": True, "family": "entity_tables", "labels": "column"}))
        return out

    def solve(self, case: Any) -> Any:
        from dclab_core import matching

        f = self.fields(case.solution.get("kind_fields"))
        return matching.evaluate(case.table, (case.tables or {}).get(f["right_table"]) if f["right_table"] else None, **_library_args(f, case.tables["matches"]))

    PRECISION, RECALL = 0.5, 0.4  # the copied bills' trap: most links it makes are copies, and it finds a good share of them

    def accept(self, run: Any) -> list[dict[str, Any]]:
        from .acceptance import check

        v, h = run["validation"], run["holdout"]
        return [check("blocking keeps the true pairs of the training and validation records", all((b or 0) >= 0.9 for b in v["blocking"].values()),
                      str(v["blocking"])),
                check("the model is chosen on the validation records, never the holdout", v["chosen"] in v["scores"], v["chosen_by"]),
                check("the chosen model beats the exact rule and the single best field on the holdout, its interval above zero",
                      all(h["beats"].values()) and bool(h["beats"]), f"F1 {h['pair_f1']}; " + "; ".join(
                          f"{n} {h['scores'][n]['f1']} (difference {h['scores'][n].get('f1_difference_ci90')})" for n in ("exact", "single_field") if n in h["scores"])),
                check("most links it makes are copies (honest repeat purchases are not), and it finds a good share of them",
                      (h["pair_recall"] or 0) >= self.RECALL and (h["pair_precision"] or 0) >= self.PRECISION,
                      f"recall {h['pair_recall']}, precision {h['pair_precision']}, every holdout bill compared with every bill")]

    def accept_engine(self, run: Any) -> list[dict[str, Any]]:
        from .acceptance import check

        records = run.records
        data, models, final = (records[k].get("evidence") or {} for k in ("data", "models", "final"))
        chosen = (records["models"].get("decision") or {}).get("chosen")
        metrics = final.get("holdout_metrics") or {}
        base = final.get("baselines") or {}
        return [check("every stage ran on the matching machine", all(records[s].get("machine") == self.key for s in records),
                      str({s: records[s].get("machine") for s in records})),
                check("a record and every record it matches are on one side of the split", bool((data.get("split") or {}).get("by_entity")),
                      str(data.get("split"))),
                check("the model chosen on the validation records is the final one, read once", final.get("model") == chosen
                      and final.get("holdout_uses_in_this_project") == 1, f"{chosen}, read {final.get('holdout_uses_in_this_project')}"),
                check("it beats the exact rule and the single best field, the bootstrap's interval of the difference above zero",
                      bool(final.get("beats")) and all((final.get("beats") or {}).values()), f"F1 {metrics.get('pair_f1')} against {base}"),
                check("blocking's recall is measured on the holdout", metrics.get("blocking_recall") is not None, str(metrics.get("blocking_recall")))]


def _read_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"{path.name} could not be read")


def _library_args(f: dict[str, Any], matches: pd.DataFrame) -> dict[str, Any]:
    return {"id_left": f["id_column"], "id_right": f["right_id_column"] or f["id_column"],
            "fields": [{k: v for k, v in x.items() if v is not None} for x in f["fields"]], "matches": matches,
            "block_equal": f["block_equal"], "window": {"column": f["window_column"], "minutes": f["window_minutes"]} if f["window_column"] else None,
            "block_text": f["block_text"], "top_k": f["top_k"]}


class MatchingMachine(Machine):
    """Matching in the engine (30.16). The split by entity is locked in the data stage; the models are chosen on the
    validation records' pairs; the holdout's matches are read once, in the final stage."""

    task_type = "matching"
    primary_metric = "pair_f1"
    higher_is_better = True
    uses_tables = True

    def __init__(self, kind: Any, solution: Any, frame: Any, project: dict[str, Any], tables: dict[str, Any] | None = None):
        from dclab_core import matching

        super().__init__(kind, solution, frame, project)
        f = self.fields
        tables = tables or {}
        if f["matches_table"] not in tables:
            raise ValueError(f"The matching machine reads the known matches from the table {f['matches_table']!r}: attach it to the project by that name")
        right = tables.get(f["right_table"]) if f["right_table"] else None
        if f["right_table"] and right is None:
            raise ValueError(f"The matching machine links to the table {f['right_table']!r}: attach it to the project by that name")
        self.ctx = matching.setup(frame, right, **_library_args(f, tables[f["matches_table"]]))
        split = self.ctx["split"]
        self.lock = {"by_entity": True, "entities": self.ctx["entities"], **{p: split[p] for p in matching.PARTS}}
        self.lock["fingerprint"] = hashlib.sha256(repr(sorted((k, str(v)) for k, v in self.lock.items())).encode()).hexdigest()[:16]

    def training_rows(self) -> Any:
        L = self.ctx["left"]
        return L[L["__part"] != "holdout"].drop(columns=["__key", "__part"])

    def holdout_description(self) -> str:
        h = self.ctx["split"]["holdout"]
        return f"a fifth of the records by entity ({h['left']} records, {h['matches']} known matches), a record and its matches on one side"

    def cv_description(self) -> str:
        v = self.ctx["split"]["validation"]
        return f"the validation records ({v['left']}, {v['matches']} known matches), by entity, before the holdout"

    def data(self, eid: str, previous: dict) -> tuple:
        s = self.ctx["split"]
        evidence = {"train_rows": s["train"]["left"], "holdout_rows": s["holdout"]["left"], "holdout_split": self.holdout_description(),
                    "cv_protocol": self.cv_description(), "split": self.lock, "source_rows": int(len(self.frame)),
                    "usable_feature_count": len(self.fields["fields"]), "known_matches": len(self.ctx["matches"]),
                    "notes_for_person": [{"title": "A record and its matches stay on one side", "proof": ["DCLAB-R02", "DCLAB-R17"],
                                          "text": f"{self.lock['entities']} entities (a record and every record it matches) are cut by entity into "
                                                  "training, validation and holdout: no record is learned from on one side and scored on the other."}]}
        claims = [claim(f"{eid}-C1", "fact", f"{len(self.ctx['matches'])} known matches; {self.holdout_description()}.", ["evidence.split"])]
        return evidence, claims, f"{s['train']['left']} training records, holdout {self.holdout_description()}", None

    def leakage(self, eid: str, previous: dict) -> tuple:
        f = self.fields
        compared = {x["column"] for x in f["fields"]}
        kept = sorted(c for c in self.frame.columns if c not in compared and c != f["id_column"] and c not in f["block_equal"] and c != f["window_column"])
        forbidden = sorted(x.column for x in self.solution.forbidden if x.column in self.frame.columns)
        why = {c: "not a field compared: a pair is judged only on the fields the person named" for c in kept}
        why.update({x.column: x.reason or "forbidden in the solution" for x in self.solution.forbidden if x.column in self.frame.columns})
        evidence = {"kept_out": kept, "why": why, "features": sorted(compared), "declared_leakage_features": forbidden,
                    "heuristic_review_candidates": [], "matches_never_a_feature": f["matches_table"],
                    "notes_for_person": [{"title": "The known matches are the outcome, never a feature", "proof": ["DCLAB-R01"],
                                          "text": f"The table {f['matches_table']} says which pairs match; no pair is scored on it."}]}
        return evidence, [claim(f"{eid}-C1", "fact", f"Compared: {', '.join(sorted(compared))}; not compared: {', '.join(kept) or 'none'}.",
                                ["evidence.features"])], f"{len(compared)} fields compared", None

    def features(self, eid: str, previous: dict) -> tuple:
        recipe = "pair_similarity"
        names = [f"{x['column']}:{x['kind']}" for x in self.fields["fields"]]
        about = (f"for each candidate pair: {', '.join(names)} (a text's TF-IDF cosine and token overlap, a number's relative "
                 "difference, a time's minutes apart, a category's equality)")
        evidence = {"selected_recipe": recipe, "best_recipe": recipe, "selected_recipe_description": about, "features": names,
                    "blocking": {"equal": self.fields["block_equal"], "window": self.ctx["window"], "text": self.fields["block_text"], "top_k": self.fields["top_k"]},
                    "stage_results": [{"recipe": recipe, "description": about, "feature_count_mean": len(names),
                                       "metrics": {self.primary_metric: {"mean": None, "std": None}}}], "selection_rule": "one design for every model"}
        decision = {"kind": "recipe", "selected": recipe, "chosen": recipe, "rule": evidence["selection_rule"],
                    "options": [{"id": recipe, "label": recipe, "description": about, "features": len(names)}]}
        return evidence, [claim(f"{eid}-C1", "fact", f"Pairs are compared on {about}.", ["evidence.features"])], about, decision

    def models(self, eid: str, previous: dict) -> tuple:
        from dclab_core import matching

        picked = matching.validate(self.ctx)
        scores, best = picked["scores"], picked["chosen"]
        ranking = sorted(({"model": m, "mean": v["f1"], "std": 0.0, "precision": v["precision"], "recall": v["recall"], "threshold": v.get("threshold")}
                          for m, v in scores.items()), key=lambda r: -(r["mean"] or 0))
        evidence = {"selected_model": best, "ranking": ranking, "validation": scores, "blocking_recall": picked["blocking"], "pairs": picked["pairs"],
                    "feature_names": picked["names"], "selection_rule": picked["chosen_by"], "threshold": scores[best].get("threshold"),
                    "notes_for_person": [{"title": f"{best} chosen on the validation records", "proof": ["DCLAB-R13", "DCLAB-R17"],
                                          "text": "Pair F1 by model: " + "; ".join(f"{r['model']} {r['mean']}" for r in ranking)
                                                  + ". The exact rule and the single best field are what a model must beat."}]}
        decision = {"kind": "model", "selected": best, "chosen": best, "rule": picked["chosen_by"],
                    "options": [{"id": r["model"], "label": r["model"], "mean": r["mean"], "std": r["std"]} for r in ranking]}
        claims = [claim(f"{eid}-C1", "fact", "Pair F1 on the validation records: " + "; ".join(f"{r['model']} {r['mean']}" for r in ranking)
                        + f". Chosen: {best}.", ["evidence.ranking", "evidence.selected_model"])]
        return evidence, claims, f"{best} chosen on the validation records", decision

    def final(self, eid: str, previous: dict) -> tuple:
        from dclab_core import matching

        locked = (((previous.get("data") or {}).get("evidence") or {}).get("split") or {}).get("fingerprint")
        if locked != self.lock["fingerprint"]:
            raise ValueError("The split was locked on other records or matches than the tables now give: rerun the data stage")
        models = (previous.get("models") or {}).get("evidence") or {}
        chosen = self.chosen(previous.get("models"))
        threshold = next((r.get("threshold") for r in models.get("ranking", []) if r["model"] == chosen), models.get("threshold"))
        held = matching.score(self.ctx, chosen, threshold, (models.get("validation") or {}).get("single_field"))
        cv = next((r["mean"] for r in models.get("ranking", []) if r["model"] == chosen), None)
        s = held["scores"]
        evidence = {"model": chosen, "feature_recipe": self.chosen(previous.get("features")),
                    "holdout_metrics": {"pair_f1": held["pair_f1"], "pair_precision": held["pair_precision"], "pair_recall": held["pair_recall"],
                                        "average_precision": held["average_precision"], "blocking_recall": held["blocking_recall"]},
                    "holdout_primary_metric_ci": {"low": (held["pair_f1_ci95"] or (None, None))[0], "high": (held["pair_f1_ci95"] or (None, None))[1],
                                                  "point": held["pair_f1"], "method": f"bootstrap over holdout entities ({matching.BOOTSTRAP} draws), 95%"},
                    "cv_selected_metric_mean": cv, "cv_to_holdout_gap": round(held["pair_f1"] - cv, 4) if cv is not None else None,
                    "baselines": {"exact": {"pair_f1": s["exact"]["f1"]}, "single_field": {"pair_f1": s["single_field"]["f1"]}},
                    "scores": s, "beats": held["beats"], "beats_exact": held["beats_exact"], "beats_single_field": held["beats_single_field"],
                    "holdout_pairs": held["holdout_pairs"], "holdout_true_pairs": held["holdout_true_pairs"],
                    "task_assessment": {"threshold": {"value": threshold, "chosen_by": "the best pair F1 on the validation records",
                                                      "chosen_on": "the validation records' candidate pairs"}} if threshold is not None else None,
                    "holdout_split": self.holdout_description(), "holdout_rows": self.ctx["split"]["holdout"]["left"],
                    "train_rows": self.ctx["split"]["train"]["left"], "selection_rule": "the model chosen on the validation records, read once here",
                    "notes_for_person": [{"title": f"Holdout pair F1 {held['pair_f1']} for {chosen}", "proof": ["DCLAB-R17", "DCLAB-R18"],
                                          "text": f"Precision {held['pair_precision']}, recall {held['pair_recall']}; the exact rule's F1 {s['exact']['f1']}, "
                                                  f"the single best field's {s['single_field']['f1']}; blocking kept {held['blocking_recall']} of the true pairs."}]}
        claims = [claim(f"{eid}-C1", "fact", f"On the holdout records {chosen} links pairs with F1 {held['pair_f1']} (precision {held['pair_precision']}, "
                        f"recall {held['pair_recall']}), against the exact rule's {s['exact']['f1']}.", ["evidence.holdout_metrics", "evidence.baselines"])]
        return evidence, claims, f"{chosen}: holdout pair F1 {held['pair_f1']}", None


KIND = Matching()
