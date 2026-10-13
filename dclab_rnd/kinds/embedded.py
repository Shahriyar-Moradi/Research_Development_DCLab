"""Images and long text through encoders (package 16.12): each item becomes a vector (cached by its content), then a
tabular model reads the vectors, with folds that keep a source (a scene, an author) on one side and near copies across
the split counted. The kind runs its own machine (``dclab_core.embeddings``), not the five tabular stages."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from .base import ProblemKind


class EmbeddedFields(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_column: str  # the scene, document source or author that must stay on one side
    item_kind: str = "image"  # "image" or "text"
    near_duplicate_threshold: float | None = None  # None: calibrated for the encoder (calibrate)


class Embedded(ProblemKind):
    key = "embedded"
    title = "Images and long text through encoders"
    description = "Photos or long documents labelled by a team: an encoder turns each into numbers a model can read."
    base_task = "binary"
    task_type = "embedded"
    specificity = 12  # below the other kinds: photos mentioned beside a forecast or a churn question are not the input
    packs = ("vision",)
    # the items must be what is classified: "classify the photos", "label each scan", "long documents by many authors"
    words = (r"\b((?:classif\w*|label\w*|sort\w*|tag\w*) (?:the |each |every |these |our )?(?:images?|photos?|pictures?|scans?|documents?)"
             r"|(?:images?|photos?|pictures?|scans?) (?:labelled|classified|of (?:shapes|defects|products|parts))"
             r"|long (?:texts?|documents?)|documents? (?:from|by) (?:several|many) (?:sources|authors))\b")
    questions = ("Where are the images or documents, and which column holds their label?",
                 "What is one item's source (a scene, a camera, a document's author), so a source stays on one side?",
                 "Which encoder may read them on this machine (a local pretrained model), or should a simple local baseline be used?")
    metrics = (("roc_auc", "ROC-AUC"), ("accuracy", "Accuracy"))
    baselines = ()
    checks = ()
    reliability = ("near copies across the split", "sources per side", "whether the encoder is pretrained")
    evidence_track = "computer-vision"  # 16.14: the research track whose evidence agents cite for this kind
    stratify = False
    runs_in_engine = False  # items, not table cells: dclab_core.embeddings runs it as a library
    open_decisions = ("a pretrained local encoder is measured on the replicas",)  # needs weights on this machine: the owner's download
    Fields = EmbeddedFields

    def applies(self, solution: Any) -> bool:
        from .base import _get

        return _get(solution, "kind") == self.key

    def case(self, seed: int = 0) -> Any:
        from dclab_core.replicas import shapes_and_notes

        from .acceptance import Case

        images, notes = shapes_and_notes.shapes(seed), shapes_and_notes.notes(seed)
        return Case(images.table, "label", "binary", "When a photo is taken: is the object a square?",
                    solution={"kind_fields": {"source_column": "scene", "item_kind": "image"}, "items": images.items,
                              "second": {"items": notes.items, "table": notes.table, "label": "urgent",
                                         "fields": {"source_column": "author", "item_kind": "text"}}},
                    facts={"images": images.facts, "notes": notes.facts})

    def solve(self, case: Any) -> Any:
        import tempfile

        from dclab_core import embeddings

        def one(items, table, label, fields, cache):
            f = self.fields(fields)
            image = f["item_kind"] == "image"
            encoder = embeddings.RandomFeatures() if image else embeddings.HashingText()
            calibration = embeddings.calibrate(items, encoder, embeddings.perturb_image if image else embeddings.perturb_text)
            threshold = f["near_duplicate_threshold"] or calibration["threshold"] or embeddings.THRESHOLD
            X, first = embeddings.encode(items, encoder, cache)
            _, again = embeddings.encode(items, encoder, cache)  # the second pass reads the cache
            y = table[label]
            y = (y == sorted(y.unique())[-1]).astype(int) if y.dtype == object else y.astype(int)
            result = embeddings.evaluate(X, y.to_numpy(), table[f["source_column"]], threshold=threshold)
            by_source = embeddings.near_duplicates(X, table[f["source_column"]].to_numpy(), threshold=threshold)
            return {"encoding": first, "again": again, "calibration": calibration, "threshold": threshold, "evaluation": result,
                    "across_sources": by_source}

        with tempfile.TemporaryDirectory() as tmp:
            cache = embeddings.Cache(tmp)
            first = one(case.solution["items"], case.table, case.target, case.solution["kind_fields"], cache)
            s = case.solution.get("second")
            second = one(s["items"], s["table"], s["label"], s["fields"], cache) if s else None
        return {"images": first, "notes": second, "facts": case.facts}

    def accept(self, run: Any) -> list[dict[str, Any]]:
        from .acceptance import check

        im, nt, f = run["images"], run["notes"] or {}, run["facts"]
        ie, ne = im["evaluation"], nt.get("evaluation") or {}
        copied = {tuple(sorted(p)) for p in f["notes"]["copied_pairs"]}
        found = {tuple(sorted(p[:2])) for p in nt.get("across_sources", {}).get("pairs", [])}
        apart = all(not set(e["train_rows"]) & set(e["holdout_rows"]) and e["fitted_rows"] == len(e["train_rows"]) and e["sources_on_both_sides"] == 0
                    for e in (ie, ne) if e)
        return [check("vectors are cached by content: a second pass encodes nothing", im["again"]["encoded_now"] == 0
                      and im["again"]["from_cache"] == im["encoding"]["distinct_contents"],
                      f"{im['encoding']['encoded_now']} encoded, then {im['again']['from_cache']} read from the cache"),
                check("a source stays on one side; the holdout's rows are never fitted", apart,
                      f"images: {ie['holdout_sources']} scenes held out; notes: {ne.get('holdout_sources')} authors held out"),
                check("shots of one scene are near copies a random split spreads, and folds by scene do not", im["calibration"]["separable"]
                      and ie["near_duplicates_across_random_folds"]["pairs_across_sides"] > 5 * max(ie["near_duplicates_across_folds"]["pairs_across_sides"], 1),
                      f"threshold {im['threshold']} (calibrated); random folds {ie['near_duplicates_across_random_folds']['pairs_across_sides']} pairs, "
                      f"folds by scene {ie['near_duplicates_across_folds']['pairs_across_sides']}"),
                check("every text pasted again by another author is found, and nothing else", copied == found,
                      f"{len(copied & found)} of {len(copied)} copies found, {len(found - copied)} other pairs, threshold {nt.get('threshold')}"),
                # folds by source (four fifths of the sources) carry the score; the holdout (a fifth: 16 scenes, 64 images) is noisier
                check("both replicas are solved on whole sources: folds by source at least 0.9, the holdout at least 0.8",
                      all((e.get("grouped_folds") or 0) >= 0.9 and (e.get("holdout") or 0) >= 0.8 for e in (ie, ne)),
                      f"images {ie['metric']} folds by scene {ie['grouped_folds']}, holdout {ie['holdout']}; notes folds by author "
                      f"{ne.get('grouped_folds')}, holdout {ne.get('holdout')}"
                      + (f"; {ne['caveat']}" if ne.get("caveat") else "")),
                check("a pretrained local encoder is measured on the replicas", bool(im["encoding"]["pretrained"]) and bool(nt.get("encoding", {}).get("pretrained")),
                      f"measured with baselines only ({im['encoding']['encoder']}, {nt.get('encoding', {}).get('encoder')}): no pretrained weights are on "
                      "this machine, and no model purpose may read workspace texts yet")]


KIND = Embedded()
