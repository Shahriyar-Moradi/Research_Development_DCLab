"""Images and long text through encoders (package 16.12): each item becomes a vector (cached by its content), then a
tabular model reads the vectors, with folds that keep a source (a scene, an author) on one side and near copies across
the split counted. Long texts run in the engine since 16.15 C: the text is a column of the project's table, the holdout
whole sources chosen from their names alone, the model chosen on folds grouped by source. Images run in the engine since 26.8,
as rows of pixel columns read by a local baseline encoder (random patch projections, not pretrained); a pretrained encoder
needs weights on this machine (the owner's download)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from .base import ProblemKind


class EmbeddedFields(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_column: str  # the scene, document source or author that must stay on one side
    item_kind: str = "image"  # "image" or "text"
    near_duplicate_threshold: float | None = None  # None: calibrated for the encoder (calibrate)
    text_column: str | None = None  # 16.15 C: the column holding each long text, in the engine
    pixel_prefix: str | None = None  # 26.8: images in the engine, one grey pixel per column (px_0000 …), a square image per row


class Embedded(ProblemKind):
    key = "embedded"
    title = "Images and long text through encoders"
    description = "Photos or long documents labelled by a team: an encoder turns each into numbers a model can read."
    base_task = "binary"
    tasks = ("binary", "multiclass", "regression")  # a yes or no, a class or a number, read from the text
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
    metrics = (("roc_auc", "ROC-AUC"), ("accuracy", "Accuracy"), ("r2", "R²"))
    baselines = ()
    checks = ()
    reliability = ("near copies across the split", "sources per side", "whether the encoder is pretrained")
    evidence_track = "computer-vision"  # 16.14: the research track whose evidence agents cite for this kind
    aliases = ('image classification', 'computer vision', 'document classification', 'duplicate photos')  # 16.14: the names this kind of problem goes by (read into its evidence record)
    stratify = False
    runs_in_engine = True  # 16.15 C: long texts, a column of the table; 26.8: images as pixel columns, a baseline encoder
    open_decisions = ("a pretrained local encoder is measured on the replicas",)  # needs weights on this machine: the owner's download
    Fields = EmbeddedFields
    export = {"notebook": False, "service": False}

    def applies(self, solution: Any) -> bool:
        from .base import _get

        return _get(solution, "kind") == self.key

    def own_machine(self) -> Any:
        return EmbeddedMachine

    def default_metric(self, solution: Any, positive_rate: float | None, pack: dict[str, Any] | None = None) -> str:
        from .base import _get

        return {"multiclass": "accuracy", "regression": "r2"}.get(_get(solution, "task"), "roc_auc")

    def columns_of(self, solution: Any) -> list[str]:
        from .base import _get

        f = _get(solution, "kind_fields") or {}
        return [c for c in [f.get("source_column"), f.get("text_column")] if c]

    def check_solution(self, solution: Any) -> None:
        k = solution.kind_fields
        if solution.task not in ("binary", "multiclass", "regression"):
            raise ValueError(f"The embedded machine answers a yes or no, a class or a number: not {solution.task!r}")
        if k["item_kind"] == "image":  # 26.8: images as pixel columns, read by a local baseline encoder (a pretrained one is the owner's download)
            if not k.get("pixel_prefix"):
                raise ValueError("In the engine images are rows of pixel columns: name their prefix (pixel_prefix, e.g. px_)")
            if str(solution.target).startswith(k["pixel_prefix"]) or str(k["source_column"]).startswith(k["pixel_prefix"]):
                raise ValueError("The source and the target are not pixel columns")
            if k["source_column"] == solution.target:
                raise ValueError("The source and the target must be different columns")
            if any(str(f.column).startswith(k["pixel_prefix"]) for f in solution.forbidden):  # the machine reads every pixel
                raise ValueError("A pixel column cannot be forbidden: the model reads the whole image (crop the images instead)")
        elif k["item_kind"] != "text" or not k.get("text_column"):
            raise ValueError("In the engine the items are long texts in a column (text_column) or images as pixel columns (pixel_prefix)")
        elif k["text_column"] in (k["source_column"], solution.target) or k["source_column"] == solution.target:
            raise ValueError("The text, the source and the target must be different columns")
        elif k["text_column"] in {f.column for f in solution.forbidden}:
            raise ValueError(f"{k['text_column']} is forbidden: the model would read nothing else")
        if solution.metric and solution.metric != self.default_metric(solution, None):
            raise ValueError(f"The embedded machine reports {self.default_metric(solution, None)} for this task: {solution.metric!r} would be ignored")

    def accept_engine(self, run: Any) -> list[dict[str, Any]]:
        from .acceptance import check, engine_run

        images = self.accept_image_engine(engine_run(self.image_case(0), self.key))  # 26.8: the images run in the engine too

        records, f = run.records, run.case.facts["notes"]
        leak, models, final = run.evidence("leakage"), records["models"], run.evidence("final")
        table = run.case.table
        held = set(final.get("holdout_row_positions") or [])
        straddle = {tuple(sorted(p)) for p in f["copied_pairs"] if (p[0] in held) != (p[1] in held)}
        counted = {tuple(sorted(p[:2])) for p in leak.get("near_duplicates_across_holdout_pairs") or []}
        return [check("the engine runs every stage as the embedded machine", all((records[s] or {}).get("machine") == self.key for s in records),
                      ", ".join(f"{s}: {(records[s] or {}).get('machine')}" for s in records)),
                check("in the engine a source stays on one side, and the holdout's rows are never fitted",
                      final.get("sources_on_both_sides") == 0 and final.get("fitted_rows") == final.get("train_rows")
                      and final.get("train_rows", 0) + final.get("holdout_rows", 0) == len(table),
                      f"{final.get('holdout_rows')} rows of {run.evidence('data').get('split', {}).get('holdout_sources')} authors held out; "
                      f"{final.get('sources_on_both_sides')} authors on both sides"),
                check("in the engine every text pasted again across the holdout line is counted, and nothing else",
                      counted == straddle, f"{len(counted & straddle)} of {len(straddle)} copies across the line found, {len(counted - straddle)} other pairs"),
                check("in the engine the model chosen on folds by author is the one the holdout scores, once",
                      final.get("model") == (models.get("decision") or {}).get("chosen") and final.get("holdout_uses_in_this_project") == 1,
                      f"chosen C={(models.get('decision') or {}).get('chosen')}, scored C={final.get('model')}"),
                check("in the engine the notes are solved on whole authors: folds by author at least 0.9, the holdout at least 0.8",
                      (run.evidence("models").get("grouped_folds") or 0) >= 0.9 and (final.get("holdout_metrics", {}).get(final.get("metric")) or 0) >= 0.8,
                      f"folds by author {run.evidence('models').get('grouped_folds')}, random folds {run.evidence('models').get('random_folds')}, "
                      f"holdout {final.get('holdout_metrics', {}).get(final.get('metric'))}")] + images

    def image_case(self, seed: int = 0) -> Any:
        """26.8: the shapes as a person's image table (one row per image, its pixels as columns), for the engine."""
        from dclab_core.replicas import shapes_and_notes

        from .acceptance import Case

        shapes = shapes_and_notes.shapes_table(seed)
        return Case(shapes.table, "label", "binary", "When a photo of a part arrives: is it a square?",
                    solution={"kind": "embedded", "kind_fields": {"source_column": "scene", "item_kind": "image", "pixel_prefix": "px_"},
                              "identifiers": ["item"], "positive_label": "square"},
                    facts={"images": shapes.facts})

    def accept_image_engine(self, run: Any) -> list[dict[str, Any]]:
        """26.8: the image case in the engine: a scene's shots on one side, no near copy across the holdout line (the shots of
        a scene are near copies a random split would spread), the model chosen on folds by scene scored once, solved."""
        from .acceptance import check

        records, leak, final, models = run.records, run.evidence("leakage"), run.evidence("final"), run.evidence("models")
        chosen = ((records.get("models") or {}).get("decision") or {}).get("chosen")  # the stage's decision, beside its evidence
        table = run.case.table
        return [check("images: the engine runs every stage as the embedded machine", all((records[s] or {}).get("machine") == self.key for s in records),
                      ", ".join(f"{s}: {(records[s] or {}).get('machine')}" for s in records)),
                check("images: a scene's shots stay on one side, and the holdout's images are never fitted",
                      final.get("sources_on_both_sides") == 0 and final.get("fitted_rows") == final.get("train_rows")
                      and final.get("train_rows", 0) + final.get("holdout_rows", 0) == len(table),
                      f"{final.get('holdout_rows')} images of {run.evidence('data').get('split', {}).get('holdout_sources')} scenes held out"),
                # a scene's own shots never straddle the line (the split is by scene); two scenes that happen to look alike may,
                # and the engine reports them with a warning rather than hide them
                check("images: the near copies are caught: most shots of one scene are flagged, images of different scenes almost never",
                      (leak.get("calibration") or {}).get("separable") is True
                      and leak.get("same_source_pairs", 0) > 0 and leak.get("same_source_pairs_flagged", 0) >= 0.9 * leak["same_source_pairs"]
                      and leak.get("different_source_pairs_flagged", 0) <= 0.001 * max(leak.get("different_source_pairs", 0), 1),
                      f"{leak.get('same_source_pairs_flagged')} of {leak.get('same_source_pairs')} same-scene pairs flagged, "
                      f"{leak.get('different_source_pairs_flagged')} of {leak.get('different_source_pairs')} other pairs; "
                      f"{leak.get('near_duplicates_across_holdout')} across the line; threshold {leak.get('threshold')}"),
                check("images: the model chosen on folds by scene is the one the holdout scores, once",
                      chosen is not None and final.get("model") == chosen and final.get("holdout_uses_in_this_project") == 1,
                      f"chosen {chosen}, scored {final.get('model')}"),
                check("images: the shapes are solved on whole scenes: folds by scene at least 0.9, the holdout at least 0.8",
                      (models.get("grouped_folds") or 0) >= 0.9 and (final.get("holdout_metrics", {}).get(final.get("metric")) or 0) >= 0.8,
                      f"folds by scene {models.get('grouped_folds')}, random folds {models.get('random_folds')}, holdout "
                      f"{final.get('holdout_metrics', {}).get(final.get('metric'))}")]

    def case(self, seed: int = 0) -> Any:
        from dclab_core.replicas import shapes_and_notes

        from .acceptance import Case

        images, notes = shapes_and_notes.shapes(seed), shapes_and_notes.notes(seed)
        table = notes.table.assign(text=notes.items)  # the long texts as a column, as a person's export holds them
        return Case(table, "urgent", "binary", "When a support note arrives: is it urgent?",
                    solution={"kind": "embedded", "kind_fields": {"source_column": "author", "item_kind": "text", "text_column": "text"},
                              "identifiers": ["item"]},
                    items={"images": {"items": images.items, "table": images.table, "label": "label",
                                      "fields": {"source_column": "scene", "item_kind": "image"}}},
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

        fields = case.solution["kind_fields"]
        with tempfile.TemporaryDirectory() as tmp:
            cache = embeddings.Cache(tmp)
            im = (case.items or {}).get("images")
            first = one(im["items"], im["table"], im["label"], im["fields"], cache)
            second = one(case.table[fields["text_column"]].tolist(), case.table, case.target, fields, cache)
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


from .machine import Machine, claim  # noqa: E402


UNSEEN = "\x00unseen"  # a held-out class the training rows never had


class EmbeddedMachine(Machine):
    """Long texts in the engine (16.15 C). Each text becomes a vector by a local baseline encoder (word counts hashed;
    no download, not pretrained); the holdout is whole sources, chosen from their names in the data stage; near copies
    across the holdout line and across folds are counted from the texts alone; the regularisation is chosen on folds
    grouped by source; the holdout's labels are read once, in the final stage."""

    task_type = "embedded"
    higher_is_better = True

    def __init__(self, kind: Any, solution: Any, frame: Any, project: dict[str, Any]):
        from dclab_core import embeddings

        super().__init__(kind, solution, frame, project)
        f = self.fields
        self.task = solution.task if solution.task in ("binary", "multiclass", "regression") else "binary"
        self.primary_metric = kind.default_metric(solution, None)
        self.text, self.source, self.target = f["text_column"], f["source_column"], solution.target
        self.image = f["item_kind"] == "image"  # 26.8
        self.pixels = embeddings.pixel_columns(frame.columns, f["pixel_prefix"]) if self.image else []
        if self.image:
            self.size = int(round(len(self.pixels) ** 0.5))
            if self.size < 8 or self.size * self.size != len(self.pixels):
                raise ValueError(f"{len(self.pixels)} pixel columns: a square image of at least 8 by 8 is needed")
        read = frame[self.pixels].notna().all(axis=1) if self.image else frame[self.text].notna()
        readable = read & frame[self.source].notna()
        # the sources are held out from the rows with a text and a source, before any label is looked at: a missing label
        # never moves a source to the other side
        self.parts = embeddings.split(frame.loc[readable, self.source], seed=0)
        keep = readable & frame[self.target].notna()
        self.dropped = {"without_text_or_source": int((~readable).sum()), "without_label": int((readable & frame[self.target].isna()).sum())}
        self.rows = frame[keep]
        self.index = self.rows.index.to_numpy()  # positions in the table as attached, for every record
        self.rows = self.rows.reset_index(drop=True)
        self.test = embeddings.holdout_mask(self.rows[self.source], self.parts)
        if not self.test.any() or self.test.all():
            raise ValueError(f"The held-out {self.source}s hold {int(self.test.sum())} of {len(self.test)} labelled rows: no honest split")
        if self.image:  # centred on the training images only: pooled rectified features are otherwise all alike (26.8)
            self.encoder = embeddings.Centred.fit(embeddings.RandomFeatures(size=self.size), self.items(self.rows[~self.test]))
            self.copies = embeddings.Pixels()  # near copies are found on the pictures, not on a classifier's features
        else:
            self.encoder = self.copies = embeddings.HashingText()
        self._C = None
        self._X = None
        self.image_decisions: list[dict[str, Any]] = []  # 26.8: written to the ledger after the leakage stage
        self.image_totals: dict[str, int] = {}

    def items(self, rows: Any) -> list[Any]:
        """The items a model reads: each row's image (its pixel columns as a square), or its text."""
        if self.image:
            from dclab_core import embeddings

            return embeddings.images_of(rows, self.pixels)
        return rows[self.text].astype(str).tolist()

    def _copy_vectors(self) -> Any:
        from dclab_core import embeddings

        if self._C is None:
            self._C, _ = embeddings.encode(self.items(self.rows), self.copies)
        return self._C

    def _copies_within(self, X: Any, threshold: float) -> dict[str, Any]:
        """26.8: on the training images, how many pairs of one source (two shots of a scene) reach the near-copy threshold,
        and how many pairs of different sources do: the trap a random split would fall into, shown on training rows."""
        train = ~self.test
        V, src = X[train], self.rows.loc[train, self.source].to_numpy()
        S = V @ V.T
        i, j = np.triu_indices(len(V), 1)
        same = src[i] == src[j]
        over = S[i, j] >= threshold
        return {"same_source_pairs": int(same.sum()), "same_source_pairs_flagged": int((same & over).sum()),
                "different_source_pairs": int((~same).sum()), "different_source_pairs_flagged": int((~same & over).sum())}

    MAX_IMAGE_DECISIONS = 1500  # training images compared for the image points (seeded sample above it)
    MAX_LABEL_RECORDS = 200  # suspect labels kept, and as many of the others

    def _image_decisions(self, threshold: float) -> list[dict[str, Any]]:
        """26.8: the image points on the training images only (a held-out image is never compared): near copies across
        sources, suspect labels, whether a mirror flip keeps the label. Keys name rows of the table as attached."""
        from ..extensions import optional

        idc = optional("dclab_core.image_decisions")  # the image decision points: the full edition's
        if idc is None:
            return []
        train = np.flatnonzero(~self.test)
        if len(train) > self.MAX_IMAGE_DECISIONS:
            train = np.sort(np.random.default_rng(0).choice(train, self.MAX_IMAGE_DECISIONS, replace=False))
        rows = self.rows.iloc[train]
        self.image_totals = {"images_compared": int(len(train)), "training_images": int((~self.test).sum())}
        out = idc.rule_decisions(self.items(rows), rows[self.source].tolist(), rows[self.target].tolist(), threshold=threshold,
                                 task="regression" if self.task == "regression" else "classification", totals=self.image_totals)

        def at(k: str) -> str:
            return str(int(self.index[train[int(k)]]))

        kept, counted = [], {True: 0, False: 0}
        for d in out:
            if d["point"] == "image.label_suspect":
                if counted[d["value"]] >= self.MAX_LABEL_RECORDS:
                    continue
                counted[d["value"]] += 1
                d = {**d, "key": f"row:{at(d['key'])}"}
            elif d["point"] == "image.near_copy":
                a, b = d["key"].split("-")
                d = {**d, "key": f"rows:{at(a)}-{at(b)}"}
            kept.append(d)
        return kept

    def _image_summary(self, threshold: float) -> dict[str, Any]:
        self.image_decisions = self._image_decisions(threshold)
        flip = next((d for d in self.image_decisions if d["point"] == "image.flip_safe"), None)
        return {"image_decisions": {**self.image_totals,  # counted before the records' caps: the figures a person reads
                                    "near_copy_pairs_recorded": sum(d["point"] == "image.near_copy" for d in self.image_decisions),
                                    "label_records_recorded": sum(d["point"] == "image.label_suspect" for d in self.image_decisions),
                                    "flip_safe": None if flip is None else flip["value"],
                                    "flip_drop": None if flip is None else flip["state"]["flip_drop"]}}

    def item_name(self) -> str:
        return f"the images ({len(self.pixels)} pixel columns)" if self.image else str(self.text)

    def X(self) -> Any:
        from dclab_core import embeddings

        if self._X is None:
            self._X, self.encoding = embeddings.encode(self.items(self.rows), self.encoder)
        return self._X

    def seen(self) -> list[str]:
        """The classes as the training rows have them; never read from a held-out row."""
        return sorted({str(v) for v in self.rows.loc[~self.test, self.target]})

    def classes(self) -> tuple[str, str]:
        """(negative, positive) as the training rows have them: the person's positive label, else the later of the two;
        never read from a held-out row."""
        seen = self.seen()
        if len(seen) != 2:
            raise ValueError(f"The training rows hold {len(seen)} classes of {self.target}: a binary task needs two (multiclass takes more)")
        positive = str(self.solution.positive_label) if self.solution.positive_label is not None else seen[-1]
        if positive not in seen:
            raise ValueError(f"The positive label {positive!r} is not among the training rows' classes")
        return next(v for v in seen if v != positive), positive

    def y(self) -> Any:
        """Binary: 1 for the positive class. Multiclass: the class names. Regression: the numbers. A held-out class the
        training rows never had is marked (-1, or ``UNSEEN``) and refused when the final stage reads it."""
        if self.task == "regression":
            values = pd.to_numeric(self.rows[self.target], errors="coerce")
            if values[~self.test].isna().any():  # the training rows' own; a held-out row's is refused where the holdout is read
                raise ValueError(f"{self.target} holds values that are not numbers: a regression needs a number on every labelled row")
            return values.to_numpy(dtype=float)
        v = self.rows[self.target].astype(str)
        if self.task == "multiclass":
            seen = self.seen()
            if len(seen) < 2:
                raise ValueError(f"The training rows hold {len(seen)} class of {self.target}: nothing to tell apart")
            return np.where(v.isin(seen), v, UNSEEN)
        negative, positive = self.classes()
        return np.where(v == positive, 1, np.where(v == negative, 0, -1))

    def _unseen(self, y: Any) -> Any:
        if self.task == "regression":
            return np.isnan(y)  # a held-out label that is not a number, read here in the final stage
        return (y == UNSEEN) if self.task == "multiclass" else (y == -1)

    def baseline(self, y: Any) -> dict[str, Any]:
        """What the holdout scores with no text read, from the training rows alone: the majority class's accuracy, the
        training mean's R², or chance's ROC-AUC."""
        from sklearn.metrics import r2_score

        train, test = ~self.test, self.test
        if self.task == "multiclass":
            majority = pd.Series(y[train]).value_counts().idxmax()
            return {"majority_class": {"accuracy": round(float(np.mean(y[test] == majority)), 4), "class": str(majority)}}
        if self.task == "regression":
            return {"training_mean": {"r2": round(float(r2_score(y[test], np.full(int(test.sum()), float(np.mean(y[train]))))), 4)}}
        return {"chance": {"roc_auc": 0.5}}

    def training_rows(self) -> Any:
        return self.rows[~self.test]

    def lock(self) -> dict[str, Any]:
        return {k: v for k, v in self.parts.items() if not k.startswith("_")}

    def holdout_description(self) -> str:
        return f"{self.parts['holdout_sources']} of {self.parts['sources']} sources ({self.source}), whole, chosen from their names"

    def cv_description(self) -> str:
        return f"{self.parts['folds']} folds grouped by {self.source} on the training rows"

    def data(self, eid: str, previous: dict) -> tuple:
        train, held = int((~self.test).sum()), int(self.test.sum())
        positive = self.classes()[1] if self.task == "binary" else None
        self.y()  # refused here, before any choice, when the training rows cannot be modelled for this task
        dropped = self.dropped
        evidence = {"split": self.lock(), "train_rows": train, "holdout_rows": held, "holdout_split": self.holdout_description(),
                    "cv_protocol": self.cv_description(), "source_rows": int(len(self.frame)), "rows_dropped": dropped,
                    "positive_class": positive, "task": self.task,
                    **({"classes": self.seen()} if self.task == "multiclass" else {}),
                    "usable_feature_count": 1, "text_column": self.text, "source_column": self.source,
                    **({"pixel_columns": len(self.pixels), "image_size": self.size} if self.image else {}),
                    "notes_for_person": [{"title": "Whole sources held out", "proof": ["DCLAB-R02"],
                                          "text": f"{train} training rows; the holdout is {self.holdout_description()}: a source is never on both sides. "
                                                  f"Left out: {dropped['without_text_or_source']} rows without a text or a {self.source}, "
                                                  f"{dropped['without_label']} without a label. "
                                                  + (f"{positive!r} is the positive class." if positive is not None else
                                                     f"{len(self.seen())} classes in the training rows." if self.task == "multiclass" else
                                                     f"{self.target} is predicted as a number.")}]}
        claims = [claim(f"{eid}-C1", "fact", f"{train} training rows; {held} rows of {self.parts['holdout_sources']} whole sources held out; "
                        f"{dropped['without_text_or_source'] + dropped['without_label']} rows left out for a missing text, source or label.",
                        ["evidence.split", "evidence.rows_dropped"])]
        return evidence, claims, f"{train} training rows, {self.parts['holdout_sources']} sources held out", None

    def leakage(self, eid: str, previous: dict) -> tuple:
        from dclab_core import embeddings

        X = self.X() if not self.image else self._copy_vectors()
        texts = self.items(self.training_rows())
        calibration = embeddings.calibrate(texts, self.copies, embeddings.perturb_image if self.image else embeddings.perturb_text,  # training items only
                                           sources=self.training_rows()[self.source].tolist() if self.image else None)
        threshold = self.fields["near_duplicate_threshold"] or calibration["threshold"] or embeddings.THRESHOLD
        across = embeddings.near_duplicates(X, np.where(self.test, "holdout", "train"), threshold=threshold)  # texts only, never a label
        not_read = [c for c in self.frame.columns if c not in (self.text, self.source, self.target) and c not in set(self.pixels)]
        forbidden = [x.column for x in self.solution.forbidden]
        evidence = {"kept_out": sorted(set(not_read) | set(forbidden)), "declared_leakage_features": sorted(set(forbidden)),
                    "why": {c: f"the model reads {self.item_name()} only" for c in not_read}, "calibration": calibration, "threshold": threshold,
                    "near_duplicates_across_holdout": across["pairs_across_sides"], "near_duplicates_method": across["method"],
                    "near_duplicates_recall": across.get("recall_at_threshold"),  # hashed past EXACT_UP_TO texts: a pair can be missed
                    "near_duplicates_across_holdout_pairs": [[int(self.index[a]), int(self.index[b]), sim] for a, b, sim in across["pairs"]],
                    **(self._copies_within(X, threshold) if self.image else {}),
                    **(self._image_summary(threshold) if self.image else {}),
                    "notes_for_person": [{"level": "warning" if across["pairs_across_sides"] else "info", "proof": ["DCLAB-R01"],
                                          "title": f"{across['pairs_across_sides']} near copies straddle the holdout line",
                                          "text": "A text pasted again by another source lets a model recognise rather than predict: the holdout "
                                                  "score may be helped by them." if across["pairs_across_sides"] else "No text is a near copy across the line."}]}
        searched = ("every pair compared" if across["method"] == "exact" else
                    f"pairs found by hashing, each found with probability {across.get('recall_at_threshold')} at the threshold: a lower count")
        claims = [claim(f"{eid}-C1", "fact", f"The model reads {self.item_name()} only; {across['pairs_across_sides']} near copies (cosine at least "
                        f"{threshold}) straddle the holdout line ({searched}).", ["evidence.near_duplicates_across_holdout", "evidence.kept_out"])]
        return evidence, claims, f"{across['pairs_across_sides']} near copies across the holdout line", None

    def features(self, eid: str, previous: dict) -> tuple:
        self.X()
        recipe = self.encoder.name
        about = (f"each image as fixed random projections of 5x5 patches, pooled ({self.encoder.name}): a local baseline, not a pretrained encoder"
                 if self.image else f"each text as {self.encoder.width} hashed word and word-pair counts: a local baseline, not a pretrained encoder")
        width = int(self.X().shape[1])
        evidence = {"selected_recipe": recipe, "best_recipe": recipe, "selected_recipe_description": about, "encoding": self.encoding,
                    "pretrained": False, "stage_results": [{"recipe": recipe, "description": about, "feature_count_mean": width,
                                                            "metrics": {self.primary_metric: {"mean": None, "std": None}}}],
                    "selection_rule": "one encoder: the local baseline (a pretrained one needs the owner's download)"}
        decision = {"kind": "recipe", "selected": recipe, "chosen": recipe, "rule": evidence["selection_rule"],
                    "options": [{"id": recipe, "label": recipe, "description": about, "features": width}]}
        return evidence, [claim(f"{eid}-C1", "fact", f"{about}.", ["evidence.encoding"], ["Not a pretrained encoder: a baseline."])], about, decision

    def models(self, eid: str, previous: dict) -> tuple:
        from dclab_core import embeddings

        train = ~self.test
        threshold = ((previous.get("leakage") or {}).get("evidence") or {}).get("threshold") or embeddings.THRESHOLD
        picked = embeddings.choose(self.X()[train], self.y()[train], self.rows[self.source][train], task=self.task, threshold=threshold)
        ranking = sorted(({"model": c, "mean": m["mean"], "std": m["std"]} for c, m in picked["by_c"].items()), key=lambda r: -(r["mean"] or 0))
        best = picked["chosen"]
        evidence = {"selected_model": best, "ranking": ranking, "selection_rule": picked["chosen_by"], "grouped_folds": picked["grouped_folds"],
                    "random_folds": picked["random_folds"], "near_duplicates_across_folds": picked["near_duplicates_across_folds"],
                    "near_duplicates_across_random_folds": picked["near_duplicates_across_random_folds"], "metric": picked["metric"],
                    "near_duplicates_method": picked["near_duplicates_method"], "near_duplicates_recall": picked["near_duplicates_recall"],
                    "notes_for_person": [{"title": f"C={best} chosen on folds by {self.source}", "proof": ["DCLAB-R13"],
                                          "text": f"Folds by {self.source}: {picked['grouped_folds']}; random folds: {picked['random_folds']} "
                                                  f"({picked['near_duplicates_across_random_folds']} near copies across random folds, "
                                                  f"{picked['near_duplicates_across_folds']} across folds by {self.source}). The holdout is unread."}]}
        decision = {"kind": "model", "selected": best, "chosen": best, "rule": picked["chosen_by"],
                    "options": [{"id": r["model"], "label": f"logistic C={r['model']}", "mean": r["mean"], "std": r["std"]} for r in ranking]}
        return evidence, [claim(f"{eid}-C1", "fact", f"{picked['metric']} over folds by {self.source}: " + "; ".join(f"C={r['model']} {r['mean']}" for r in ranking)
                                + f". Chosen: C={best}.", ["evidence.ranking"])], f"C={best} chosen on folds by {self.source}", decision

    def final(self, eid: str, previous: dict) -> tuple:
        from dclab_core import embeddings

        locked = ((previous.get("data") or {}).get("evidence") or {}).get("split") or {}
        if locked.get("holdout_fingerprint") != self.parts["holdout_fingerprint"]:
            raise ValueError("The held-out sources changed since the data stage: rerun it")
        chosen = self.chosen(previous.get("models"))
        y = self.y()
        unknown = sorted({str(v) for v in self.rows.loc[self.test & self._unseen(y), self.target]})
        if unknown:
            raise ValueError(f"The held-out rows hold values that are not numbers: {', '.join(unknown[:5])}" if self.task == "regression" else
                             f"The held-out rows hold labels the training rows never had: {', '.join(unknown[:5])}")
        scored = embeddings.score(self.X(), y, self.rows[self.source], self.parts, chosen, task=self.task)
        cv = next((r["mean"] for r in ((previous.get("models") or {}).get("evidence") or {}).get("ranking", []) if r["model"] == chosen), None)
        across = ((previous.get("leakage") or {}).get("evidence") or {}).get("near_duplicates_across_holdout") or 0
        metric = scored["metric"]
        evidence = {"model": chosen, "feature_recipe": self.chosen(previous.get("features")), "metric": metric, "holdout_metrics": {metric: scored["holdout"]},
                    "holdout_primary_metric_ci": {"low": None, "high": None, "point": scored["holdout"], "method": "not computed"},
                    "cv_selected_metric_mean": cv, "cv_to_holdout_gap": round(scored["holdout"] - cv, 4) if cv is not None and scored["holdout"] is not None else None,
                    "fitted_rows": scored["fitted_rows"], "train_rows": scored["train_rows"], "holdout_rows": scored["holdout_rows"],
                    "sources_on_both_sides": scored["sources_on_both_sides"], "holdout_row_positions": [int(i) for i in self.index[self.test]],
                    "why_no_score": scored["why_no_score"],
                    "holdout_split": self.holdout_description(), "baselines": self.baseline(y), "selection_rule": "the regularisation chosen on folds by source, read once here",
                    "notes_for_person": [{"title": (f"Holdout {metric} {scored['holdout']} on unseen {self.source}s" if scored["holdout"] is not None
                                                    else f"No holdout {metric}: {scored['why_no_score']}"), "proof": ["DCLAB-R17"],
                                          "text": f"Folds by {self.source} gave {cv}. "
                                                  + (f"{across} near copies straddle the line: the score may be helped by them." if across else "")}]}
        said = (f"On {self.parts['holdout_sources']} held-out {self.source}s the model scores {metric} {scored['holdout']}." if scored["holdout"] is not None
                else f"On {self.parts['holdout_sources']} held-out {self.source}s no {metric} can be read: {scored['why_no_score']}.")
        claims = [claim(f"{eid}-C1", "fact", said, ["evidence.holdout_metrics"], ["A baseline encoder, not a pretrained one.",
                                                                                  f"Against no text read: {self.baseline(y)}.",
                                                       *([f"{across} near copies straddle the holdout line."] if across else []),
                                                       "Not approved for production from this evidence alone."])]
        return evidence, claims, f"C={chosen}: holdout {metric} {scored['holdout']}", {
            "kind": "final", "selected": "defaults", "chosen": "defaults", "rule": evidence["selection_rule"], "options": []}


KIND = Embedded()
