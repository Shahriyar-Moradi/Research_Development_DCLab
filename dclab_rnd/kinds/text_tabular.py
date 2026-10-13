"""Text next to structured fields, with a yes or no outcome: reviews, tickets, notes."""

from __future__ import annotations

from typing import Any

from .base import _get
from .binary import Binary


class TextTabular(Binary):
    key = "text_tabular"
    title = "Text + tabular (binary)"
    description = "A yes or no from free text and the structured fields that come with it."
    task_type = "text_tabular_binary"
    specificity = 10
    packs = ("text",)
    questions = ("What should the model predict from the text, and which structured fields come with each text?",
                 "Is the text written before the prediction moment, or could it describe the outcome?")
    baselines = ("majority_class",)
    reliability = ("calibration", "text written after the outcome", "drift in vocabulary")
    evidence_track = "nlp-and-text"
    aliases = ('text and tabular', 'free text with numbers', 'review text', 'text features')  # 16.14: the names this kind of problem goes by (read into its evidence record)
    export = {"notebook": True, "service": False,
              "why": "a text project's vocabulary is learned from its rows: its service export is not built yet"}

    def detect(self, problem: str, profile: dict[str, Any] | None = None) -> str | None:
        texts = (profile or {}).get("text_candidates") or []
        return f"long text in {', '.join(texts[:3])}" if texts else None

    def machine(self, solution: Any, text_columns: tuple[str, ...]) -> str:
        return "text_tabular_binary" if text_columns else "binary_imbalanced"

    def applies(self, solution: Any) -> bool:
        return _get(solution, "task") == "binary" and bool(_get(solution, "text_columns"))


    def case(self, seed: int = 0) -> Any:
        from dclab_core.replicas import product_reviews

        from .acceptance import Case

        replica = product_reviews.generate(seed)
        return Case(replica.reviews, "negative", "binary", "When the review is posted, before anyone answers it.", {"positive_label": "1"}, facts=replica.facts)

    def accept(self, run: Any) -> list[dict[str, Any]]:
        from .acceptance import check, kept_out

        chosen = (run.records["features"].get("decision") or {}).get("chosen")
        options = {o["label"]: o.get("mean") for o in (run.records["features"].get("decision") or {}).get("options", [])}
        auc = run.holdout("roc_auc")
        from .acceptance import modelled_text

        return [kept_out(run, "shop_reply", "written after the review", roles=("blocked",)),
                kept_out(run, "review_id", "an identifier"),
                check("the review text is modelled", run.role("review_text") == "text" and "review_text" in modelled_text(run),
                      f"its role: {run.role('review_text')}"),
                check("a text recipe is chosen over the structured fields alone", chosen in ("tfidf_word", "combined"), str(chosen)),
                check("the chosen recipe beats the structured fields by 0.1 training-CV", chosen in options and "tabular" in options
                      and options[chosen] is not None and options["tabular"] is not None and options[chosen] >= options["tabular"] + 0.1,
                      f"{options.get(chosen)} against {options.get('tabular')}"),
                check("the holdout ROC-AUC is at least 0.8", auc is not None and auc >= 0.8, str(auc)),
                check("no copy of the label is left (holdout ROC-AUC below 0.99)", auc is not None and auc < 0.99, str(auc))]


KIND = TextTabular()
