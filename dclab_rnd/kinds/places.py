"""Places (package 16.10): a regression whose rows sit on a map, where nearby rows have alike targets (prices across a
city). It runs in the engine as a regression, with one difference that matters: the split keeps whole blocks of places
on one side (``Solution.spatial``), chosen by dclab_core's plan when the training rows show that nearby targets are
alike. The ``maps`` Domain pack runs as this kind."""

from __future__ import annotations

from typing import Any

from .base import _get
from .regression import Regression


class Places(Regression):
    key = "places"
    title = "Places: values over a map"
    description = "A value at a place (a price, a demand) where nearby places are alike: tested on areas the model has not seen."
    specificity = 10
    packs = ("maps",)
    words = r"\b(spatial|on (?:a|the) map|across (?:the )?(?:city|region|country)|by (?:district|neighbourhood|area)|(?:flat|house|home|rent|land) prices?)\b"
    questions = ("Which columns hold the latitude and the longitude?",
                 "Will the model be used in areas it has seen, or in new ones (a new city, a new district)?")
    reliability = ("blocks of places on each side of the split", "how much a random split flatters", "areas with few rows")
    evidence_track = "places-and-spatial"  # 16.14: the research track whose evidence agents cite for this kind
    aliases = ('spatial data', 'geographic data', 'location', 'spatial cross validation', 'nearby houses')  # 16.14: the names this kind of problem goes by (read into its evidence record)

    def applies(self, solution: Any) -> bool:
        return _get(solution, "task") == "regression" and bool(_get(solution, "spatial"))

    def detect(self, problem: str, profile: dict[str, Any] | None = None) -> str | None:
        found = super().detect(problem, profile)
        if found and profile is not None:
            from dclab_core.behaviour import coordinate_pairs

            names = [c.get("name") if isinstance(c, dict) else str(c) for c in (profile.get("columns") or [])]
            if not coordinate_pairs([n for n in names if n]):
                return None  # a map's words without coordinates: nothing to split by
        return found

    def case(self, seed: int = 0) -> Any:
        from dclab_core.replicas import city_prices

        from .acceptance import Case

        replica = city_prices.generate(seed)
        return Case(replica.flats, "price", "regression", "When a flat is listed: its place, size, rooms and age are known.",
                    pack="maps", facts=replica.facts)

    def accept(self, run: Any) -> list[dict[str, Any]]:
        from .acceptance import check

        spatial = run.solution.get("spatial") or {}
        ev = run.evidence("data").get("spatial") or {}
        gap = ev.get("random_split_optimism") or {}
        mae = run.holdout("mae")
        median = (run.evidence("final").get("baselines") or {}).get("training_median", {}).get("mae")
        return [check("the plan finds that nearby flats are alike and the engine splits by blocks of places", bool(spatial)
                      and run.evidence("data").get("problem_kind") == self.key and "group" in str(run.evidence("data").get("holdout_split", "")).lower(),
                      f"blocks of {spatial.get('block_km')} km from {spatial.get('lat_column')}, {spatial.get('lon_column')}; "
                      f"holdout: {run.evidence('data').get('holdout_split')}"),
                check("no block of places is on both sides of the holdout", ev.get("blocks_on_both_sides") == 0 and (ev.get("holdout_blocks") or 0) > 0,
                      f"{ev.get('train_blocks')} training blocks, {ev.get('holdout_blocks')} held out, {ev.get('blocks_on_both_sides')} on both sides"),
                check("a random split's optimism against blocks is measured on the training rows", (gap.get("optimism") or 0) > 0.02,
                      f"{gap.get('metric')}: random folds {gap.get('random')}, blocks {gap.get('blocked')}, optimism {gap.get('optimism')}"),
                check("the holdout MAE is below the training median's", mae is not None and median is not None and mae < median, f"{mae} against {median}")]


KIND = Places()
