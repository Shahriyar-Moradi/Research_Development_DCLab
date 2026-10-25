"""Classical decision classifiers, trained (package 22.2): ``make decision-models``.

For each point of the generated decision sets (``make decisions-gen``), a classifier chosen by table-grouped
out-of-fold accuracy (``dclab_core.decision_models.train``), measured on the dev rows beside the rule. Each card is
written to ``research/llm-fine-tuning/experiments/decisions/models_v1/<point>.json`` (committed: what was trained on
what, and how it compares with the rule); the fitted models to ``models_v1/<point>.joblib`` (rebuilt, not committed).
The sealed rows are never read here: 22.3 reads them once, under a gate fixed before.

    python -m dclab_rnd.decision_models_build [--points p1,p2] [--quick]
    python -m dclab_rnd.decision_models_build --all        # 29.4: every generation, a student or a reason it has none
    python -m dclab_rnd.decision_models_build --compare    # 29.4: every student beside its rule on dev, from the cards
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SETS = ROOT / "research" / "llm-fine-tuning" / "experiments" / "decisions" / "generated_v1" / "decisions"
OUT = ROOT / "research" / "llm-fine-tuning" / "experiments" / "decisions" / "models_v1"
IMAGE_SETS = ROOT / "research" / "llm-fine-tuning" / "experiments" / "decisions" / "generated_images_v1" / "decisions"  # 26.8
IMAGE_OUT = ROOT / "research" / "llm-fine-tuning" / "experiments" / "decisions" / "models_images_v1"
V2_SETS = ROOT / "research" / "llm-fine-tuning" / "experiments" / "decisions" / "generated_v2" / "decisions"  # 29.2
V2_OUT = ROOT / "research" / "llm-fine-tuning" / "experiments" / "decisions" / "models_v2"


GENERATIONS = {"v1": (SETS, OUT), "v2": (V2_SETS, V2_OUT), "images": (IMAGE_SETS, IMAGE_OUT)}
MIN_TRAIN = 50  # fewer training rows than this: no student (a reason, not a model)


def rows(point: str, sets: Path = SETS) -> list[dict[str, Any]]:
    """The training and dev rows of a point; the sealed file is not opened."""
    path = Path(sets) / point / "rows.jsonl"
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


def why_not(point: str, data: list[dict[str, Any]]) -> str | None:
    """Why a point gets no student (29.4), or None: labels from a source whose fields are not excluded yet, too few
    training rows, or a single class among them."""
    sources = {(r.get("label") or {}).get("source") for r in data if r.get("label")}
    if sources - {"benchmark"}:
        return f"labels from {sorted(sources - {'benchmark'})} need the fields they are measured from excluded (22.4)"
    train = [r for r in data if r["split"] == "train" and (r.get("label") or {}).get("value") is not None]
    if len(train) < MIN_TRAIN:
        return f"{len(train)} training rows (at least {MIN_TRAIN})"
    if len({str(r["label"]["value"]) for r in train}) < 2:
        return "a single class among the training rows"
    return None


def build_all(quick: bool = False, generations: tuple[str, ...] = tuple(GENERATIONS)) -> dict[str, dict[str, Any]]:
    """29.4: every generation's points: a student (its card written, its fitted file hashed) or a reason it has none,
    in each generation's ``SKIPPED.json``. One command, whatever a point holds."""
    out: dict[str, dict[str, Any]] = {}
    for gen in generations:
        sets, folder = GENERATIONS[gen]
        if not Path(sets).exists():
            out[gen] = {"_": "no decision sets (make decisions-gen first)"}
            continue
        skipped: dict[str, str] = {}
        done: dict[str, Any] = {}
        for point in sorted(p.name for p in Path(sets).iterdir() if p.is_dir()):
            reason = why_not(point, rows(point, sets))
            if reason is not None:
                skipped[point] = reason
                continue
            try:
                done.update(build([point], quick, sets=sets, out=folder))
            except Exception as exc:  # noqa: BLE001 - one point that cannot be trained never costs the others
                skipped[point] = f"{type(exc).__name__}: {str(exc)[:300]}"
        Path(folder).mkdir(parents=True, exist_ok=True)
        (Path(folder) / "SKIPPED.json").write_text(json.dumps(skipped, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        out[gen] = {**{k: c for k, c in done.items()}, **{k: f"skipped: {v}" for k, v in skipped.items()}}
    return out


def compare(generations: tuple[str, ...] = tuple(GENERATIONS)) -> list[str]:
    """29.4: every student beside its rule on dev, from the committed cards (nothing trained)."""
    lines = []
    for gen in generations:
        _, folder = GENERATIONS[gen]
        for card in sorted(Path(folder).glob("*.json")):
            if card.stem in ("fitted", "SKIPPED"):
                continue
            c = json.loads(card.read_text(encoding="utf-8"))
            d, r = c.get("dev") or {}, c.get("rule_dev") or {}
            lines.append(f"{gen:<7} {c['point']:<26} {c['family']:<9} dev {d.get('accuracy')} {d.get('accuracy_ci')}  rule {r.get('accuracy')} {r.get('accuracy_ci')}")
        skipped = Path(folder) / "SKIPPED.json"
        for point, why in (json.loads(skipped.read_text(encoding="utf-8")).items() if skipped.exists() else []):
            lines.append(f"{gen:<7} {point:<26} no student: {why}")
    return lines


def build(points: list[str] | None = None, quick: bool = False, sets: Path = SETS, out: Path = OUT) -> dict[str, Any]:
    import joblib

    from dclab_core.decision_models import train

    points = points or sorted(p.name for p in Path(sets).iterdir() if p.is_dir())
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cards = {}
    for point in points:
        data = rows(point, sets)
        sources = {(r.get("label") or {}).get("source") for r in data if r.get("label")}
        if sources - {"benchmark"}:  # a measured label comes from a state field, which must then be left out
            raise ValueError(f"{point}: labels from {sorted(sources - {'benchmark'})} need the fields they are measured "
                             "from excluded (22.4); this build trains on benchmark labels only")
        model, card = train(point, [r for r in data if r["split"] == "train"], [r for r in data if r["split"] == "dev"],
                            **({"families": ("logistic",), "strengths": (1.0,)} if quick else {}))
        (out / f"{point}.json").write_text(card.model_dump_json(indent=1) + "\n", encoding="utf-8")
        joblib.dump(model, out / f"{point}.joblib")
        cards[point] = card
    record_fitted(out)
    return cards


V3_SETS = ROOT / "research" / "llm-fine-tuning" / "experiments" / "decisions" / "generated_v3" / "decisions"  # step 5 (2026-10-25)
V3_OUT = ROOT / "research" / "llm-fine-tuning" / "experiments" / "decisions" / "models_v3"  # the mapped students (ledger.mapped)


def build_mapped(sets: Path = V3_SETS, out: Path = V3_OUT, bases: Path = V2_OUT) -> dict[str, Any]:
    """Step 5 (2026-10-25): each mapped student (``ledger.mapped``: a trained student of another point answering this one)
    saved like any student, its card measured on the point's own dev rows beside the rule. Nothing is trained; the base
    student is read only when its bytes match its own ``fitted.json``. The sealed rows are never read here."""
    import hashlib

    import joblib

    from dclab_core.decision_models import ModelCard, metrics

    from .ledger.mapped import MAPS, MappedAnswerer

    hashes = json.loads((Path(bases) / "fitted.json").read_text(encoding="utf-8"))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cards = {}
    for point, (base_point, _, _, _) in MAPS.items():
        fitted = Path(bases) / f"{base_point}.joblib"
        if hashlib.sha256(fitted.read_bytes()).hexdigest() != hashes.get(base_point):
            raise ValueError(f"{base_point}: the fitted student is not the one fitted.json names: never unpickled")
        base = joblib.load(fitted)
        base_card = json.loads((Path(bases) / f"{base_point}.json").read_text(encoding="utf-8"))
        model = MappedAnswerer(base, point)
        dev = [r for r in rows(point, sets) if r["split"] == "dev" and (r.get("label") or {}).get("value") is not None]
        truth, groups = [r["label"]["value"] for r in dev], [r["group"] for r in dev]
        answers = [getattr(model.answer(point, r["state"], list(r.get("options") or [])), "value", None) for r in dev]
        card = ModelCard(point=point, name=model.name, version=model.version, family=base_card["family"], strength=base_card["strength"],
                         features=base_card["features"], fields=sorted({f for r in dev for f in r["state"]}), trained_on=base_card["trained_on"],
                         trained_set=base_card["trained_set"], groups=base_card["groups"], label_sources=base_card.get("label_sources") or {},
                         chosen_by=f"not trained on this point: the {base_point} student ({base_card['name']}@{base_card['version']}) answers, "
                                   f"its answer mapped (ledger.mapped)",
                         calibration=f"the {base_point} student's own", dev=metrics(truth, answers, groups=groups) if dev else None,
                         rule_dev=metrics(truth, [r.get("value_used") for r in dev], groups=groups) if dev else None,
                         limits=["Measured on dev rows of generated_v3 (seeds generated_v2 never drew); the sealed partition is read once, by the gate.",
                                 f"The base student was trained on generated_v2's {base_point} homework: invented tables, not real ones.",
                                 "Its confidence is the base student's for its own answer, not calibrated for this point."])
        (out / f"{point}.json").write_text(card.model_dump_json(indent=1) + "\n", encoding="utf-8")
        joblib.dump(model, out / f"{point}.joblib")
        cards[point] = card
    record_fitted(out)
    return cards


def record_fitted(out: Path) -> dict[str, str]:
    """``fitted.json``: each fitted model's SHA-256, committed beside the cards. A model file is unpickled only when its
    bytes match (22.4): unpickling runs code, so a file swapped in the folder is refused before it is read."""
    import hashlib
    import json

    out = Path(out)
    hashes = {p.stem: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob("*.joblib"))}
    (out / "fitted.json").write_text(json.dumps(hashes, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return hashes


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.decision_models_build", description=__doc__.split("\n")[0])
    parser.add_argument("--points", default="")
    parser.add_argument("--quick", action="store_true", help="one candidate per point (for a quick look)")
    parser.add_argument("--images", action="store_true", help="26.8: the image points' sets, to models_images_v1/")
    parser.add_argument("--v2", action="store_true", help="29.2: the table-level points' sets, to models_v2/")
    parser.add_argument("--all", action="store_true", help="29.4: every generation, a student or a reason (SKIPPED.json) for every point")
    parser.add_argument("--compare", action="store_true", help="29.4: every student beside its rule on dev, from the cards")
    parser.add_argument("--mapped", action="store_true", help="step 5: the mapped students (ledger.mapped) on generated_v3, to models_v3/")
    args = parser.parse_args(argv)
    if args.mapped:
        for point, c in build_mapped().items():
            d, r = c.dev, c.rule_dev
            print(f"{point:<26} mapped dev accuracy {d.accuracy} {d.accuracy_ci} precision {d.precision} recall {d.recall} | rule {r.accuracy} / {r.precision} / {r.recall}")
        return 0
    if args.compare:
        print("\n".join(compare()))
        return 0
    if args.all:
        for gen, made in build_all(args.quick).items():
            for point, c in made.items():
                print(f"{gen:<7} {point:<26} " + (c if isinstance(c, str) else f"{c.family} dev {c.dev.accuracy} rule {c.rule_dev.accuracy}"))
        return 0
    sets, out = (V2_SETS, V2_OUT) if args.v2 else (IMAGE_SETS, IMAGE_OUT) if args.images else (SETS, OUT)
    if not sets.exists():
        print(f"No decision sets: run make decisions-gen{' ARGS=--images' if args.images else ''} first.", file=sys.stderr)
        return 2
    cards = build([p for p in args.points.split(",") if p] or None, args.quick, sets=sets, out=out)
    for point, c in cards.items():
        d, r = c.dev, c.rule_dev
        print(f"{point:<26} {c.family}:{c.strength:<5} dev accuracy {d.accuracy} [{d.accuracy_ci[0]}, {d.accuracy_ci[1]}]  "
              f"precision {d.precision} recall {d.recall} | rule {r.accuracy} / {r.precision} / {r.recall}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
