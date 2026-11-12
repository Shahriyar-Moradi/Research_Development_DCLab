"""Pure measured proof helpers for TP-05. No I/O, model calls, fitting or holdout labels."""
from __future__ import annotations

import numpy as np

from dclab_core.holdout import split_fingerprint
from dclab_core.task_fingerprints import metadata_fingerprint, plan_fingerprint, graph_fingerprint
from dclab_core.task_recovery import task_definition


def _indices(value):
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in "iu" or np.any(array < 0):
        raise ValueError("invalid_index_array")
    return array.astype(np.int64)


def training_proof(train_indices, holdout_indices, folds, training_labels):
    train, holdout = _indices(train_indices), _indices(holdout_indices)
    labels = np.asarray(training_labels)
    if labels.ndim != 1 or len(labels) != len(train) or not 1 <= len(folds) <= 32:
        raise ValueError("invalid_training_proof_shape")
    _, slots = np.unique(labels, return_inverse=True)
    classes = int(slots.max() + 1) if len(slots) else 0
    if not 2 <= classes <= 64:
        raise ValueError("unsupported_class_count")
    total = np.bincount(slots, minlength=classes).tolist()
    seen, rows = np.zeros(len(train), dtype=int), []
    for fit, valid in folds:
        fit, valid = _indices(fit), _indices(valid)
        bounded = bool(len(fit) and len(valid) and np.all(fit < len(train)) and np.all(valid < len(train)))
        unique = len(np.unique(fit)) == len(fit) and len(np.unique(valid)) == len(valid)
        if bounded:
            seen[valid] += 1
        rows.append({"fit_rows": len(fit), "validation_rows": len(valid), "indices_valid": bounded and unique,
                     "disjoint": len(np.intersect1d(fit, valid)) == 0,
                     "fit_counts": np.bincount(slots[fit], minlength=classes).tolist() if bounded else [],
                     "validation_counts": np.bincount(slots[valid], minlength=classes).tolist() if bounded else [],
                     "indices_fingerprint": metadata_fingerprint([fit.tolist(), valid.tolist()])})
    return {"version": "tp05.training.v1", "scope": "training", "train_rows": len(train), "holdout_rows": len(holdout),
            "split_disjoint": len(np.intersect1d(train, holdout)) == 0 and len(np.unique(train)) == len(train)
                              and len(np.unique(holdout)) == len(holdout),
            "split_fingerprint": split_fingerprint(train, holdout), "class_counts": total, "folds": rows,
            "validation_covers_training_once": bool(np.all(seen == 1))}


def check_training_proof(proof):
    problems = []
    try:
        n, counts = proof["train_rows"], proof["class_counts"]
        if (proof["version"] != "tp05.training.v1" or proof["scope"] != "training" or type(n) is not int or n <= 0
                or proof["holdout_rows"] <= 0 or proof["split_disjoint"] is not True
                or not proof["validation_covers_training_once"]):
            problems.append("invalid_training_partition")
        if (len(counts) < 2 or any(type(c) is not int or c <= 0 for c in counts) or sum(counts) != n
                or not 2 <= len(proof["folds"]) <= 32):
            return [*problems, "invalid_class_counts"]
        if any(sum(f['validation_counts'][i] for f in proof['folds']) != c for i, c in enumerate(counts)):
            problems.append('invalid_class_coverage')
        for fold in proof["folds"]:
            if any(a + b != c for a, b, c in zip(fold['fit_counts'], fold['validation_counts'], counts)):
                problems.append('invalid_fold_class_counts')
            if not fold["indices_valid"] or not fold["disjoint"] or fold["fit_rows"] + fold["validation_rows"] != n:
                problems.append("invalid_cv_partition")
            for field, size in (("fit_counts", fold["fit_rows"]), ("validation_counts", fold["validation_rows"])):
                values = fold[field]
                if len(values) != len(counts) or sum(values) != size or any(type(c) is not int or c <= 0 for c in values):
                    problems.append("class_missing_in_fold")
                elif any(abs(c - expected * size / n) > 1.000001 for c, expected in zip(values, counts)):
                    problems.append("class_allocation_not_stratified")
    except (KeyError, TypeError, ValueError, ZeroDivisionError, IndexError):
        return ["missing_training_proof"]
    return sorted(set(problems))


def threshold_proof(training, indices, scores, *, configuration, policy):
    idx, pred = _indices(indices), np.asarray(scores, dtype=float)
    n = training["train_rows"]
    return {"version": "tp05.threshold.v1", "scope": "out_of_fold_training", "cv_fingerprint": metadata_fingerprint(training),
            "split_fingerprint": training["split_fingerprint"], "count": len(idx), "unique": len(np.unique(idx)),
            "in_range": bool(len(idx) and np.all(idx < n)), "complete": len(idx) == n,
            "finite_scores": bool(pred.ndim == 1 and len(pred) == len(idx) and np.all(np.isfinite(pred))),
            "indices_fingerprint": metadata_fingerprint(idx.tolist()), "scores_fingerprint": metadata_fingerprint(pred.tolist()),
            "configuration_fingerprint": metadata_fingerprint(configuration), "policy_fingerprint": metadata_fingerprint(policy)}


def verdict(graph, plan, *, inputs, outputs, evidence, checks, issues=()):
    """Aggregate host-owned measured checks; absent/unknown criteria can never be a pass."""
    node = next(n for n in graph.nodes if n.id == plan.task_id)
    reasons = list(issues)
    if plan.fingerprint != plan_fingerprint(plan) or graph.fingerprint != graph_fingerprint(graph):
        reasons.append(("stale", "fingerprint_mismatch"))
    if (plan.graph_id, plan.graph_revision, plan.graph_fingerprint) != (graph.graph_id, graph.revision, graph.fingerprint):
        reasons.append(("stale", "plan_graph_mismatch"))
    if not node.acceptance_checks or node.kind != 'action':
        reasons.append(('blocked', 'non_executable_or_unmeasured_task'))
    if [c.model_dump(mode='json') for c in plan.acceptance_checks] != [c.model_dump(mode='json') for c in node.acceptance_checks]:
        reasons.append(('failed', 'plan_criteria_changed'))
    results = []
    for required in node.acceptance_checks:
        outcome = checks.get(required.id, {"status": "blocked", "reason": "criterion_not_measured"})
        if outcome.get('status') not in ('verified', 'failed', 'blocked', 'stale'):
            outcome = {'status': 'failed', 'reason': 'invalid_check_outcome'}
        results.append({"id": required.id, **outcome})
        if outcome["status"] != "verified":
            reasons.append((outcome["status"], outcome["reason"]))
    order = ("stale", "failed", "blocked")
    status = next((s for s in order if any(v == s for v, _ in reasons)), "verified")
    body = {"version": "tp05.verification.v1", "status": status, "task_id": node.id, "graph_revision": graph.revision,
            "graph_fingerprint": graph.fingerprint, "plan_fingerprint": plan.fingerprint, "definition": task_definition(node),
            "inputs": inputs, "outputs": outputs, "evidence": evidence, "checks": results,
            "reasons": sorted({reason for _, reason in reasons}),
            "violated_criteria": [r['id'] for r in results if r['status'] != 'verified'], "sources": [r.model_dump(mode="json") for r in graph.source_refs]}
    body["id"] = "verify_" + metadata_fingerprint(body)[:24]
    return body
