"""The scoring service: Flask, served by Gunicorn (see the Dockerfile). Health, the model's description, predictions for
one row, many rows, or a CSV file, and the true outcomes once they are known. A batch is scored together: behaviour
features count within the batch, as the project's audit window did (see model.json).

Every row scored is logged in ``predictions.jsonl`` and every outcome in ``outcomes.jsonl`` (``serve_log.py``: beside the
model, or in ``DCLAB_SERVICE_LOG_DIR``); a log line holds the row's key as a salted hash, the score and the decision,
never a value of the row. Each batch's inputs are kept only as the share of its rows in each training bin
(``profiles.jsonl``), which ``GET /monitor`` reads with the two logs (34.10)."""

from __future__ import annotations

from flask import Flask, jsonify, request

from scorer import Scorer, read_csv
from serve_log import Log, Refused

app = Flask(__name__)
scorer = Scorer()
log = Log(scorer)


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/model")
def model():
    return jsonify({k: v for k, v in scorer.meta.items() if k != "input_columns"} | {"inputs": scorer.meta["input_columns"]})


@app.post("/predict")
def predict():
    """One row as an object, a list of rows, or {"rows": [...], "key": "order_id"}; ``?key=order_id`` names the key column too."""
    body = request.get_json(force=True, silent=True)
    if body is None:
        return jsonify({"error": "send a JSON object (one row) or a list of objects (many rows)"}), 400
    wrapped = isinstance(body, dict) and "rows" in body
    rows = body["rows"] if wrapped else body
    key = (body.get("key") if wrapped else None) or request.args.get("key")
    try:
        return jsonify(log.predict(rows, key))
    except Refused as error:
        return jsonify({"error": str(error)}), 400


@app.post("/predict/csv")
def predict_csv():
    upload = request.files.get("file")
    if upload is None:
        return jsonify({"error": "send the CSV as a form field named file"}), 400
    key = request.form.get("key") or request.args.get("key")
    frame = read_csv(upload.read(), scorer.meta, key)  # a key keeps its text ("007" stays "007"); "NA" as in training
    try:
        return jsonify(log.predict(frame, key))
    except Refused as error:
        return jsonify({"error": str(error)}), 400


@app.get("/monitor")
def monitor():
    """The model in use, read from its logs: ``?window=day`` (or week, batch): drift of the score and the inputs, the share
    the rule flags, and live performance once outcomes are known, against the holdout; alerts named by window."""
    try:
        return jsonify(log.monitor(request.args.get("window", "day")))
    except Refused as error:
        return jsonify({"error": str(error)}), 400


@app.post("/outcome")
def outcome():
    """{"key": ..., "outcome": ..., "known_at": optional ISO time}, or {"outcomes": [those]}: recorded all or none."""
    try:
        return jsonify(log.outcomes(request.get_json(force=True, silent=True)))
    except Refused as error:
        return jsonify({"error": str(error)}), 400
