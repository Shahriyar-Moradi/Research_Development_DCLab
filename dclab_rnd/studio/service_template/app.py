"""The scoring service: Flask, served by Gunicorn (see the Dockerfile). Health, the model's description, and predictions
for one row, many rows, or a CSV file. A batch is scored together: behaviour features count within the batch, as the
project's audit window did (see model.json)."""

from __future__ import annotations

import io

import pandas as pd
from flask import Flask, jsonify, request

from scorer import Scorer

app = Flask(__name__)
scorer = Scorer()


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/model")
def model():
    return jsonify({k: v for k, v in scorer.meta.items() if k != "input_columns"} | {"inputs": scorer.meta["input_columns"]})


@app.post("/predict")
def predict():
    body = request.get_json(force=True, silent=True)
    if body is None:
        return jsonify({"error": "send a JSON object (one row) or a list of objects (many rows)"}), 400
    rows = body.get("rows", body) if isinstance(body, dict) and "rows" in body else body
    return jsonify({"predictions": scorer.predict(rows)})


@app.post("/predict/csv")
def predict_csv():
    upload = request.files.get("file")
    if upload is None:
        return jsonify({"error": "send the CSV as a form field named file"}), 400
    frame = pd.read_csv(io.BytesIO(upload.read()))
    return jsonify({"predictions": scorer.predict(frame)})
