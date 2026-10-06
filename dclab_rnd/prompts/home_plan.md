Read the user's problem sentence for a machine-learning project. For each field say what the sentence
already tells: target (what exactly is predicted, with its time window), prediction_moment (when the prediction is
made), action (what is done with each prediction and what a wrong one costs). Answer with JSON only:
{"target": {"status": "stated|inferred|unknown", "value": "...", "quote": "..."}, "prediction_moment": {...}, "action": {...}}
"stated": the sentence says it; "inferred": it clearly implies it; "unknown": it does not say. The quote must be
words copied exactly from the sentence. Never guess a value the sentence does not support.
