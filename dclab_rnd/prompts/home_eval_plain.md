A person wrote one request to a data tool. Say what kind of help it asks for and what it states.
Tracks: "prediction" (a model that predicts something not yet known), "analysis" (numbers about what already happened),
"preparation" (cleaning, combining or reshaping data), "reporting" (a report or dashboard that is refreshed),
"business" (a business question or decision: why something changed, what to do), "advice" (an explanation, no data work).
Exits: "ml_project", "analysis", "dataset", "dashboard", "decision_brief", "answer".
Answer with one JSON object and nothing else: {"tracks": [the tracks it needs, the main one first], "exit": one exit,
"fields": {name: the words of the request that say it}} with only the fields the request states, from: target,
prediction_moment, action, measure, breakdown, period, filters, inputs, output_grain, rules, destination, kpis,
audience, refresh, decisions_supported, decision, options, success_measure, period_compared, constraints, topic.
The tables lists the column names of the person's files; it holds no values.
