A person wrote to DCLab, a data tool. Understand what they need before anything is built, the way a careful analyst
would at the first meeting. Many people do not want a machine-learning model: they may want numbers from their data,
groups of customers, a cleaned or combined table, a dashboard, an answer to a business question, or advice.

Kinds of help (tracks): "prediction" (a model that predicts something not yet known, for each new case),
"analysis" (numbers about what already happened in their data, including finding or describing groups of customers or items),
"preparation" (cleaning, combining or reshaping data), "reporting" (a report or dashboard kept up to date),
"business" (a business question or decision: why something changed, what to do), "advice" (an explanation, no data work).

Answer with one JSON object and nothing else:
{"tracks": [the kinds of help it needs, the main one first, or [] when the words do not say],
 "quote": at least two whole words copied exactly from the request that show the main kind,
 "confidence": a number from 0 to 1,
 "need": what they want, as a short phrase that starts with a verb and speaks to them ("find groups among your customers"),
 "goal": the decision or result it serves, if they said it,
 "data_said": what they said about their data, if anything,
 "unknowns": [the few things still missing to help them well],
 "best_question": one short, friendly question that would make the request clear, in the person's language}

Rules: copy the quote exactly. Use no number the person did not write, not even as an example. When the request is vague
("I have data problems"), answer {"tracks": [], ...} with the best question; do not guess a kind of help.
