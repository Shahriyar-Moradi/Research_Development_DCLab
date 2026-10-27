A person wrote a request to DCLab, a data tool. Say what kind of help it asks for.
Tracks: "prediction" (a model that predicts something not yet known), "analysis" (numbers about what already happened
in their data), "preparation" (cleaning, combining or reshaping data), "reporting" (a report or dashboard kept up to date),
"business" (a business question or decision: why something changed, what to do), "advice" (an explanation, no data work).
Answer with one JSON object and nothing else: {"tracks": [the tracks it needs, the main one first], "quote": the words
of the request, copied exactly, that show the main track, "confidence": a number from 0 to 1}.
The quote must be at least two whole words copied from the request. If the request does not say, answer {"tracks": []}.
