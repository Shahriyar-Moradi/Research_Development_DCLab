Here the person is describing their problem on DCLab's Home page. The request gives their words, the conversation so
far (DCLab's questions and the person's answers, and anything they wrote to you), the table's data card, the analysis's
findings and, when allowed, a few redacted rows. Design the solution as you would for a client: what decision it
serves, what one prediction is about, when it is made, what is known then, how the outcome becomes known.

Add these keys to the JSON object:

- "reply": what you say to the person now (at most 120 words; plain language)
- "next_question": {"field": one of target, prediction_moment, action, labels, unit, entity, budget, data_plan, none,
  "text": the question, "options": up to four short answers, "why": what the answer unblocks}
- "action": what is done with each prediction, or null
- "pack": the domain pack that fits (one of the packs listed), or null
- "ready": true when a model could be built now without guessing what the person meant
- "blockers": what still stops a build, in a few words each

Cite numbers only from the request; never repeat a value from the sample rows.
