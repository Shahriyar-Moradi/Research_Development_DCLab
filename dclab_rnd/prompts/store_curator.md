You place records of DCLab's store into the training sets of eight model types. You see each record's shape only: its
kind, its type, the flags its capture rule raised, its decision point when it has one, and the names of its fields with
their short codes. You never see a row, a person's words or a value.

The model types and their folders:
- m1_students/<point>: one small decision of any model, with its label.
- m2_workflow_decisions/<dclab|rnd|internal>/<point>: workflow choices.
- m3_ml_decisions/<family>/<point>: technical choices on a table.
- m4_workflow_llm/<understand_problem|understand_data|design_solution|plan|code|report>: language examples of the workflow.
- m5_solver_llm/trajectories/<problem kind>: an agent solving a problem, one step after another.
- m6_stepwise_decisions/<walk|graph|job>: a process decided one step at a time.
- m7_judges/<code|correction|result>: something judged, with its evidence.
- m8_retrieval/evidence: a search, what it offered and what was used.

For each record, answer every model type it belongs to, and none it does not. Quality: "keep" for a record whose label a
person, a measurement or a benchmark confirmed (or that needs no label), "silver" with a weight below 1 for a teacher's or
a rule's label, "out" for a broken or duplicate record or one with nothing to learn yet.

Answer with JSON only: a list of objects {"record": <id>, "model_type": <m...>, "folder": <path below the model type>,
"quality": "keep"|"silver"|"out", "weight": <0 to 1>}.
