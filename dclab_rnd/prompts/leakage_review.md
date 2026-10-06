You review a machine-learning table for leakage. The prediction is made at the moment described below.
For each column that would NOT be known at that moment (written later, or derived from the outcome), and for each
column the audit flagged that you think IS known then, return an item. Answer with JSON only:
{"columns": [{"column": "...", "verdict": "after the moment" | "available", "reason": "one sentence that quotes words of the moment", "records": ["DCLAB-R01"]}]}
Cite only record ids from the list given. Leave out columns you are unsure about. You see names and summaries,
never values.
