# Brief: eight held-out people with no data to share

**For a separate session.** The session that builds the no-data design must never read these people, their key or any
one person's result. Give this file, and nothing else from that work, to a fresh session.

## What to write

Eight written people who bring a problem to a data-science assistant and have **no table to share**: they have no data
yet, or cannot or will not share it. Each is one opening message, as a real person would write it, 60 to 140 words,
with the facts and numbers that person would really have. Write them from your own knowledge of businesses. Do not copy
or lightly reword any existing case.

The eight must include:

- two who never use the words "no data" or "dataset" (it is clear from what they say);
- two who refuse to share for privacy, legal or company-policy reasons;
- two whose real need is **not** a prediction model (a business decision, a process change, a measurement, a plan for
  what to record);
- one who writes in Persian;
- one who asks for made-up (simulated) data or a demonstration on it;
- one plain control: a clear prediction problem with the numbers needed to reason about cost and benefit.

One person may cover two of these. Vary the industries. In at least three, the obvious first answer is wrong or
incomplete (for example: the outcome is only seen for some cases, the action changes the outcome, the number they ask
to predict is not the decision, the rate they quote has no period).

## Files and their exact shape

`dclab_rnd/agent_eval/harness_ablation/cases_nodata_held.json`: a list of eight objects with exactly these fields:

```json
{"id": "NH-01", "stratum": "short_snake_case_name", "upload": null,
 "opening": "the person's message",
 "reply": "what this person says if asked again for a file, in their own language, one or two sentences: they have none or will not share it, and ask to continue without it"}
```

`dclab_rnd/agent_eval/harness_ablation/answer_key_nodata_held.json`: an object keyed by the same ids:

```json
{"NH-01": {
  "must_understand": ["2 to 5 points a good answer must show it understood, each a full sentence about this person"],
  "acceptable_approaches": ["1 to 3 sound ways to help this person now, without data"],
  "forbidden": ["things a good answer must not do, specific to this person",
                "tells the person it cannot help, or cannot go further, until they upload or share a table"],
  "right_is_model": true,
  "in_six_tracks": true,
  "route": "advice",
  "also_right": ["business", "prediction", "analysis"],
  "adaptation_check": null}}
```

- `right_is_model`: false when a model is not the right answer for this person now.
- `route` is one of `prediction`, `analysis`, `preparation`, `reporting`, `business`, `advice`; use `business` when the
  need is a business decision, otherwise `advice`. `also_right` lists the others that would be acceptable.
- A key never asks for a figure the person did not give. A good answer for these people uses only the person's own
  numbers (and sums of them), says what cannot be known without real data, and never invents performance or savings.
- For the person who asks for made-up data: a good answer first asks how the data should look and proposes a design for
  their yes, creates nothing in its first message, and says that a score on made-up data says nothing about the real
  business.

## Do not read

`dclab_rnd/nodata/`, `dclab_rnd/draft/no_data*.py`, `dclab_rnd/agents/`, `dclab_rnd/prompts/`,
`cases_nodata.json`, `answer_key_nodata.json`, `cases_dev3.json`, `answer_key_dev3.json`, anything under
`evidence/campaigns/` or `agent_runs/`. You may read `cases_dev.json` for the tone of an opening only.

## When done

```bash
.venv/bin/python -m dclab_rnd.agent_eval.harness_ablation --set nodata_held --freeze
```

It checks the files and writes `cases_nodata_held.sha256`. Commit the three files. Tell the other session only:
"the held-out no-data people are ready".
