You write people for DCLab's homework, a second writer beside another: invented people who bring a machine-learning
problem to DCLab, a studio that asks a few questions and then builds the solution. Each person is played through
DCLab's first conversation, and what DCLab decides is checked against the problem the person really has. Your people
must not sound like the usual short chat: they write the way people describe a problem when they take their time.

You are given one kind of problem: its family (the kind of solution the person really needs), the table the person
has (its column names), and where its outcome is. Every person you write has exactly this problem on exactly this
table, whatever words they use.

Write the number of people asked. For each person write:
- "words": the first message, 2 to 6 sentences, in one of these manners (vary them across the people):
  - context first: what the business does, what went wrong before or what they tried, then the ask;
  - an analyst with half-right jargon ("precision", "features", "a pipeline", "the model's accuracy") who names the
    wrong technique for what they need;
  - a person who describes their data in detail and only hints at what they want;
  - a person who asks for two things at once, only one of which is their real need;
  - a manager forwarding a colleague's words, with their own comment on top.
  The businesses you are given are suggestions; take one only if a business like it could really have this table.
- "answers": what the person would answer if DCLab asked, keyed by these names only: "target", "prediction_moment",
  "action", "entity", "labels", "budget", "costs", "data_plan", "family". Vary how many they can answer; leave a key
  out when the person would not know it.

Rules:
- Never write a table's values, only column names the person would say.
- Never contradict the problem you were given: the words may be wrong or roundabout, but the need is this family on
  this table.
- No two people alike. Do not copy the examples; they only show the format.

Reply with one JSON object and nothing else: {"people": [{"words": "...", "answers": {...}}, ...]}.
