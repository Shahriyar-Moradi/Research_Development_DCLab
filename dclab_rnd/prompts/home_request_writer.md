You write first messages that people type into DCLab's Home chat, a data tool that helps with predictions, numbers
from data, data preparation, dashboards, business questions and advice. Each message is the person's very first words,
typed into a chat box, before DCLab has asked anything.

You are given a list of slots. Each slot fixes, for one person:
- "kind": the kind of help this person really needs (described in "kind_means"). Every message you write must truly
  need this kind of help, whatever words the person uses;
- "need": what exactly they want inside that kind (described in "need_means");
- "second_kind" (only in some slots): a smaller second thing they also ask for, of another kind; their main goal is
  still "kind";
- "language": "en" (English), "fa" (Persian, in Persian script) or "mixed" (described in "language_means");
- "style": how they write (described in "style_means");
- "business": the person's line of work; use it for the details.

Write one message per slot, exactly as that person would type it.
- Many people never use the textbook word for what they need: in about half of the messages, do not use it (for
  example, do not say "predict", "analyse", "dashboard", "clean" or "advice"; say what they want in their own words).
- Vary openings, length within the style, tone and details. No two messages alike; no message starts like another.
- Never write a table's values, a person's name or any personal data. Column or file names are fine.
- Do not explain or label the message; write only the person's words.

Reply with one JSON object and nothing else: {"requests": [{"slot": "<the slot id>", "text": "<the message>"}, ...]},
one entry per slot, in the order given.
