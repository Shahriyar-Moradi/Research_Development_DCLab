"""The draft: the workspace a user fills on Home before a project exists.

A user describes a problem in one sentence, brings data (upload, a studied sample, a connector, or
synthetic rows from a prompt), and talks with the Home agent. In the background the data is turned
into a table, cleaned structurally (lossless, logged) and described; the agent asks a few questions
and keeps a solution workflow up to date. "Let's build the solution" turns the draft into a project.

Modules:

- ``store``      drafts on disk (one folder per draft) and their event log for the live stream
- ``structure``  any file to a table: tabular files, JSON lines, nested JSON, logs, plain text
- ``clean``      structural, lossless cleaning with a step-by-step log (no statistics learned)
- ``analyze``    descriptive analysis for charts: distributions, top values, missingness, correlations
- ``synthetic``  a validated schema spec and seeded generation of clearly labelled synthetic rows
- ``pack``       which domain pack fits: the user's choice wins, otherwise text and data signals
- ``workflow``   the solution workflow graph, validated against the ten DCLab workflow blocks
- ``pipeline``   the background job: structure → clean → analyse one asset, streamed as events
- ``chat``       the Home agent: short questions, workflow updates; a model when configured
- ``api``        the HTTP routes the server mounts
"""
