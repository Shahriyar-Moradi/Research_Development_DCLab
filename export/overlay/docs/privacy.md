# Privacy

Everything dclabai records stays on your machine (or in your PostgreSQL): projects, tables, chats, the decisions it made and
the audit trail. Nothing is sent anywhere except the requests to the model endpoint you configure, and those carry column
names, counts and scores, never a row of your data (each purpose's limit is written in `dclabai/models/settings.py`).

There is no telemetry. A future version may offer an explicit, opt-in command to share redacted records with the
maintainers, off by default and previewed before anything is sent.
