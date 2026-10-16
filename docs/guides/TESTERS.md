# Trusted friends testing the DCLab R&D studio

Package 25.11. Each friend clones the repository and runs DCLab on their own machine, with their own Anthropic key.
Everything they do is recorded in their workspace folder, and `make share` sends it to the owner's private records
repository; the owner's `make collect` brings every tester's records into the owner's workspace.

## The owner, once

1. Make a **private** GitHub repository for the records, for example `dclab-tester-records`, with a first commit
   (a README).
2. Invite each friend to it with write access (Settings → Collaborators), and give them read access to this code
   repository.
3. Clone the records repository on your machine, for example next to this one, and put its path in `.env`:
   `DCLAB_TESTER_RECORDS=/path/to/dclab-tester-records`.

## Each friend, once

```bash
git clone git@github.com:Shahriyar-Moradi/Research_Development_DCLab.git
cd Research_Development_DCLab
python -m venv .venv && .venv/bin/pip install -r requirements/base.txt pyarrow   # the README's Install
python -m dclab_rnd.testers notice
make tester-setup NAME="Ali Rezaei" EMAIL=ali@example.com REPO=git@github.com:OWNER/dclab-tester-records.git AGREE=yes
```

`tester-setup` refuses without `AGREE=yes`. It gives the tester an id, turns the workspace's consent on (improve and
train), turns Claude beside DCLab on, caps the month at 15 (`DCLAB_WORKSPACE_MONTHLY_EUR` in their `.env`) and clones
the records repository into their workspace. Then the friend adds their own key to `.env`
(`ANTHROPIC_API_KEY=…`, `DCLAB_LLM_PROVIDER=anthropic`), keeps `DCLAB_DATABASE_URL` empty (a tester's workspace keeps its
records in files; a share refuses otherwise) and starts DCLab with `make notebook`.

## Each friend, now and then

```bash
make share                      # asks once per table (again when its content changes); ARGS="--tables no" never sends tables
make share ARGS=--forget-tables # decide every table again
```

Only the files that changed since the last share are sent, whole: the session record, the decision ledger, the
audit, the twin folder (Claude beside DCLab: both answers per step, Claude's conversation, its reading and thinking,
the picks, the budget raises), the intern's sessions, the projects' and drafts' records, the datasets built, and the
tables they said yes to. When they said no to a table, the values of its cells are taken out of the records that quote
them (a column's examples, previews, top values and categories). A file they deleted is deleted on the owner's side
too. Never `.env`, a key, or a file over 45 MB. If they are offline, the bundle is committed and sent by the next
`make share`. A share needs the workspace's consent on: turning it off stops sharing.

## The owner, now and then

```bash
make collect                    # reads DCLAB_TESTER_RECORDS from .env, pulls, rebuilds each tester under agent_runs/testers/<id>/
```

Each tester's folder is laid out like a workspace; a file that does not match its manifest's hash, a link, or a path
outside the tester's folder is never applied, and a bundle whose manifest cannot be read is skipped and said.
Any tester with write access could write into another tester's folder of a shared repository: if that matters, make one
repository per tester. The **Evaluation** tab on Admin includes every collected tester
(Claude beside DCLab step by step, and every model's precision, recall, F1 and accuracy); `make collect` ends with a
table per tester (sessions, drafts, projects, Claude's turns, decisions, model spend); the session record of one tester
is read with `DCLAB_AGENT_HOME=agent_runs/testers/<id> python -m dclab_rnd.experience sessions`.

## What a tester agrees to

`python -m dclab_rnd.testers notice` prints it: their problems, messages, every answer and decision DCLab and Claude
make, their projects and results, and the cost of each model request are recorded and sent, as they are, to the
owner's private repository, to improve DCLab and train its models; a table goes only with their yes; their key never
leaves their machine; they can stop sharing and ask the owner to delete what they sent (delete their folder in the
records repository and under `agent_runs/testers/`).

## Testing on a shared server instead

If friends should use one server rather than their own machines: turn sign-in on (`DCLAB_AUTH=password`,
`DCLAB_SECRET_KEY`, PostgreSQL, `python -m dclab_rnd.storage upgrade`, then
`python -m dclab_rnd.accounts add --email you@example.com --role owner`), add each friend on Admin → Members (each
person's monthly model budget is 15 by default, set per person there), and serve it over HTTPS with its host name in
`DCLAB_ALLOWED_HOSTS` (a cloud deployment from `deploy/`, or a tunnel; the owner's to set up). Everything each person
does is then kept in that server's workspace under their account.
