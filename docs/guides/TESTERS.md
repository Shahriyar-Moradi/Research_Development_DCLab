# Trusted friends testing the DCLab R&D studio

How to let a few trusted people sign in and test the studio, and where everything they do is kept. Nothing here is
run by tooling: each step is the owner's.

## 1. Turn sign-in on

Accounts need PostgreSQL (`make dev` already uses the local `dclab_dev` database).

1. In `.env`: `DCLAB_AUTH=password`, `DCLAB_SECRET_KEY=<a long random value>` and `DCLAB_DATABASE_URL` (as `make dev`
   sets it). Keep `ANTHROPIC_API_KEY` there for Claude.
2. Bring the schema up to date: `python -m dclab_rnd.storage upgrade`.
3. Make yourself the first owner: `python -m dclab_rnd.accounts add --email you@example.com --role owner` (the
   password is asked for, never typed on the command line).
4. Restart the server. A sign-in page now comes before anything of the workspace.

## 2. Add each friend

Admin → **Members** → the email, a name, the role and a first password (10 characters or more), then **Add**. For
testers, the role **ML engineer** (they can create drafts and projects and run stages) or **Reviewer** (they can
also label decisions). Every addition, role change and removal is in the audit log; removing a member ends their
sessions and API tokens at once.

## 3. Their budgets

Every signed-in person may spend **15 a month** on models by default (`DCLAB_LIMIT_USER_MONTHLY_EUR`; prices are
entered in dollars). Set one person's own on Admin → **Members** → *Model budget a month* → **Save** (audited). On
Home, **Add €1.50** raises one draft's Claude budget within what is left of the person's month. When a person's month
is used up, every model request for them is refused with that reason; the rules go on without the model.

## 4. Claude beside DCLab

Admin → **Budgets & data** → *Claude beside DCLab* (or `PUT /api/twin/settings`): `on` runs Claude, as the AI-first expert, beside every Home conversation;
`sample` also shows it up to 20 redacted rows (columns that hold a person's data left out). Each draft's Claude spends
at most its session budget (1.5 by default). A project's page has **Claude beside** for a whole twin run.

## 5. What is kept, and where

Every action of every person is kept with their account id:

| What | Where |
|---|---|
| Every visit: each page action, message, answer, decision, model request, job, error | the session record (`experience_event` on PostgreSQL); Admin → **Sessions** finds a person's sessions |
| Every decision DCLab made, with the rule's, the model's and the person's answers and the labels | the decision ledger; Admin → **Decisions** |
| Who did what (members, switches, budgets, approvals, deletions) | the audit log; Admin → **Audit log** |
| Claude beside the system: both answers to each Home step, Claude's conversation, its reading, its thinking, the person's picks, budget raises | `<workspace>/twin/` (its `README.md` says each file; `session.json` names the people) |
| Whole twin runs: both lanes, the comparison, Claude's expert record | `<workspace>/twin/runs/<run>/` |
| How Claude and the system did, and every model's precision, recall, F1 and accuracy | Admin → **Evaluation** |
| Training sets built from all of the above | `make datasets`, only under the workspace's train consent (Admin → consent), `<workspace>/datasets/` |

Tell your testers before they start that their sessions are kept and, with the train switch on, may become training
data for DCLab's models (redacted, never a table's values). A session can be deleted on request (Admin → Sessions),
which removes its events and decisions; the audit keeps that it was deleted.

## 6. Reaching the server

Friends must reach your server over the network: a cloud deployment (`deploy/`, the owner's to plan and apply), or a
tunnel to your machine. Either way serve it over HTTPS and list its host name in `DCLAB_ALLOWED_HOSTS`; the server
refuses other host names without sign-in.
