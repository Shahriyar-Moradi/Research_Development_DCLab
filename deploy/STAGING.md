# Staging in a cloud, step by step (package 14.7)

Nothing here runs from an assistant's session: planning needs the provider's credentials, and **applying needs the owner's
explicit yes.** Production is a separate yes. The cloud code itself is in this folder (README.md); this page is the order of
the work and what to record.

## The owner decides first

1. **The cloud** (`aws/` or `gcp/`), the **region**, the **account or project**, a **domain** for staging and who holds its
   DNS. (The code was written for both clouds; region and domain were left open on purpose.)
2. The model key goes into the cloud's secret store, never into a file in the repository (README, "Before the first plan").

## Then, in this order

| Step | Command (a person, with credentials) | Record |
|---|---|---|
| 1. Plan | `cd deploy/<cloud>/staging && tofu init && tofu plan -out staging.plan` | The plan's summary and the estimate of the monthly cost, in the playbook row |
| 2. Apply (**only after the owner's yes**) | `tofu apply staging.plan` | The address of staging |
| 3. Flows | `make staging-check URL=https://<staging address>` | The result of the product flows and of the 19-page browser check |
| 4. Backup and restore drill | `make backup`, then `make restore FROM=backups/<time> HOME_DIR=<new folder>` against a new, empty database | That the counts and hashes in the manifest match after the restore |
| 5. Stop the bill | `tofu destroy` when the test is done, unless the owner keeps staging | What was removed |

`make staging-check` runs the end-to-end flows (`scripts/product_e2e.py --base URL`) and the browser checks
(`scripts/browser_e2e.py --base URL`) against the running address. Two things differ from the local run:

- **Sign-in.** Staging asks for it (the server refuses `DCLAB_AUTH=none` on a host name), so both scripts need an API token of a member
  of a test workspace in `DCLAB_E2E_TOKEN` (environment only; make one with the accounts tool, and revoke it afterwards). The token
  header is tested; the checks have **never run against a signed-in server**, so the first run may need a fix, and that is part of the work.
- **Model requests.** On a server you started, whatever it is configured with applies: with the model key in staging's secret store
  (step 2 above) the flows **do send model requests** and spend a little. To keep them free, start staging for the check without the key,
  or accept the cost, which is small (the flows use made-up data).

## Done when

Staging passes the flows and the browser checks, and the plan, its cost and the drill are written in the playbook's 14.7 row.
