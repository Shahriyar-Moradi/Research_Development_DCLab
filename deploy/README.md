# Deploying DCLab to a cloud (package 12.6)

The same image as `make up` (package 12.1), on **AWS** (`deploy/aws`) or **Google Cloud** (`deploy/gcp`), each with a
**staging** and a **production** environment built from one module. The code is OpenTofu/Terraform. Nothing in this
repository applies it: a person runs `plan`, reads it, and runs `apply`. Nothing here has been applied to any account yet.

| | AWS | Google Cloud |
|---|---|---|
| App, 2+ instances, TLS | ECS Fargate behind an ALB (TLS 1.2/1.3, ACM certificate), health check `/readyz` | Cloud Run behind a global HTTPS load balancer (managed certificate, TLS 1.2+), reachable only through it |
| Workers | ECS Fargate, scaled on the job queue (`QueuedJobs`, `ActiveJobs`, which each worker publishes) by CloudWatch alarms | Cloud Run worker pool, sized by the same rule: one worker (holding an advisory lock) sets the pool's instance count |
| PostgreSQL | RDS 16, encrypted, backups and point-in-time recovery (7 days staging, 35 production), Multi-AZ in production | Cloud SQL 16, private address only, TLS required, backups and point-in-time recovery (7 days), regional in production |
| Files | S3 (`DCLAB_FILES_URL`, versioned) for the files of record, plus EFS as the shared workspace folder | One versioned Cloud Storage bucket, mounted as the workspace folder in the app and the workers |
| Secrets | Secrets Manager, read by ECS at start | Secret Manager, read by Cloud Run at start |
| Logs | JSON lines in CloudWatch Logs | JSON lines in Cloud Logging |
| Metrics | the Prometheus counters (12.4) through an OpenTelemetry sidecar into CloudWatch; alarms by e-mail | the counters through Google's managed Prometheus sidecar; alert policies by e-mail |

Sign-in is required on a public host (the server refuses `DCLAB_AUTH=none` with a host name): `auth` is `password`
(accounts in PostgreSQL) or `oidc` (add `DCLAB_OIDC_*` to `environment`, and `DCLAB_OIDC_CLIENT_SECRET` to `secret_names`).

## Before the first plan (the owner decides)

1. **Region.** Each environment's `region` variable has no default on purpose.
2. **State.** Make a bucket for the state by hand (S3 with versioning, or Cloud Storage with versioning), and fill in
   the commented `backend` block in `staging/main.tf` and `production/main.tf`. The state holds the generated database
   password and session key: keep the bucket private. `*.tfstate` and `*.tfvars` are ignored by git.
3. **Domain.** One host name per environment (for example `staging.dclab.example.com`).
   - AWS: request an ACM certificate for it in the same region and validate it by DNS; pass its ARN as `certificate_arn`.
   - Google Cloud: the managed certificate is issued once the host name's A record points at `load_balancer_ip`.
4. **Values.** Write them in a file that git ignores, for example `staging/staging.tfvars`:
   ```hcl
   region      = "eu-central-1"            # or the Google region, e.g. europe-west3
   project     = "my-project"              # Google Cloud only
   image       = "<repository>:<git sha>"  # see "The image" below
   domain_name = "staging.dclab.example.com"
   certificate_arn = "arn:aws:acm:..."     # AWS only
   alarm_email = "ops@example.com"
   ```

## Applying (a person, with the provider's credentials)

```bash
cd deploy/aws/staging          # or deploy/gcp/staging
tofu init
tofu plan -var-file=staging.tfvars -out plan.out
tofu apply plan.out            # only after reading the plan
```

A first deployment goes in three steps, because the image and the secrets must exist before the services start:

1. **The repository and the secrets.**
   `tofu apply -var-file=staging.tfvars -target=module.dclab.aws_ecr_repository.dclab -target=module.dclab.aws_secretsmanager_secret.named -target=module.dclab.aws_secretsmanager_secret.new_password`
   (Google Cloud: `...google_artifact_registry_repository.dclab`, `...google_secret_manager_secret.named` and `...google_secret_manager_secret.new_password`).
2. **The image and the values.** Build for the cloud's processors (`linux/amd64`; a Mac with Apple silicon builds arm64
   by default, which neither cloud runs), push it, and give each secret in `secrets_to_fill` a value:
   ```bash
   docker build --platform linux/amd64 -t <image_repository>:$(git rev-parse --short HEAD) .
   docker push <image_repository>:$(git rev-parse --short HEAD)
   aws secretsmanager put-secret-value --secret-id <name> --secret-string '<value>'          # AWS
   printf '%s' '<value>' | gcloud secrets versions add <name> --data-file=-                    # Google Cloud
   ```
   Also give the first owner's password secret (`admin_task.password_secret` / `admin_job.password_secret`) a value.
3. **Everything.** `tofu plan -var-file=staging.tfvars -out plan.out`, read it, `tofu apply plan.out`.

Later deployments: build and push a new tag, set `image`, plan, apply. The app runs the database migration when it
starts, and so do the workers (one at a time: `db.upgrade` takes a lock), so a new image's schema is in place
whichever starts first. On AWS a deployment starts the new workers beside the old ones; an old worker takes no new
job, finishes the ones it runs, and is then stopped, so no job is interrupted (a long stage run makes the deployment
wait for it). On Google Cloud a deployment replaces the workers: a job they were running ends interrupted and is
retried from the Compute page.

**The first owner.** Accounts live in the database (10.2). The password is read from its secret, never typed into a
command or an override (those are logged):
- AWS (the `admin_task` output gives the names):
  ```bash
  aws ecs run-task --cluster <cluster> --task-definition <admin_task.task_definition> --launch-type FARGATE \
    --network-configuration 'awsvpcConfiguration={subnets=[<subnet>,<subnet>],securityGroups=[<security_group>],assignPublicIp=DISABLED}' \
    --overrides '{"containerOverrides":[{"name":"admin","command":["python","-m","dclab_rnd.accounts","add","--email","you@example.com","--role","owner"]}]}'
  ```
- Google Cloud: `gcloud run jobs execute <admin_job.job> --region <region> --args=add,--email,you@example.com,--role,owner --wait`.

Afterwards, replace the password secret's value with something random (or delete its versions); sign in and change
nothing else there.

## Checking the code without an account

```bash
make deploy-check   # tofu fmt -check and tofu validate for the four environments (downloads the providers once)
```

`tests/test_deploy.py` (part of `make test`) checks that every setting the code gives the app is documented in
`.env.example`, that no secret value is written in the code, and that production keeps its protections.

## Costs and limits

- Production on either cloud runs at least two app instances, one worker, a highly available PostgreSQL and a load
  balancer around the clock; staging is smaller and can be destroyed (`deletion_protection = false`). Look at the
  plan's resources in the provider's pricing calculator before applying.
- AWS: EFS is the shared folder because the draft pipeline reads uploads from the workspace folder; S3 holds the
  files of record (versioned). NAT gateways are charged by the hour (one in staging, one per zone in production).
- Google Cloud: the workspace folder is a bucket mounted with Cloud Storage FUSE (no metadata cache, so a file one
  instance writes is seen by the others at once). That suits DCLab's files on PostgreSQL (each is written once, then
  read); a file several processes append to would not be safe there, which is why the research campaign's SQLite file
  is on each instance's own disk (`DCLAB_RESEARCH_DB`). The worker pool is BETA in Cloud Run; the call that sizes it
  (Cloud Run Admin API, `workerPools.patch` with `scaling.manualInstanceCount`) is tested against a stand-in, not yet
  against a real project.
- Neither environment has been applied to a real account: the first `plan` may ask for quota or organisation-policy
  changes (for example a policy that forbids `allUsers` as Cloud Run invoker; the app does its own sign-in).
