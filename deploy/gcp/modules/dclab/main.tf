# DCLab on Google Cloud (package 12.6): the image from 12.1 behind an HTTPS load balancer.
#
#   network     a VPC with private access to Google's services; Cloud Run reaches it by Direct VPC egress
#   database    Cloud SQL PostgreSQL 16 on a private address, automated backups and point-in-time recovery,
#               regional (two zones) when asked, TLS required
#   files       one versioned Cloud Storage bucket holds the workspace folder: the tables, uploads and exports
#               (9.3's object storage), mounted at /workspace in the app and the workers (Cloud Storage FUSE), so
#               both see the same files, as the compose volume does on one machine
#   app         a Cloud Run service, app_min..app_max instances, behind a global HTTPS load balancer with a
#               Google-managed certificate; reachable only through the load balancer
#   workers     a Cloud Run worker pool, worker_min..worker_max instances, scaled by Cloud Run; each worker also
#               publishes the queue's length (custom.googleapis.com/dclab/queued_jobs), and an alert says when
#               jobs keep waiting
#   secrets     Secret Manager: the database URL and the session key are made here; the others are created
#               empty and filled by you before the services are made (README)
#   monitoring  JSON logs in Cloud Logging (Cloud Run's own), the Prometheus counters through Google's managed
#               Prometheus sidecar, alert policies to an e-mail address
#
# Nothing here is applied by DCLab's tooling: `tofu plan` / `tofu apply` are run by a person (deploy/README.md).

terraform {
  required_version = ">= 1.6"
  required_providers {
    google = { source = "hashicorp/google", version = "~> 8.5" }
    random = { source = "hashicorp/random", version = "~> 3.6" }
  }
}

locals {
  name   = "${var.name}-${var.deployment}"
  labels = merge(var.labels, { application = "dclab", deployment = var.deployment })
}

resource "google_project_service" "apis" {
  for_each = toset(["run.googleapis.com", "sqladmin.googleapis.com", "servicenetworking.googleapis.com", "compute.googleapis.com",
  "secretmanager.googleapis.com", "artifactregistry.googleapis.com", "monitoring.googleapis.com", "logging.googleapis.com"])
  project            = var.project
  service            = each.key
  disable_on_destroy = false
}
