# DCLab on AWS (package 12.6): the image from 12.1 behind an HTTPS load balancer.
#
#   network     a VPC over two zones: public subnets for the load balancer and the NAT, private ones for the rest
#   database    RDS PostgreSQL 16, encrypted, automated backups with point-in-time recovery, Multi-AZ when asked
#   files       S3 (DCLAB_FILES_URL: the files of record, versioned) and EFS (the shared workspace folder that the
#               app and the workers both see, as the compose volume does on one machine)
#   app         ECS Fargate, app_min..app_max tasks behind an ALB with TLS 1.2/1.3, health check /readyz
#   workers     ECS Fargate, worker_min..worker_max tasks, scaled on the job queue's length (QueuedJobs, published
#               by the workers: dclab_rnd/jobs/queue_metric.py)
#   secrets     Secrets Manager: the database URL and the session key are made here; the others (model keys, OIDC)
#               are created empty and filled by you (README); tasks read them at start, nothing is in plain env
#   monitoring  JSON logs in CloudWatch Logs, the Prometheus counters through an OpenTelemetry sidecar into
#               CloudWatch metrics, alarms to an e-mail address
#
# Nothing here is applied by DCLab's tooling: `tofu plan` / `tofu apply` are run by a person (deploy/README.md).

terraform {
  required_version = ">= 1.6"
  required_providers {
    aws    = { source = "hashicorp/aws", version = "~> 6.0" }
    random = { source = "hashicorp/random", version = "~> 3.6" }
  }
}

data "aws_availability_zones" "zones" {
  state = "available"
}

data "aws_region" "current" {}

data "aws_caller_identity" "current" {}

locals {
  name   = "${var.name}-${var.deployment}"
  zones  = slice(data.aws_availability_zones.zones.names, 0, 2)
  region = data.aws_region.current.region
  tags   = merge(var.tags, { Application = "dclab", Deployment = var.deployment })
}
