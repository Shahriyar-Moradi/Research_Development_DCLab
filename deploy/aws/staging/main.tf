# DCLab staging on AWS: small, cheap to keep, deletable. `tofu plan` here, never apply without the owner's yes.
terraform {
  required_version = ">= 1.6"
  # State in S3, made once by hand (README): uncomment and fill before the first plan.
  # backend "s3" {
  #   bucket       = "<your-state-bucket>"
  #   key          = "dclab/staging.tfstate"
  #   region       = "<region>"
  #   use_lockfile = true
  #   encrypt      = true
  # }
}

variable "region" {
  description = "The AWS region (decided before the first plan; for example eu-central-1)."
  type        = string
}

variable "image" {
  type = string
}

variable "domain_name" {
  type = string
}

variable "certificate_arn" {
  type = string
}

variable "alarm_email" {
  type    = string
  default = ""
}

provider "aws" {
  region = var.region
  default_tags {
    tags = { Application = "dclab", Deployment = "staging", ManagedBy = "opentofu" }
  }
}

module "dclab" {
  source              = "../modules/dclab"
  deployment          = "staging"
  image               = var.image
  domain_name         = var.domain_name
  certificate_arn     = var.certificate_arn
  alarm_email         = var.alarm_email
  vpc_cidr            = "10.40.0.0/16"
  nat_per_zone        = false
  app_min             = 2
  app_max             = 2
  worker_min          = 1
  worker_max          = 2
  worker_memory       = 4096
  db_instance_class   = "db.t4g.small"
  db_storage_gb       = 20
  db_multi_az         = false
  db_backup_days      = 7
  deletion_protection = false
  log_days            = 14
}

output "dclab" {
  value = module.dclab
}
