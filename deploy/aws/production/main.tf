# DCLab production on AWS: two zones, Multi-AZ database, 35 days of point-in-time recovery, protected from deletion. `tofu plan` here, never apply without the owner's yes.
terraform {
  required_version = ">= 1.6"
  # State in S3, made once by hand (README): uncomment and fill before the first plan.
  # backend "s3" {
  #   bucket       = "<your-state-bucket>"
  #   key          = "dclab/production.tfstate"
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
    tags = { Application = "dclab", Deployment = "production", ManagedBy = "opentofu" }
  }
}

module "dclab" {
  source              = "../modules/dclab"
  deployment          = "production"
  image               = var.image
  domain_name         = var.domain_name
  certificate_arn     = var.certificate_arn
  alarm_email         = var.alarm_email
  vpc_cidr            = "10.41.0.0/16"
  nat_per_zone        = true
  app_min             = 2
  app_max             = 6
  worker_min          = 1
  worker_max          = 8
  worker_memory       = 8192
  db_instance_class   = "db.m7g.large"
  db_storage_gb       = 100
  db_multi_az         = true
  db_backup_days      = 35
  deletion_protection = true
  log_days            = 90
}

output "dclab" {
  value = module.dclab
}
