# DCLab staging on Google Cloud: small, cheap to keep, deletable. `tofu plan` here, never apply without the owner's yes.
terraform {
  required_version = ">= 1.6"
  # State in Cloud Storage, made once by hand (README): uncomment and fill before the first plan.
  # backend "gcs" {
  #   bucket = "<your-state-bucket>"
  #   prefix = "dclab/staging"
  # }
}

variable "project" {
  type = string
}

variable "region" {
  description = "The region (decided before the first plan; for example europe-west3)."
  type        = string
}

variable "image" {
  type = string
}

variable "domain_name" {
  type = string
}

variable "alarm_email" {
  type    = string
  default = ""
}

provider "google" {
  project        = var.project
  region         = var.region
  default_labels = { application = "dclab", deployment = "staging", managed-by = "opentofu" }
}

module "dclab" {
  source              = "../modules/dclab"
  project             = var.project
  region              = var.region
  deployment          = "staging"
  image               = var.image
  domain_name         = var.domain_name
  alarm_email         = var.alarm_email
  subnet_cidr         = "10.50.0.0/20"
  app_min             = 2
  app_max             = 2
  worker_min          = 1
  worker_max          = 2
  worker_memory       = "4Gi"
  db_tier             = "db-custom-1-3840"
  db_disk_gb          = 20
  db_regional         = false
  db_backups_kept     = 7
  db_log_days         = 7
  deletion_protection = false
}

output "dclab" {
  value = module.dclab
}
