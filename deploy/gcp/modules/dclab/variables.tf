variable "project" {
  description = "The Google Cloud project id."
  type        = string
}

variable "region" {
  description = "The region (decided before the first plan; for example europe-west3)."
  type        = string
}

variable "name" {
  type    = string
  default = "dclab"
}

variable "deployment" {
  description = "staging or production: part of every name, and the deployment label of the queue metric."
  type        = string
  validation {
    condition     = contains(["staging", "production"], var.deployment)
    error_message = "deployment is staging or production."
  }
}

variable "image" {
  description = "The image to run (for example <region>-docker.pkg.dev/<project>/dclab-production/dclab:<git sha>)."
  type        = string
}

variable "domain_name" {
  description = "The host name people open; the managed certificate is issued for it once its DNS points at the load balancer."
  type        = string
}

variable "auth" {
  description = "How people sign in: password or oidc (sign-in is required on a public host; the server refuses none)."
  type        = string
  default     = "password"
  validation {
    condition     = contains(["password", "oidc"], var.auth)
    error_message = "auth is password or oidc."
  }
}

variable "environment" {
  description = "More settings for the app and the workers, by name (see .env.example); never a secret."
  type        = map(string)
  default     = {}
}

variable "secret_names" {
  description = "Settings kept in Secret Manager and filled by you before the services are made (README), for example OPENAI_API_KEY."
  type        = list(string)
  default     = ["OPENAI_API_KEY"]
}

variable "subnet_cidr" {
  type    = string
  default = "10.50.0.0/20"
}

variable "app_min" {
  description = "App instances at least (two or more behind the load balancer)."
  type        = number
  default     = 2
  validation {
    condition     = var.app_min >= 2
    error_message = "Run at least two app instances."
  }
}

variable "app_max" {
  type    = number
  default = 4
}

variable "app_cpu" {
  type    = string
  default = "1"
}

variable "app_memory" {
  type    = string
  default = "2Gi"
}

variable "worker_min" {
  type    = number
  default = 1
}

variable "worker_max" {
  type    = number
  default = 4
}

variable "worker_cpu" {
  type    = string
  default = "2"
}

variable "worker_memory" {
  description = "A stage run holds a table and its models in memory."
  type        = string
  default     = "8Gi"
}

variable "db_tier" {
  type    = string
  default = "db-custom-2-7680"
}

variable "db_disk_gb" {
  type    = number
  default = 50
}

variable "db_regional" {
  description = "A standby in a second zone that takes over by itself (production)."
  type        = bool
  default     = false
}

variable "db_backups_kept" {
  description = "Daily backups kept."
  type        = number
  default     = 7
}

variable "db_log_days" {
  description = "Days of transaction logs: point-in-time recovery reaches back this far (1 to 7)."
  type        = number
  default     = 7
  validation {
    condition     = var.db_log_days >= 1 && var.db_log_days <= 7
    error_message = "db_log_days is 1 to 7."
  }
}

variable "deletion_protection" {
  description = "Refuse to delete the database, the bucket and the services (production)."
  type        = bool
  default     = true
}

variable "alarm_email" {
  description = "Where alerts go (empty: the policies are made, nobody is told)."
  type        = string
  default     = ""
}

variable "prometheus_sidecar_image" {
  description = "Google's managed Prometheus sidecar for Cloud Run; it reads the counters on 127.0.0.1:8080/metrics (12.4)."
  type        = string
  default     = "us-docker.pkg.dev/cloud-ops-agents-artifacts/cloud-run-gmp-sidecar/cloud-run-gmp-sidecar:1.2.0"
}

variable "labels" {
  type    = map(string)
  default = {}
}
