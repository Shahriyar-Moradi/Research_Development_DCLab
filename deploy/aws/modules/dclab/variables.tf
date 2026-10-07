variable "name" {
  description = "Prefix of every resource's name."
  type        = string
  default     = "dclab"
}

variable "deployment" {
  description = "staging or production: part of every name, and the Deployment dimension of the queue metric."
  type        = string
  validation {
    condition     = contains(["staging", "production"], var.deployment)
    error_message = "deployment is staging or production."
  }
}

variable "image" {
  description = "The image to run, from 12.1's Dockerfile (for example <account>.dkr.ecr.<region>.amazonaws.com/dclab-production:<git sha>)."
  type        = string
}

variable "domain_name" {
  description = "The host name people open (for example dclab.example.com); the server answers to it only (DCLAB_ALLOWED_HOSTS)."
  type        = string
}

variable "certificate_arn" {
  description = "An ACM certificate for domain_name, issued in this region (README: request it, validate it by DNS)."
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
  description = "Settings kept in Secrets Manager and filled by you after the first apply (README), for example OPENAI_API_KEY."
  type        = list(string)
  default     = ["OPENAI_API_KEY"]
}

variable "vpc_cidr" {
  type    = string
  default = "10.40.0.0/16"
}

variable "nat_per_zone" {
  description = "One NAT gateway per zone (production) or one for both (cheaper; staging)."
  type        = bool
  default     = false
}

variable "app_min" {
  description = "App tasks at least (two or more behind the load balancer)."
  type        = number
  default     = 2
  validation {
    condition     = var.app_min >= 2
    error_message = "Run at least two app tasks."
  }
}

variable "app_max" {
  type    = number
  default = 4
}

variable "app_cpu" {
  type    = number
  default = 1024
}

variable "app_memory" {
  type    = number
  default = 2048
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
  type    = number
  default = 2048
}

variable "worker_memory" {
  description = "A stage run holds a table and its models in memory."
  type        = number
  default     = 8192
}

variable "db_instance_class" {
  type    = string
  default = "db.t4g.medium"
}

variable "db_storage_gb" {
  type    = number
  default = 50
}

variable "db_multi_az" {
  type    = bool
  default = false
}

variable "db_backup_days" {
  description = "Automated backups kept this many days; point-in-time recovery reaches back as far (1 to 35)."
  type        = number
  default     = 7
  validation {
    condition     = var.db_backup_days >= 1 && var.db_backup_days <= 35
    error_message = "db_backup_days is 1 to 35 (0 would turn backups and point-in-time recovery off)."
  }
}

variable "deletion_protection" {
  description = "Refuse to delete the database and the load balancer, and keep a final snapshot (production)."
  type        = bool
  default     = true
}

variable "log_days" {
  type    = number
  default = 30
}

variable "alarm_email" {
  description = "Where alarms go (empty: alarms are made, nobody is told)."
  type        = string
  default     = ""
}

variable "tags" {
  type    = map(string)
  default = {}
}
