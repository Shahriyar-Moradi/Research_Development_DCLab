# ---------------------------------------------------------------------------------------------------- network
resource "google_compute_network" "main" {
  project                 = var.project
  name                    = local.name
  auto_create_subnetworks = false
  depends_on              = [google_project_service.apis]
}

resource "google_compute_subnetwork" "main" {
  project                  = var.project
  name                     = local.name
  region                   = var.region
  network                  = google_compute_network.main.id
  ip_cidr_range            = var.subnet_cidr
  private_ip_google_access = true
}

# Cloud SQL's private address comes from a range peered with Google's network
resource "google_compute_global_address" "services" {
  project       = var.project
  name          = "${local.name}-services"
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 16
  network       = google_compute_network.main.id
}

resource "google_service_networking_connection" "services" {
  network                 = google_compute_network.main.id
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.services.name]
}

# ---------------------------------------------------------------------------------------------------- PostgreSQL
resource "random_password" "db" {
  length  = 32
  special = false # it goes into a URL
}

resource "google_sql_database_instance" "main" {
  project             = var.project
  name                = local.name
  region              = var.region
  database_version    = "POSTGRES_16"
  deletion_protection = var.deletion_protection
  settings {
    tier                        = var.db_tier
    edition                     = "ENTERPRISE"
    availability_type           = var.db_regional ? "REGIONAL" : "ZONAL"
    disk_type                   = "PD_SSD"
    disk_size                   = var.db_disk_gb
    disk_autoresize             = true
    deletion_protection_enabled = var.deletion_protection
    user_labels                 = local.labels
    backup_configuration {
      enabled                        = true
      start_time                     = "02:00"
      point_in_time_recovery_enabled = true
      transaction_log_retention_days = var.db_log_days
      backup_retention_settings {
        retained_backups = var.db_backups_kept
      }
    }
    ip_configuration {
      ipv4_enabled    = false # no public address
      private_network = google_compute_network.main.id
      ssl_mode        = "ENCRYPTED_ONLY"
    }
    maintenance_window {
      day  = 7
      hour = 3
    }
    insights_config {
      query_insights_enabled = true
    }
  }
  depends_on = [google_service_networking_connection.services]
}

resource "google_sql_database" "dclab" {
  project  = var.project
  name     = "dclab"
  instance = google_sql_database_instance.main.name
}

resource "google_sql_user" "dclab" {
  project  = var.project
  name     = "dclab"
  instance = google_sql_database_instance.main.name
  password = random_password.db.result
}

# ---------------------------------------------------------------------------------------------------- files
resource "google_storage_bucket" "workspace" {
  project                     = var.project
  name                        = "${var.project}-${local.name}-workspace"
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  force_destroy               = !var.deletion_protection
  labels                      = local.labels
  versioning {
    enabled = true # an overwritten or deleted table can be restored (12.5 relies on it for files in a bucket)
  }
  lifecycle_rule {
    condition {
      days_since_noncurrent_time = 90
      with_state                 = "ARCHIVED"
    }
    action {
      type = "Delete"
    }
  }
  depends_on = [google_project_service.apis]
}

resource "google_artifact_registry_repository" "dclab" {
  project       = var.project
  location      = var.region
  repository_id = local.name
  format        = "DOCKER"
  labels        = local.labels
  cleanup_policies {
    id     = "keep-30"
    action = "KEEP"
    most_recent_versions {
      keep_count = 30
    }
  }
  depends_on = [google_project_service.apis]
}

# ---------------------------------------------------------------------------------------------------- secrets
resource "google_secret_manager_secret" "database_url" {
  project   = var.project
  secret_id = "${local.name}-DCLAB_DATABASE_URL"
  labels    = local.labels
  replication {
    auto {}
  }
  depends_on = [google_project_service.apis]
}

resource "google_secret_manager_secret_version" "database_url" {
  secret      = google_secret_manager_secret.database_url.id
  secret_data = "postgresql+psycopg://dclab:${random_password.db.result}@${google_sql_database_instance.main.private_ip_address}:5432/dclab?sslmode=require"
}

resource "random_password" "session" {
  length  = 48
  special = false
}

resource "google_secret_manager_secret" "secret_key" {
  project   = var.project
  secret_id = "${local.name}-DCLAB_SECRET_KEY"
  labels    = local.labels
  replication {
    auto {}
  }
  depends_on = [google_project_service.apis]
}

resource "google_secret_manager_secret_version" "secret_key" {
  secret      = google_secret_manager_secret.secret_key.id
  secret_data = random_password.session.result
}

# the first owner's password, read by the one-off admin job only (README); made empty, filled by you, disabled after
resource "google_secret_manager_secret" "new_password" {
  project    = var.project
  secret_id  = "${local.name}-DCLAB_NEW_PASSWORD"
  labels     = local.labels
  depends_on = [google_project_service.apis]
  replication {
    auto {}
  }
}

# made empty: you add a version to each (README) before the services are made
resource "google_secret_manager_secret" "named" {
  for_each   = toset(var.secret_names)
  project    = var.project
  secret_id  = "${local.name}-${each.key}"
  labels     = local.labels
  depends_on = [google_project_service.apis]
  replication {
    auto {}
  }
}
