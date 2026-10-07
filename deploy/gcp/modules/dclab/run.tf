# ---------------------------------------------------------------------------------------------------- identity
resource "google_service_account" "run" {
  project      = var.project
  account_id   = substr("${local.name}-run", 0, 30)
  display_name = "DCLab ${var.deployment}: the app and the workers"
}

resource "google_storage_bucket_iam_member" "workspace" {
  bucket = google_storage_bucket.workspace.name
  role   = "roles/storage.objectUser"
  member = "serviceAccount:${google_service_account.run.email}"
}

resource "google_secret_manager_secret_iam_member" "read" {
  for_each = merge({ DCLAB_DATABASE_URL = google_secret_manager_secret.database_url.id, DCLAB_SECRET_KEY = google_secret_manager_secret.secret_key.id,
    DCLAB_NEW_PASSWORD = google_secret_manager_secret.new_password.id },
  { for name, s in google_secret_manager_secret.named : name => s.id })
  secret_id = each.value
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.run.email}"
}

resource "google_project_iam_member" "run" {
  for_each = toset(["roles/monitoring.metricWriter", "roles/logging.logWriter", "roles/cloudtrace.agent"])
  project  = var.project
  role     = each.key
  member   = "serviceAccount:${google_service_account.run.email}"
}

locals {
  environment = merge({
    DCLAB_AGENT_HOME    = "/workspace" # the bucket, mounted: the tables and uploads are objects in it
    DCLAB_WORKER        = "external"
    DCLAB_AUTH          = var.auth
    DCLAB_ALLOWED_HOSTS = var.domain_name
    DCLAB_COOKIE_SECURE = "1"
    DCLAB_LOG_FORMAT    = "json"
    DCLAB_METRICS_PORT  = "8080" # where the managed Prometheus sidecar reads them
    DCLAB_METRICS_HOST  = "127.0.0.1"
    DCLAB_DEPLOYMENT    = var.deployment
    DCLAB_RESEARCH_DB   = "/tmp/research.sqlite3" # SQLite off the mounted bucket
  }, var.environment)
  secrets = merge({ DCLAB_DATABASE_URL = google_secret_manager_secret.database_url.secret_id, DCLAB_SECRET_KEY = google_secret_manager_secret.secret_key.secret_id },
  { for name, s in google_secret_manager_secret.named : name => s.secret_id })
  # the image's user (Dockerfile): files written through the mount are its own
  # no metadata cache: a file a worker just wrote is seen by the app at once
  workspace_mount_options = ["uid=10001", "gid=10001", "file-mode=660", "dir-mode=770", "implicit-dirs", "metadata-cache-ttl-secs=0"]
  worker_pool             = "projects/${var.project}/locations/${var.region}/workerPools/${local.name}-worker"
}

# ---------------------------------------------------------------------------------------------------- the app
resource "google_cloud_run_v2_service" "app" {
  project             = var.project
  name                = "${local.name}-app"
  location            = var.region
  ingress             = "INGRESS_TRAFFIC_INTERNAL_LOAD_BALANCER" # through the HTTPS load balancer only
  deletion_protection = var.deletion_protection
  labels              = local.labels
  scaling {
    min_instance_count = var.app_min
    max_instance_count = var.app_max
  }
  template {
    service_account                  = google_service_account.run.email
    execution_environment            = "EXECUTION_ENVIRONMENT_GEN2" # needed for the bucket mount
    timeout                          = "3600s"                      # the Home page's event stream stays open
    max_instance_request_concurrency = 80
    labels                           = local.labels
    vpc_access {
      egress = "PRIVATE_RANGES_ONLY" # the database by its private address; the internet (model APIs) directly
      network_interfaces {
        network    = google_compute_network.main.id
        subnetwork = google_compute_subnetwork.main.id
      }
    }
    volumes {
      name = "workspace"
      gcs {
        bucket        = google_storage_bucket.workspace.name
        read_only     = false
        mount_options = local.workspace_mount_options
      }
    }
    containers {
      name  = "app"
      image = var.image
      ports {
        container_port = 8765
      }
      resources {
        limits = { cpu = var.app_cpu, memory = var.app_memory }
      }
      dynamic "env" {
        for_each = local.environment
        content {
          name  = env.key
          value = env.value
        }
      }
      dynamic "env" {
        for_each = local.secrets
        content {
          name = env.key
          value_source {
            secret_key_ref {
              secret  = env.value
              version = "latest"
            }
          }
        }
      }
      volume_mounts {
        name       = "workspace"
        mount_path = "/workspace"
      }
      startup_probe {
        http_get {
          path = "/readyz" # the database answers and the files can be written (12.4); the first instance migrates first
        }
        initial_delay_seconds = 5
        period_seconds        = 10
        failure_threshold     = 30
      }
      liveness_probe {
        http_get {
          path = "/healthz"
        }
        period_seconds = 30
      }
    }
    containers {
      name       = "metrics"
      image      = var.prometheus_sidecar_image
      depends_on = ["app"]
      resources {
        limits = { cpu = "0.25", memory = "256Mi" }
      }
    }
  }
  depends_on = [google_project_iam_member.run, google_secret_manager_secret_iam_member.read, google_storage_bucket_iam_member.workspace]
}

# the load balancer passes requests on without a Google identity: the app does its own sign-in (DCLAB_AUTH)
resource "google_cloud_run_v2_service_iam_member" "public" {
  project  = var.project
  location = var.region
  name     = google_cloud_run_v2_service.app.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# ---------------------------------------------------------------------------------------------------- the workers
resource "google_cloud_run_v2_worker_pool" "worker" {
  project             = var.project
  name                = "${local.name}-worker"
  location            = var.region
  launch_stage        = "BETA"
  deletion_protection = var.deletion_protection
  labels              = local.labels
  # The size follows the job queue: one worker (the one holding an advisory lock) sets it every minute by the same rule as
  # on AWS (dclab_rnd/jobs/queue_metric.py): jobs waiting two minutes add workers, fifteen minutes with nothing waiting or
  # running removes one, so no busy worker is stopped. OpenTofu sets the first size and then leaves it to that rule.
  scaling {
    scaling_mode          = "MANUAL"
    manual_instance_count = var.worker_min
  }
  lifecycle {
    ignore_changes = [scaling[0].manual_instance_count]
  }
  template {
    service_account = google_service_account.run.email
    labels          = local.labels
    vpc_access {
      egress = "PRIVATE_RANGES_ONLY"
      network_interfaces {
        network    = google_compute_network.main.id
        subnetwork = google_compute_subnetwork.main.id
      }
    }
    volumes {
      name = "workspace"
      gcs {
        bucket        = google_storage_bucket.workspace.name
        read_only     = false
        mount_options = local.workspace_mount_options
      }
    }
    containers {
      name    = "worker"
      image   = var.image
      command = ["sh", "-c", "python -m dclab_rnd.storage upgrade && exec python -m dclab_rnd.worker"] # a new image's migration first (db.upgrade takes a lock)
      resources {
        limits = { cpu = var.worker_cpu, memory = var.worker_memory }
      }
      dynamic "env" {
        for_each = merge(local.environment, { DCLAB_QUEUE_METRIC = "gcp", DCLAB_WORKER_POOL = local.worker_pool,
        DCLAB_WORKERS_MIN = tostring(var.worker_min), DCLAB_WORKERS_MAX = tostring(var.worker_max) })
        content {
          name  = env.key
          value = env.value
        }
      }
      dynamic "env" {
        for_each = local.secrets
        content {
          name = env.key
          value_source {
            secret_key_ref {
              secret  = env.value
              version = "latest"
            }
          }
        }
      }
      volume_mounts {
        name       = "workspace"
        mount_path = "/workspace"
      }
    }
    containers {
      name       = "metrics"
      image      = var.prometheus_sidecar_image
      depends_on = ["worker"]
      resources {
        limits = { cpu = "0.25", memory = "256Mi" }
      }
    }
  }
  depends_on = [google_project_iam_member.run, google_secret_manager_secret_iam_member.read, google_storage_bucket_iam_member.workspace]
}

# the elected worker sets the pool's size (and acts as the pool's own service account when it does)
resource "google_cloud_run_v2_worker_pool_iam_member" "scaler" {
  project  = var.project
  location = var.region
  name     = google_cloud_run_v2_worker_pool.worker.name
  role     = "roles/run.developer"
  member   = "serviceAccount:${google_service_account.run.email}"
}

resource "google_service_account_iam_member" "scaler_acts_as_itself" {
  service_account_id = google_service_account.run.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.run.email}"
}

# ---------------------------------------------------------------------------------------------------- the first owner
# a one-off job for the accounts command, run by a person (README); the password comes from its own secret
resource "google_cloud_run_v2_job" "admin" {
  project             = var.project
  name                = "${local.name}-admin"
  location            = var.region
  deletion_protection = false
  labels              = local.labels
  template {
    template {
      service_account = google_service_account.run.email
      max_retries     = 0
      vpc_access {
        egress = "PRIVATE_RANGES_ONLY"
        network_interfaces {
          network    = google_compute_network.main.id
          subnetwork = google_compute_subnetwork.main.id
        }
      }
      containers {
        image   = var.image
        command = ["python", "-m", "dclab_rnd.accounts"]
        args    = ["list"]
        dynamic "env" {
          for_each = local.environment
          content {
            name  = env.key
            value = env.value
          }
        }
        dynamic "env" {
          for_each = merge(local.secrets, { DCLAB_NEW_PASSWORD = google_secret_manager_secret.new_password.secret_id })
          content {
            name = env.key
            value_source {
              secret_key_ref {
                secret  = env.value
                version = "latest"
              }
            }
          }
        }
      }
    }
  }
  depends_on = [google_secret_manager_secret_iam_member.read]
}

# ---------------------------------------------------------------------------------------------------- load balancer
resource "google_compute_region_network_endpoint_group" "app" {
  project               = var.project
  name                  = "${local.name}-app"
  region                = var.region
  network_endpoint_type = "SERVERLESS"
  cloud_run {
    service = google_cloud_run_v2_service.app.name
  }
}

resource "google_compute_backend_service" "app" {
  project               = var.project
  name                  = "${local.name}-app"
  load_balancing_scheme = "EXTERNAL_MANAGED"
  protocol              = "HTTPS"
  backend {
    group = google_compute_region_network_endpoint_group.app.id
  }
  log_config {
    enable      = true
    sample_rate = 1
  }
}

resource "google_compute_url_map" "https" {
  project         = var.project
  name            = "${local.name}-https"
  default_service = google_compute_backend_service.app.id
}

resource "google_compute_managed_ssl_certificate" "app" {
  project = var.project
  name    = "${local.name}-cert"
  managed {
    domains = [var.domain_name]
  }
}

resource "google_compute_ssl_policy" "modern" {
  project         = var.project
  name            = "${local.name}-tls"
  profile         = "MODERN"
  min_tls_version = "TLS_1_2"
}

resource "google_compute_target_https_proxy" "app" {
  project          = var.project
  name             = "${local.name}-https"
  url_map          = google_compute_url_map.https.id
  ssl_certificates = [google_compute_managed_ssl_certificate.app.id]
  ssl_policy       = google_compute_ssl_policy.modern.id
}

resource "google_compute_global_address" "app" {
  project = var.project
  name    = "${local.name}-ip"
}

resource "google_compute_global_forwarding_rule" "https" {
  project               = var.project
  name                  = "${local.name}-https"
  load_balancing_scheme = "EXTERNAL_MANAGED"
  ip_address            = google_compute_global_address.app.address
  port_range            = "443"
  target                = google_compute_target_https_proxy.app.id
}

resource "google_compute_url_map" "redirect" {
  project = var.project
  name    = "${local.name}-redirect"
  default_url_redirect {
    https_redirect         = true
    redirect_response_code = "MOVED_PERMANENTLY_DEFAULT"
    strip_query            = false
  }
}

resource "google_compute_target_http_proxy" "redirect" {
  project = var.project
  name    = "${local.name}-redirect"
  url_map = google_compute_url_map.redirect.id
}

resource "google_compute_global_forwarding_rule" "http" {
  project               = var.project
  name                  = "${local.name}-http"
  load_balancing_scheme = "EXTERNAL_MANAGED"
  ip_address            = google_compute_global_address.app.address
  port_range            = "80"
  target                = google_compute_target_http_proxy.redirect.id
}
