resource "google_monitoring_notification_channel" "email" {
  count        = var.alarm_email == "" ? 0 : 1
  project      = var.project
  display_name = "DCLab ${var.deployment} alerts"
  type         = "email"
  labels       = { email_address = var.alarm_email }
}

# the queue, as the workers write it (dclab_rnd/jobs/queue_metric.py): declared so the alert below can use it
resource "google_monitoring_metric_descriptor" "jobs" {
  for_each     = { queued_jobs = "Jobs waiting to be claimed", active_jobs = "Jobs waiting or running" }
  project      = var.project
  type         = "custom.googleapis.com/dclab/${each.key}"
  display_name = "DCLab ${lower(each.value)}"
  description  = "${each.value}, as each worker counts them once a minute."
  metric_kind  = "GAUGE"
  value_type   = "INT64"
  unit         = "1"
  labels {
    key         = "deployment"
    description = "staging or production"
  }
  labels {
    key         = "instance"
    description = "the worker that counted"
  }
  depends_on = [google_project_service.apis]
}

locals {
  channels = google_monitoring_notification_channel.email[*].id
  alerts = {
    server-errors = {
      title  = "The app answered 5xx more than 20 times in 5 minutes"
      filter = "resource.type = \"cloud_run_revision\" AND resource.labels.service_name = \"${local.name}-app\" AND metric.type = \"run.googleapis.com/request_count\" AND metric.labels.response_code_class = \"5xx\""
      align  = "ALIGN_SUM", threshold = 20, duration = "0s"
    }
    database-cpu = {
      title  = "The database is above 80% CPU for 15 minutes"
      filter = "resource.type = \"cloudsql_database\" AND resource.labels.database_id = \"${var.project}:${local.name}\" AND metric.type = \"cloudsql.googleapis.com/database/cpu/utilization\""
      align  = "ALIGN_MEAN", threshold = 0.8, duration = "900s"
    }
    database-disk = {
      title  = "The database disk is more than 85% full"
      filter = "resource.type = \"cloudsql_database\" AND resource.labels.database_id = \"${var.project}:${local.name}\" AND metric.type = \"cloudsql.googleapis.com/database/disk/utilization\""
      align  = "ALIGN_MAX", threshold = 0.85, duration = "300s"
    }
    jobs-backlog = {
      title  = "More than 20 jobs waited for 15 minutes: raise worker_max, or the workers are failing or not scaling"
      filter = "resource.type = \"global\" AND metric.type = \"custom.googleapis.com/dclab/queued_jobs\" AND metric.labels.deployment = \"${var.deployment}\""
      align  = "ALIGN_MAX", threshold = 20, duration = "900s"
    }
  }
}

resource "google_monitoring_alert_policy" "health" {
  for_each              = local.alerts
  project               = var.project
  display_name          = "${local.name}: ${each.value.title}"
  combiner              = "OR"
  notification_channels = local.channels
  user_labels           = local.labels
  conditions {
    display_name = each.value.title
    condition_threshold {
      filter          = each.value.filter
      comparison      = "COMPARISON_GT"
      threshold_value = each.value.threshold
      duration        = each.value.duration
      aggregations {
        alignment_period     = "300s"
        per_series_aligner   = each.value.align
        cross_series_reducer = "REDUCE_MAX"
      }
    }
  }
  depends_on = [google_project_service.apis, google_monitoring_metric_descriptor.jobs]
}
