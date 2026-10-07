output "load_balancer_ip" {
  description = "Point domain_name at this (an A record); the certificate is issued once it does."
  value       = google_compute_global_address.app.address
}

output "image_repository" {
  description = "Push the image here (README), then set image to <this>/dclab:<tag>."
  value       = "${var.region}-docker.pkg.dev/${var.project}/${google_artifact_registry_repository.dclab.repository_id}"
}

output "workspace_bucket" {
  value = google_storage_bucket.workspace.name
}

output "database_instance" {
  value = google_sql_database_instance.main.connection_name
}

output "secrets_to_fill" {
  description = "Secrets made empty: add a version to each before the services are made (README)."
  value       = { for name, s in google_secret_manager_secret.named : name => s.secret_id }
}

output "admin_job" {
  description = "The one-off job for the accounts command (README: the first owner)."
  value = {
    job             = google_cloud_run_v2_job.admin.name
    password_secret = google_secret_manager_secret.new_password.secret_id
  }
}
