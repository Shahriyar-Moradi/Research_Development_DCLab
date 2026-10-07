output "load_balancer_dns" {
  description = "Point domain_name at this (a CNAME, or a Route 53 alias)."
  value       = aws_lb.app.dns_name
}

output "image_repository" {
  description = "Push the image here (README), then set image to <this>:<tag>."
  value       = aws_ecr_repository.dclab.repository_url
}

output "files_bucket" {
  value = aws_s3_bucket.files.bucket
}

output "database_endpoint" {
  value = aws_db_instance.main.address
}

output "secrets_to_fill" {
  description = "Secrets made empty: put each value before the tasks can start (README)."
  value       = { for name, s in aws_secretsmanager_secret.named : name => s.name }
}

output "cluster" {
  value = aws_ecs_cluster.main.name
}

output "admin_task" {
  description = "The one-off task for the accounts command (README: the first owner)."
  value = {
    task_definition = aws_ecs_task_definition.admin.family
    subnets         = aws_subnet.private[*].id
    security_group  = aws_security_group.tasks.id
    password_secret = aws_secretsmanager_secret.new_password.name
  }
}
