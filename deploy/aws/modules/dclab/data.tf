# ---------------------------------------------------------------------------------------------------- PostgreSQL
resource "random_password" "db" {
  length  = 32
  special = false # it goes into a URL
}

resource "aws_db_subnet_group" "main" {
  name       = local.name
  subnet_ids = aws_subnet.private[*].id
  tags       = local.tags
}

resource "aws_db_instance" "main" {
  identifier                   = local.name
  engine                       = "postgres"
  engine_version               = "16"
  instance_class               = var.db_instance_class
  allocated_storage            = var.db_storage_gb
  max_allocated_storage        = var.db_storage_gb * 4
  storage_type                 = "gp3"
  storage_encrypted            = true
  db_name                      = "dclab"
  username                     = "dclab"
  password                     = random_password.db.result
  db_subnet_group_name         = aws_db_subnet_group.main.name
  vpc_security_group_ids       = [aws_security_group.data.id]
  publicly_accessible          = false
  multi_az                     = var.db_multi_az
  backup_retention_period      = var.db_backup_days # automated backups and point-in-time recovery within these days
  backup_window                = "02:00-03:00"
  maintenance_window           = "sun:03:30-sun:04:30"
  copy_tags_to_snapshot        = true
  deletion_protection          = var.deletion_protection
  skip_final_snapshot          = !var.deletion_protection
  final_snapshot_identifier    = var.deletion_protection ? "${local.name}-final" : null
  auto_minor_version_upgrade   = true
  performance_insights_enabled = true
  tags                         = local.tags
}

# ---------------------------------------------------------------------------------------------------- files
resource "aws_s3_bucket" "files" {
  bucket_prefix = "${local.name}-files-"
  force_destroy = !var.deletion_protection
  tags          = local.tags
}

resource "aws_s3_bucket_versioning" "files" {
  bucket = aws_s3_bucket.files.id
  versioning_configuration {
    status = "Enabled" # an overwritten or deleted table can be restored (12.5 relies on it for files in a bucket)
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "files" {
  bucket = aws_s3_bucket.files.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "files" {
  bucket                  = aws_s3_bucket.files.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "files" {
  bucket = aws_s3_bucket.files.id
  rule {
    id     = "old-versions"
    status = "Enabled"
    filter {}
    noncurrent_version_expiration {
      noncurrent_days = 90
    }
  }
}

# the workspace folder, shared by the app and the workers (the compose volume's role on one machine)
resource "aws_efs_file_system" "workspace" {
  encrypted        = true
  performance_mode = "generalPurpose"
  throughput_mode  = "elastic"
  tags             = merge(local.tags, { Name = "${local.name}-workspace" })
}

resource "aws_efs_backup_policy" "workspace" {
  file_system_id = aws_efs_file_system.workspace.id
  backup_policy {
    status = "ENABLED"
  }
}

resource "aws_efs_mount_target" "workspace" {
  count           = 2
  file_system_id  = aws_efs_file_system.workspace.id
  subnet_id       = aws_subnet.private[count.index].id
  security_groups = [aws_security_group.data.id]
}

resource "aws_efs_access_point" "workspace" {
  file_system_id = aws_efs_file_system.workspace.id
  posix_user {
    uid = 10001 # the image's user (Dockerfile)
    gid = 10001
  }
  root_directory {
    path = "/workspace"
    creation_info {
      owner_uid   = 10001
      owner_gid   = 10001
      permissions = "750"
    }
  }
  tags = local.tags
}

# ---------------------------------------------------------------------------------------------------- images
resource "aws_ecr_repository" "dclab" {
  name                 = local.name
  image_tag_mutability = "IMMUTABLE"
  force_delete         = !var.deletion_protection
  image_scanning_configuration {
    scan_on_push = true
  }
  tags = local.tags
}

resource "aws_ecr_lifecycle_policy" "dclab" {
  repository = aws_ecr_repository.dclab.name
  policy = jsonencode({ rules = [{
    rulePriority = 1, description = "keep the 30 newest images",
    selection    = { tagStatus = "any", countType = "imageCountMoreThan", countNumber = 30 },
    action       = { type = "expire" }
  }] })
}

# ---------------------------------------------------------------------------------------------------- secrets
resource "aws_secretsmanager_secret" "database_url" {
  name_prefix = "${local.name}/DCLAB_DATABASE_URL-"
  tags        = local.tags
}

resource "aws_secretsmanager_secret_version" "database_url" {
  secret_id     = aws_secretsmanager_secret.database_url.id
  secret_string = "postgresql+psycopg://dclab:${random_password.db.result}@${aws_db_instance.main.address}:5432/dclab?sslmode=require"
}

resource "random_password" "session" {
  length  = 48
  special = false
}

resource "aws_secretsmanager_secret" "secret_key" {
  name_prefix = "${local.name}/DCLAB_SECRET_KEY-"
  tags        = local.tags
}

resource "aws_secretsmanager_secret_version" "secret_key" {
  secret_id     = aws_secretsmanager_secret.secret_key.id
  secret_string = random_password.session.result
}

# the first owner's password, read by the one-off admin task only (README); made empty, filled by you, emptied after
resource "aws_secretsmanager_secret" "new_password" {
  name_prefix = "${local.name}/DCLAB_NEW_PASSWORD-"
  tags        = local.tags
}

# made empty: you put each value (README) before the services are made
resource "aws_secretsmanager_secret" "named" {
  for_each    = toset(var.secret_names)
  name_prefix = "${local.name}/${each.key}-"
  tags        = local.tags
}
