variable "collector_image" {
  description = "The OpenTelemetry collector that reads the Prometheus counters (12.4) and sends them to CloudWatch."
  type        = string
  default     = "public.ecr.aws/aws-observability/aws-otel-collector:v0.40.0"
}

resource "aws_cloudwatch_log_group" "app" {
  name              = "/dclab/${local.name}/app"
  retention_in_days = var.log_days
  tags              = local.tags
}

resource "aws_cloudwatch_log_group" "worker" {
  name              = "/dclab/${local.name}/worker"
  retention_in_days = var.log_days
  tags              = local.tags
}

resource "aws_cloudwatch_log_group" "metrics" {
  name              = "/dclab/${local.name}/metrics"
  retention_in_days = var.log_days
  tags              = local.tags
}

# ---------------------------------------------------------------------------------------------------- roles
data "aws_iam_policy_document" "tasks_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

# what ECS itself needs to start a task: pull the image, write its logs, read its secrets
resource "aws_iam_role" "execution" {
  name_prefix        = "${local.name}-exec-"
  assume_role_policy = data.aws_iam_policy_document.tasks_assume.json
  tags               = local.tags
}

resource "aws_iam_role_policy_attachment" "execution" {
  role       = aws_iam_role.execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

data "aws_iam_policy_document" "read_secrets" {
  statement {
    actions = ["secretsmanager:GetSecretValue"]
    resources = concat([aws_secretsmanager_secret.database_url.arn, aws_secretsmanager_secret.secret_key.arn, aws_secretsmanager_secret.new_password.arn],
    [for s in aws_secretsmanager_secret.named : s.arn])
  }
}

resource "aws_iam_role_policy" "read_secrets" {
  role   = aws_iam_role.execution.id
  policy = data.aws_iam_policy_document.read_secrets.json
}

# what DCLab does once running: its bucket, the shared folder, its own metrics and log lines
resource "aws_iam_role" "task" {
  name_prefix        = "${local.name}-task-"
  assume_role_policy = data.aws_iam_policy_document.tasks_assume.json
  tags               = local.tags
}

data "aws_iam_policy_document" "task" {
  statement {
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.files.arn]
  }
  statement {
    actions   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
    resources = ["${aws_s3_bucket.files.arn}/*"]
  }
  statement {
    actions   = ["elasticfilesystem:ClientMount", "elasticfilesystem:ClientWrite"]
    resources = [aws_efs_file_system.workspace.arn]
    condition {
      test     = "StringEquals"
      variable = "elasticfilesystem:AccessPointArn"
      values   = [aws_efs_access_point.workspace.arn]
    }
  }
  statement {
    actions   = ["cloudwatch:PutMetricData"] # the queue (workers) and the Prometheus counters (the collector), in DCLab's namespaces only
    resources = ["*"]
    condition {
      test     = "StringEquals"
      variable = "cloudwatch:namespace"
      values   = ["DCLab", "DCLab/Prometheus"]
    }
  }
  statement {
    actions   = ["ecs:DescribeServices"] # a worker asks whether a deployment replaces it
    resources = ["arn:aws:ecs:${local.region}:${data.aws_caller_identity.current.account_id}:service/${local.name}/worker"]
  }
  statement {
    actions   = ["ecs:UpdateTaskProtection", "ecs:GetTaskProtection"] # a worker running a job is not stopped by scaling in
    resources = ["arn:aws:ecs:${local.region}:${data.aws_caller_identity.current.account_id}:task/${local.name}/*"]
  }
  statement {
    actions   = ["logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams"]
    resources = ["${aws_cloudwatch_log_group.metrics.arn}:*"]
  }
}

resource "aws_iam_role_policy" "task" {
  role   = aws_iam_role.task.id
  policy = data.aws_iam_policy_document.task.json
}

# ---------------------------------------------------------------------------------------------------- load balancer
resource "aws_lb" "app" {
  name                       = substr(local.name, 0, 32)
  load_balancer_type         = "application"
  internal                   = false
  security_groups            = [aws_security_group.alb.id]
  subnets                    = aws_subnet.public[*].id
  drop_invalid_header_fields = true
  enable_deletion_protection = var.deletion_protection
  idle_timeout               = 300 # the Home page's event stream stays open
  tags                       = local.tags
}

resource "aws_lb_target_group" "app" {
  name_prefix          = "dclab-"
  port                 = 8765
  protocol             = "HTTP"
  target_type          = "ip"
  vpc_id               = aws_vpc.main.id
  deregistration_delay = 30
  health_check {
    path                = "/readyz" # the database answers and the files can be written (12.4)
    matcher             = "200"
    interval            = 15
    timeout             = 6
    healthy_threshold   = 2
    unhealthy_threshold = 3
  }
  tags = local.tags
}

resource "aws_lb_listener" "https" {
  load_balancer_arn = aws_lb.app.arn
  port              = 443
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS13-1-2-2021-06"
  certificate_arn   = var.certificate_arn
  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.app.arn
  }
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.app.arn
  port              = 80
  protocol          = "HTTP"
  default_action {
    type = "redirect"
    redirect {
      protocol    = "HTTPS"
      port        = "443"
      status_code = "HTTP_301"
    }
  }
}

# ---------------------------------------------------------------------------------------------------- tasks
locals {
  environment = merge({
    DCLAB_AGENT_HOME    = "/workspace"
    DCLAB_FILES_URL     = "s3://${aws_s3_bucket.files.bucket}/files"
    DCLAB_WORKER        = "external"
    DCLAB_AUTH          = var.auth
    DCLAB_ALLOWED_HOSTS = var.domain_name
    DCLAB_COOKIE_SECURE = "1"
    DCLAB_LOG_FORMAT    = "json"
    DCLAB_METRICS_HOST  = "127.0.0.1" # the collector beside it in the task; nothing else reaches it
    DCLAB_DEPLOYMENT    = var.deployment
    DCLAB_RESEARCH_DB   = "/tmp/research.sqlite3" # SQLite off the shared folder (EFS)
    AWS_DEFAULT_REGION  = local.region
  }, var.environment)
  secrets = concat([
    { name = "DCLAB_DATABASE_URL", valueFrom = aws_secretsmanager_secret.database_url.arn },
    { name = "DCLAB_SECRET_KEY", valueFrom = aws_secretsmanager_secret.secret_key.arn },
  ], [for name, s in aws_secretsmanager_secret.named : { name = name, valueFrom = s.arn }])

  collector_config = <<-YAML
    receivers:
      prometheus:
        config:
          scrape_configs:
            - job_name: dclab
              scrape_interval: 60s
              static_configs:
                - targets: ["127.0.0.1:METRICS_PORT"]
    exporters:
      awsemf:
        namespace: DCLab/Prometheus
        log_group_name: ${aws_cloudwatch_log_group.metrics.name}
        dimension_rollup_option: ZeroAndSingleDimensionRollup  # the alarms read the totals
        resource_to_telemetry_conversion:
          enabled: false
    service:
      pipelines:
        metrics:
          receivers: [prometheus]
          exporters: [awsemf]
  YAML

  volume = {
    name = "workspace"
    efs = {
      file_system_id     = aws_efs_file_system.workspace.id
      transit_encryption = "ENABLED"
      access_point_id    = aws_efs_access_point.workspace.id
    }
  }
}

resource "aws_ecs_cluster" "main" {
  name = local.name
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
  tags = local.tags
}

resource "aws_ecs_task_definition" "app" {
  family                   = "${local.name}-app"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.app_cpu
  memory                   = var.app_memory
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn
  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "X86_64"
  }
  volume {
    name = local.volume.name
    efs_volume_configuration {
      file_system_id     = local.volume.efs.file_system_id
      transit_encryption = local.volume.efs.transit_encryption
      authorization_config {
        access_point_id = local.volume.efs.access_point_id
        iam             = "ENABLED"
      }
    }
  }
  container_definitions = jsonencode([
    {
      name         = "app"
      image        = var.image
      essential    = true
      portMappings = [{ containerPort = 8765, protocol = "tcp" }]
      environment  = [for k, v in merge(local.environment, { DCLAB_METRICS_PORT = "9464" }) : { name = k, value = v }]
      secrets      = local.secrets
      mountPoints  = [{ sourceVolume = "workspace", containerPath = "/workspace", readOnly = false }]
      logConfiguration = {
        logDriver = "awslogs"
        options   = { awslogs-group = aws_cloudwatch_log_group.app.name, awslogs-region = local.region, awslogs-stream-prefix = "app" }
      }
    },
    {
      name        = "metrics"
      image       = var.collector_image
      essential   = false
      environment = [{ name = "AOT_CONFIG_CONTENT", value = replace(local.collector_config, "METRICS_PORT", "9464") }]
      logConfiguration = {
        logDriver = "awslogs"
        options   = { awslogs-group = aws_cloudwatch_log_group.metrics.name, awslogs-region = local.region, awslogs-stream-prefix = "app" }
      }
    },
  ])
  tags = local.tags
}

resource "aws_ecs_task_definition" "worker" {
  family                   = "${local.name}-worker"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.worker_cpu
  memory                   = var.worker_memory
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn
  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "X86_64"
  }
  volume {
    name = local.volume.name
    efs_volume_configuration {
      file_system_id     = local.volume.efs.file_system_id
      transit_encryption = local.volume.efs.transit_encryption
      authorization_config {
        access_point_id = local.volume.efs.access_point_id
        iam             = "ENABLED"
      }
    }
  }
  container_definitions = jsonencode([
    {
      name        = "worker"
      image       = var.image
      essential   = true
      command     = ["sh", "-c", "python -m dclab_rnd.storage upgrade && exec python -m dclab_rnd.worker"] # a new image's migration runs first, whoever starts first (db.upgrade takes a lock)
      stopTimeout = 120                                                                                    # a worker running a job is protected; a deployment waits for it (see the worker service)
      environment = [for k, v in merge(local.environment, { DCLAB_METRICS_PORT = "9465", DCLAB_QUEUE_METRIC = "cloudwatch", DCLAB_ECS_SERVICE = "${local.name}/worker" }) : { name = k, value = v }]
      secrets     = local.secrets
      mountPoints = [{ sourceVolume = "workspace", containerPath = "/workspace", readOnly = false }]
      logConfiguration = {
        logDriver = "awslogs"
        options   = { awslogs-group = aws_cloudwatch_log_group.worker.name, awslogs-region = local.region, awslogs-stream-prefix = "worker" }
      }
    },
    {
      name        = "metrics"
      image       = var.collector_image
      essential   = false
      environment = [{ name = "AOT_CONFIG_CONTENT", value = replace(local.collector_config, "METRICS_PORT", "9465") }]
      logConfiguration = {
        logDriver = "awslogs"
        options   = { awslogs-group = aws_cloudwatch_log_group.metrics.name, awslogs-region = local.region, awslogs-stream-prefix = "worker" }
      }
    },
  ])
  tags = local.tags
}

# a one-off task for the accounts command (the first owner): run by a person with `aws ecs run-task` (README)
resource "aws_ecs_task_definition" "admin" {
  family                   = "${local.name}-admin"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = 512
  memory                   = 1024
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn
  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "X86_64"
  }
  container_definitions = jsonencode([{
    name        = "admin"
    image       = var.image
    essential   = true
    command     = ["python", "-m", "dclab_rnd.accounts", "list"]
    environment = [for k, v in local.environment : { name = k, value = v }]
    secrets     = concat(local.secrets, [{ name = "DCLAB_NEW_PASSWORD", valueFrom = aws_secretsmanager_secret.new_password.arn }])
    logConfiguration = {
      logDriver = "awslogs"
      options   = { awslogs-group = aws_cloudwatch_log_group.app.name, awslogs-region = local.region, awslogs-stream-prefix = "admin" }
    }
  }])
  tags = local.tags
}

resource "aws_ecs_service" "app" {
  name                               = "app"
  cluster                            = aws_ecs_cluster.main.id
  task_definition                    = aws_ecs_task_definition.app.arn
  desired_count                      = var.app_min
  launch_type                        = "FARGATE"
  deployment_minimum_healthy_percent = 100
  deployment_maximum_percent         = 200
  health_check_grace_period_seconds  = 120 # the first instance migrates the database before it answers (db.upgrade takes a lock)
  enable_execute_command             = false
  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }
  network_configuration {
    subnets          = aws_subnet.private[*].id
    security_groups  = [aws_security_group.tasks.id]
    assign_public_ip = false
  }
  load_balancer {
    target_group_arn = aws_lb_target_group.app.arn
    container_name   = "app"
    container_port   = 8765
  }
  lifecycle {
    ignore_changes = [desired_count] # the autoscaler owns it
  }
  depends_on = [aws_lb_listener.https, aws_efs_mount_target.workspace]
  tags       = local.tags
}

resource "aws_ecs_service" "worker" {
  name            = "worker"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.worker.arn
  desired_count   = var.worker_min
  launch_type     = "FARGATE"
  # A deployment starts the new workers beside the old ones (200%). An old worker sees it is superseded
  # (queue_metric.superseded), claims nothing new, finishes its jobs, drops its scale-in protection, and is stopped:
  # no job is interrupted, and the deployment does not wait on a worker that keeps taking new jobs.
  deployment_minimum_healthy_percent = 100
  deployment_maximum_percent         = 200
  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }
  network_configuration {
    subnets          = aws_subnet.private[*].id
    security_groups  = [aws_security_group.tasks.id]
    assign_public_ip = false
  }
  lifecycle {
    ignore_changes = [desired_count]
  }
  depends_on = [aws_efs_mount_target.workspace]
  tags       = local.tags
}
