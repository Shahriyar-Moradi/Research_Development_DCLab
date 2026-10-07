# ---------------------------------------------------------------------------------------------------- app: on CPU
resource "aws_appautoscaling_target" "app" {
  service_namespace  = "ecs"
  resource_id        = "service/${aws_ecs_cluster.main.name}/${aws_ecs_service.app.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  min_capacity       = var.app_min
  max_capacity       = var.app_max
}

resource "aws_appautoscaling_policy" "app_cpu" {
  name               = "${local.name}-app-cpu"
  policy_type        = "TargetTrackingScaling"
  service_namespace  = aws_appautoscaling_target.app.service_namespace
  resource_id        = aws_appautoscaling_target.app.resource_id
  scalable_dimension = aws_appautoscaling_target.app.scalable_dimension
  target_tracking_scaling_policy_configuration {
    target_value       = 60
    scale_in_cooldown  = 300
    scale_out_cooldown = 60
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
  }
}

# ---------------------------------------------------------------------------------------------------- workers: on the queue
# Each worker publishes the jobs waiting (QueuedJobs) and the jobs waiting or running (ActiveJobs) once a minute
# (dclab_rnd/jobs/queue_metric.py, the same rule the Google Cloud workers follow). Jobs waiting for two minutes add
# workers (one, or two when more than five wait); fifteen minutes with no job waiting or running removes one. A worker
# running a job also holds ECS scale-in protection (queue_metric.TaskProtection), so scaling in never stops it.
resource "aws_appautoscaling_target" "worker" {
  service_namespace  = "ecs"
  resource_id        = "service/${aws_ecs_cluster.main.name}/${aws_ecs_service.worker.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  min_capacity       = var.worker_min
  max_capacity       = var.worker_max
}

resource "aws_appautoscaling_policy" "worker_out" {
  name               = "${local.name}-worker-out"
  policy_type        = "StepScaling"
  service_namespace  = aws_appautoscaling_target.worker.service_namespace
  resource_id        = aws_appautoscaling_target.worker.resource_id
  scalable_dimension = aws_appautoscaling_target.worker.scalable_dimension
  step_scaling_policy_configuration {
    adjustment_type         = "ChangeInCapacity"
    cooldown                = 120
    metric_aggregation_type = "Maximum"
    step_adjustment {
      metric_interval_lower_bound = 0
      metric_interval_upper_bound = 5
      scaling_adjustment          = 1
    }
    step_adjustment {
      metric_interval_lower_bound = 5
      scaling_adjustment          = 2
    }
  }
}

resource "aws_appautoscaling_policy" "worker_in" {
  name               = "${local.name}-worker-in"
  policy_type        = "StepScaling"
  service_namespace  = aws_appautoscaling_target.worker.service_namespace
  resource_id        = aws_appautoscaling_target.worker.resource_id
  scalable_dimension = aws_appautoscaling_target.worker.scalable_dimension
  step_scaling_policy_configuration {
    adjustment_type         = "ChangeInCapacity"
    cooldown                = 300
    metric_aggregation_type = "Maximum"
    step_adjustment {
      metric_interval_upper_bound = 0
      scaling_adjustment          = -1
    }
  }
}

resource "aws_cloudwatch_metric_alarm" "queue_waiting" {
  alarm_name          = "${local.name}-jobs-waiting"
  alarm_description   = "Jobs wait: add workers"
  namespace           = "DCLab"
  metric_name         = "QueuedJobs"
  dimensions          = { Deployment = var.deployment }
  statistic           = "Maximum"
  period              = 60
  evaluation_periods  = 2
  threshold           = 0
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_appautoscaling_policy.worker_out.arn]
  tags                = local.tags
}

resource "aws_cloudwatch_metric_alarm" "queue_empty" {
  alarm_name          = "${local.name}-jobs-none-active"
  alarm_description   = "No job waited or ran for 15 minutes: remove a worker (never below worker_min)"
  namespace           = "DCLab"
  metric_name         = "ActiveJobs"
  dimensions          = { Deployment = var.deployment }
  statistic           = "Maximum"
  period              = 60
  evaluation_periods  = 15
  threshold           = 0
  comparison_operator = "LessThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_appautoscaling_policy.worker_in.arn]
  tags                = local.tags
}
