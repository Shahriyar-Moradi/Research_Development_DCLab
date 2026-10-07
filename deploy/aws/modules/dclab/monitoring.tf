resource "aws_sns_topic" "alarms" {
  name = "${local.name}-alarms"
  tags = local.tags
}

resource "aws_sns_topic_subscription" "email" {
  count     = var.alarm_email == "" ? 0 : 1
  topic_arn = aws_sns_topic.alarms.arn
  protocol  = "email" # AWS mails a confirmation link first
  endpoint  = var.alarm_email
}

locals {
  alarms = {
    server-errors = {
      description = "The app answered 5xx more than 20 times in 5 minutes"
      namespace   = "AWS/ApplicationELB", metric = "HTTPCode_Target_5XX_Count", statistic = "Sum", threshold = 20, operator = "GreaterThanThreshold", periods = 1
      dimensions  = { LoadBalancer = aws_lb.app.arn_suffix }
    }
    unhealthy-app = {
      description = "An app task fails its readiness check (/readyz)"
      namespace   = "AWS/ApplicationELB", metric = "UnHealthyHostCount", statistic = "Maximum", threshold = 0, operator = "GreaterThanThreshold", periods = 3
      dimensions  = { LoadBalancer = aws_lb.app.arn_suffix, TargetGroup = aws_lb_target_group.app.arn_suffix }
    }
    database-cpu = {
      description = "The database is above 80% CPU for 15 minutes"
      namespace   = "AWS/RDS", metric = "CPUUtilization", statistic = "Average", threshold = 80, operator = "GreaterThanThreshold", periods = 3
      dimensions  = { DBInstanceIdentifier = aws_db_instance.main.identifier }
    }
    database-storage = {
      description = "The database has less than 5 GB free"
      namespace   = "AWS/RDS", metric = "FreeStorageSpace", statistic = "Minimum", threshold = 5 * 1024 * 1024 * 1024, operator = "LessThanThreshold", periods = 1
      dimensions  = { DBInstanceIdentifier = aws_db_instance.main.identifier }
    }
    jobs-backlog = {
      description = "More than 20 jobs waited for 15 minutes: worker_max may be too low"
      namespace   = "DCLab", metric = "QueuedJobs", statistic = "Maximum", threshold = 20, operator = "GreaterThanThreshold", periods = 3
      dimensions  = { Deployment = var.deployment }
    }
    job-failures = {
      description = "More than 5 jobs failed or were interrupted in 5 minutes (dclab_job_failures_total)"
      namespace   = "DCLab/Prometheus", metric = "dclab_job_failures_total", statistic = "Sum", threshold = 5, operator = "GreaterThanThreshold", periods = 1
      dimensions  = {}
    }
  }
}

resource "aws_cloudwatch_metric_alarm" "health" {
  for_each            = local.alarms
  alarm_name          = "${local.name}-${each.key}"
  alarm_description   = each.value.description
  namespace           = each.value.namespace
  metric_name         = each.value.metric
  statistic           = each.value.statistic
  dimensions          = each.value.dimensions
  period              = 300
  evaluation_periods  = each.value.periods
  threshold           = each.value.threshold
  comparison_operator = each.value.operator
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alarms.arn]
  ok_actions          = [aws_sns_topic.alarms.arn]
  tags                = local.tags
}
