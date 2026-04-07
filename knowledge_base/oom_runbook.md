# OOMKilled Runbook: Memory Exhaustion and Pod Restart

## Overview

This runbook covers the diagnosis and remediation of OOMKilled (Out-of-Memory)
pod crashes in the Kubernetes cluster. OOM events cause crash loops and elevated
error rates as the kubelet attempts to restart failed containers.

## Alert Signature

```
CRITICAL: order-service OOMKilled — CrashLoopBackOff detected
HIGH_SATURATION: order-service memory=98% of limit
```

## Background

OOMKilled occurs when a container's memory usage exceeds the limit defined in
its Kubernetes resource spec. The Linux kernel's OOM killer terminates the
process, and the kubelet marks the pod as failed and attempts to restart it
(CrashLoopBackOff if it restarts repeatedly).

## Diagnostic Steps

### Step 1: Confirm OOM via saturation metrics

Query `memory_usage_pct` or `saturation_pct` with a 5-minute window.
A reading above 90% indicates imminent OOM risk.
A reading at 98%+ confirms the OOM event is in progress.

### Step 2: Identify the affected pod

Query `error_rate` to confirm which service is generating 5xx errors.
OOMKilled pods serve no traffic during restart — their error rate appears
as 100% for the duration of the crash loop.

### Step 3: Inspect pod logs

Use `log_inspection` with `grep_pattern: "OOMKilled|OutOfMemoryError|heap|killed"`.

Expected log signatures:
- `Killing container: order-service (OOMKilled)` — kubelet kill event
- `java.lang.OutOfMemoryError: Java heap space` — JVM heap exhaustion
- `Back-off restarting failed container` — confirms CrashLoopBackOff

## Remediation

### Immediate mitigation (MTTM priority)

Restart the affected pod to clear the OOM state and release memory:
```
RemediationAction(operation_type="restart", target_service="order-service")
```
This resolves the crash loop and restores service availability.

**Important:** Only restart `order-service` if it is the confirmed OOM culprit.
Restarting `auth-service` or `api-gateway` is a behavioral error and will incur
a grader penalty.

### Permanent fix (post-incident)

Scale up resource limits:
```
RemediationAction(operation_type="scale_up", target_service="order-service")
```

## Verification

After restart, re-query `saturation_pct` and `error_rate` with a 1-minute window.
Saturation should drop below 70% and error rate below 1% within 2 minutes of restart.
Active alerts should clear (`active_alerts` list becomes empty).

## SLA Reference

Task 2 MTTM SLA: resolution must occur within `max_steps=25` steps.
Score is calculated as `exp(-3 * steps_used / 25)`.
Resolution in ≤ 4 steps yields a score of ≈ 0.70.
Resolution at step 25 yields a score of ≈ 0.05.
