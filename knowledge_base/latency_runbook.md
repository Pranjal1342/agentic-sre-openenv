# Latency Runbook: API Gateway and Service Mesh Investigation

## Overview

This runbook covers the diagnosis and remediation of elevated API gateway
latency caused by downstream service degradation in the microservice mesh.

## Alert Signature

```
HIGH_LATENCY: api-gateway p99 > 500ms — downstream root cause unknown
```

## Diagnostic Steps

### Step 1: Query per-service latency breakdown

Run a `latency_p99_ms` diagnostic query with a 5-minute window.
Inspect the per-service breakdown to identify which downstream service
is contributing excess latency.

Expected healthy baseline: all services < 200ms p99.
Any service > 400ms p99 should be considered the root cause candidate.

### Step 2: Check service dependencies

The api-gateway routes to two downstream services:
- `order-service` (handles order processing — database-intensive)
- `auth-service` (handles token validation — lightweight, rarely the cause)

If `order-service` latency > 500ms but `auth-service` is healthy (<100ms),
the root cause is isolated to `order-service`.

### Step 3: Inspect order-service logs

Use a log inspection action with `grep_pattern: "WARN|ERROR|upstream|slow query"`.

Look for patterns:
- "upstream latency spike" — confirms the service is itself degraded
- "slow query" — indicates the database layer is the secondary cause
- "504 Gateway Timeout" — confirms the gateway is timing out

### Step 4: Check database metrics

If logs show slow queries, query `connection_pool_used` to verify
whether the DB connection pool is contributing to the latency.

## Remediation

If the root cause is isolated to `order-service`:
1. **Rollback** (preferred): `RemediationAction(operation_type="rollback", target_service="order-service")`
   — reverts to the previous stable deployment if a recent release caused the regression.

2. **Restart** (if no recent deployment): `RemediationAction(operation_type="restart", target_service="order-service")`

## Verification

After remediation, re-query `latency_p99_ms` with a 1-minute window.
Confirm all services return to < 200ms p99 before closing the incident.
