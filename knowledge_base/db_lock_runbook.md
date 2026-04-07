# DB Lock Contention Runbook: PostgreSQL Connection Pool Exhaustion

## Overview

This runbook covers the diagnosis and remediation of PostgreSQL connection pool
exhaustion caused by long-running transaction locks (deadlocks). This is a
cascading failure scenario: the DB blockage causes order-service errors,
which degrade api-gateway latency.

## Alert Signature

```
CRITICAL: DB connection pool exhausted (18/20 used)
DB_LOCK_CONTENTION: blocking PID=4821 holding lock for >45s
HIGH_ERROR_RATE: order-service 5xx=17.3%
```

## Root Cause Pattern

A long-running UPDATE query holds an exclusive row-level lock, blocking
other transactions from acquiring the same resource. As blocked connections
accumulate, the connection pool exhausts, causing new requests to fail
with "too many connections" errors, cascading across all dependent services.

## Diagnostic Steps

### Stage 1: Detect pool exhaustion

Query `connection_pool_used` to observe connection pool saturation.
A reading of 85%+ (17+ of 20 connections) is a critical threshold.

Also query `saturation_pct` and `error_rate` to confirm the cascading impact.

### Stage 2: Identify the blocking PID

Run a log inspection with `grep_pattern: "lock|deadlock|pid|blocking"`.

Then query `connection_pool_used` — this also returns the pg_stat_activity
output showing the blocking PID and its query text.

Alternatively, query `pg_stat_activity` explicitly:
```sql
SELECT pid, state, wait_event_type, left(query,60)
FROM pg_stat_activity
WHERE wait_event_type='Lock';
```

Look for the PID holding the lock with `wait_ms > 10000`.
The blocking PID in this scenario is **4821**.

### Stage 3: Terminate the blocking PID

Execute a rollback remediation targeting the specific PID:
```
RemediationAction(
    operation_type="rollback",
    target_service="db:pid:4821"
)
```

This simulates `SELECT pg_terminate_backend(4821)`.
On success, the lock is released, several connections are freed,
and cascading errors begin to resolve.

### Stage 4: Prevent recurrence — scale the connection pool

To prevent future pool exhaustion, scale the max_connections:
```
RemediationAction(
    operation_type="scale_up",
    target_service="connection-pool"
)
```

This executes:
```sql
ALTER SYSTEM SET max_connections = 50;
SELECT pg_reload_conf();
```

## Grading (Partial Credit)

Each stage earns 0.25 partial credit (total max = 1.0):
- Stage 1 (0.25): Queried pool/saturation metrics
- Stage 2 (0.25): Identified blocking PID 4821 in output
- Stage 3 (0.25): PID 4821 terminated (verified by DB state)
- Stage 4 (0.25): Connection pool scaled beyond default (20)

## Verification

After Stages 3 and 4:
- Query `connection_pool_used`: pool_used should drop significantly
- Query `error_rate`: order-service 5xx should fall below 1%
- Query `latency_p99_ms`: api-gateway latency should normalise
- `active_alerts` list should become empty (all alerts cleared)
