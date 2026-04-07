"""
agents/code_agent.py
Code-Agent — handles LogInspectionAction by returning deterministic
synthetic log output based on the current infra fault state.
Operates during INVESTIGATION_STATE.
"""

import re
from mock_infra.database import MockDatabase
from mock_infra.service_mesh import MockServiceMesh

# Synthetic log templates per fault type
_LOG_TEMPLATES: dict[str, list[str]] = {
    "latency": [
        "2026-04-06T12:01:03Z [order-service] WARN  upstream latency spike: db responded in 823ms",
        "2026-04-06T12:01:04Z [order-service] WARN  upstream latency spike: db responded in 917ms",
        "2026-04-06T12:01:05Z [api-gateway]   ERROR upstream timeout: order-service did not respond within 1000ms",
        "2026-04-06T12:01:06Z [order-service] WARN  slow query: SELECT * FROM orders WHERE status='pending' [823ms]",
        "2026-04-06T12:01:07Z [auth-service]  INFO  request completed in 34ms",
        "2026-04-06T12:01:08Z [api-gateway]   ERROR 504 Gateway Timeout → order-service",
    ],
    "connection_pool_exhaustion": [
        "2026-04-06T12:01:01Z [db]            ERROR too many connections (18/20 pool used)",
        "2026-04-06T12:01:02Z [order-service] ERROR connection refused: db pool exhausted — retrying…",
        "2026-04-06T12:01:03Z [order-service] ERROR DEADLOCK detected on pid=4821",
        "2026-04-06T12:01:04Z [db]            WARN  lock contention: UPDATE orders SET status='processing'",
        "2026-04-06T12:01:05Z [order-service] ERROR SQLAlchemyError: could not obtain connection from pool",
        "2026-04-06T12:01:06Z [db]            INFO  pg_stat_activity: pid 4821 blocking 4 other queries",
        "2026-04-06T12:01:07Z [auth-service]  WARN  db query degraded: 1240ms (normal: 40ms)",
    ],
    "oom_killed": [
        "2026-04-06T12:01:01Z [order-service] WARN  memory usage at 94% of limit (limit: 512Mi)",
        "2026-04-06T12:01:02Z [order-service] WARN  memory usage at 97% of limit — approaching OOM",
        "2026-04-06T12:01:03Z [kubelet]       INFO  Killing container: order-service (OOMKilled)",
        "2026-04-06T12:01:04Z [kubelet]       INFO  Back-off restarting failed container order-service",
        "2026-04-06T12:01:05Z [order-service] INFO  container started (attempt 3)",
        "2026-04-06T12:01:06Z [order-service] ERROR java.lang.OutOfMemoryError: Java heap space",
    ],
    "default": [
        "2026-04-06T12:01:00Z [api-gateway]   INFO  request received GET /api/orders",
        "2026-04-06T12:01:00Z [auth-service]  INFO  token validated",
        "2026-04-06T12:01:01Z [order-service] INFO  processing order #88241",
        "2026-04-06T12:01:02Z [db]            INFO  query completed in 38ms",
    ],
}


class CodeAgent:
    """
    Specialised sub-agent for log inspection and code analysis.
    Returns deterministic synthetic log output shaped by the current fault state.
    """

    def __init__(self, db: MockDatabase, mesh: MockServiceMesh) -> None:
        self._db = db
        self._mesh = mesh

    def execute(self, tail_lines: int, grep_pattern: str) -> tuple[str, str, int]:
        """
        Execute a log inspection action.

        Returns:
            (stdout, stderr, exit_code)
        """
        try:
            # Determine which log template matches current fault
            db_fault = self._db.fault_type
            mesh_faults = list(self._mesh.faults.values())
            mesh_fault_type = mesh_faults[0].fault_type if mesh_faults else None

            template_key = "default"
            if db_fault == "connection_pool_exhaustion":
                template_key = "connection_pool_exhaustion"
            elif mesh_fault_type == "oom_killed":
                template_key = "oom_killed"
            elif mesh_fault_type == "latency":
                template_key = "latency"

            lines: list[str] = _LOG_TEMPLATES.get(template_key, _LOG_TEMPLATES["default"])

            # Apply grep filter
            if grep_pattern:
                try:
                    pattern = re.compile(grep_pattern, re.IGNORECASE)
                    filtered = [l for l in lines if pattern.search(l)]
                except re.error as e:
                    return "", f"grep: invalid pattern: {e}", 2
            else:
                filtered = lines

            # Apply tail limit
            result = filtered[-tail_lines:]
            if not result:
                return f"(no lines matched grep pattern '{grep_pattern}')", "", 0

            stdout = "\n".join(result)
            return stdout, "", 0

        except Exception as exc:
            return "", f"CodeAgent error: {exc}", 1
