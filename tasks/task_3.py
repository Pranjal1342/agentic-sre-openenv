"""
tasks/task_3.py
Task 3 — Cascading Failure and Root Cause Resolution (Hard)
max_steps: 40

Scenario:
  DB connection pool is exhausted (18/20 connections used).
  PID 4821 holds a long-running UPDATE lock blocking 4 other queries.
  Cascading: order-service errors spike, api-gateway latency degrades.

Agent objective (four-stage, partial credit per stage):
  Stage 1 (0.25): Detect high saturation / pool exhaustion via telemetry.
  Stage 2 (0.25): Inspect logs or pg_stat_activity to identify PID 4821.
  Stage 3 (0.25): Execute RemediationAction(rollback, 'db:pid:4821') to kill lock.
  Stage 4 (0.25): Execute RemediationAction(scale_up, 'connection-pool') to prevent recurrence.

Grader:
  Deterministic state inspection of MockDatabase — no LLM self-reporting.
  partial_score = stages_completed / 4
"""

import random
from mock_infra.database import MockDatabase
from mock_infra.service_mesh import MockServiceMesh

TASK_ID = "task_3"
MAX_STEPS = 40
BLOCKING_PID = 4821
POOL_INITIAL_USED = 18
POOL_SCALE_TARGET = 50


def setup(db: MockDatabase, mesh: MockServiceMesh, rng: random.Random) -> None:
    """Inject cascading DB failure: pool exhaustion + deadlock PID."""
    db.reset()
    mesh.reset()
    db.inject_fault(
        "connection_pool_exhaustion",
        used=POOL_INITIAL_USED,
        blocking_pid=BLOCKING_PID,
        wait_ms=45000,
    )
    # Cascading effect: services see high error rates due to pool exhaustion
    mesh.inject_fault("order-service", fault_type="http_500", value=17.3)
    mesh.inject_fault("auth-service", fault_type="latency", value=400.0)


def get_initial_alerts() -> list[str]:
    return [
        f"CRITICAL: DB connection pool exhausted ({POOL_INITIAL_USED}/20 used)",
        f"DB_LOCK_CONTENTION: blocking PID={BLOCKING_PID} holding lock for >45s",
        "HIGH_ERROR_RATE: order-service 5xx=17.3%",
    ]


class StageTracker:
    """
    Tracks multi-stage completion for Task 3 partial credit grading.
    Inspects MockDatabase state directly — no agent self-reporting.
    """

    def __init__(self) -> None:
        self.stage_1_done = False  # Pool exhaustion detected via telemetry
        self.stage_2_done = False  # PID identified via logs / pg_stat_activity
        self.stage_3_done = False  # PID killed (pool lock cleared)
        self.stage_4_done = False  # Pool scaled up

    @property
    def stages_completed(self) -> int:
        return sum([self.stage_1_done, self.stage_2_done, self.stage_3_done, self.stage_4_done])

    @property
    def partial_score(self) -> float:
        return self.stages_completed / 4.0

    @property
    def fully_resolved(self) -> bool:
        return self.stages_completed == 4

    def update(self, action: dict, stdout: str, db: MockDatabase) -> list[str]:
        """
        Update stage completion based on action + current DB state.
        Returns list of newly completed stage names.
        """
        newly_done: list[str] = []
        action_type = action.get("action_type", "")
        metric = action.get("metric_identifier", "")

        # Stage 1: agent queries saturation or connection pool metrics
        if not self.stage_1_done and action_type == "diagnostic_query":
            if metric in ("connection_pool_used", "saturation_pct", "error_rate"):
                self.stage_1_done = True
                newly_done.append("Stage 1: pool exhaustion detected")

        # Stage 2: agent inspects logs or pg_stat_activity and sees PID 4821
        if not self.stage_2_done:
            if action_type in ("log_inspection", "diagnostic_query"):
                if str(BLOCKING_PID) in stdout:
                    self.stage_2_done = True
                    newly_done.append(f"Stage 2: blocking PID {BLOCKING_PID} identified")

        # Stage 3: PID killed — verified by DB state (lock no longer present)
        if not self.stage_3_done and BLOCKING_PID not in db.active_locks:
            if self.stage_2_done:  # Must have found PID first
                self.stage_3_done = True
                newly_done.append(f"Stage 3: PID {BLOCKING_PID} terminated")

        # Stage 4: Pool scaled — pool_max > initial default
        if not self.stage_4_done and db.pool_max > MockDatabase.DEFAULT_POOL_MAX:
            self.stage_4_done = True
            newly_done.append(f"Stage 4: connection pool scaled to {db.pool_max}")

        return newly_done
