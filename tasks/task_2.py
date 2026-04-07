"""
tasks/task_2.py
Task 2 — Active Mitigation of Resource Exhaustion (Medium)
max_steps: 25

Scenario:
  order-service is in an OOMKilled crash loop.
  Memory saturation = 98%, 5xx error rate is high.

Agent objective (two-phase):
  1. Diagnose: query saturation_pct / memory_usage_pct to confirm OOM.
  2. Mitigate: execute RemediationAction(operation_type='restart',
               target_service='order-service').
  3. Verify: re-query golden_signals to confirm resolution.

Grader:
  Uses nonlinear exponential decay — exact blueprint formula:
  Score = e^(-1.45 * t_m / T_max)
  Fast resolution → high score. Slow resolution → lower score.
  Restart of the WRONG service = behavioral penalty.
"""

import math
import random
from mock_infra.database import MockDatabase
from mock_infra.service_mesh import MockServiceMesh

TASK_ID = "task_2"
MAX_STEPS = 25
TARGET_SERVICE = "order-service"
WRONG_TARGETS = {"auth-service", "api-gateway", "db"}


def setup(db: MockDatabase, mesh: MockServiceMesh, rng: random.Random) -> None:
    """Inject OOMKilled fault on order-service."""
    db.reset()
    mesh.reset()
    mesh.inject_fault(TARGET_SERVICE, fault_type="oom_killed", value=98.0)


def get_initial_alerts() -> list[str]:
    return [
        "CRITICAL: order-service OOMKilled — CrashLoopBackOff detected",
        "HIGH_SATURATION: order-service memory=98% of limit (512Mi)",
    ]


def check_resolution(action: dict, mesh: MockServiceMesh) -> tuple[bool, str]:
    """
    Resolution: agent restarted the correct service and the fault is cleared.
    """
    if (
        action.get("action_type") == "remediation"
        and action.get("operation_type") == "restart"
        and action.get("target_service") == TARGET_SERVICE
        and mesh.is_healthy()
    ):
        return True, f"OOMKilled resolved: {TARGET_SERVICE} successfully restarted."
    return False, ""


def wrong_target_penalty(action: dict) -> float:
    """Penalty for restarting a service that is not the OOM culprit."""
    if (
        action.get("action_type") == "remediation"
        and action.get("target_service") in WRONG_TARGETS
    ):
        return 0.25
    return 0.0


def compute_mttm_score(steps_used: int) -> float:
    """
    Nonlinear exponential decay MTTM score — exact blueprint formula:

      Score = e^(-1.45 * (t_m / T_max))

    Where:
      t_m   = steps_used to achieve mitigation
      T_max = MAX_STEPS = 25
      -1.45 = blueprint-specified decay constant

    Step examples (T_max=25):
      step  1 → exp(-1.45 × 0.04) ≈ 0.944  (very fast)
      step  5 → exp(-1.45 × 0.20) ≈ 0.748  (fast)
      step 12 → exp(-1.45 × 0.48) ≈ 0.500  (moderate)
      step 25 → exp(-1.45 × 1.00) ≈ 0.235  (just at deadline)

    Delegates to graders/grader.py as the single source of truth.
    """
    from graders.grader import compute_task2_mttm_bonus
    return compute_task2_mttm_bonus(steps_used, MAX_STEPS)
