"""
tasks/task_1.py
Task 1 — Diagnostic Triage and Localisation (Easy)
max_steps: 15

Scenario:
  An alert fires for elevated p99 latency at the API gateway level.
  Root cause: latency fault injected specifically into 'order-service'.
  auth-service is healthy — the agent must NOT waste steps there.

Agent objective:
  Trace the latency spike through the service mesh and correctly
  identify 'order-service' as the root cause.

Grader:
  - Milestone bonus for querying latency on order-service specifically.
  - Step-decay penalty for querying auth-service (wrong service).
  - Episode resolves when the agent issues a diagnostic query proving
    order-service is the latency source (exit_code=0, metric > threshold).
"""

import random
from mock_infra.database import MockDatabase
from mock_infra.service_mesh import MockServiceMesh
from mock_infra.telemetry import MockTelemetry

TASK_ID = "task_1"
MAX_STEPS = 15
LATENCY_FAULT_MS = 780.0   # Injected latency on order-service
ROOT_CAUSE_SERVICE = "order-service"
WRONG_SERVICE = "auth-service"


def setup(db: MockDatabase, mesh: MockServiceMesh, rng: random.Random) -> None:
    """
    Inject the Task 1 fault scenario into the mock infrastructure.
    Called once at episode reset.
    """
    db.reset()
    mesh.reset()
    mesh.inject_fault(ROOT_CAUSE_SERVICE, fault_type="latency", value=LATENCY_FAULT_MS)


def get_initial_alerts() -> list[str]:
    return [
        "HIGH_LATENCY: api-gateway p99 > 500ms — downstream root cause unknown",
    ]


def check_resolution(action: dict, stdout: str) -> tuple[bool, str]:
    """
    Determine if the agent's action constitutes task resolution.

    Resolution condition: agent issued a DiagnosticQueryAction targeting
    'latency_p99_ms' and the response mentions 'order-service' explicitly.
    """
    if action.get("action_type") != "diagnostic_query":
        return False, ""
    if action.get("metric_identifier") not in ("latency_p99_ms", "latency"):
        return False, ""
    # Check that the agent received the per-service breakdown showing order-service
    if "order-service" in stdout and action.get("time_window"):
        return True, f"Root cause identified: {ROOT_CAUSE_SERVICE} latency confirmed."
    return False, ""


def wrong_service_penalty(action: dict) -> float:
    """Return a penalty if the agent queries the wrong (unaffected) service."""
    if action.get("action_type") == "diagnostic_query":
        # auth-service is healthy — querying it is wasted exploration
        if WRONG_SERVICE in action.get("metric_identifier", ""):
            return 0.15
    return 0.0
