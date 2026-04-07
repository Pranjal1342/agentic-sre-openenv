"""
server/pipeline.py
CI/CD/CM/CC Pipeline — Continuous Integration, Continuous Deployment,
Continuous Monitoring, and Continuous Correction.

Blueprint reference:
  "The simulated environment serves as a testbed for the next evolution of
   DevOps workflows: shifting from traditional CI/CD to a CI/CD/CM/CC
   architecture. The AI agent acts as the active 'Correction' engine,
   autonomously closing the loop by interpreting monitoring data, applying
   targeted remediation commands, and verifying the fix."

Pipeline Phases:
  ┌─────────────────────────────────────────────────────────┐
  │  CI  → CD  → CM  → CC  → CM (verify)  → [resolved]     │
  │  ↑                  │                                    │
  │  └──── fault still active (re-enters CC) ───────────────┘
  └─────────────────────────────────────────────────────────┘

  Phase 1 — CI (Continuous Integration):
    A code change is "merged and built." Represented in the environment
    as a fault being injected (simulating a bad deployment reaching prod).

  Phase 2 — CD (Continuous Deployment):
    The failing build is deployed. The environment records the deployment
    event and stamps the incident with the responsible task/seed so the
    grader can attribute causality deterministically.

  Phase 3 — CM (Continuous Monitoring):
    MockTelemetry continuously reads MockDatabase + MockServiceMesh state.
    It emits golden signals, computes Burn Rate, fires alerts when thresholds
    are breached. The monitoring loop drives the CC phase by surfacing alerts
    for the agent to act on.

  Phase 4 — CC (Continuous Correction):
    The SRE-Agent receives the alerts, works through the FSM
    (TRIAGE → INVESTIGATION → REMEDIATION → VERIFICATION), and applies
    targeted remediation commands. After each action, CM re-evaluates the
    system's health. When Health Score exceeds the recovery threshold,
    the CC phase signals resolution and the pipeline closes the loop.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional
import logging

from mock_infra.telemetry import MockTelemetry

logger = logging.getLogger(__name__)

# Health Score threshold above which the system is considered recovered
RECOVERY_HEALTH_THRESHOLD = 0.85


class PipelinePhase(Enum):
    """The four phases of the CI/CD/CM/CC loop."""
    CI = "continuous_integration"   # Fault injection (bad deploy event)
    CD = "continuous_deployment"    # Deployment recorded; incident created
    CM = "continuous_monitoring"    # Telemetry watching; alerts firing
    CC = "continuous_correction"    # Agent actively remediating


@dataclass
class DeploymentEvent:
    """
    Represents a CI/CD deployment event that introduced a fault.
    Links the simulated 'bad deploy' to the RL task and episode seed.
    """
    task_id: str
    seed: int
    fault_description: str
    commit_sha: str = "a3f9c12"    # Simulated commit that introduced the bug
    deployed_at_step: int = 0


@dataclass
class PipelineState:
    """
    Tracks the current position in the CI/CD/CM/CC loop across an episode.
    Updated at each step and used by the app.py to enrich episode metadata.
    """
    current_phase: PipelinePhase = PipelinePhase.CI
    deployment_event: Optional[DeploymentEvent] = None
    monitoring_cycles: int = 0          # How many CM ticks have run
    correction_attempts: int = 0        # How many CC actions the agent has taken
    alerts_fired: list[str] = field(default_factory=list)
    corrections_applied: list[str] = field(default_factory=list)
    resolved: bool = False
    phase_log: list[tuple[int, str, str]] = field(default_factory=list)  # (step, phase, event)

    def log_event(self, step: int, event: str) -> None:
        self.phase_log.append((step, self.current_phase.value, event))
        logger.info("[Pipeline:%s] step=%d — %s", self.current_phase.value, step, event)


class CICDCMCCPipeline:
    """
    Orchestrates the four-phase CI/CD/CM/CC loop for a single episode.

    Lifecycle per episode:
      1. on_reset()    — CI + CD phases: record the deployment event
      2. on_cm_tick()  — CM phase: read telemetry, fire alerts
      3. on_cc_action()— CC phase: record the agent's correction attempt
      4. on_verify()   — CM re-evaluation: check if system recovered
    """

    def __init__(self, telemetry: MockTelemetry) -> None:
        self._telemetry = telemetry
        self.state = PipelineState()

    # ── Phase 1 + 2: CI → CD ─────────────────────────────────────────────────

    def on_reset(self, task_id: str, seed: int, fault_description: str) -> None:
        """
        CI phase: A code change has been 'merged' (fault injected).
        CD phase: The bad build has been 'deployed' (environment is now degraded).
        """
        self.state = PipelineState()

        # CI — represent the integration of a bad change
        self.state.current_phase = PipelinePhase.CI
        self.state.log_event(0, f"CI: bad change merged — {fault_description}")

        # CD — represent the deployment reaching production
        self.state.current_phase = PipelinePhase.CD
        self.state.deployment_event = DeploymentEvent(
            task_id=task_id,
            seed=seed,
            fault_description=fault_description,
        )
        self.state.log_event(0, f"CD: fault deployed to prod (seed={seed})")

        # Immediately enter CM — monitoring always runs
        self.state.current_phase = PipelinePhase.CM
        self.state.log_event(0, "CM: monitoring loop started")

    # ── Phase 3: CM ───────────────────────────────────────────────────────────

    def on_cm_tick(self, step: int) -> list[str]:
        """
        Continuous Monitoring tick — called every step.
        Reads live telemetry and returns newly fired alerts.
        Transitions to CC if alerts are present.
        """
        self.state.monitoring_cycles += 1
        alerts = self._telemetry.get_active_alerts()

        new_alerts = [a for a in alerts if a not in self.state.alerts_fired]
        if new_alerts:
            for alert in new_alerts:
                self.state.alerts_fired.append(alert)
                self.state.log_event(step, f"CM ALERT: {alert}")

        # If active alerts exist and agent hasn't started CC, transition
        if alerts and self.state.current_phase == PipelinePhase.CM:
            self.state.current_phase = PipelinePhase.CC
            self.state.log_event(step, "CC: agent correction loop activated")

        return new_alerts

    # ── Phase 4: CC ───────────────────────────────────────────────────────────

    def on_cc_action(self, step: int, action: dict, stdout: str) -> None:
        """
        Continuous Correction step — the agent applied a correction.
        Records what was applied and what the system observed.
        """
        self.state.correction_attempts += 1
        action_type = action.get("action_type", "unknown")
        summary = f"{action_type}"

        if action_type == "remediation":
            op = action.get("operation_type", "?")
            target = action.get("target_service", "?")
            summary = f"remediation:{op}→{target}"

        self.state.corrections_applied.append(summary)
        self.state.log_event(step, f"CC ACTION #{self.state.correction_attempts}: {summary}")

    # ── Verification: CM re-evaluates after CC ────────────────────────────────

    def on_verify(self, step: int) -> tuple[bool, float]:
        """
        Post-correction CM verification step.
        Returns (recovered: bool, health_score: float).

        Recovery condition: Health Score > RECOVERY_HEALTH_THRESHOLD
        and active_alerts list is empty.
        """
        health = self._telemetry.compute_health_score()
        alerts = self._telemetry.get_active_alerts()

        if health >= RECOVERY_HEALTH_THRESHOLD and not alerts:
            self.state.resolved = True
            self.state.current_phase = PipelinePhase.CM  # Back to monitoring (steady state)
            self.state.log_event(
                step,
                f"CC RESOLVED: health={health:.3f} — pipeline loop closed. "
                f"Corrections applied: {self.state.correction_attempts}",
            )

        return self.state.resolved, health

    # ── Summary ───────────────────────────────────────────────────────────────

    def get_pipeline_summary(self) -> dict:
        """
        Returns a structured summary of the pipeline run.
        Exposed via GET /state and included in StepResult.info.
        """
        return {
            "pipeline_phase": self.state.current_phase.value,
            "monitoring_cycles": self.state.monitoring_cycles,
            "correction_attempts": self.state.correction_attempts,
            "alerts_fired_total": len(self.state.alerts_fired),
            "corrections_applied": self.state.corrections_applied,
            "resolved": self.state.resolved,
            "deployment": {
                "task_id": self.state.deployment_event.task_id if self.state.deployment_event else None,
                "fault": self.state.deployment_event.fault_description if self.state.deployment_event else None,
                "commit_sha": self.state.deployment_event.commit_sha if self.state.deployment_event else None,
            },
        }

    @property
    def current_phase(self) -> str:
        return self.state.current_phase.value
