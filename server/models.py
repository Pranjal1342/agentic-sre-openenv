"""
server/models.py
Pydantic v2 data models — the canonical type-safe contract between the RL
agent and the SRE environment server.  (Replaces server/schemas.py)
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Annotated, Literal, List, Dict, Union, Any

# ==========================================
# 1. THE ACTION SPACE
# Discriminated union of typed tool actions.
# ==========================================

class DiagnosticQueryAction(BaseModel):
    """Query live telemetry metrics from the MockTelemetry layer."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "action_type": "diagnostic_query",
                "metric_identifier": "latency_p99_ms",
                "time_window": "5m",
            }
        }
    )

    action_type: Literal["diagnostic_query"] = "diagnostic_query"
    metric_identifier: str = Field(
        ...,
        description=(
            "Prometheus-compatible metric name. Valid: 'cpu_usage', 'error_rate', "
            "'latency_p99_ms', 'memory_usage_pct', 'connection_pool_used', 'traffic_rps'."
        ),
    )
    time_window: str = Field(
        ...,
        description="Prometheus duration format look-back window, e.g. '1m', '5m', '1h'.",
        pattern=r"^\d+[mhs]$",
    )


class LogInspectionAction(BaseModel):
    """Safely retrieve and filter application log lines."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "action_type": "log_inspection",
                "tail_lines": 50,
                "grep_pattern": "ERROR|deadlock|OOMKilled",
            }
        }
    )

    action_type: Literal["log_inspection"] = "log_inspection"
    tail_lines: int = Field(..., ge=1, le=500, description="Number of tail lines to return (1–500).")
    grep_pattern: str = Field(
        default="",
        description="Optional regex filter. Empty string returns all lines.",
    )


class RemediationAction(BaseModel):
    """
    Execute a state-changing remediation command.
    Literal typing structurally prevents hallucinated destructive operations.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "action_type": "remediation",
                "operation_type": "restart",
                "target_service": "order-service",
            }
        }
    )

    action_type: Literal["remediation"] = "remediation"
    operation_type: Literal["restart", "rollback", "scale_up"] = Field(
        ...,
        description=(
            "'restart' — restart a crashed pod; "
            "'rollback' — revert to previous deployment revision; "
            "'scale_up' — increase replica/resource limits."
        ),
    )
    target_service: str = Field(
        ...,
        description=(
            "The microservice, pod ID, or resource to target. "
            "Examples: 'order-service', 'db:pid:4821', 'connection-pool'."
        ),
    )


# Discriminated union — FastAPI/Pydantic resolves the correct model
# from the 'action_type' discriminator field.
AgentAction = Annotated[
    Union[DiagnosticQueryAction, LogInspectionAction, RemediationAction],
    Field(discriminator="action_type"),
]


# ==========================================
# 2. THE OBSERVATION SPACE
# ==========================================

class ObservationModel(BaseModel):
    """Complete sensory state returned after every reset() or step()."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "command_stdout": "latency_p99_ms{aggregated} = 847.3ms (window: 5m)",
                "command_stderr": "",
                "exit_code": 0,
                "active_alerts": ["HIGH_LATENCY: p99=847ms (threshold: 200ms)"],
                "golden_signals": {
                    "latency_p99_ms": 847.3,
                    "traffic_rps": 312.0,
                    "error_rate_pct": 2.1,
                    "saturation_pct": 61.0,
                },
                "rolling_summary": "",
            }
        }
    )

    command_stdout: str = Field(
        default="",
        description="Standard output of the previously executed action.",
    )
    command_stderr: str = Field(
        default="",
        description="Standard error output. Non-empty stderr means the agent made a syntax error.",
    )
    exit_code: int = Field(
        default=0,
        description="Numeric exit code. 0 = success, non-zero = failure.",
    )
    active_alerts: List[str] = Field(
        default_factory=list,
        description="All currently firing incident alerts. Empty list = incident resolved.",
    )
    golden_signals: Dict[str, float] = Field(
        default_factory=dict,
        description="Four Golden Signals snapshot: latency_p99_ms, traffic_rps, error_rate_pct, saturation_pct.",
    )
    rolling_summary: str = Field(
        default="",
        description="Auto-compressed episodic history (updated every 5 steps) to prevent context window saturation.",
    )


# ==========================================
# 3. THE REWARD SPACE
# ==========================================

class RewardBreakdown(BaseModel):
    """
    Named decomposition of the reward signal.

    Exact blueprint formula: R_t = α·ΔH_t + β·M_t + λ·E_t − γ·P_t − δ

    Blueprint-specified weights (constrain reward to [−0.5, 1.5]):
      α = 1.0  (health delta)
      β = 0.2  (milestone bonus)
      λ = 0.15 (action efficiency)
      γ = 0.5  (behavioral penalty)
      δ = 0.01 (time-step penalty)
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "health_delta": 0.12,
                "milestone_bonus": 0.20,
                "action_efficiency": 0.10,
                "behavioral_penalty": 0.0,
                "time_step_penalty": -0.01,
            }
        }
    )

    health_delta: float = Field(
        default=0.0,
        description="α·ΔH_t — reward for improving Health Score H_t. Weight α=1.0.",
    )
    milestone_bonus: float = Field(
        default=0.0,
        description="β·M_t — bonus for hitting a critical deductive milestone. Weight β=0.2.",
    )
    action_efficiency: float = Field(
        default=0.0,
        description="λ·E_t — Et=unique_queries/total_queries, penalises loops. Weight λ=0.15.",
    )
    behavioral_penalty: float = Field(
        default=0.0,
        description="γ·P_t — penalty for syntax errors or wrong service targets. Weight γ=0.5.",
    )
    time_step_penalty: float = Field(
        default=-0.01,
        description="δ — constant negative reward every step to encourage fast resolution. δ=0.01.",
    )


class RewardModel(BaseModel):
    """Type-safe reward returned by step()."""

    model_config = ConfigDict(
        json_schema_extra={"example": {"value": 0.48, "breakdown": {}}}
    )

    value: float = Field(..., description="Scalar reward consumed by the RL training loop.")
    breakdown: RewardBreakdown = Field(
        ...,
        description="Named decomposition for interpretability and debugging.",
    )


class StepResult(BaseModel):
    """Comprehensive tuple returned by step(action)."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "observation": {},
                "reward": {"value": 0.48, "breakdown": {}},
                "done": False,
                "info": {"fsm_state": "INVESTIGATION_STATE", "step_count": 3},
            }
        }
    )

    observation: ObservationModel
    reward: RewardModel
    done: bool = Field(
        ...,
        description="True when episode ended by resolution, max_steps, or fatal error.",
    )
    info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Auxiliary metadata: fsm_state, step_count, termination reason.",
    )


# ==========================================
# 4. REQUEST MODELS
# ==========================================

class ResetRequest(BaseModel):
    """Request body for POST /reset."""

    model_config = ConfigDict(
        json_schema_extra={"example": {"task_id": "task_1", "seed": 42}}
    )

    task_id: Literal["task_1", "task_2", "task_3"] = Field(
        ...,
        description=(
            "task_1=Diagnostic Triage (easy, 15 steps); "
            "task_2=OOMKilled Mitigation (medium, 25 steps); "
            "task_3=Cascading DB Failure (hard, 40 steps)."
        ),
    )
    seed: int = Field(
        default=42,
        description="Primary episode seed for deterministic fault injection.",
    )


class StepRequest(BaseModel):
    """Request body for POST /step."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "action": {
                    "action_type": "diagnostic_query",
                    "metric_identifier": "latency_p99_ms",
                    "time_window": "5m",
                }
            }
        }
    )

    action: AgentAction = Field(..., description="The typed action the agent executes this step.")
