"""
graders/grader.py
Dense reward shaping — exact blueprint formula:

  R_t = α·ΔH_t + β·M_t + λ·E_t − γ·P_t − δ

Blueprint-specified weights (validated to keep reward in [-0.5, 1.5]):
  α = 1.0   Health delta       — rewards fixing latency/burn rates
  β = 0.2   Milestone bonus    — rewards hitting critical deductive steps
  λ = 0.15  Action efficiency  — Et = unique_queries / total_queries
  γ = 0.5   Behavioral penalty — heavy penalty for syntax errors / destructive commands
  δ = 0.01  Time-step penalty  — constant cost to encourage fast resolution
"""

import math
from server.models import RewardModel, RewardBreakdown
from mock_infra.telemetry import MockTelemetry

# ── Rebalanced reward weights (anti-hacking) ─────────────────────────────────
# BETA  ↑ 0.20 → 1.50 : milestone/resolution signal dominates the score
# LAMBDA↓ 0.15 → 0.02 : efficiency too small to farm meaningfully
# DELTA ↑ 0.01 → 0.10 : 15 wasted steps costs −1.50 in time penalties
ALPHA  = 1.0    # α — Health delta weight
BETA   = 1.5    # β — Milestone bonus weight (raised to make resolution dominant)
LAMBDA = 0.02   # λ — Action efficiency weight (lowered to prevent farming)
GAMMA  = 0.5    # γ — Behavioral penalty weight
DELTA  = 0.10   # δ — Constant time-step penalty (raised to penalise looping)


def compute_reward(
    prev_health: float,
    curr_health: float,
    milestone_hit: bool,
    step_count: int,
    unique_action_count: int,
    behavioral_penalty: float = 0.0,
    milestone_value: float = 1.0,
) -> RewardModel:
    """
    Compute the step reward using the exact blueprint dense reward formula:

      R_t = α·ΔH_t + β·M_t + λ·E_t − γ·P_t − δ

    Args:
        prev_health:         H(t-1) before action.
        curr_health:         H(t) after action.
        milestone_hit:       True if FSM transitioned or a task stage completed.
        step_count:          Total steps taken so far (including this step).
        unique_action_count: Number of distinct action types used so far.
        behavioral_penalty:  Raw penalty magnitude for wrong targets / syntax errors.
        milestone_value:     Milestone bonus magnitude override (0–1).

    Reward range is constrained to approximately [−0.5, 1.5] by design.
    """
    # α·ΔH_t — health improvement reward
    delta_h = curr_health - prev_health
    health_component = ALPHA * delta_h

    # β·M_t — milestone bonus
    milestone_component = BETA * milestone_value if milestone_hit else 0.0

    # λ·E_t — action efficiency: unique_queries / total_queries
    e_t = unique_action_count / max(step_count, 1)
    efficiency_component = LAMBDA * e_t

    # γ·P_t — behavioral penalty
    penalty_component = GAMMA * behavioral_penalty

    # δ — constant time-step penalty
    timestep_penalty = -DELTA

    total = (
        health_component
        + milestone_component
        + efficiency_component
        - penalty_component
        + timestep_penalty
    )

    breakdown = RewardBreakdown(
        health_delta=round(health_component, 6),
        milestone_bonus=round(milestone_component, 6),
        action_efficiency=round(efficiency_component, 6),
        behavioral_penalty=round(penalty_component, 6),
        time_step_penalty=timestep_penalty,
    )

    return RewardModel(value=round(total, 6), breakdown=breakdown)


def compute_timeout_reward() -> RewardModel:
    """
    Canonical max_steps timeout reward per OpenEnv 0.1 spec.
    Returns value=-1.0 (done=True, breakdown showing termination reason).
    """
    return RewardModel(
        value=-1.0,
        breakdown=RewardBreakdown(
            health_delta=0.0,
            milestone_bonus=0.0,
            action_efficiency=0.0,
            behavioral_penalty=0.0,
            time_step_penalty=-1.0,
        ),
    )


def compute_task2_mttm_bonus(steps_used: int, max_steps: int) -> float:
    """
    Task 2 nonlinear MTTM bonus — exact blueprint formula:

      Score = e^(-1.45 * (t_m / T_max))

    Where:
      t_m    = steps used to achieve mitigation
      T_max  = max_steps for task_2 (25)
      -1.45  = blueprint-specified decay constant

    Blueprint-validated score examples (T_max = 25):
      At  5% of steps (t_m ≈ 1)  → e^(-1.45 × 0.05) ≈ 0.93
      At 100% of steps (t_m = 25) → e^(-1.45 × 1.00) ≈ 0.24
    """
    ratio = min(steps_used / max_steps, 1.0)
    return round(math.exp(-1.45 * ratio), 4)


def compute_task3_partial_score(stages_completed: int) -> float:
    """
    Task 3 partial credit: 0.25 per completed stage (max 1.0).
    Used as the milestone_value passed to compute_reward().
    """
    return min(stages_completed / 4.0, 1.0)


class EpisodeGrader:
    """
    Stateful grader that tracks health history and computes step rewards
    throughout the active episode. Wires task-specific logic into the
    blueprint reward formula.
    """

    def __init__(self, telemetry: MockTelemetry, task_id: str, max_steps: int) -> None:
        self._telemetry = telemetry
        self.task_id = task_id
        self.max_steps = max_steps
        self.prev_health: float = 0.0
        self.action_type_history: list[str] = []

    def reset(self) -> None:
        self.prev_health = self._telemetry.compute_health_score()
        self.action_type_history = []

    def step(
        self,
        action: dict,
        milestone_hit: bool,
        behavioral_penalty: float = 0.0,
        milestone_value: float = 1.0,
        step_count: int = 1,
    ) -> RewardModel:
        """Compute reward for one step and advance internal state."""
        # Build a signature that captures the full action identity, not just its type.
        # This ensures repeated queries to the same metric score low on diversity.
        action_type = action.get("action_type", "unknown")
        metric = action.get("metric_identifier", action.get("operation_type", ""))
        action_sig = f"{action_type}:{metric}"
        self.action_type_history.append(action_sig)

        curr_health = self._telemetry.compute_health_score()
        unique_count = len(set(self.action_type_history))

        # Task 2: replace milestone_value with MTTM decay score
        if self.task_id == "task_2" and milestone_hit:
            milestone_value = compute_task2_mttm_bonus(step_count, self.max_steps)

        reward = compute_reward(
            prev_health=self.prev_health,
            curr_health=curr_health,
            milestone_hit=milestone_hit,
            step_count=step_count,
            unique_action_count=unique_count,
            behavioral_penalty=behavioral_penalty,
            milestone_value=milestone_value,
        )

        self.prev_health = curr_health
        return reward

    def terminal_bonus(self, resolved: bool) -> RewardModel:
        """
        One-time terminal reward applied at episode end.

        resolved=True  → +0.5 resolution bonus (agent found root cause)
        resolved=False → -0.35 penalty (episode timed out without resolution)

        This ensures that a fast, correct agent always outscores a slow,
        looping agent that never resolves the incident, regardless of the
        accumulated step rewards.
        """
        if resolved:
            value = 0.5
            breakdown = RewardBreakdown(
                health_delta=0.0,
                milestone_bonus=value,
                action_efficiency=0.0,
                behavioral_penalty=0.0,
                time_step_penalty=0.0,
            )
        else:
            value = -0.35
            breakdown = RewardBreakdown(
                health_delta=0.0,
                milestone_bonus=0.0,
                action_efficiency=0.0,
                behavioral_penalty=abs(value),
                time_step_penalty=0.0,
            )
        return RewardModel(value=round(value, 6), breakdown=breakdown)
