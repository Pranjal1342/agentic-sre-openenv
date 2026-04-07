"""
server/fsm.py
Deterministic Enum-based Finite State Machine orchestrator.
Governs state transitions among the four specialised agent roles.

States: TRIAGE → INVESTIGATION → REMEDIATION → VERIFICATION
"""

from enum import Enum


class AgentState(Enum):
    TRIAGE = "TRIAGE_STATE"
    INVESTIGATION = "INVESTIGATION_STATE"
    REMEDIATION = "REMEDIATION_STATE"
    VERIFICATION = "VERIFICATION_STATE"


VALID_TRANSITIONS: dict[AgentState, list[AgentState]] = {
    AgentState.TRIAGE: [AgentState.INVESTIGATION],
    AgentState.INVESTIGATION: [AgentState.INVESTIGATION, AgentState.REMEDIATION],
    AgentState.REMEDIATION: [AgentState.VERIFICATION],
    AgentState.VERIFICATION: [],  # Terminal
}

ACTION_STATE_HINT: dict[str, AgentState] = {
    "diagnostic_query": AgentState.INVESTIGATION,
    "log_inspection": AgentState.INVESTIGATION,
    "remediation": AgentState.REMEDIATION,
}


class FSMOrchestrator:
    """
    Tracks the agent's diagnostic/remediation progress through the FSM.
    Awards milestone bonuses on valid forward state transitions.
    Serves as a programmatic guardrail preventing unstructured loops.
    """

    def __init__(self) -> None:
        self.current_state: AgentState = AgentState.TRIAGE
        self.transition_log: list[tuple[AgentState, AgentState]] = []
        self._milestone_hit: bool = False

    def reset(self) -> None:
        self.current_state = AgentState.TRIAGE
        self.transition_log = []
        self._milestone_hit = False

    def process_action(self, action_type: str) -> tuple[AgentState, bool]:
        """
        Advance the FSM based on the agent's action type.
        Returns (current_state, transition_occurred).
        """
        self._milestone_hit = False
        target = ACTION_STATE_HINT.get(action_type)
        if target is None:
            return self.current_state, False

        allowed = VALID_TRANSITIONS.get(self.current_state, [])
        if target not in allowed:
            return self.current_state, False

        if target == self.current_state:
            # Staying in INVESTIGATION — valid but not a milestone
            return self.current_state, False

        self._apply_transition(target)
        return self.current_state, True

    def advance_to_verification(self) -> bool:
        """Force REMEDIATION → VERIFICATION after grader confirms success."""
        if self.current_state == AgentState.REMEDIATION:
            self._apply_transition(AgentState.VERIFICATION)
            return True
        return False

    def _apply_transition(self, target: AgentState) -> None:
        old = self.current_state
        self.current_state = target
        self.transition_log.append((old, target))
        self._milestone_hit = True

    @property
    def state_name(self) -> str:
        return self.current_state.value

    @property
    def transitions_completed(self) -> int:
        return len(self.transition_log)

    @property
    def is_terminal(self) -> bool:
        return self.current_state == AgentState.VERIFICATION

    def milestone_just_hit(self) -> bool:
        return self._milestone_hit

    def __repr__(self) -> str:
        return f"FSMOrchestrator(state={self.state_name}, transitions={self.transitions_completed})"
