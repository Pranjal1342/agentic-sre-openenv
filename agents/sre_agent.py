"""
agents/sre_agent.py
SRE-Agent — primary orchestrator that routes actions to the appropriate
sub-agent (DataAgent or CodeAgent) based on action type, and validates
all operations through the FSM guardrail.
"""

from server.fsm import FSMOrchestrator
from agents.data_agent import DataAgent
from agents.code_agent import CodeAgent
from agents.quarantine_agent import QuarantineAgent
from mock_infra.database import MockDatabase
from mock_infra.service_mesh import MockServiceMesh
from mock_infra.telemetry import MockTelemetry


class SREAgent:
    """
    Central orchestrator that delegates to specialised sub-agents based on
    the action type and FSM state, then sanitises all outputs through the
    QuarantineAgent security boundary.
    """

    def __init__(
        self,
        fsm: FSMOrchestrator,
        db: MockDatabase,
        mesh: MockServiceMesh,
        telemetry: MockTelemetry,
    ) -> None:
        self.fsm = fsm
        self._data_agent = DataAgent(telemetry=telemetry, db=db)
        self._code_agent = CodeAgent(db=db, mesh=mesh)
        self._quarantine = QuarantineAgent()
        self._mesh = mesh
        self._db = db

    def reset(self) -> None:
        self.fsm.reset()
        self._quarantine.reset()

    def dispatch(self, action: dict) -> tuple[str, str, int, bool]:
        """
        Route the action to the correct sub-agent, run FSM transition,
        and sanitise output through QuarantineAgent.

        Returns:
            (stdout, stderr, exit_code, fsm_milestone_hit)
        """
        action_type = action.get("action_type", "")

        # 1. Advance FSM
        _, milestone = self.fsm.process_action(action_type)

        # 2. Dispatch to sub-agent
        raw_stdout, raw_stderr, exit_code = self._execute(action, action_type)

        # 3. Security boundary — sanitise before returning
        stdout, stderr = self._quarantine.sanitize_observation(raw_stdout, raw_stderr)

        return stdout, stderr, exit_code, milestone

    def execute_remediation(self, operation_type: str, target_service: str) -> tuple[str, str, int]:
        """
        Execute a RemediationAction directly against mock infrastructure.
        Returns (stdout, stderr, exit_code).
        """
        stdout, stderr, exit_code = "", "", 0

        if operation_type == "restart":
            ok, msg = self._mesh.restart_service(target_service)
            if not ok:
                # Might be a DB-level target
                ok, msg = self._handle_db_remediation(target_service)
            stdout, exit_code = msg, (0 if ok else 1)

        elif operation_type == "rollback":
            ok, msg = self._mesh.rollback_service(target_service)
            if not ok:
                ok, msg = self._handle_db_remediation(target_service)
            stdout, exit_code = msg, (0 if ok else 1)

        elif operation_type == "scale_up":
            if target_service == "connection-pool":
                stdout = self._db.scale_connection_pool(new_max=50)
            else:
                ok, msg = self._mesh.scale_up_service(target_service)
                stdout, exit_code = msg, (0 if ok else 1)

        else:
            stderr = f"Unknown operation_type '{operation_type}'"
            exit_code = 1

        stdout, stderr = self._quarantine.sanitize_observation(stdout, stderr)
        return stdout, stderr, exit_code

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _execute(self, action: dict, action_type: str) -> tuple[str, str, int]:
        if action_type == "diagnostic_query":
            return self._data_agent.execute(
                metric_identifier=action.get("metric_identifier", ""),
                time_window=action.get("time_window", "5m"),
            )
        elif action_type == "log_inspection":
            return self._code_agent.execute(
                tail_lines=action.get("tail_lines", 50),
                grep_pattern=action.get("grep_pattern", ""),
            )
        elif action_type == "remediation":
            return self.execute_remediation(
                operation_type=action.get("operation_type", ""),
                target_service=action.get("target_service", ""),
            )
        return "", f"Unknown action_type '{action_type}'", 1

    def _handle_db_remediation(self, target: str) -> tuple[bool, str]:
        """Handle targets like 'db:pid:4821'."""
        if target.startswith("db:pid:"):
            try:
                pid = int(target.split(":")[-1])
                return self._db.kill_pid(pid)
            except ValueError:
                return False, f"Invalid PID format: '{target}'"
        return False, f"Unknown target '{target}'"
