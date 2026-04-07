"""
agents/data_agent.py
Data-Agent — handles DiagnosticQueryAction by querying MockTelemetry.
Operates during INVESTIGATION_STATE, surfacing metric and DB data.
"""

from mock_infra.telemetry import MockTelemetry
from mock_infra.database import MockDatabase


class DataAgent:
    """
    Specialised sub-agent for telemetry and tracing queries.
    Delegates to MockTelemetry.query_metric() and MockDatabase helpers.
    """

    def __init__(self, telemetry: MockTelemetry, db: MockDatabase) -> None:
        self._telemetry = telemetry
        self._db = db

    def execute(self, metric_identifier: str, time_window: str) -> tuple[str, str, int]:
        """
        Execute a diagnostic query.

        Returns:
            (stdout, stderr, exit_code)
        """
        try:
            # Special handler: pg_stat_activity surfaces DB lock details
            if metric_identifier == "pg_stat_activity":
                return self._db.get_pg_stat_activity(), "", 0

            result = self._telemetry.query_metric(metric_identifier, time_window)
            return result, "", 0

        except Exception as exc:
            return "", f"DataAgent error: {exc}", 1
