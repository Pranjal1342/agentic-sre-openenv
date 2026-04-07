"""
mock_infra/database.py
MockDatabase — simulates PostgreSQL connection pool exhaustion,
transaction lock contention, and PID-level query termination.
Used by Task 3 (cascading failure) as the ground-truth state store.
"""

import random
from dataclasses import dataclass
from typing import Optional


@dataclass
class ActiveLock:
    pid: int
    query: str
    wait_ms: int
    is_blocking: bool = True


class MockDatabase:
    """
    Lightweight in-memory simulation of a PostgreSQL instance.

    Tracks:
      - Connection pool (used / max)
      - Active transaction locks (pid → ActiveLock)
      - Lock contention flag

    The grader for Task 3 inspects this object's state directly to
    award partial credit — no hallucination, pure deterministic state.
    """

    DEFAULT_POOL_MAX = 20

    def __init__(self, rng: random.Random) -> None:
        self._rng = rng
        self.pool_used: int = 0
        self.pool_max: int = self.DEFAULT_POOL_MAX
        self.active_locks: dict[int, ActiveLock] = {}
        self.lock_contention: bool = False
        self.fault_type: Optional[str] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self.pool_used = 0
        self.pool_max = self.DEFAULT_POOL_MAX
        self.active_locks = {}
        self.lock_contention = False
        self.fault_type = None

    def inject_fault(self, fault_type: str, **kwargs) -> None:
        """
        Inject a specific fault scenario.

        fault_type='connection_pool_exhaustion':
            Partially exhausts the connection pool and injects a deadlock.
        """
        self.fault_type = fault_type
        if fault_type == "connection_pool_exhaustion":
            self.pool_used = kwargs.get("used", 18)
            pid = kwargs.get("blocking_pid", 4821)
            query = kwargs.get(
                "query",
                "UPDATE orders SET status='processing' WHERE id IN "
                "(SELECT id FROM orders FOR UPDATE SKIP LOCKED)",
            )
            self.active_locks[pid] = ActiveLock(
                pid=pid,
                query=query,
                wait_ms=kwargs.get("wait_ms", 45000),
                is_blocking=True,
            )
            self.lock_contention = True

    # ------------------------------------------------------------------
    # Agent-callable operations
    # ------------------------------------------------------------------

    def kill_pid(self, pid: int) -> tuple[bool, str]:
        """
        Terminate a blocking query by PID.
        Simulates: SELECT pg_terminate_backend(pid).
        Returns (success, stdout_message).
        """
        if pid in self.active_locks:
            lock = self.active_locks.pop(pid)
            # Terminating the lock frees several connections
            freed = self._rng.randint(3, 6)
            self.pool_used = max(0, self.pool_used - freed)
            self.lock_contention = len(self.active_locks) > 0
            return True, (
                f"SELECT pg_terminate_backend({pid});\n"
                f" pg_terminate_backend\n"
                f"----------------------\n"
                f" t\n"
                f"(1 row)\n\n"
                f"Terminated query (held for {lock.wait_ms}ms):\n"
                f"  {lock.query[:80]}…\n"
                f"Connection pool freed: {freed} connections released."
            )
        return False, (
            f"SELECT pg_terminate_backend({pid});\n"
            f" pg_terminate_backend\n"
            f"----------------------\n"
            f" f\n"
            f"(1 row)\n"
            f"ERROR: PID {pid} not found in pg_stat_activity."
        )

    def scale_connection_pool(self, new_max: int) -> str:
        """
        Update max_connections. Simulates ALTER SYSTEM + pg_reload_conf().
        """
        new_max = min(max(new_max, self.pool_used + 2), 200)
        old = self.pool_max
        self.pool_max = new_max
        return (
            f"ALTER SYSTEM SET max_connections = {new_max};\n"
            f"SELECT pg_reload_conf();\n"
            f" pg_reload_conf\n"
            f"----------------\n"
            f" t\n"
            f"(1 row)\n\n"
            f"Connection pool updated: {old} → {new_max} max connections."
        )

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def get_metrics(self) -> dict:
        return {
            "pool_used": self.pool_used,
            "pool_max": self.pool_max,
            "pool_pct": round(self.pool_used / self.pool_max * 100, 1),
            "active_locks": len(self.active_locks),
            "lock_contention": self.lock_contention,
            "blocking_pids": [p for p, l in self.active_locks.items() if l.is_blocking],
        }

    def get_pg_stat_activity(self) -> str:
        """Returns a simulated pg_stat_activity result table."""
        if not self.active_locks:
            return (
                "SELECT pid, state, wait_event_type, left(query,60) FROM pg_stat_activity "
                "WHERE wait_event_type='Lock';\n pid | state | wait_event_type | query\n"
                "-----+-------+-----------------+-------\n(0 rows)"
            )
        header = (
            "SELECT pid, state, wait_event_type, left(query,60) FROM pg_stat_activity "
            "WHERE wait_event_type='Lock';\n"
            " pid  | state  | wait_event_type | wait_ms | query\n"
            "------+--------+-----------------+---------+-------"
        )
        rows = [
            f" {pid}  | active | Lock            | {lk.wait_ms:<7} | {lk.query[:55]}…"
            for pid, lk in self.active_locks.items()
        ]
        return header + "\n" + "\n".join(rows) + f"\n({len(self.active_locks)} row(s))"

    def is_healthy(self) -> bool:
        return (
            not self.lock_contention
            and self.pool_used < int(self.pool_max * 0.8)
        )
