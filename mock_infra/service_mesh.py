"""
mock_infra/service_mesh.py
MockServiceMesh — simulates a 4-service network topology with
controlled fault injection (HTTP 500, latency spikes, timeouts).

Topology:
  api-gateway → auth-service → db
              → order-service → db
"""

import random
from dataclasses import dataclass
from typing import Optional

SERVICES = ["api-gateway", "auth-service", "order-service", "db"]

SERVICE_DEPS: dict[str, list[str]] = {
    "api-gateway": ["auth-service", "order-service"],
    "auth-service": ["db"],
    "order-service": ["db"],
    "db": [],
}

FAULT_TYPES = {"latency", "http_500", "connection_timeout", "oom_killed"}


@dataclass
class ServiceFault:
    service: str
    fault_type: str
    value: float  # ms for latency, pct for error rate


class MockServiceMesh:
    """
    Simulates an HTTP service mesh with deterministic fault injection.

    Uses a Dual-RNG architecture:
      - primary_rng  → fault values (deterministic per episode seed)
      - network_rng  → jitter overlay (reproducible but independent)
    """

    def __init__(self, primary_rng: random.Random, network_rng: random.Random) -> None:
        self._rng = primary_rng
        self._net_rng = network_rng
        self.faults: dict[str, ServiceFault] = {}
        self._base_latency: dict[str, float] = {
            svc: self._rng.uniform(10, 40) for svc in SERVICES
        }
        self._saturation: dict[str, float] = {svc: 30.0 for svc in SERVICES}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self.faults = {}
        self._base_latency = {svc: self._rng.uniform(10, 40) for svc in SERVICES}
        self._saturation = {svc: 30.0 for svc in SERVICES}

    def inject_fault(self, service: str, fault_type: str, value: float = 0.0) -> None:
        if service not in SERVICES:
            raise ValueError(f"Unknown service '{service}'. Valid: {SERVICES}")
        if fault_type not in FAULT_TYPES:
            raise ValueError(f"Unknown fault_type '{fault_type}'. Valid: {FAULT_TYPES}")
        self.faults[service] = ServiceFault(
            service=service, fault_type=fault_type, value=value
        )
        if fault_type == "oom_killed":
            self._saturation[service] = 98.0
        if fault_type == "latency":
            # Propagate upstream: dependents also degrade slightly
            for dep_svc, deps in SERVICE_DEPS.items():
                if service in deps and dep_svc not in self.faults:
                    self._base_latency[dep_svc] += value * 0.3

    # ------------------------------------------------------------------
    # Metric accessors (read by MockTelemetry)
    # ------------------------------------------------------------------

    def get_latency(self, service: str) -> float:
        """Current p99 latency including fault and network jitter."""
        base = self._base_latency.get(service, 20.0)
        jitter = self._net_rng.uniform(5, 50)
        fault = self.faults.get(service)
        if fault and fault.fault_type == "latency":
            return base + fault.value + jitter
        return base + jitter

    def get_error_rate(self, service: str) -> float:
        """Current HTTP 5xx error rate (%)."""
        fault = self.faults.get(service)
        if fault and fault.fault_type == "http_500":
            return fault.value
        if fault and fault.fault_type == "oom_killed":
            return self._rng.uniform(5, 20)
        return self._rng.uniform(0.0, 0.8)

    def get_saturation(self, service: str) -> float:
        """Memory/CPU saturation % for a service."""
        return self._saturation.get(service, 30.0)

    # ------------------------------------------------------------------
    # Agent-callable operations
    # ------------------------------------------------------------------

    def restart_service(self, service: str) -> tuple[bool, str]:
        if service not in SERVICES:
            return False, f"Error: unknown service '{service}'"
        self.faults.pop(service, None)
        self._saturation[service] = 30.0
        return True, (
            f"kubectl rollout restart deployment/{service}\n"
            f"Waiting for deployment '{service}' rollout to finish…\n"
            f"deployment.apps/{service} successfully rolled out"
        )

    def rollback_service(self, service: str) -> tuple[bool, str]:
        if service not in SERVICES:
            return False, f"Error: unknown service '{service}'"
        self.faults.pop(service, None)
        return True, (
            f"kubectl rollout undo deployment/{service}\n"
            f"deployment.apps/{service} rolled back to previous revision"
        )

    def scale_up_service(self, service: str) -> tuple[bool, str]:
        if service not in SERVICES:
            return False, f"Error: unknown service '{service}'"
        return True, (
            f"kubectl scale deployment/{service} --replicas=3\n"
            f"deployment.apps/{service} scaled"
        )

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def get_service_status(self, service: str) -> dict:
        return {
            "service": service,
            "latency_ms": round(self.get_latency(service), 2),
            "error_rate_pct": round(self.get_error_rate(service), 2),
            "saturation_pct": round(self.get_saturation(service), 1),
            "dependencies": SERVICE_DEPS.get(service, []),
            "fault": (
                {"type": f.fault_type, "value": f.value}
                if (f := self.faults.get(service))
                else None
            ),
        }

    def get_topology(self) -> dict:
        return {
            "services": SERVICES,
            "dependencies": SERVICE_DEPS,
            "faults": {s: {"type": f.fault_type, "value": f.value} for s, f in self.faults.items()},
        }

    def is_healthy(self) -> bool:
        return len(self.faults) == 0
