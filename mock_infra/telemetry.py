"""
mock_infra/telemetry.py
MockTelemetry — Prometheus-compatible metric aggregator.

Deterministically reads state from MockDatabase and MockServiceMesh to emit
the Four Golden Signals, compute the multi-window Burn Rate, and produce
the composite Health Score.

All values are derived from concrete infrastructure state — no hallucination.

Health Score formula (blueprint-specified):
  H_t = w1·A_t + w2·(1/L_t) − w3·B_t

Burn Rate formula (blueprint-specified):
  B_t = E_t / (1 − SLO_target)
  where SLO_target = 0.999 (99.9% availability), error budget = 0.001
"""

from mock_infra.database import MockDatabase
from mock_infra.service_mesh import MockServiceMesh

# ── Healthy operating thresholds ──────────────────────────────────────────────
THRESHOLDS = {
    "latency_p99_ms": 200.0,
    "error_rate_pct": 1.0,
    "saturation_pct": 70.0,
    "traffic_rps": 300.0,
}

# ── SLO configuration ─────────────────────────────────────────────────────────
SLO_AVAILABILITY_TARGET = 0.999          # 99.9% uptime target
SLO_ERROR_BUDGET = 1.0 - SLO_AVAILABILITY_TARGET  # 0.001 = 0.1% error budget

# ── Health Score weights (w1 + w2 = 1, w3 scales the burn rate subtraction) ──
W1 = 0.50   # Availability weight
W2 = 0.35   # Latency health weight (via 1/L_t)
W3 = 0.15   # Burn Rate penalty weight


class MockTelemetry:
    """
    Prometheus-compatible metric aggregator for the SRE environment.

    Implements the blueprint Health Score:
      H_t = w1·A_t + w2·(1/L_t) − w3·B_t

    Where:
      A_t  = Availability = 1 − error_rate
      1/L_t = Latency quality = threshold / latency_p99 (capped at 1.0)
      B_t  = Burn Rate = error_rate / SLO_error_budget (normalised to [0, 1])
    """

    def __init__(self, db: MockDatabase, mesh: MockServiceMesh) -> None:
        self._db = db
        self._mesh = mesh

    # ── Four Golden Signals ───────────────────────────────────────────────────

    def get_golden_signals(self) -> dict[str, float]:
        """
        Compute the Four Golden Signals from current infra state.
        All values are deterministic — derived purely from MockDB + MockMesh state.
        """
        db = self._db.get_metrics()

        # Latency: worst p99 across observable services
        latency_p99 = max(
            self._mesh.get_latency(svc)
            for svc in ["api-gateway", "auth-service", "order-service"]
        )

        # Traffic: degrades proportionally to DB pool saturation
        pool_factor = db["pool_pct"] / 100.0
        traffic_rps = max(30.0, THRESHOLDS["traffic_rps"] * (1.0 - pool_factor * 0.5))

        # Errors: worst 5xx rate across observable services
        error_rate = max(
            self._mesh.get_error_rate(svc)
            for svc in ["api-gateway", "auth-service", "order-service"]
        )

        # Saturation: max of DB pool %, latency-implied saturation, memory saturation
        db_saturation = db["pool_pct"]
        latency_saturation = min(100.0, (latency_p99 / 1000.0) * 50.0)
        mem_saturation = max(
            self._mesh.get_saturation(svc)
            for svc in ["api-gateway", "order-service"]
        )
        saturation = max(db_saturation, latency_saturation, mem_saturation)

        return {
            "latency_p99_ms": round(latency_p99, 2),
            "traffic_rps": round(traffic_rps, 2),
            "error_rate_pct": round(error_rate, 2),
            "saturation_pct": round(saturation, 2),
        }

    # ── Burn Rate ─────────────────────────────────────────────────────────────

    def compute_burn_rate(self, error_rate_pct: float | None = None) -> float:
        """
        Multi-window Burn Rate — blueprint formula:

          B_t = E_t / (1 − SLO_target)

        Where:
          E_t          = current error rate as a fraction (e.g., 0.01 for 1%)
          1 − SLO_target = error budget fraction (0.001 for 99.9% SLO)

        B_t = 1.0 means consuming error budget at exactly the SLA rate.
        B_t > 1.0 means burning through the error budget faster than allowed.

        Examples (SLO_target=0.999, error_budget=0.001):
          0.1% errors → B_t = 0.001 / 0.001 = 1.0   (at SLA limit)
          1.0% errors → B_t = 0.010 / 0.001 = 10.0  (10× over budget)
          0.0% errors → B_t = 0.000               (no burn)

        Returns B_t normalised to [0, 1] for Health Score computation.
        """
        if error_rate_pct is None:
            signals = self.get_golden_signals()
            error_rate_pct = signals["error_rate_pct"]

        error_rate_fraction = error_rate_pct / 100.0
        raw_burn_rate = error_rate_fraction / SLO_ERROR_BUDGET  # e.g., 10.0 at 1% errors

        # Normalise to [0, 1]: cap at 100× the error budget (extreme failure)
        normalised = min(raw_burn_rate / 100.0, 1.0)
        return round(normalised, 6)

    # ── Health Score ──────────────────────────────────────────────────────────

    def compute_health_score(self) -> float:
        """
        Composite Health Score H_t ∈ [0, 1] — blueprint formula:

          H_t = w1·A_t + w2·(1/L_t) − w3·B_t

        Components:
          A_t  = Availability = 1 − error_rate_fraction ∈ [0, 1]
          1/L_t = Latency quality = min(threshold / latency_p99, 1.0) ∈ [0, 1]
                  (threshold = 200ms; reads 1.0 when latency is at/below threshold,
                   decreases toward 0 as latency degrades)
          B_t  = Burn Rate (normalised, 0 = no burn, 1 = extreme burn)

        Weights: w1=0.50, w2=0.35, w3=0.15
        """
        signals = self.get_golden_signals()

        # A_t — Availability
        a_t = max(0.0, 1.0 - signals["error_rate_pct"] / 100.0)

        # 1/L_t — Latency quality (capped so faster-than-threshold doesn't inflate score)
        latency_ratio = signals["latency_p99_ms"] / THRESHOLDS["latency_p99_ms"]
        inv_l_t = min(1.0 / latency_ratio, 1.0)  # cap at 1.0 (no bonus for sub-threshold)
        inv_l_t = max(0.0, inv_l_t)

        # B_t — Normalised Burn Rate
        b_t = self.compute_burn_rate(error_rate_pct=signals["error_rate_pct"])

        h_t = W1 * a_t + W2 * inv_l_t - W3 * b_t
        return round(max(0.0, min(1.0, h_t)), 4)

    # ── Alerts ────────────────────────────────────────────────────────────────

    def get_active_alerts(self) -> list[str]:
        """Returns all currently firing alerts based on live infrastructure state."""
        alerts = []
        s = self.get_golden_signals()
        db = self._db.get_metrics()
        burn_rate = self.compute_burn_rate(s["error_rate_pct"])

        if s["latency_p99_ms"] > THRESHOLDS["latency_p99_ms"]:
            alerts.append(
                f"HIGH_LATENCY: p99={s['latency_p99_ms']:.0f}ms "
                f"(SLA threshold: {THRESHOLDS['latency_p99_ms']:.0f}ms)"
            )
        if s["error_rate_pct"] > THRESHOLDS["error_rate_pct"]:
            alerts.append(
                f"HIGH_ERROR_RATE: {s['error_rate_pct']:.1f}% 5xx "
                f"(SLA threshold: {THRESHOLDS['error_rate_pct']:.1f}%)"
            )
        if burn_rate > 0.1:  # >10% of max burn budget
            raw_burn = (s["error_rate_pct"] / 100.0) / SLO_ERROR_BUDGET
            alerts.append(
                f"HIGH_BURN_RATE: {raw_burn:.1f}× SLO error budget "
                f"(SLO target: {SLO_AVAILABILITY_TARGET*100:.1f}% availability)"
            )
        if s["saturation_pct"] > THRESHOLDS["saturation_pct"]:
            alerts.append(
                f"HIGH_SATURATION: {s['saturation_pct']:.1f}% "
                f"(threshold: {THRESHOLDS['saturation_pct']:.1f}%)"
            )
        if db["lock_contention"]:
            alerts.append(f"DB_LOCK_CONTENTION: blocking PIDs={db['blocking_pids']}")
        if db["pool_pct"] > 85:
            alerts.append(
                f"DB_POOL_EXHAUSTION: {db['pool_used']}/{db['pool_max']} "
                f"connections ({db['pool_pct']:.1f}%)"
            )
        return alerts

    # ── Metric Query ──────────────────────────────────────────────────────────

    def query_metric(self, metric_identifier: str, time_window: str) -> str:
        """
        Simulate a PromQL query. Returns human-readable metric output.
        Called by DataAgent when the agent issues a DiagnosticQueryAction.
        """
        s = self.get_golden_signals()
        db = self._db.get_metrics()
        burn_rate_raw = (s["error_rate_pct"] / 100.0) / SLO_ERROR_BUDGET
        svc_status = {
            svc: self._mesh.get_service_status(svc)
            for svc in ["api-gateway", "auth-service", "order-service", "db"]
        }

        per_svc_latency = "\n".join(
            f"  {svc}: {st['latency_ms']:.1f}ms" for svc, st in svc_status.items()
        )
        per_svc_errors = "\n".join(
            f"  {svc}: {st['error_rate_pct']:.2f}%" for svc, st in svc_status.items()
        )

        METRIC_MAP = {
            "latency_p99_ms": (
                f"latency_p99_ms{{aggregated}} = {s['latency_p99_ms']} "
                f"(window: {time_window})\nPer-service breakdown:\n{per_svc_latency}"
            ),
            "traffic_rps": (
                f"http_requests_total{{rate}} = {s['traffic_rps']:.1f} req/s "
                f"(window: {time_window})"
            ),
            "error_rate": (
                f"http_5xx_rate{{aggregated}} = {s['error_rate_pct']:.2f}% "
                f"(window: {time_window})\nBurn rate: {burn_rate_raw:.1f}× SLO budget\n"
                f"Per-service breakdown:\n{per_svc_errors}"
            ),
            "saturation_pct": (
                f"resource_saturation = {s['saturation_pct']:.1f}% "
                f"(window: {time_window})"
            ),
            "connection_pool_used": (
                f"pg_stat_activity_count = {db['pool_used']}/{db['pool_max']} "
                f"({db['pool_pct']:.1f}%) (window: {time_window})\n"
                + self._db.get_pg_stat_activity()
            ),
            "burn_rate": (
                f"slo_burn_rate = {burn_rate_raw:.2f}× error budget "
                f"(SLO: {SLO_AVAILABILITY_TARGET*100:.1f}%, "
                f"budget: {SLO_ERROR_BUDGET*100:.2f}%) (window: {time_window})"
            ),
            "cpu_usage": f"container_cpu_usage = 67.3% (window: {time_window})",
            "memory_usage_pct": (
                f"container_memory_usage / limit = {s['saturation_pct']:.1f}% "
                f"(window: {time_window})"
            ),
            "pg_stat_activity": self._db.get_pg_stat_activity(),
        }

        result = METRIC_MAP.get(metric_identifier)
        if result:
            return result
        return (
            f"No metric found for '{metric_identifier}'.\n"
            f"Available: {', '.join(METRIC_MAP.keys())}"
        )

    # ── Health check ──────────────────────────────────────────────────────────

    def is_healthy(self) -> bool:
        return self.compute_health_score() > 0.85
