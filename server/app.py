"""
server/app.py
OpenEnv 0.1-compliant FastAPI server — fully wired to all subsystems.

Endpoints:
  POST /reset  → Initialise episode, inject fault, return ObservationModel
  POST /step   → Execute AgentAction, compute reward, return StepResult
  GET  /state  → Query episodic metadata (no step advance)
  WS   /ws     → Persistent WebSocket for TRL/VeRL training loops
  GET  /health → Docker HEALTHCHECK liveness probe
"""

import json
import random
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional

import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, status
from fastapi.responses import JSONResponse

from server.models import (
    ResetRequest, StepRequest, StepResult,
    ObservationModel, RewardModel, RewardBreakdown,
)
from server.fsm import FSMOrchestrator
from server.pipeline import CICDCMCCPipeline
from mock_infra.database import MockDatabase
from mock_infra.service_mesh import MockServiceMesh
from mock_infra.telemetry import MockTelemetry
from agents.sre_agent import SREAgent
from graders.grader import EpisodeGrader, compute_timeout_reward
from server.models import RewardBreakdown
import tasks.task_1 as task_1_mod
import tasks.task_2 as task_2_mod
import tasks.task_3 as task_3_mod
from tasks.task_3 import StageTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def _load_config() -> dict:
    try:
        with open("openenv.yaml") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {"max_steps": {"task_1": 15, "task_2": 25, "task_3": 40},
                "step_timeout_seconds": 30, "episode_seed": 42, "network_shim_seed": 99}

CONFIG = _load_config()
MAX_STEPS: dict = CONFIG.get("max_steps", {"task_1": 15, "task_2": 25, "task_3": 40})

# ---------------------------------------------------------------------------
# Episode State
# ---------------------------------------------------------------------------
class EpisodeState:
    def __init__(self) -> None:
        self.active: bool = False
        self.task_id: Optional[str] = None
        self.seed: int = 42
        self.step_count: int = 0
        self.max_steps: int = 15
        self.fsm: FSMOrchestrator = FSMOrchestrator()
        self.db: Optional[MockDatabase] = None
        self.mesh: Optional[MockServiceMesh] = None
        self.telemetry: Optional[MockTelemetry] = None
        self.agent: Optional[SREAgent] = None
        self.grader: Optional[EpisodeGrader] = None
        self.stage_tracker: Optional[StageTracker] = None
        self.pipeline: Optional[CICDCMCCPipeline] = None
        self.history: list[dict] = []
        self.unique_action_types: set[str] = set()
        self.rolling_summary: str = ""
        self.prev_health: float = 0.0

    def reset(self, task_id: str, seed: int) -> None:
        self.active = True
        self.task_id = task_id
        self.seed = seed
        self.step_count = 0
        self.max_steps = MAX_STEPS.get(task_id, 15)
        self.history = []
        self.unique_action_types = set()
        self.rolling_summary = ""

        primary_rng = random.Random(seed)
        network_seed = CONFIG.get("network_shim_seed", 99)
        network_rng = random.Random(network_seed)

        self.db = MockDatabase(rng=primary_rng)
        self.mesh = MockServiceMesh(primary_rng=primary_rng, network_rng=network_rng)
        self.telemetry = MockTelemetry(db=self.db, mesh=self.mesh)
        self.fsm = FSMOrchestrator()
        self.agent = SREAgent(fsm=self.fsm, db=self.db, mesh=self.mesh, telemetry=self.telemetry)
        self.grader = EpisodeGrader(telemetry=self.telemetry, task_id=task_id, max_steps=self.max_steps)
        self.stage_tracker = StageTracker() if task_id == "task_3" else None
        self.pipeline = CICDCMCCPipeline(telemetry=self.telemetry)

        # Inject task-specific fault — this IS the CI/CD event (bad deploy)
        fault_desc = "unknown fault"
        if task_id == "task_1":
            task_1_mod.setup(self.db, self.mesh, primary_rng)
            fault_desc = "latency fault injected on order-service (simulated bad deploy)"
        elif task_id == "task_2":
            task_2_mod.setup(self.db, self.mesh, primary_rng)
            fault_desc = "OOMKilled crash loop on order-service (simulated bad deploy)"
        elif task_id == "task_3":
            task_3_mod.setup(self.db, self.mesh, primary_rng)
            fault_desc = "DB pool exhaustion + lock contention (simulated bad deploy)"

        # CI → CD phases: record the deployment event
        self.pipeline.on_reset(task_id=task_id, seed=seed, fault_description=fault_desc)

        self.grader.reset()
        self.prev_health = self.telemetry.compute_health_score()

    def get_initial_alerts(self) -> list[str]:
        if self.task_id == "task_1":
            return task_1_mod.get_initial_alerts()
        elif self.task_id == "task_2":
            return task_2_mod.get_initial_alerts()
        elif self.task_id == "task_3":
            return task_3_mod.get_initial_alerts()
        return []

    def to_info_dict(self) -> dict:
        base = {
            "task_id": self.task_id,
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "fsm_state": self.fsm.state_name,
            "seed": self.seed,
        }
        if self.pipeline:
            base["cicdcmcc"] = self.pipeline.get_pipeline_summary()
        return base

    def _update_rolling_summary(self, action: dict, stdout: str, step: int) -> None:
        """Compress history every 5 steps to prevent context window saturation."""
        if step % 5 != 0:
            return
        recent = self.history[-5:]
        action_types = [h.get("action_type", "?") for h in recent]
        self.rolling_summary = (
            f"[Summary at step {step}] FSM: {self.fsm.state_name} | "
            f"Recent actions: {action_types} | "
            f"Health: {self.telemetry.compute_health_score():.3f}"
        )


EPISODE = EpisodeState()


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(application: FastAPI):
    logger.info("Agentic SRE Environment starting — config: max_steps=%s", MAX_STEPS)
    # Pre-load RAG engine (non-blocking — fails gracefully if index not built)
    try:
        from rag.engine import get_engine
        engine = get_engine()
        if engine.is_loaded:
            logger.info("RAG engine loaded successfully.")
        else:
            logger.warning("RAG index not found — run `python rag/offline_index.py` to build it.")
    except Exception as e:
        logger.warning("RAG engine unavailable: %s", e)
    yield
    logger.info("Agentic SRE Environment shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Agentic SRE Environment",
    description=(
        "OpenEnv 0.1-compliant RL environment for Agentic SRE incident remediation. "
        "Implements reset(), step(), and state() with persistent WebSocket support."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Health probe
# ---------------------------------------------------------------------------
@app.get("/health", tags=["Infrastructure"])
async def health_check():
    return {
        "status": "healthy",
        "episode_active": EPISODE.active,
        "task_id": EPISODE.task_id,
        "spec_version": CONFIG.get("spec_version", "1"),
    }


# ---------------------------------------------------------------------------
# POST /reset
# ---------------------------------------------------------------------------
@app.post("/reset", response_model=ObservationModel, tags=["OpenEnv"])
async def reset(request: ResetRequest) -> ObservationModel:
    """
    Initialise a pristine episode: reset infra, inject fault, return initial observation.
    Corresponds to OpenEnv reset().
    """
    logger.info("reset() task_id=%s seed=%d", request.task_id, request.seed)
    EPISODE.reset(task_id=request.task_id, seed=request.seed)

    signals = EPISODE.telemetry.get_golden_signals()
    alerts = EPISODE.get_initial_alerts()

    return ObservationModel(
        command_stdout=f"Episode initialised. Task: {request.task_id} | Seed: {request.seed} | "
                       f"Health: {EPISODE.prev_health:.3f}",
        command_stderr="",
        exit_code=0,
        active_alerts=alerts,
        golden_signals=signals,
        rolling_summary="",
    )


# ---------------------------------------------------------------------------
# POST /step
# ---------------------------------------------------------------------------
@app.post("/step", response_model=StepResult, tags=["OpenEnv"])
async def step(request: StepRequest) -> StepResult:
    """
    Execute the agent's typed action, compute reward, advance simulation.
    Enforces max_steps: returns done=True, reward=-1.0 on timeout.
    Corresponds to OpenEnv step(action).
    """
    if not EPISODE.active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No active episode. Call POST /reset first.",
        )

    # Max-steps enforcement
    if EPISODE.step_count >= EPISODE.max_steps:
        EPISODE.active = False
        logger.warning("max_steps=%d exceeded — terminating.", EPISODE.max_steps)
        return StepResult(
            observation=ObservationModel(
                command_stderr="Episode terminated: max_steps exceeded.",
                exit_code=1,
                active_alerts=["TIMEOUT: episode limit reached"],
                golden_signals=EPISODE.telemetry.get_golden_signals(),
                rolling_summary=EPISODE.rolling_summary,
            ),
            reward=compute_timeout_reward(),
            done=True,
            info={"termination": "max_steps_exceeded", **EPISODE.to_info_dict()},
        )

    EPISODE.step_count += 1
    action_dict = request.action.model_dump()
    action_type = action_dict.get("action_type", "")
    logger.info("step %d/%d action=%s", EPISODE.step_count, EPISODE.max_steps, action_dict)

    EPISODE.unique_action_types.add(action_type)
    EPISODE.history.append(action_dict)

    # ── Execute via SREAgent ──────────────────────────────────────────────
    stdout, stderr, exit_code, milestone_hit = EPISODE.agent.dispatch(action_dict)

    # ── CM: Continuous Monitoring tick ────────────────────────────────────
    EPISODE.pipeline.on_cm_tick(step=EPISODE.step_count)

    # ── CC: Record this correction attempt ───────────────────────────────
    EPISODE.pipeline.on_cc_action(step=EPISODE.step_count, action=action_dict, stdout=stdout)

    # ── Task-specific resolution check ───────────────────────────────────
    resolved = False
    behavioral_penalty = 0.0
    milestone_value = 1.0

    if EPISODE.task_id == "task_1":
        resolved, _ = task_1_mod.check_resolution(action_dict, stdout)
        behavioral_penalty = task_1_mod.wrong_service_penalty(action_dict)

    elif EPISODE.task_id == "task_2":
        resolved, _ = task_2_mod.check_resolution(action_dict, EPISODE.mesh)
        behavioral_penalty = task_2_mod.wrong_target_penalty(action_dict)

    elif EPISODE.task_id == "task_3" and EPISODE.stage_tracker:
        newly_done = EPISODE.stage_tracker.update(action_dict, stdout, EPISODE.db)
        if newly_done:
            milestone_hit = True
            milestone_value = EPISODE.stage_tracker.partial_score
            logger.info("Task 3 stages: %s", newly_done)
        resolved = EPISODE.stage_tracker.fully_resolved

    # ── CM: Verify after CC action (pipeline loop closure check) ─────────
    pipeline_resolved, post_health = EPISODE.pipeline.on_verify(step=EPISODE.step_count)
    resolved = resolved or pipeline_resolved

    # ── FSM: advance to VERIFICATION if resolved ─────────────────────────
    if resolved:
        EPISODE.fsm.advance_to_verification()
        EPISODE.active = False

    # ── Compute step reward ───────────────────────────────────────────────
    reward = EPISODE.grader.step(
        action=action_dict,
        milestone_hit=milestone_hit,
        behavioral_penalty=behavioral_penalty,
        milestone_value=milestone_value,
        step_count=EPISODE.step_count,
    )

    # ── Update rolling summary (includes pipeline phase) ─────────────────
    EPISODE._update_rolling_summary(action_dict, stdout, EPISODE.step_count)

    done = resolved or (EPISODE.step_count >= EPISODE.max_steps)
    info = EPISODE.to_info_dict()
    if done:
        info["termination"] = "resolved" if resolved else "max_steps_exceeded"
        # Apply terminal bonus/penalty so resolved agents always outscore looping ones
        terminal = EPISODE.grader.terminal_bonus(resolved=resolved)
        reward = RewardModel(
            value=round(reward.value + terminal.value, 6),
            breakdown=RewardBreakdown(
                health_delta=reward.breakdown.health_delta,
                milestone_bonus=round(reward.breakdown.milestone_bonus + terminal.breakdown.milestone_bonus, 6),
                action_efficiency=reward.breakdown.action_efficiency,
                behavioral_penalty=round(reward.breakdown.behavioral_penalty + terminal.breakdown.behavioral_penalty, 6),
                time_step_penalty=reward.breakdown.time_step_penalty,
            ),
        )

    signals = EPISODE.telemetry.get_golden_signals()
    alerts = EPISODE.telemetry.get_active_alerts()

    return StepResult(
        observation=ObservationModel(
            command_stdout=stdout,
            command_stderr=stderr,
            exit_code=exit_code,
            active_alerts=alerts,
            golden_signals=signals,
            rolling_summary=EPISODE.rolling_summary,
        ),
        reward=reward,
        done=done,
        info=info,
    )


# ---------------------------------------------------------------------------
# GET /state
# ---------------------------------------------------------------------------
@app.get("/state", tags=["OpenEnv"])
async def state() -> JSONResponse:
    """
    Return episodic metadata without advancing the step counter.
    Includes full CI/CD/CM/CC pipeline summary.
    """
    if not EPISODE.active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No active episode. Call POST /reset first.",
        )
    return JSONResponse(content=EPISODE.to_info_dict())


# ---------------------------------------------------------------------------
# WS /ws — Persistent WebSocket (TRL / VeRL compatible)
# ---------------------------------------------------------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Persistent WebSocket wrapping reset/step/state.
    Message: {"method": "reset"|"step"|"state", "payload": {...}}
    """
    await websocket.accept()
    logger.info("WebSocket connected: %s", websocket.client)
    timeout = CONFIG.get("step_timeout_seconds", 30)

    try:
        while True:
            raw = await asyncio.wait_for(websocket.receive_text(), timeout=timeout)
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"error": "Invalid JSON"}))
                continue

            method = msg.get("method")
            payload = msg.get("payload", {})

            if method == "reset":
                obs = await reset(ResetRequest(**payload))
                await websocket.send_text(obs.model_dump_json())
            elif method == "step":
                result = await step(StepRequest(**payload))
                await websocket.send_text(result.model_dump_json())
            elif method == "state":
                result = await state()
                await websocket.send_text(result.body.decode())
            else:
                await websocket.send_text(
                    json.dumps({"error": f"Unknown method '{method}'. Use: reset|step|state"})
                )

    except asyncio.TimeoutError:
        await websocket.close(code=1001, reason="step_timeout")
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception as exc:
        logger.exception("WebSocket error: %s", exc)
        await websocket.close(code=1011, reason="internal_error")