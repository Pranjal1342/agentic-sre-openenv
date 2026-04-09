![alt text](<React Dashboard to-2026-04-07-172555.png>)

# System Architecture — Agentic SRE OpenEnv

The architecture of this environment is built around one core principle: every layer the agent touches must behave like real infrastructure. That means stateful services with genuine failure modes, a reward signal grounded in production SRE mathematics, documentation retrieval from hierarchically-parsed runbooks, and an orchestration layer that enforces the same access controls a real on-call engineer would face. The sections below walk through each component layer by layer — what it does, why it's designed the way it is, and how it connects to everything around it.

---

## Color Legend (Architecture Diagram)

The diagram uses consistent color coding to distinguish concerns at a glance:

- **Blue** — client-facing communication paths (HTTP, WebSocket, agent I/O)
- **Green** — core state and orchestration logic (FSM, health scoring, episode management)
- **Red** — agent and AI decision paths (LLM calls, action parsing, model fallback)
- **Orange** — infrastructure layer connections (mock services, telemetry, CI/CD)
- **Yellow dashed** — feedback and failure propagation cycles (cascading faults, RL loop)

---

## 1. Entry Point — FastAPI Server (`server/app.py`)

The environment exposes two communication channels from a single FastAPI process: a standard HTTP REST interface and a persistent WebSocket connection, both running on port 7860.

The HTTP interface serves three OpenEnv-spec endpoints. `POST /reset` initializes or reinitializes an episode, seeding the fault scenario and resetting infrastructure state to a clean baseline. `POST /step` accepts a structured action payload, runs it through the full execution pipeline, and returns the next observation along with a reward signal. `GET /state` allows the current environment state to be inspected without advancing the episode useful for debugging and for external evaluators checking environment health between runs.

Every action and observation passing through these endpoints is validated against Pydantic models defined in `server/models.py`. This enforces strict type contracts at the boundary, which prevents malformed agent outputs from causing silent failures deep in the pipeline.

The WebSocket endpoint at `/ws` runs the same step/reset loop but over a persistent connection. This is what RL training libraries like TRL and VeRL use during continuous rollout collection the client sends a JSON action frame, the server processes it through the full pipeline, and the resulting observation frame is streamed back within the same open connection. Keeping the connection alive eliminates per-request TCP handshake overhead during high-frequency training, which matters when you're running thousands of environment steps per training iteration.

---

## 2. Orchestration — FSM (`server/fsm.py`)

Every active episode is managed by a Finite State Machine that tracks the current incident phase and enforces valid state transitions. The FSM exists because one of the most common failure modes for LLM agents in SRE scenarios is jumping straight to remediation before understanding the fault — exactly what a tired junior engineer does at 2am, and exactly what gets people paged twice.

The state graph moves through: `IDLE → ALERTING → TRIAGING → REMEDIATING → VALIDATING → RESOLVED` (or `ESCALATED` if the step budget is exhausted without resolution). Each state exposes a different subset of the action space. In `TRIAGING`, only `diagnostic_query` and `log_inspection` actions are valid. Attempting a `remediation` action in that state returns a structured error observation and applies the destructive operation penalty (γ = 0.5), which directly trains the agent to respect the diagnostic phase.

The FSM also enforces deployment locks during the `REMEDIATING` state — a lock mechanism that mirrors the real-world practice of acquiring a change freeze token before touching production services during an active incident. An agent that attempts to restart a locked service is penalized, not silently ignored.

---

## 3. CI/CD/CM/CC Pipeline Tracker (`server/pipeline.py`)

This is one of the less obvious but more important components in the system. Most RL environments are stateless within a step — the world resets cleanly between actions. Real infrastructure doesn't work that way. A deployment that was 70% applied when you initiated a rollback doesn't cleanly undo itself. A config change that propagated to three of five pods before you caught it leaves the cluster in a split-brain state.

`pipeline.py` simulates this by maintaining a stateful tracker of the infrastructure lifecycle across steps within an episode. It tracks the current deployment phase (build, test, deploy, monitor), whether a change lock is active, which services have received configuration updates, and whether a rollback is in a clean or partial state. The agent's observation includes this pipeline context, forcing it to reason about the *current state of infrastructure operations* rather than treating each step as independent.

This is what the CI/CD/CM/CC label means: Continuous Integration (build and test state), Continuous Deployment (rollout progress and lock status), Configuration Management (per-service config drift tracking), and Continuous Compliance (whether current infra state violates defined SLOs).

---

## 4. Mock Infrastructure Layer (`mock_infra/`)

The mock infrastructure layer is where fault scenarios are physically simulated. It has three components that correspond to the three observable dimensions of a microservice mesh:

**`service_mesh.py`** simulates an Envoy-style proxy layer between services. It injects configurable latency faults, connection timeouts, and circuit breaker trips. Latency values are not hardcoded — they're generated by the Network RNG at episode initialization, which introduces stochastic variation in the magnitude and spread of latency spikes. This means the agent can't memorize "latency spike always means restart auth-db" it has to actually read the metric values and reason about thresholds.

**`database.py`** simulates a PostgreSQL connection pool with configurable pool size, active connection count, and lock state. It supports operations like `acquire_connection`, `release_connection`, `inspect_locks`, and `drain_pool`. When the pool is exhausted or a lock is held, subsequent queries return realistic error messages rather than generic failures — the agent sees `ERROR: remaining connection slots are reserved for non-replication superuser connections` rather than `ERROR: database unavailable`.

**`telemetry.py`** computes the Four Golden Signals — latency (p99 response time), traffic (requests per second), error rate (5xx responses / total), and saturation (resource utilization across CPU, memory, and connection pool) — and from these derives the composite Health Score H_t used in the reward function. In Task 4, this is the component that fails: `prometheus-server` eviction caused by the fluentd sidecar's memory leak means `telemetry.py` returns null or error for all metric queries, and the agent's primary observability window goes dark.

---

## 5. Dual RNG System

The environment uses two separate random number generators at episode initialization, each seeded independently to give fine-grained control over reproducibility and variance.

The **Primary RNG** handles deterministic fault injection — it selects which task scenario runs, determines the specific failure mode (OOMKill vs. pool exhaustion vs. latency spike), and sets the initial health score and alert payload. Using a fixed seed for the Primary RNG produces a fully reproducible episode, which is what the baseline evaluation script uses.

The **Network RNG** introduces stochastic variation into the Envoy mesh layer — jitter in latency values, randomized packet loss rates, and variable connection timeout windows. This ensures that even within a fixed task scenario, the agent encounters slightly different metric profiles on each episode, preventing overfitting to specific numeric thresholds. The Network RNG is seeded separately from the Primary RNG so you can hold the fault scenario constant while varying the network noise, or vice versa — useful for ablation studies during training.

---

## 6. RAG Knowledge Engine (`rag/`)

The RAG pipeline connects the agent's current observation to the `knowledge_base/` of SRE runbooks and post-mortems. It has three stages:

**Offline indexing** (`rag/offline_index.py`) runs at Docker build time. It uses `unstructured` to parse SRE runbook PDFs, preserving document hierarchy — headings, subheadings, tables, and code blocks are retained as structural metadata rather than being flattened into a single text blob. This is what "layout-aware" means: a runbook section titled "DB Pool Exhaustion — Immediate Mitigation Steps" is treated as a semantically distinct chunk, not just another paragraph.

**Dense embedding** uses `sentence-transformers` to convert each parsed chunk into a vector embedding. The model is chosen for its performance on technical documentation retrieval rather than general-purpose semantic similarity.

**FAISS indexing** builds a nearest-neighbor index over these embeddings and bakes it directly into the Docker image. At inference time, the agent's current alert signature is embedded and queried against the FAISS index — the top-k matching runbook chunks are retrieved and injected into the agent's observation as the `rag_context` field. Because the index lives in the container rather than being fetched from a remote service, retrieval latency is effectively zero and there are no network dependencies during RL training.

The result is that the agent doesn't just see raw metrics — it sees metrics alongside the relevant mitigation playbook for the alert it's looking at. This tests whether the agent can synthesize structured observability data with natural language procedural guidance, which is exactly what a human SRE does when they open a runbook mid-incident.

---

## 7. Reward & Grading Layer (`graders/`)

The graders implement the dense reward function described in the README:

```
R_t = α·ΔH_t + β·M_t + λ·E_t − γ·P_t − δ
```

Each grader is task-specific but shares the same base computation structure. After each step, the grader:

1. Computes `ΔH_t` — the change in composite health score from the previous step to the current step, weighted by α = 1.0. This is the primary reward signal.
2. Computes `M_t` — a non-linear MTTM (Mean Time To Mitigation) bonus, weighted by β = 0.2. The bonus is larger for faster resolutions and decays as steps accumulate, actively incentivizing efficient diagnosis over brute-force iteration.
3. Computes `E_t` — an exploration reward for querying previously unexamined metrics, weighted by λ = 0.15. This counters the tendency of undertrained agents to repeatedly query the same metric.
4. Applies `P_t` — a penalty for destructive or invalid operations, weighted by γ = 0.5. This is the heaviest single-step penalty in the function, reflecting the asymmetry in production: a bad restart is far more costly than a missed diagnostic step.
5. Subtracts δ = 0.01 per step as a constant time penalty. This is small enough not to dominate the signal but large enough to prevent the agent from entering idle loops while the episode timer runs down.

The grader returns a `StepResult` object containing the scalar reward, updated observation, `done` flag, and diagnostic metadata (which penalty terms fired and why). This metadata is surfaced in the `[STEP]` log output for evaluation transparency.

---

## 8. Multi-Agent Layer (`agents/`)

The `agents/` directory implements the workforce model — multiple specialized agents that can operate within the same episode. The current design separates diagnostic and remediation concerns: a Diagnostician agent whose action space is restricted to read-only queries and log inspection, and a Remediator agent that can execute state-changing operations but only after the Diagnostician has flagged a root cause.

This mirrors how actual SRE teams operate under incident command — the engineer who owns the diagnostic phase is different from the engineer who owns the change execution, with a handoff moment that prevents untested remediations from being applied to a system that isn't fully understood yet.

---

## 9. RL Training Loop Integration

The environment is designed to plug directly into standard RL post-training frameworks. The WebSocket interface maps cleanly to the rollout collection pattern used by TRL's `GRPOTrainer` and VeRL's distributed training loop. An episode proceeds as: `reset()` → repeated `step()` calls until `done=True` → episode score returned → policy gradient update.

The episode score returned at termination is bounded to `[0.0, 1.0]` for successful resolution, or a hard `-1.0` penalty with `done=True` for timeout. Both outcomes feed back into the RL training loop, giving the policy optimizer a clean scalar signal for every trajectory regardless of how the episode ended.
