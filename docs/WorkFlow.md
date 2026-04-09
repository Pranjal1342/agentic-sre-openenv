![alt text](<sre_incident_lifecycle_flowchart (1).png>)

# Incident Lifecycle Workflow — Agentic SRE OpenEnv

This document walks through the full lifecycle of a single episode from fault injection to policy update, following the four phases shown in the workflow diagram. Each phase maps to a distinct concern: generating the incident, triaging it, retrieving the relevant response strategy, and executing and validating the fix. Understanding this flow end-to-end is important because the environment is not a simple request-response system it's a stateful pipeline where the output of each phase directly shapes what the agent sees and can do in the next.

---

## Phase 1 — Fault Detection

Every episode begins with fault injection, and the environment uses two independent random number generators to construct the failure scenario. This dual RNG design is deliberate — it separates the *type* of fault from the *network conditions* surrounding it, giving fine grained control over what's reproducible and what's noisy.

The **Primary RNG** is responsible for deterministic fault injection. At episode reset, it selects the active task scenario, determines the specific failure mode (a database lock, a memory leak, a latency spike), sets the initial health score baseline, and composes the first Prometheus alert payload. Seeding the Primary RNG to a fixed value produces a fully reproducible episode — the same fault, the same alert text, the same initial health state every time. This is what the baseline evaluation script uses when generating the benchmark scores.

In parallel, the **Network RNG** introduces stochastic noise into the Envoy proxy layer. It randomizes latency jitter, packet loss variance, and connection timeout windows across the service mesh. The key point is that the Network RNG operates independently from the Primary RNG you can hold the fault scenario fixed (same Primary RNG seed) while varying the network conditions, or vary both simultaneously. This prevents the agent from overfitting to specific numeric values: even in Task 1 (Gateway Latency Triage), the p99 latency value is different every episode even though the root cause is always the same upstream service.

Both RNGs feed into the **Mock Infrastructure** and **Mock Telemetry** layers respectively. The mock infrastructure (`mock_infra/service_mesh.py`, `database.py`) applies the injected faults to the service state increasing connection pool utilization, tripping circuit breakers, or triggering pod evictions. The mock telemetry layer (`mock_infra/telemetry.py`) observes this infrastructure state and computes the Four Golden Signals: latency, traffic, error rate, and saturation.

Once the telemetry layer detects that one or more signals have crossed their alert thresholds, it fires a **Prometheus alert** ; a structured alert payload containing the alert name, severity, affected service, and the metric values that triggered it. This alert is the handoff point between Phase 1 and Phase 2. The environment is now in an active incident state, and the agent's first observation contains this alert as the primary input.

---

## Phase 2 — Action and Triage

When the Prometheus alert fires, control passes to the **SRE Agent**, which begins processing the active alert payload. The agent's job in this phase is strictly diagnostic it must understand what's broken before it can attempt to fix anything.

The agent submits its actions to the environment via the `POST /step` endpoint. In the triage phase, the action space is constrained to read-only operations: `diagnostic_query` (querying specific metrics with a configurable time window) and `log_inspection` (retrieving recent log lines from a named service, with optional grep filtering). This constraint is enforced by the FSM orchestrator in Phase 4 any attempt to submit a `remediation` action before the FSM has transitioned out of the `TRIAGING` state is blocked and penalized.

The **AI model** component here refers to the LLM being evaluated. It receives the current observation  the active alert, current metric values, recent log output from previous steps, and the RAG context injected by Phase 3 and produces a structured JSON action. The "Mock fallback (429 errors)" annotation in the diagram refers to the environment's handling of rate-limit responses from the model API during training: when the LLM returns a 429, the environment substitutes a fallback observation rather than crashing the episode, keeping the training loop alive.

The observation the agent receives is always structured around the **Four Golden Signals**: latency (p99 response time in milliseconds), traffic (requests per second at the affected service), error rate (proportion of 5xx responses), and saturation (the highest resource utilization across CPU, memory, and connection pool usage). These four signals are intentionally chosen because they are the standard observability primitives that SRE teams use to define SLOs  the agent is being trained on the same mental model that production engineers operate with.

One important constraint governs this entire phase: the **strict JSON schema**. The agent cannot submit free-form text actions. Every action must conform to the typed Pydantic schema defined in `server/models.py` specifying `action_type`, `metric` or `target`, and any supporting parameters. This is a deliberate design choice: it forces the LLM to produce structured, parseable outputs rather than reasoning in natural language, which is closer to how production automation tooling actually works. Actions that fail schema validation are rejected at the API boundary before they ever reach the execution layer.

---

## Phase 3 — Strategic Approach (RAG Pipeline)

Once the triage phase has produced an understanding of what the active alert looks like which service is affected, which signals are elevated, what the log output says — that context flows into the **RAG pipeline** to retrieve the relevant mitigation strategy from the knowledge base.

The pipeline starts with **Sentence Transformers**, which convert the active alert signature and current observation context into a dense vector embedding. The embedding model is chosen for technical documentation retrieval rather than general semantic similarity it performs better on the kind of domain-specific language that appears in SRE runbooks ("connection pool exhaustion", "OOMKilled", "p99 latency spike upstream").

This embedding is then queried against a **FAISS index** for nearest-neighbor lookup. The FAISS index is built offline at Docker build time (`rag/offline_index.py`) and baked directly into the container image. This eliminates any runtime network dependency for retrieval there is no external vector database call, no cold-start latency on first query. The index lives in memory and search is effectively instantaneous, which matters on the constrained evaluation infra (2 vCPU, 8GB RAM).

The index was built from the **Knowledge Base** a collection of SRE runbooks and post-mortem documents in `knowledge_base/`. These were parsed offline using `unstructured`, which preserves document hierarchy (headings, subheadings, tables, code blocks) rather than flattening everything into raw text. The result is that chunks in the index carry structural metadata: a section titled "DB Pool Exhaustion Immediate Mitigation Steps" is stored as a semantically distinct unit, not merged with the surrounding prose. This is what "layout-aware" means — the parser understands that a numbered mitigation step list under a specific heading is more valuable as a retrieval unit than the paragraph before it.

The top-k retrieved chunks are assembled into a **Layout-aware RAG result** the approved mitigation strategy for the current alert pattern and injected directly into the agent's observation as the `rag_context` field. The agent therefore enters Phase 4 not just with raw metrics and log output, but with the relevant procedural guidance from the runbook that matches its current fault signature. This is the difference between an agent that guesses at remediation steps and one that reasons about them in context.

---

## Phase 4 — Execution and Validation

With a mitigation strategy in hand from the RAG pipeline, the agent can now submit a **Remediation Command**; a state-changing action with `action_type: remediation` and the target operation (restart, scale, rollback, drain) and affected service. This is where the environment's most consequential logic lives.

Before any remediation command executes, it passes through the **FSM Orchestrator** (`server/fsm.py`). The FSM has two jobs here: validating that the requested operation is legal in the current state, and blocking destructive operations that would violate the deployment lock. If the infrastructure is in a locked state for example, a rolling deployment is 60% complete a restart command against the deployment target is blocked and returns a structured error. The agent must either wait for the lock to clear, initiate a proper rollback through the CI/CD tracker, or find an alternative remediation path. Operations that bypass this validation (invalid action types, malformed parameters, attempts to restart services that don't exist) incur the γ = 0.5 penalty, the heaviest in the reward function.

Validated commands are then handed to the **CI/CD/CM/CC Pipeline** (`server/pipeline.py`), which applies the operation to the mock infrastructure state and updates the pipeline tracker. A restart operation, for example, updates the affected service's uptime clock, resets its connection pool state, and triggers a health re-check in the telemetry layer. A rollback operation walks the deployment state backward and reapplies the previous configuration. The pipeline tracker maintains the history of these operations within the episode, so the agent's observation always reflects the actual current state of infrastructure operations  and not a static snapshot.

After execution, the **Episode Grader** (`graders/`) computes the step reward using the full dense reward formula. It evaluates the change in composite health score (ΔH_t), applies the MTTM speed bonus if the agent is converging toward resolution faster than the step-penalty baseline would suggest, credits exploration bonuses for newly queried metrics, and applies any applicable penalties for this step. The result is packaged into a `StepResult` object containing the scalar reward, updated observation, done flag, and penalty metadata, which is returned to the agent via the `/step` response.

The grader then checks the **Resolved?** condition whether the composite health score H_t has returned above the resolution threshold and all active alerts have cleared. If yes, the episode terminates cleanly with a final score in the range `[0.0, 1.0]` reflecting the quality of the resolution: how fast it happened, how many destructive operations were avoided, and how completely the health score recovered. If the step budget runs out before resolution, the episode terminates with a **Timeout** a hard penalty of −1.0 and `done=True` signaling to the policy optimizer that this trajectory was a failure.

Both outcomes feed into the **RL Training Loop**, which collects the episode trajectory (the full sequence of observations, actions, and rewards) and uses it for a policy gradient update. The updated policy is used in the next episode, which begins a new reset cycle and the loop continues.

The "next episode" arrow on the right side of the diagram closes this cycle: the policy update from one episode directly shapes the action distribution for the next, which is exactly how RL post-training works in practice.

---

## Summary

The four phases form a coherent pipeline that mirrors real incident response:

| Phase | What happens | Key components |
|---|---|---|
| **1 — Fault Detection** | Dual RNG injects a fault, telemetry detects it, Prometheus alert fires | Primary RNG, Network RNG, mock_infra, telemetry |
| **2 — Action and Triage** | Agent reads the alert, queries metrics and logs under JSON schema constraints | SRE agent, AI model, Four Golden Signals |
| **3 — RAG Strategy** | Alert is embedded, FAISS retrieves matching runbook chunks, context injected into observation | Sentence Transformers, FAISS, knowledge_base |
| **4 — Execution and Validation** | Remediation runs through FSM and CI/CD pipeline, grader scores the outcome, RL loop updates the policy | FSM, pipeline.py, graders, RL training loop |
