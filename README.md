---
title: Agentic SRE OpenEnv
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# Agentic SRE OpenEnv: Production-Grade RL Environment for Incident Remediation

Welcome to the **Agentic SRE Environment**! This repository contains a fully containerized, stateful reinforcement learning environment built on the **[OpenEnv 0.1](https://github.com/huggingface/openenv)** specification. 

Historically, Reinforcement Learning researchers relied on simple game sandboxes (like OpenAI Gym/CartPole) which fail to evaluate modern tool-using LLM Agents. This project simulates a real-world **Site Reliability Engineering (SRE) incident response** scenario — complete with a mocked microservice mesh, database locks, and latency spikes — forcing AI Agents to triage, diagnose, and remediate multi-stage outages just like a human engineer.

> [!TIP]
> **View this on Hugging Face Spaces!** 
> This environment is built to be deployed seamlessly with the Hugging Face Docker SDK, exposing a WebSocket (`/ws`) for continuous RL training via libraries like TRL or VeRL.

---

## Layout-Aware RAG Stack
The environment simulates real SRE documentation retrieval using a highly optimized RAG pipeline:
1. **Hierarchical Parsing:** Uses `unstructured` to parse SRE Runbook PDFs offline, retaining visual document hierarchies and tables rather than treating them as flat text chunks.
2. **Dense Vector Embeddings:** Utilizes `sentence-transformers` for precise semantic embedding of runbook clauses to match alert signatures.
3. **Zero-Latency FAISS Indexing:** Bakes the exact pre-computed nearest-neighbor `faiss-cpu` index directly into the Docker image, eliminating runtime network dependencies and ensuring instant retrieval.

---

## Continuous Correction (CI/CD/CM) Pipeline
The environment goes beyond stateless terminal commands by implementing a stateful `pipeline.py` tracker. It simulates a live Continuous Integration and Configuration Management lifecycle, forcing agents to respect deployment locks, rollback procedures, and stateful infrastructure changes during an active incident.

---

## Environment Design & Mathematics

### Reward Shaping Formula
Returns precise, bounded rewards between `[-0.5, 1.5]` using a dense mathematical formulation:

$$R_t = \alpha\Delta H_t + \beta M_t + \lambda E_t - \gamma P_t - \delta$$

* **$\alpha = 1.0$**: Strongest alignment with restoring the composite system health score ($\Delta H_t$).
* **$\beta = 0.2$**: Bonus for rapid non-linear Mean-Time-To-Mitigation (MTTM) resolving speed.
* **$\lambda = 0.15$**: Reward for efficient systematic exploration (finding root-cause metrics).
* **$\gamma = 0.5$**: Heavy penalty for destructive operations or syntactically invalid commands.
* **$\delta = 0.01$**: Constant step-penalty to actively penalize endless loop hallucination.

### Composite Health Score
The system state ($\Delta H_t$) is governed by a weighted average of the 4 Golden Signals:

$$H_t = w_1 A_t + w_2\left(\frac{1}{L_t}\right) - w_3 B_t$$
*(Where $A_t$ is Availability, $L_t$ is Latency, and $B_t$ is the calculated Error Budget Burn Rate)*

---

## Action & Observation Space Definitions

| Action Type | Parameters | Example Schema |
| :--- | :--- | :--- |
| `diagnostic_query` | `metric`, `time_window` | `{"action_type": "diagnostic_query", "metric": "latency_p99"}` |
| `log_inspection` | `target`, `lines`, `grep` | `{"action_type": "log_inspection", "target": "api-gateway", "lines": 50}` |
| `remediation` | `operation`, `service` | `{"action_type": "remediation", "operation": "restart", "service": "auth-db"}` |

| Observation Field | Type | Description |
| :--- | :--- | :--- |
| `active_alerts` | `Array[String]` | Currently firing Prometheus/PagerDuty alerts |
| `system_health` | `Float (0.0-1.0)` | The evaluated composite Health Score $H_t$ |
| `stdout` / `stderr` | `String` | Deterministic CLI/Log output from the previous action |

---

## Task Scenarios

| Task ID | Description | Difficulty | Expected Score |
| :--- | :--- | :--- | :--- |
| **`task_1`** | **Gateway Latency Triage:** A single upstream latency spike with a clear causal chain. Logs are readable, metrics are available, and a targeted restart resolves the issue. This establishes the baseline, it tests whether the agent can navigate the observation space, identify a localized fault signal, and execute a correct remediation without over engineering the solution. | Easy | `0.65` – `0.95` |
| **`task_2`** | **OOMKilled Loop:** A memory leak causing progressive pod evictions. Restarting without scaling or rolling back the offending deployment re-triggers the same eviction cycle. The agent must recognize the repeating failure pattern, resist the instinct to apply the same fix twice, and choose between two valid but non-equivalent remediation paths (scale-out vs. rollback).| Medium | `0.40` – `0.75` |
| **`task_3`** | **DB Pool Exhaustion:** A PostgreSQL schema lock caused by pool exhaustion looks superficially similar to task 1 but restarting the database here doesn't release the lock cleanly; it risks corrupting in flight transactions and cascades failures downstream. The heavy penalty weight (γ = 0.5) for destructive operations directly targets this failure mode. The agent must inspect connection pool state, identify the locking query, and drain connections gracefully before any restart. | Hard | `0.15` – `0.50` |
| **`task_4`** | **Telemetry Collapse:** A memory leak in `logging-fluentd-sidecar` evicts `prometheus-server`. The observability stack itself is the failure.The agent must recover the logging pipeline before it can diagnose anything else. This tests meta-reasoning under zero-visibility; a scenario with no clean optimal path and accordingly low expected scores.. | Hard | `< 0.15` |

---

## Baseline Evaluation Scores
Demonstrating the post-training environment validity on `task_1` (using the included baseline comparison script):

* **Pretrained LLM (Vanilla Prompt):** `0.0841` *(Failed to resolve, exhausted step limits)*
* **SRE Expert (Few-Shot Prompt):** `0.8725` *(Resolved optimally in 3 steps)*
* **Training Delta:** `+0.7884` *(Proving meaningful, wide RL gradients are available)*

---

## Project Structure

```text
  agentic-sre-openenv/
├── openenv.yaml                # Standardized OpenEnv metadata manifest
├── Dockerfile                  # Two-stage Docker build with built-in FAISS
├── inference.py                # Baseline evaluation script
├── .env                        # Environment configuration map
├── server/                     # Core FastAPI Server & OpenEnv logic
│   ├── app.py                  # HTTP & WebSocket endpoints (/step, /reset)
│   ├── pipeline.py             # CI/CD/CM/CC lifecycle tracker
│   ├── models.py               # Pydantic Action/Observation schemas
│   └── fsm.py                  # Finite State Machine Orchestrator
├── mock_infra/                 # Deterministic execution layer
│   ├── service_mesh.py         # Mocked Envoy mesh (latency faults)
│   ├── database.py             # Mock PostgreSQL (connection pools)
│   └── telemetry.py            # Golden Signal & Burn Rate computation
├── agents/                     # Multi-Agent workforce logic
├── graders/                    # Dense Reward Computation (MTTM formulas)
├── rag/                        # Knowledge Engine & FAISS logic
├── tasks/                      # Progressive incident scenarios
└── knowledge_base/             # SRE runbooks ingested by the RAG system
```
> For a deep dive into the RAG stack, Dual RNG, and FSM orchestration, check out our [System Architecture Documentation](docs/ARCHITECTURE.md) and [WorkFlow](docs/WorkFlow.md).

---

## 🚀 How to Run Locally

```bash
# 1. Install Dependencies
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt

# 2. Generate the Offline RAG Index
python rag/offline_index.py

# 3. Spin up the Environment Server (Default OpenEnv HTTP/WS)
uvicorn server.app:app --port 7860

# 4. Run the Baseline Assessment Runner
python inference.py
```
