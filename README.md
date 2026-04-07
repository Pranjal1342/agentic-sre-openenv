---
title: Agentic SRE OpenEnv
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
---

# 🚀 Agentic SRE OpenEnv: Production-Grade RL Environment for Incident Remediation

Welcome to the **Agentic SRE Environment**! This repository contains a fully containerized, stateful reinforcement learning environment built on the **[OpenEnv 0.1](https://github.com/huggingface/openenv)** specification. 

Historically, Reinforcement Learning researchers relied on simple game sandboxes (like OpenAI Gym/CartPole) which fail to evaluate modern tool-using LLM Agents. This project simulates a real-world **Site Reliability Engineering (SRE) incident response** scenario — complete with a mocked microservice mesh, database locks, and latency spikes — forcing AI Agents to triage, diagnose, and remediate multi-stage outages just like a human engineer.

> [!TIP]
> **View this on Hugging Face Spaces!** 
> This environment is built to be deployed seamlessly with the Hugging Face Docker SDK, exposing a WebSocket (`/ws`) for continuous RL training via libraries like TRL or VeRL.

---

## 🏗️ Architectural Highlights

This environment was meticulously designed for determinism, stable RL gradients, and true multi-agent execution:

- **Type-Safe Pydantic Boundaries**: All Agent `Actions` (Diagnostic Queries, Log Inspections, Remediation) and `Observations` (Golden Signals, Stdout, Stderr) are strictly typed to prevent format hallucination.
- **Dual-RNG Determinism**: Fault injection is strictly seeded for reproducibility, while a secondary "Network Shim" injects reproducible latency jitter to prevent agents from merely memorizing paths.
- **Stateful FSM Orchestration**: A central Orchestrator dictates state transitions (`TRIAGE` → `INVESTIGATION` → `REMEDIATION` → `VERIFICATION`), preventing endless, unstructured internal looping.
- **Continuous Correction Pipeline (CI/CD/CM/CC)**: Automatically tracks the incident lifecycle. The mock infrastructure natively computes the Four Golden Signals and tracks error budget **Burn Rates**.
- **Layout-Aware Hierarchical RAG**: Utilizes `unstructured` and FAISS to parse SRE Runbook PDFs offline, retaining visual hierarchies that flat-text chunking destroys.
- **Dense Mathematical Reward Shaping**: Returns precise rewards between `[-0.5, 1.5]` using the blueprint formula: `R_t = α·ΔH_t + β·M_t + λ·E_t − γ·P_t − δ`, explicitly rewarding progress and penalizing destructive syntax.

---

## 📂 Project Structure

```text
📦 agentic-sre-openenv/
├── openenv.yaml                # Standardized OpenEnv metadata manifest
├── Dockerfile                  # Two-stage Docker build with built-in FAISS
├── inference.py                # Baseline evaluation script (Agent execution run)
├── .env                        # Environment configuration map
├── server/                     # Core FastAPI Server & OpenEnv logic
│   ├── app.py                  # HTTP & WebSocket endpoints (/step, /reset)
│   ├── pipeline.py             # CI/CD/CM/CC lifecycle tracker
│   ├── models.py               # Pydantic schemas (Action, Observation, Reward)
│   └── fsm.py                  # Finite State Machine Orchestrator
├── mock_infra/                 # Deterministic execution layer
│   ├── service_mesh.py         # Mocked Envoy mesh (latency, HTTP faults)
│   ├── database.py             # Mock PostgreSQL (locks, connection pools)
│   └── telemetry.py            # Golden Signal & Burn Rate computation
├── agents/                     # Multi-Agent workforce
│   ├── sre_agent.py            # Primary Planner & Orchestrator
│   ├── data_agent.py           # Metric/PromQL interrogation
│   ├── code_agent.py           # Log & Trace analysis
│   └── quarantine_agent.py     # Regex sanitization & security boundary
├── graders/                    # Dense Reward Computation
│   └── grader.py               # Implements exact mathematical reward constraints
├── rag/                        # Knowledge Engine
│   ├── offline_index.py        # Offline FAISS index generator
│   └── engine.py               # Low-latency runtime embedding retrieval
├── tasks/                      # Progressive incident scenarios
│   ├── task_1.py               # Gateway latency triage (Easy)
│   ├── task_2.py               # OOMKilled loop mitigation (Medium)
│   └── task_3.py               # Cascading DB pool exhaustion (Hard)
└── knowledge_base/             # SRE runbooks ingested by the RAG system
```

---

## 🚀 How to Run Locally

### 1. Installation
Clone the repository and install the dependencies. The numerical backend aggressively pins `numpy < 2.0` and utilizes a CPU-only PyTorch wheel to keep the Docker image inherently lean.

```bash
# Create a virtual environment
python -m venv venv
source venv/Scripts/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Generate the Offline RAG Index
To ensure the Docker container starts immediately without network calls, the FAISS index must be built locally into the `assets/` folder first.

```bash
python rag/offline_index.py
```

### 3. Spin up the Environment Server
```bash
uvicorn server.app:app --port 8000
```

### 4. Run the Baseline Assessment
To test your agent against the environment, run the baseline script in a separate terminal. Note: you can provide an OpenAI/HuggingFace API token, or leave it blank to execute the deterministic 'Mock LLM' fallback!

```bash
python inference.py
```

---

## 🎯 Hackathon Judging Criteria Satisfied
* **Real-World Utility:** Directly simulates the `$M/hr` problem of corporate incident response.
* **OpenEnv Compliance:** Natively implements `openenv.yaml` schemas, strict episode bounding, and Websocket streaming.
* **Grader Quality:** Leverages non-linear exponential decay ($e^{-1.45 * t_m / T_{max}}$) for Mean-Time-To-Mitigation (MTTM) reward components.
* **Environment Design:** Protected from context-window bloat via rolling context summaries; isolated agent security using a dedicated `QuarantineAgent`.
