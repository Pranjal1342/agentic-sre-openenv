---
title: Agentic SRE OpenEnv
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
---

# Agentic SRE Incident Remediation Environment

This is a production-grade, [OpenEnv 0.1-compliant](https://github.com/huggingface/openenv) reinforcement learning environment for SRE incident remediation. 

It provides a stateful, containerized sandbox where LLM agents must navigate simulated infrastructure (database connection pools, microservice mesh) to diagnose and resolve production outages.

## Architecture
- **Server:** FastAPI backend exposing `/step`, `/reset`, `/state`, and a persistent `/ws` WebSocket for TRL/VeRL training.
- **Mock Infrastructure:** Deterministically simulates PostgreSQL locks and Envoy service mesh latency.
- **Telemetry Layer:** Dynamically calculates a composite Health Score from the 4 Golden Signals.
- **Grader:** Uses a dense-reward function bounding values between `[-0.5, 1.5]`.

## Usage
Deploy this Space using Hugging Face Docker SDK, then point your RL loop to its WebSocket endpoint URL.
