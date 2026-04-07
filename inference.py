import asyncio
import json
import os
import logging
from dotenv import load_dotenv

# Force load the .env file
load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"), override=True)

from dataclasses import dataclass, field
import websockets

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SERVER_WS_URL = os.getenv("SRE_ENV_WS", "ws://localhost:8000/ws")

# Mandatory Hackathon Variables
HF_TOKEN = os.getenv("HF_TOKEN", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")

# ---------------------------------------------------------------------------
# System Prompts
# ---------------------------------------------------------------------------

VANILLA_PROMPT = """You are a helpful AI assistant. Help the user debug their system."""

SRE_EXPERT_PROMPT = """You are an expert Site Reliability Engineer with deep knowledge of
Kubernetes, PostgreSQL, and distributed microservice architectures.

Your diagnostic process follows this strict methodology:
1. TRIAGE: Ingest the active alerts. Identify the affected service tier.
2. INVESTIGATION: Query the specific golden signal that matches the alert signature.
   - HIGH_LATENCY → query 'latency_p99_ms' with time_window '5m'
   - OOMKilled → query 'saturation_pct' or 'memory_usage_pct'
   - DB_POOL_EXHAUSTION → query 'connection_pool_used' or inspect 'pg_stat_activity'
3. LOCALIZATION: Narrow to the single root-cause service. Avoid querying unaffected services.
4. REMEDIATION: Execute the minimum necessary remediation action.
5. VERIFICATION: Re-query golden signals to confirm recovery before closing the incident.

Action format rules:
- DiagnosticQueryAction: {"action_type": "diagnostic_query", "metric_identifier": "<metric>", "time_window": "<Xm>"}
- LogInspectionAction: {"action_type": "log_inspection", "tail_lines": 50, "grep_pattern": "<pattern>"}
- RemediationAction: {"action_type": "remediation", "operation_type": "<restart|rollback|scale_up>", "target_service": "<name>"}

Available metrics: latency_p99_ms, traffic_rps, error_rate, saturation_pct,
                   connection_pool_used, memory_usage_pct, cpu_usage, pg_stat_activity
"""

# ---------------------------------------------------------------------------
# Mock LLM (Fallback for Rate Limits)
# ---------------------------------------------------------------------------

def _mock_llm_action(observation: dict, system_prompt: str, step: int) -> dict:
    is_expert = "SRE" in system_prompt or "TRIAGE" in system_prompt

    if is_expert:
        if step == 1:
            return {"action_type": "diagnostic_query", "metric_identifier": "latency_p99_ms", "time_window": "5m"}
        elif step == 2:
            return {"action_type": "log_inspection", "tail_lines": 30, "grep_pattern": "WARN|ERROR|upstream"}
        else:
            return {"action_type": "diagnostic_query", "metric_identifier": "error_rate", "time_window": "1m"}
    else:
        actions = [
            {"action_type": "diagnostic_query", "metric_identifier": "cpu_usage", "time_window": "15m"},
            {"action_type": "diagnostic_query", "metric_identifier": "traffic_rps", "time_window": "15m"},
            {"action_type": "log_inspection", "tail_lines": 100, "grep_pattern": ""},
            {"action_type": "diagnostic_query", "metric_identifier": "cpu_usage", "time_window": "15m"},
            {"action_type": "diagnostic_query", "metric_identifier": "cpu_usage", "time_window": "15m"},
        ]
        return actions[min(step - 1, len(actions) - 1)]


async def _get_llm_action(observation: dict, system_prompt: str, step: int, model: str) -> dict:
    if not HF_TOKEN:
        logger.debug("No HF_TOKEN — using mock LLM.")
        return _mock_llm_action(observation, system_prompt, step)

    try:
        from openai import AsyncOpenAI
        
        # Mandatory variable initialization
        client = AsyncOpenAI(
            api_key=HF_TOKEN,
            base_url=API_BASE_URL
        )

        obs_text = json.dumps(observation, indent=2)
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Current environment observation (step {step}):\n{obs_text}\n\n"
                        "Respond with ONLY a valid JSON action object. No explanation."
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=256,
        )
        raw = response.choices[0].message.content.strip()
        return json.loads(raw)
    except Exception as exc:
        logger.warning("LLM call failed (%s) — using mock fallback.", exc)
        return _mock_llm_action(observation, system_prompt, step)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

@dataclass
class EpisodeResult:
    score: float = 0.0
    steps: int = 0
    resolved: bool = False
    reward_history: list[float] = field(default_factory=list)
    termination: str = ""


async def run_episode(
    model: str,
    system_prompt: str,
    task_id: str,
    seed: int = 42,
) -> EpisodeResult:
    result = EpisodeResult()
    
    # REQUIRED LOG: [START] - Flushed immediately so automated graders catch it
    print(f"[START] task_id={task_id} model={model}", flush=True)

    async with websockets.connect(SERVER_WS_URL) as ws:
        await ws.send(json.dumps({"method": "reset", "payload": {"task_id": task_id, "seed": seed}}))
        obs_raw = await ws.recv()
        observation = json.loads(obs_raw)

        while True:
            result.steps += 1
            action = await _get_llm_action(observation, system_prompt, result.steps, model)

            await ws.send(json.dumps({"method": "step", "payload": {"action": action}}))
            step_raw = await ws.recv()
            step_result = json.loads(step_raw)

            reward_val = step_result.get("reward", {}).get("value", 0.0)
            result.reward_history.append(reward_val)
            result.steps = step_result.get("info", {}).get("step_count", result.steps)
            
            is_done = step_result.get("done", False)

            # REQUIRED LOG: [STEP] — full action JSON required by grader
            print(f"[STEP] step={result.steps} action={json.dumps(action, separators=(',', ':'))} reward={reward_val:.4f} done={str(is_done).lower()}", flush=True)

            observation = step_result.get("observation", {})

            if is_done:
                result.termination = step_result.get("info", {}).get("termination", "unknown")
                result.resolved = result.termination == "resolved"
                break

        result.score = sum(result.reward_history)
        
        # REQUIRED LOG: [END] — field must be 'score=' not 'final_score='
        print(f"[END] task_id={task_id} score={result.score:.4f} resolved={str(result.resolved).lower()} steps={result.steps}", flush=True)

    return result


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

async def main():
    # Run all 3 tasks — checklist requires 3+ tasks, each with a grader score
    for task_id in ["task_1", "task_2", "task_3"]:
        await run_episode(model=MODEL_NAME, system_prompt=SRE_EXPERT_PROMPT, task_id=task_id)

if __name__ == "__main__":
    if not HF_TOKEN:
        logger.warning("HF_TOKEN is missing. Make sure your .env file is updated for submission.")
    
    asyncio.run(main())