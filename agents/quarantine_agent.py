"""
agents/quarantine_agent.py
Security boundary between untrusted telemetry/log data and the planner agents.

Responsibilities:
  1. Strip ANSI escape sequences (terminal colour codes)
  2. Strip HTML / script tags (XSS injection via log lines)
  3. Strip common prompt-injection patterns
  4. Enforce a hard 4096-character cap on all sanitised output
  5. Log every sanitisation event for auditability
"""

import re
import logging

logger = logging.getLogger(__name__)

# ─── Injection patterns ───────────────────────────────────────────────────────
_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("ansi_escape",   re.compile(r"\x1b\[[0-9;]*[mABCDEFGHJKLMSTfils]")),
    ("html_script",   re.compile(r"<script[\s\S]*?>[\s\S]*?</script>", re.IGNORECASE)),
    ("html_tag",      re.compile(r"<[^>]+>")),
    ("null_byte",     re.compile(r"\x00")),
    ("carriage_ret",  re.compile(r"\r")),
    # Prompt injection: attempts to override system context
    ("prompt_inject", re.compile(
        r"(?i)(ignore\s+(all\s+)?previous\s+instructions?|"
        r"you\s+are\s+now|act\s+as|system\s*:\s*)",
    )),
    # Path traversal in log output
    ("path_traversal", re.compile(r"\.\./|\.\.\\")),
    # Shell metacharacters that should never appear in telemetry
    ("shell_meta",    re.compile(r"[`$]\(|&&|\|\|")),
]

MAX_LOG_CHARS = 4096


class QuarantineAgent:
    """
    Sanitises untrusted log and telemetry strings before they reach
    the planning agents, preventing prompt injection and buffer overflows.
    """

    def __init__(self, max_chars: int = MAX_LOG_CHARS) -> None:
        self.max_chars = max_chars
        self._sanitisation_events: list[dict] = []

    def sanitize(self, raw: str, source: str = "unknown") -> str:
        """
        Clean `raw` of all injection patterns and truncate to max_chars.

        Args:
            raw:    The untrusted string (log line, metric output, etc.)
            source: Label for audit logging (e.g. 'order-service-logs').

        Returns:
            The sanitised, length-capped string.
        """
        cleaned = raw
        triggered: list[str] = []

        for name, pattern in _PATTERNS:
            new_cleaned = pattern.sub("", cleaned)
            if new_cleaned != cleaned:
                triggered.append(name)
            cleaned = new_cleaned

        # Collapse excess whitespace introduced by stripping
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

        # Hard 4096-char cap — prevents catastrophic buffer overflows
        truncated = False
        if len(cleaned) > self.max_chars:
            cleaned = cleaned[: self.max_chars]
            truncated = True

        if triggered or truncated:
            event = {
                "source": source,
                "patterns_stripped": triggered,
                "truncated": truncated,
                "original_len": len(raw),
                "sanitised_len": len(cleaned),
            }
            self._sanitisation_events.append(event)
            logger.warning("QuarantineAgent sanitised output from '%s': %s", source, event)

        return cleaned

    def sanitize_observation(self, stdout: str, stderr: str) -> tuple[str, str]:
        """Convenience wrapper — sanitises both stdout and stderr at once."""
        return self.sanitize(stdout, "stdout"), self.sanitize(stderr, "stderr")

    @property
    def audit_log(self) -> list[dict]:
        return list(self._sanitisation_events)

    def reset(self) -> None:
        self._sanitisation_events = []
