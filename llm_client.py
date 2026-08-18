import copy
import json
import os
import threading
import time
from typing import Dict, List

import requests


DEFAULT_API_BASE = "https://chat.intern-ai.org.cn/api/v1/chat/completions"
DEFAULT_MODEL = "intern-s2-preview"


class InternChatClient:
    """Small OpenAI-compatible chat client for the competition sample."""

    def __init__(
        self,
        timeout: int = 120,
        retry: int = 3,
    ) -> None:
        raw_api_key = os.environ.get("INTERN_API_KEY")
        if not raw_api_key:
            raise RuntimeError("Missing API key. Set INTERN_API_KEY.")
        self.authorization = (
            raw_api_key if raw_api_key.startswith("Bearer ") else f"Bearer {raw_api_key}"
        )
        self.api_base = os.environ.get("INTERN_API_BASE", DEFAULT_API_BASE)
        self.model = os.environ.get("INTERN_MODEL", DEFAULT_MODEL)
        self.timeout = timeout
        self.retry = retry
        self._telemetry = threading.local()

    def reset_telemetry(self, run_context: Dict = None) -> None:
        """Start an isolated telemetry buffer for the current solve thread."""
        self._telemetry.calls = []
        self._telemetry.run_context = dict(run_context or {})
        self._telemetry.call_context = {}

    def set_telemetry_context(self, **context) -> None:
        """Attach a node label to the next logical chat call in this thread."""
        self._telemetry.call_context = dict(context)

    def get_telemetry(self) -> List[Dict]:
        return copy.deepcopy(getattr(self._telemetry, "calls", []))

    def _append_telemetry(self, record: Dict) -> None:
        if not hasattr(self._telemetry, "calls"):
            self.reset_telemetry()
        record.setdefault("call_index", len(self._telemetry.calls))
        self._telemetry.calls.append(record)

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": self.authorization,
        }

        started_at = time.perf_counter()
        call_context = dict(getattr(self._telemetry, "call_context", {}))
        run_context = dict(getattr(self._telemetry, "run_context", {}))
        self._telemetry.call_context = {}
        attempt_errors = []
        last_error = None
        for attempt in range(self.retry):
            attempt_started_at = time.perf_counter()
            try:
                response = requests.post(
                    self.api_base,
                    headers=headers,
                    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                choice = data["choices"][0]
                message = choice["message"]
                content = message.get("content", "")
                usage = data.get("usage", {}) or {}
                finish_reason = choice.get("finish_reason", choice.get("stop_reason"))
                record = {
                    **run_context,
                    **call_context,
                    "model": self.model,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "finish_reason": finish_reason,
                    "prompt_tokens": usage.get("prompt_tokens", usage.get("input_tokens")),
                    "completion_tokens": usage.get(
                        "completion_tokens", usage.get("output_tokens")
                    ),
                    "total_tokens": usage.get("total_tokens"),
                    "latency_seconds": round(time.perf_counter() - started_at, 6),
                    "http_attempts": attempt + 1,
                    "attempt_errors": attempt_errors,
                    "response_chars": len(content) if isinstance(content, str) else 0,
                    "truncated": finish_reason in {"length", "max_tokens"},
                    "error": None,
                }
                if os.environ.get("RESEARCH_LOG_FULL", "0") == "1":
                    record["messages"] = copy.deepcopy(messages)
                    record["response_content"] = content
                    if message.get("reasoning_content") is not None:
                        record["reasoning_content"] = message.get("reasoning_content")
                self._append_telemetry(record)
                return content
            except Exception as exc:  # noqa: BLE001 - keep sample robust and simple.
                last_error = exc
                attempt_errors.append(
                    {
                        "attempt": attempt + 1,
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "latency_seconds": round(
                            time.perf_counter() - attempt_started_at, 6
                        ),
                    }
                )
                if attempt + 1 < self.retry:
                    time.sleep(2**attempt)

        self._append_telemetry(
            {
                **run_context,
                **call_context,
                "model": self.model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "finish_reason": "error",
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
                "latency_seconds": round(time.perf_counter() - started_at, 6),
                "http_attempts": self.retry,
                "attempt_errors": attempt_errors,
                "response_chars": 0,
                "truncated": False,
                "error": {
                    "type": type(last_error).__name__,
                    "message": str(last_error),
                },
            }
        )
        raise RuntimeError(f"Chat completion failed after {self.retry} attempts: {last_error}")
