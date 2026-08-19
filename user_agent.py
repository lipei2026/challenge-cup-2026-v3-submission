"""Competition entry point for the frozen V3 adaptive-compute agent."""

import time
from typing import Dict

from user_agent_v3 import ReasoningAgent as FrozenV3ReasoningAgent
from user_agent_v3 import V3AgentConfig


EMERGENCY_SOLVE_PROMPT = """你是严谨的数学解题者。上一条内部推理链因运行时异常未能返回答案。
请直接独立解决当前题目，输出一份简洁、完整、可独立判分的正式解答。
证明题必须保留必要证明，推导题保留关键步骤，计算题给出明确结果。
不要讨论异常、Agent、Prompt、节点或内部思考。最后一行写成：
FINAL_ANSWER: 具体答案或结论
"""


class _ChatOnlyClient:
    """Expose only the client API guaranteed by the competition runner."""

    def __init__(self, client) -> None:
        if client is None:
            raise TypeError("ReasoningAgent requires the official client")
        self._client = client

    def chat(self, messages, temperature, max_tokens):
        return self._client.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )


class ReasoningAgent(FrozenV3ReasoningAgent):
    """Frozen V3 with a bounded top-level fallback for submission reliability."""

    VERSION = "V3-AdaptiveCompute-Submission"

    def __init__(self, client, config: V3AgentConfig | None = None, *args, **kwargs) -> None:
        # Frozen V3 has optional local telemetry hooks. The competition only
        # guarantees chat(), so do not expose platform-client internals to V3.
        super().__init__(client=_ChatOnlyClient(client), config=config)

    def solve(self, problem: str, metadata: Dict) -> Dict:
        started = time.perf_counter()
        try:
            result = super().solve(problem, metadata)
            final_response = result.get("final_response", "")
            if isinstance(final_response, str) and final_response.strip():
                return result
            failure_type = "EmptyFinalResponse"
        except Exception as exc:  # Keep one failed node from invalidating the task.
            failure_type = type(exc).__name__

        return self._emergency_solve(problem, failure_type, time.perf_counter() - started)

    def _emergency_solve(
        self, problem: str, failure_type: str, elapsed_seconds: float
    ) -> Dict:
        try:
            response = self._chat(
                EMERGENCY_SOLVE_PROMPT,
                f"题目：\n{problem}",
                temperature=0.1,
                max_tokens=4096,
                call_label="submission_emergency_fallback",
            )
            final_response = str(response or "").strip()
        except Exception:
            final_response = "未能在运行时限内生成完整解答。\nFINAL_ANSWER: 无法确定"
        if not final_response:
            final_response = "未能生成完整解答。\nFINAL_ANSWER: 无法确定"
        return {
            "final_response": final_response,
            "trace": [
                {
                    "step": "submission_emergency_fallback",
                    "content": {
                        "trigger": failure_type,
                        "elapsed_before_fallback_seconds": round(elapsed_seconds, 6),
                        "fallback_chat_calls": 1,
                        "retry_control": "platform_managed",
                    },
                }
            ],
        }


__all__ = ["ReasoningAgent", "V3AgentConfig"]
