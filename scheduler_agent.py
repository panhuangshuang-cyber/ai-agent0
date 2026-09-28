"""Tool-free scheduling agent for the document QA pipeline."""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Sequence

from agent_types import (
    SCHEDULER_INTENTS,
    SchedulePlan,
    clean_text,
    decision_arguments,
    first_message,
    function_tool,
    history_block,
    tool_choice,
)


SCHEDULE_DECISION_TOOL = function_tool(
    "schedule_decision",
    "Classify the turn and return a routing plan.",
    {
        "intent": {"type": "string", "enum": sorted(SCHEDULER_INTENTS)},
        "reply": {"type": "string"},
        "question": {"type": "string"},
        "doc_hint": {"type": "string"},
        "sheet_hint": {"type": "string"},
        "calc_goal": {"type": "string"},
    },
    ["intent", "question"],
)

SYSTEM_PROMPT = """你是调度代理。你只判断本轮属于闲聊、追问、查表还是统计，并且必须调用 schedule_decision。
intent 只能是 chat、clarify、lookup、analyze。
只有 chat 或 clarify 才能填写 reply；查表问题不要在 reply 里回答。
question 去掉寒暄但保留用户用词。doc_hint 必须保持用户指代文档的原话，不得替换成记忆中的正式名称。
sheet_hint 保留用户对子表的原话，没有就填空字符串。analyze 要在 calc_goal 写清计算目标；lookup 的 calc_goal 必须为空。
记忆中的别名只供理解，禁止改写进 doc_hint。你看不到文档，也不能编造查表结论。"""


class SchedulerAgent:
    def __init__(self, chat: Callable[..., Awaitable[Any]] | None = None):
        self._chat = chat

    async def _complete(self, messages: list[dict[str, Any]], forced: bool) -> Any:
        choice = tool_choice("schedule_decision") if forced else None
        if self._chat is not None:
            return await self._chat(messages, [SCHEDULE_DECISION_TOOL], choice)
        from llm_client import chat_with_fallback
        response, _model = await chat_with_fallback(
            messages,
            tools=[SCHEDULE_DECISION_TOOL],
            tool_choice=choice,
        )
        return response

    async def _ask(self, messages: list[dict[str, Any]]) -> Any:
        try:
            return await self._complete(messages, True)
        except Exception as exc:
            logging.warning("scheduler retry without tool_choice: %s", exc)
            return await self._complete(messages, False)

    async def run(
        self,
        user_text: str,
        prior_turns: Sequence[dict[str, Any]] | None,
        memory_text: str,
    ) -> SchedulePlan:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"已有记忆：\n{memory_text or '（无）'}\n\n"
                    f"最近对话：\n{history_block(prior_turns)}\n\n"
                    f"用户本轮原话：\n{clean_text(user_text)}"
                ),
            },
        ]

        for attempt in range(2):
            response = await self._ask(messages)
            message = first_message(response)
            raw = decision_arguments(message, "schedule_decision")
            plan = self._normalize(raw)
            if self._valid(plan):
                logging.info("agent=scheduler intent=%s attempt=%s", plan.intent, attempt + 1)
                return plan
            messages.append({"role": "assistant", "content": str(getattr(message, "content", "") or "")})
            messages.append({
                "role": "user",
                "content": "上一次没有给出合法计划。请严格调用 schedule_decision，并满足字段约束。",
            })

        raise ValueError("scheduler returned no valid plan")

    @staticmethod
    def _normalize(raw: dict[str, Any] | None) -> SchedulePlan | None:
        if not isinstance(raw, dict):
            return None
        raw_intent = clean_text(raw.get("intent"))
        if raw_intent not in SCHEDULER_INTENTS:
            return None
        plan = SchedulePlan(
            intent=raw_intent,
            reply=clean_text(raw.get("reply")),
            question=clean_text(raw.get("question")),
            doc_hint=clean_text(raw.get("doc_hint")),
            sheet_hint=clean_text(raw.get("sheet_hint")),
            calc_goal=clean_text(raw.get("calc_goal")),
        )
        if plan.calc_goal:
            plan.intent = "analyze"
        elif plan.intent == "analyze":
            plan.intent = "lookup"
        if plan.intent in {"lookup", "analyze"}:
            plan.reply = ""
        if plan.intent == "lookup":
            plan.calc_goal = ""
        return plan

    @staticmethod
    def _valid(plan: SchedulePlan | None) -> bool:
        if plan is None or plan.intent not in SCHEDULER_INTENTS or not plan.question:
            return False
        if plan.intent in {"chat", "clarify"}:
            return bool(plan.reply)
        return True


scheduler_agent = SchedulerAgent()


async def run(user_text: str, prior_turns, memory_text: str) -> SchedulePlan:
    return await scheduler_agent.run(user_text, prior_turns, memory_text)
