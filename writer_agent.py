"""Write-proposal agent. Tool-free by design: it cannot touch the sheet.

The agent only names a column and a new value. The cell address is computed by
app.py from the sandbox's real row index plus located.columns, and nothing is
written until the user confirms a second time.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Awaitable, Callable

from agent_types import (
    AnalysisResult,
    LocatedResult,
    SchedulePlan,
    WriteProposal,
    clean_text,
    decision_arguments,
    first_message,
    function_tool,
    tool_choice,
)


WRITE_PROPOSAL_TOOL = function_tool(
    "write_proposal",
    "Name the column to change and the value to write. Never returns a cell address.",
    {
        "status": {"type": "string", "enum": ["ok", "unclear"]},
        "column": {"type": "string"},
        "new_value": {"type": "string"},
        "reason": {"type": "string"},
    },
    ["status", "column", "new_value"],
)

SYSTEM_PROMPT = """你是写入提案代理。你只提出「改哪一列、改成什么值」，必须调用 write_proposal。
你没有任何写入权限，也不要声称已经改好了；系统会先把你的提案给用户确认。
column 必须逐字复制给定 columns 里的一项，不得改写、翻译、加空格或自己编造列名。
new_value 只写要放进单元格的内容本身，不要带列名、行号、引号或任何解释。
不要指定单元格地址、行号或范围，目标行已经由系统确定。
用户没有明确给出新值、或说不清要改哪一列时，status 填 unclear，column 和 new_value 留空字符串。
new_value 不能以 = + - @ 开头（会被表格当成公式），遇到这种要求直接 unclear。
只改一个值；用户一次要求改多列时，选他最先提到的那一列，其余在 reason 里说明没有包含。"""


class WriterAgent:
    def __init__(self, chat: Callable[..., Awaitable[Any]] | None = None):
        self._chat = chat

    async def _complete(self, messages: list[dict[str, Any]], forced: bool) -> Any:
        choice = tool_choice("write_proposal") if forced else None
        if self._chat is not None:
            return await self._chat(messages, [WRITE_PROPOSAL_TOOL], choice)
        from llm_client import chat_with_fallback
        response, _model = await chat_with_fallback(
            messages,
            tools=[WRITE_PROPOSAL_TOOL],
            tool_choice=choice,
        )
        return response

    async def _ask(self, messages: list[dict[str, Any]]) -> Any:
        try:
            return await self._complete(messages, True)
        except Exception as exc:
            logging.warning("writer retry without tool_choice: %s", exc)
            return await self._complete(messages, False)

    async def run(
        self,
        plan: SchedulePlan,
        located: LocatedResult,
        analysis: AnalysisResult | None,
        row_numbers: list[int],
    ) -> WriteProposal:
        """Deliberately takes no MCP session — this agent cannot execute anything."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps({
                    "question": plan.question,
                    "write_goal": plan.write_goal,
                    "doc_title": located.doc_title,
                    "sheet_title": located.sheet_title,
                    "columns": located.columns,
                    "matched_row_count": len(row_numbers),
                    "matched_rows": row_numbers[:20],
                    "locate_output": clean_text(getattr(analysis, "output", "") or "")[:1500],
                }, ensure_ascii=False),
            },
        ]

        for attempt in range(2):
            response = await self._ask(messages)
            message = first_message(response)
            raw = decision_arguments(message, "write_proposal")
            proposal = self._normalize(raw)
            if proposal is not None:
                logging.info(
                    "agent=writer status=%s attempt=%s column=%s",
                    proposal.status,
                    attempt + 1,
                    proposal.column[:40],
                )
                return proposal
            messages.append({"role": "assistant", "content": str(getattr(message, "content", "") or "")})
            messages.append({
                "role": "user",
                "content": "上一次没有给出合法提案。请严格调用 write_proposal，column 必须逐字复制 columns 里的一项。",
            })

        return WriteProposal(status="error", reason="写入提案模型没有返回合法结果")

    @staticmethod
    def _normalize(raw: dict[str, Any] | None) -> WriteProposal | None:
        if not isinstance(raw, dict):
            return None
        status = clean_text(raw.get("status"))
        if status not in {"ok", "unclear"}:
            return None
        column = str(raw.get("column") or "")
        new_value = raw.get("new_value")
        if not isinstance(new_value, str):
            new_value = "" if new_value is None else str(new_value)
        if status == "ok" and (not column.strip() or not new_value.strip()):
            return None
        return WriteProposal(
            status=status,
            column=column.strip(),
            new_value=new_value,
            reason=clean_text(raw.get("reason")),
        )


writer_agent = WriterAgent()


async def run(
    plan: SchedulePlan,
    located: LocatedResult,
    analysis: AnalysisResult | None,
    row_numbers: list[int],
) -> WriteProposal:
    return await writer_agent.run(plan, located, analysis, row_numbers)
