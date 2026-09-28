"""Final, tool-free answer writer."""

from __future__ import annotations

import json
from typing import Any, Awaitable, Callable

from agent_types import AnalysisResult, LocatedResult, SchedulePlan, clip_text, first_message


SYSTEM_PROMPT = """你是答复代理。只使用材料中明确给出的事实，材料没有的数字不要写。
文档标题和子表名必须与材料逐字相同。不要描述代理、工具、代码或内部流程。
不要写记忆确认句，主程序会统一添加。统计失败时明确说明没有算出来，不得根据样例补数字。
材料标记 truncated=true 时，必须说明展示的是前若干行，不是全表。"""


class AnswerAgent:
    def __init__(self, chat: Callable[..., Awaitable[Any]] | None = None):
        self._chat = chat

    async def _complete(self, messages: list[dict[str, Any]]) -> Any:
        if self._chat is not None:
            return await self._chat(messages, None, None)
        from llm_client import chat_with_fallback
        response, _model = await chat_with_fallback(messages, tools=None, tool_choice=None)
        return response

    async def run(
        self,
        user_text: str,
        plan: SchedulePlan,
        located: LocatedResult,
        analysis: AnalysisResult | None,
    ) -> str:
        partial_notes: list[str] = []
        # 分析成功时，统计是在分析工具自己读取的数据上完成的；定位阶段只读了
        # 几行预览，它的“只读取了前 N 行”不代表统计范围，不能再提示给用户。
        analysis_ok = plan.intent == "analyze" and analysis is not None and analysis.status == "ok"
        if analysis_ok:
            truncated = bool(analysis.partial)
            note = analysis.partial_note if analysis.partial else ""
        else:
            truncated = located.truncated
            note = located.note
        if not analysis_ok and located.truncated and located.note:
            partial_notes.append(located.note)
        if analysis is not None and analysis.partial and analysis.partial_note:
            partial_notes.append(analysis.partial_note)
        material: dict[str, Any] = {
            "user_text": user_text,
            "question": plan.question,
            "doc_title": located.doc_title,
            "sheet_title": located.sheet_title,
            "columns": located.columns,
            "read_range": located.read_range,
            "truncated": truncated,
            "note": note,
            "partial_notes": partial_notes,
        }
        if plan.intent == "analyze":
            material["analysis"] = {
                "status": analysis.status if analysis else "error",
                "output": clip_text(analysis.output if analysis else "没有分析结果", 4000),
                "partial": bool(analysis and analysis.partial),
                "partial_note": analysis.partial_note if analysis else "",
            }
        else:
            material["rows"] = located.rows[:20]

        response = await self._complete([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(material, ensure_ascii=False)},
        ])
        message = first_message(response)
        text = str(getattr(message, "content", "") or "").strip()
        for note in partial_notes:
            if note and note not in text:
                text = (text + "\n\n" + note).strip()
        return text


answer_agent = AnswerAgent()


async def run(user_text: str, plan: SchedulePlan, located: LocatedResult, analysis: AnalysisResult | None) -> str:
    return await answer_agent.run(user_text, plan, located, analysis)
