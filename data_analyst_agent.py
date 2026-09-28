"""Restricted pandas analysis agent."""

from __future__ import annotations

import json
import logging
from typing import Any, Awaitable, Callable

from agent_types import (
    ANALYSIS_READ_LIMIT_ROWS,
    ANALYST_SCOPE_REJECTION,
    ANALYST_TOOL_NAMES,
    AnalysisResult,
    LocatedResult,
    SchedulePlan,
    assistant_tool_message,
    clip_text,
    first_message,
    function_tool,
    iter_tool_calls,
    parse_tool_arguments,
    safe_mcp_call,
    tool_call_name,
    tool_choice,
    tool_result_message,
    validate_analysis_code,
)


def _as_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _partial_info(payload: dict[str, Any], located: LocatedResult) -> tuple[bool, str]:
    """分析 payload 是否只读了部分行。"""
    rows_read = _as_int(payload.get("rows_read"))
    total_rows = _as_int(payload.get("total_rows"))
    truncated = bool(payload.get("truncated"))
    if total_rows is None and located.sheet_row_count:
        total_rows = _as_int(located.sheet_row_count)
    if rows_read is None and total_rows is not None and truncated:
        rows_read = min(total_rows, ANALYSIS_READ_LIMIT_ROWS)
    partial = bool(
        truncated
        or (rows_read is not None and total_rows is not None and rows_read < total_rows)
        or (total_rows is not None and total_rows > ANALYSIS_READ_LIMIT_ROWS)
    )
    if not partial:
        return False, ""
    if rows_read is not None and total_rows is not None:
        note = f"注意：表格只读取了前 {rows_read} 行（共 {total_rows} 行），统计结果可能不完整。"
    elif total_rows is not None:
        note = f"注意：表格只读取了前 {ANALYSIS_READ_LIMIT_ROWS} 行（共 {total_rows} 行），统计结果可能不完整。"
    else:
        note = "注意：表格只读取了前部分行，统计结果可能不完整。"
    return True, note


ANALYZE_TOOL = function_tool(
    "analyze_sheet_pandas",
    "Execute restricted pandas code against the already located sheet. Variables df and pd already exist.",
    {
        "doc_title": {"type": "string"},
        "python_code": {"type": "string"},
        "sheet_name": {"type": "string"},
        "file_id": {"type": "string"},
        "sheet_id": {"type": "string"},
    },
    ["doc_title", "python_code", "sheet_name"],
)

SYSTEM_PROMPT = """你是数据分析代理。只为给定文档和子表编写 pandas 计算代码。
变量 df 和 pd 已经存在，不要 import。代码必须 print 最终结果。
必须调用 analyze_sheet_pandas；不要搜索文档，不要修改表格，不要根据样例行心算答案。
收到拒绝或执行错误后，改写代码再试。"""


class DataAnalystAgent:
    def __init__(self, chat: Callable[..., Awaitable[Any]] | None = None):
        self._chat = chat

    async def _complete(self, messages: list[dict[str, Any]], forced: bool) -> Any:
        choice = tool_choice("analyze_sheet_pandas") if forced else None
        if self._chat is not None:
            return await self._chat(messages, [ANALYZE_TOOL], choice)
        from llm_client import chat_with_fallback
        response, _model = await chat_with_fallback(
            messages,
            tools=[ANALYZE_TOOL],
            tool_choice=choice,
        )
        return response

    async def _ask(self, messages: list[dict[str, Any]]) -> Any:
        try:
            return await self._complete(messages, True)
        except Exception as exc:
            logging.warning("analyst retry without tool_choice: %s", exc)
            return await self._complete(messages, False)

    async def run(self, session: Any, plan: SchedulePlan, located: LocatedResult) -> AnalysisResult:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps({
                    "question": plan.question,
                    "calc_goal": plan.calc_goal,
                    "doc_title": located.doc_title,
                    "sheet_title": located.sheet_title,
                    "columns": located.columns,
                    "sample_rows": located.rows[:3],
                }, ensure_ascii=False),
            },
        ]
        attempts = 0
        last_executed_code = ""
        last_output = ""

        while attempts < 3:
            response = await self._ask(messages)
            message = first_message(response)
            calls = iter_tool_calls(message)
            if not calls:
                attempts += 1
                last_output = "数据分析没有调用 analyze_sheet_pandas"
                messages.append({"role": "assistant", "content": str(getattr(message, "content", "") or "")})
                messages.append({"role": "user", "content": last_output})
                continue
            if len(calls) != 1:
                attempts += 1
                last_output = "每次只能调用一次 analyze_sheet_pandas"
                messages.append({"role": "assistant", "content": str(getattr(message, "content", "") or "")})
                messages.append({"role": "user", "content": last_output})
                continue

            call = calls[0]
            name = tool_call_name(call)
            args = parse_tool_arguments(call)
            attempts += 1
            if name != "analyze_sheet_pandas":
                last_output = ANALYST_SCOPE_REJECTION
                messages.append(assistant_tool_message(message))
                messages.append(tool_result_message(call, name, last_output))
                continue
            if args is None:
                last_output = "工具参数不是合法 JSON"
                messages.append(assistant_tool_message(message))
                messages.append(tool_result_message(call, name, last_output))
                continue

            proposed_code = str(args.get("python_code", "") or "")
            rejection = validate_analysis_code(proposed_code)
            if rejection:
                last_output = rejection
                messages.append(assistant_tool_message(message))
                messages.append(tool_result_message(call, name, rejection))
                continue

            last_executed_code = proposed_code
            safe_args = {
                "doc_title": located.doc_title,
                "python_code": last_executed_code,
                "sheet_name": located.sheet_title,
                "file_id": located.file_id,
                "sheet_id": located.sheet_id,
            }
            executed = await safe_mcp_call(
                session,
                name,
                safe_args,
                ANALYST_TOOL_NAMES,
                ANALYST_SCOPE_REJECTION,
            )
            payload = executed.payload if isinstance(executed.payload, dict) else {}
            output = payload.get("code_output") if isinstance(payload, dict) else None
            if output is None:
                output = executed.text
            last_output = clip_text(output, 4000)
            error_value = payload.get("error") if isinstance(payload, dict) else None
            has_error = (
                not executed.ok
                or bool(error_value)
                or last_output.startswith("执行代码出错")
            )
            partial, partial_note = _partial_info(payload, located)
            logging.info("agent=analyst tool=%s attempt=%s ok=%s", name, attempts, not has_error)
            if not has_error:
                return AnalysisResult(
                    status="ok",
                    doc_title=located.doc_title,
                    sheet_title=located.sheet_title,
                    code=last_executed_code,
                    output=last_output,
                    attempts=attempts,
                    partial=partial,
                    partial_note=partial_note,
                )
            messages.append(assistant_tool_message(message))
            messages.append(tool_result_message(call, name, last_output or "分析执行失败"))

        return AnalysisResult(
            status="error",
            doc_title=located.doc_title,
            sheet_title=located.sheet_title,
            code=last_executed_code,
            output=clip_text(last_output, 4000),
            attempts=attempts,
            partial=False,
            partial_note="",
        )


data_analyst_agent = DataAnalystAgent()


async def run(session: Any, plan: SchedulePlan, located: LocatedResult) -> AnalysisResult:
    return await data_analyst_agent.run(session, plan, located)
