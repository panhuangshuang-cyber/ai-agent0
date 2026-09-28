"""Read-only document and sheet locator."""

from __future__ import annotations

import json
import logging
from typing import Any, Awaitable, Callable, Iterable

from agent_types import (
    LOCATOR_SCOPE_REJECTION,
    LOCATOR_TOOL_NAMES,
    LocatedResult,
    SchedulePlan,
    clean_text,
    decision_arguments,
    first_message,
    function_tool,
    is_truncation_note as is_truncation_note_local,
    iter_tool_calls,
    normalize_rows,
    parse_memory_aliases,
    safe_mcp_call,
    tool_call_name,
    tool_choice,
)


LOCATE_DECISION_TOOL = function_tool(
    "locate_decision",
    "Choose one document title from the supplied title list, or report ambiguity/not found.",
    {
        "status": {"type": "string", "enum": ["found", "ambiguous", "not_found"]},
        "doc_title": {"type": "string"},
        "sheet_hint": {"type": "string"},
        "candidates": {"type": "array", "items": {"type": "string"}},
    },
    ["status", "doc_title", "sheet_hint", "candidates"],
)

SYSTEM_PROMPT = """你是文档定位代理。你只根据给定的表格标题列表、用户问题、两个 hint 和记忆别名定位文档。
必须调用 locate_decision。不得计算，不得组织最终回答，不得修改任何文档。
doc_title 必须逐字复制标题列表中的完整标题；不能唯一确定时返回 ambiguous 或 not_found，禁止选第一份。
记忆别名用于理解正式名称，但不要改写用户原始 doc_hint。"""


class DocLocatorAgent:
    def __init__(self, chat: Callable[..., Awaitable[Any]] | None = None):
        self._chat = chat

    async def _complete(self, messages: list[dict[str, Any]], forced: bool) -> Any:
        choice = tool_choice("locate_decision") if forced else None
        if self._chat is not None:
            return await self._chat(messages, [LOCATE_DECISION_TOOL], choice)
        from llm_client import chat_with_fallback
        response, _model = await chat_with_fallback(
            messages,
            tools=[LOCATE_DECISION_TOOL],
            tool_choice=choice,
        )
        return response

    async def _ask(self, messages: list[dict[str, Any]]) -> Any:
        try:
            return await self._complete(messages, True)
        except Exception as exc:
            logging.warning("locator retry without tool_choice: %s", exc)
            return await self._complete(messages, False)

    async def run(self, session: Any, plan: SchedulePlan, memory_text: str) -> LocatedResult:
        docs_call = await safe_mcp_call(
            session,
            "list_docs",
            {"folder_id": "/", "limit": 100, "file_type": "sheet", "is_owner": 0},
            LOCATOR_TOOL_NAMES,
            LOCATOR_SCOPE_REJECTION,
        )
        if not docs_call.ok:
            logging.info("agent=locator tool=list_docs attempt=1 ok=false")
            return LocatedResult(status="error", note="文档列表获取失败，请稍后重试。")

        docs = _extract_docs(docs_call.payload)
        titles = [item["title"] for item in docs]
        aliases = parse_memory_aliases(memory_text)
        formal_hint = _formal_hint(plan.doc_hint, aliases)
        preferred = _contains_matches(titles, formal_hint)
        fallback = _contains_matches(titles, plan.doc_hint)
        if len(preferred) > 1 and fallback:
            narrowed = [title for title in preferred if title in fallback]
            deterministic = narrowed or preferred
        else:
            deterministic = preferred if preferred else fallback

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps({
                    "question": plan.question,
                    "doc_hint": plan.doc_hint,
                    "sheet_hint": plan.sheet_hint,
                    "memory": memory_text,
                    "titles": titles,
                }, ensure_ascii=False),
            },
        ]
        if len(deterministic) != 1:
            # 0 个（not_found）或多个（ambiguous）时不需要再问模型，直接返回。
            if len(deterministic) > 1:
                return LocatedResult(status="ambiguous", candidates=deterministic)
            return LocatedResult(status="not_found")

        selected_title = deterministic[0]
        model_sheet_hint = ""
        decision: dict[str, Any] | None = None
        model_attempts = 0
        llm_failed = False
        while model_attempts < 4:
            model_attempts += 1
            try:
                response = await self._ask(messages)
            except Exception:
                logging.warning("定位模型调用失败，使用确定性候选继续", exc_info=True)
                llm_failed = True
                break
            message = first_message(response)
            calls = iter_tool_calls(message)
            wrong_names = [tool_call_name(call) for call in calls if tool_call_name(call) != "locate_decision"]
            if wrong_names:
                logging.info("agent=locator tool=%s attempt=%s rejected=true", wrong_names[0], model_attempts)
                messages.append({"role": "assistant", "content": str(getattr(message, "content", "") or "")})
                messages.append({"role": "user", "content": LOCATOR_SCOPE_REJECTION})
                continue
            decision = decision_arguments(message, "locate_decision")
            if decision is not None:
                break
            messages.append({"role": "assistant", "content": str(getattr(message, "content", "") or "")})
            messages.append({"role": "user", "content": "没有收到合法的 locate_decision 参数，请重新定位。"})

        if decision is None:
            if llm_failed:
                # 模型挂了/超时：已有唯一确定性候选，直接用它继续。
                logging.warning("定位模型不可用，使用确定性候选继续: %s", selected_title)
            else:
                return LocatedResult(status="error", note="定位模型没有返回合法结果")

        if decision is not None:
            model_sheet_hint = clean_text(decision.get("sheet_hint"))
            if clean_text(decision.get("doc_title")) != selected_title:
                return LocatedResult(status="not_found")
        doc = next(item for item in docs if item["title"] == selected_title)

        sheets_call = await safe_mcp_call(
            session,
            "list_sheets",
            {"file_id": doc["id"]},
            LOCATOR_TOOL_NAMES,
            LOCATOR_SCOPE_REJECTION,
        )
        if not sheets_call.ok:
            logging.warning("list_sheets 失败: %s", selected_title)
            return LocatedResult(status="error", doc_title=selected_title, file_id=doc["id"],
                                 note="子表列表获取失败，请稍后重试。")
        sheets = _extract_sheets(sheets_call.payload)
        sheet_titles = [sheet["title"] for sheet in sheets]
        sheet_hint = clean_text(plan.sheet_hint)
        # 模型返回的 sheet_hint：plan 为空时，只有它能命中真实子表才采用
        # （命中 1 张，或精确等于某张子表标题）；否则忽略模型擅自的选择，
        # 保持原有的 ambiguous 行为。plan 非空时同理，只接受精确匹配的细化。
        if not sheet_hint and model_sheet_hint:
            model_candidates = [s for s in sheets if _contains(s["title"], model_sheet_hint)]
            if len(model_candidates) == 1 or _exact_match(sheet_titles, model_sheet_hint):
                sheet_hint = model_sheet_hint
        elif sheet_hint and model_sheet_hint:
            if model_sheet_hint != sheet_hint and _exact_match(sheet_titles, model_sheet_hint):
                sheet_hint = model_sheet_hint

        if not sheet_hint:
            sheet_matches = sheets if len(sheets) == 1 else []
            if len(sheets) > 1:
                return LocatedResult(
                    status="ambiguous",
                    doc_title=selected_title,
                    file_id=doc["id"],
                    candidates=[sheet["title"] for sheet in sheets],
                )
        else:
            sheet_matches = [sheet for sheet in sheets if _contains(sheet["title"], sheet_hint)]

        if len(sheet_matches) > 1:
            return LocatedResult(
                status="ambiguous",
                doc_title=selected_title,
                file_id=doc["id"],
                candidates=[sheet["title"] for sheet in sheet_matches],
            )
        if not sheet_matches:
            return LocatedResult(status="not_found", doc_title=selected_title, file_id=doc["id"])

        sheet = sheet_matches[0]
        row_limit = 4 if plan.intent == "analyze" else 21
        read_call = await safe_mcp_call(
            session,
            "search_and_read_sheet",
            {
                "doc_title": selected_title,
                "sheet_name": sheet["title"],
                "max_rows": row_limit,
                "max_cols": 30,
            },
            LOCATOR_TOOL_NAMES,
            LOCATOR_SCOPE_REJECTION,
        )
        if not read_call.ok:
            read_call = await safe_mcp_call(
                session,
                "read_sheet",
                {
                    "file_id": doc["id"],
                    "sheet_id": sheet["id"],
                    "cell_range": f"A1:AD{row_limit}",
                },
                LOCATOR_TOOL_NAMES,
                LOCATOR_SCOPE_REJECTION,
            )
        if not read_call.ok:
            logging.warning("表格读取失败: %s/%s", selected_title, sheet["title"])
            return LocatedResult(
                status="error",
                doc_title=selected_title,
                file_id=doc["id"],
                sheet_title=sheet["title"],
                sheet_id=sheet["id"],
                note="表格内容读取失败，请稍后重试。",
            )

        payload = read_call.payload if isinstance(read_call.payload, dict) else {"data": read_call.payload}
        columns, rows, payload_truncated = normalize_rows(payload, row_limit, 30)
        row_count = sheet.get("row_count")
        data_rows_returned = max(0, len(rows))
        total_known: int | None = row_count if isinstance(row_count, int) else None
        if total_known is None:
            for key in ("total_rows", "totalRows", "row_count", "rowCount"):
                value = payload.get(key)
                try:
                    if value is not None and str(value) != "":
                        total_known = int(value)
                        break
                except (TypeError, ValueError):
                    continue
        # “收到行数 == 上限”且总数未知时视为可能截断；总数已知时以总数为准；
        # payload 明确说被封顶时永远标记截断。
        at_cap = data_rows_returned >= max(0, row_limit - 1)
        payload_says_capped = is_truncation_note_local(payload.get("note"))
        truncated = bool(
            payload.get("truncated")
            or payload.get("has_more")
            or payload_truncated
            or payload_says_capped
            or (total_known is not None and total_known > data_rows_returned + 1)
            or (total_known is None and at_cap)
        )
        read_range = clean_text(payload.get("read_range") or payload.get("range"))
        if not read_range:
            read_range = f"A1:AD{min(row_limit, len(rows) + 1)}"
        note = clean_text(payload.get("note"))
        if truncated and not note:
            if total_known is not None:
                note = f"只读取了前 {data_rows_returned} 行（共 {total_known} 行），统计结果可能不完整。"
            else:
                note = f"只读取了前 {data_rows_returned} 行，统计结果可能不完整。"

        logging.info(
            "agent=locator intent=%s status=found tool=%s model_attempts=%s",
            plan.intent,
            read_call.name,
            model_attempts,
        )
        return LocatedResult(
            status="found",
            doc_title=selected_title,
            file_id=doc["id"],
            sheet_title=sheet["title"],
            sheet_id=sheet["id"],
            columns=columns,
            read_range=read_range,
            sheet_row_count=row_count,
            rows=rows[:3] if plan.intent == "analyze" else rows[:20],
            truncated=truncated,
            note=note,
        )


def _unwrap_list(payload: Any, keys: Iterable[str]) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            nested = _unwrap_list(value, keys)
            if nested:
                return nested
    for value in payload.values():
        if isinstance(value, list) and all(isinstance(item, dict) for item in value):
            return value
        if isinstance(value, dict):
            nested = _unwrap_list(value, keys)
            if nested:
                return nested
    return []


def _extract_docs(payload: Any) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for item in _unwrap_list(payload, ("list", "docs", "documents", "files", "items", "data")):
        if not isinstance(item, dict):
            continue
        raw_title = item.get("title") or item.get("name") or item.get("doc_title")
        raw_id = item.get("id") or item.get("file_id") or item.get("fileId")
        title = str(raw_title) if raw_title is not None else ""
        file_id = str(raw_id) if raw_id is not None else ""
        if title and file_id:
            result.append({"title": title, "id": file_id})
    return result


def _extract_sheets(payload: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in _unwrap_list(payload, ("list", "sheets", "worksheets", "items", "data")):
        if not isinstance(item, dict):
            continue
        raw_title = item.get("title") or item.get("name") or item.get("sheet_name")
        raw_id = item.get("id") or item.get("sheet_id") or item.get("sheetId")
        title = str(raw_title) if raw_title is not None else ""
        sheet_id = str(raw_id) if raw_id is not None else ""
        count = item.get("row_count", item.get("rowCount"))
        try:
            count = int(count) if count is not None and count != "" else None
        except (TypeError, ValueError):
            count = None
        if title and sheet_id:
            result.append({"title": title, "id": sheet_id, "row_count": count})
    return result


def _formal_hint(raw_hint: str, aliases: dict[str, str]) -> str:
    hint = clean_text(raw_hint)
    if hint in aliases:
        return aliases[hint]
    contained = [(key, value) for key, value in aliases.items() if key and key in hint]
    if not contained:
        return hint
    contained.sort(key=lambda pair: len(pair[0]), reverse=True)
    return contained[0][1]


def _contains(title: str, hint: str) -> bool:
    """hint 是 title 的子串（或精确相等）才算命中。

    精确（大小写/空白归一化后相等）直接命中；除此之外只允许
    “hint 是 title 的子串”（title 包含 hint）。title 仅仅是 hint
    的子串（例如 title="销售"、hint="销售存档"）不算命中。
    """
    title_norm = clean_text(title).casefold()
    hint_norm = clean_text(hint).casefold()
    if not hint_norm or not title_norm:
        return False
    if title_norm == hint_norm:
        return True
    return hint_norm in title_norm


def _exact_match(titles: list[str], hint: str) -> str | None:
    hint_norm = clean_text(hint).casefold()
    if not hint_norm:
        return None
    for title in titles:
        if clean_text(title).casefold() == hint_norm:
            return title
    return None


def _contains_matches(titles: list[str], hint: str) -> list[str]:
    hint_norm = clean_text(hint).casefold()
    if not hint_norm:
        return []
    exact = _exact_match(titles, hint)
    if exact is not None:
        return [exact]
    return [title for title in titles if _contains(title, hint)]


doc_locator_agent = DocLocatorAgent()


async def run(session: Any, plan: SchedulePlan, memory_text: str) -> LocatedResult:
    return await doc_locator_agent.run(session, plan, memory_text)
