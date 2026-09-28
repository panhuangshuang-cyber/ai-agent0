"""Four-agent orchestration for Tencent Docs chat.

This file contains the replacement for the old 15-turn all-tools loop. The UI hook
keeps the public ``chat_interface(message, history)`` signature.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import logging
import os
import sqlite3
from types import SimpleNamespace
from typing import Any, AsyncIterator, Callable

from agent_types import AnalysisResult, LocatedResult, SchedulePlan, clean_text
from answer_agent import answer_agent
from data_analyst_agent import data_analyst_agent
from doc_locator_agent import doc_locator_agent
from scheduler_agent import scheduler_agent


DB_FILE = "/home/ubuntu/tencent-docs-web/chat_history.db"
MCP_SERVER_SCRIPT = "/home/ubuntu/tencent-docs-mcp/server.py"
MCP_PYTHON_BIN = "/home/ubuntu/tencent-docs-mcp/.venv/bin/python"

SCHEDULER_FAILURE = "这轮没有分清是闲聊还是查表，没有查文档。"
LOCATOR_FAILURE = "文档列表没有取到，这轮没有查表。"
ANALYST_FAILURE = "数据分析没有完成。"
ANSWER_FAILURE = "已经定位到《{doc_title}》的「{sheet_title}」，但没有组织出回答。"


@dataclass
class Runtime:
    memory_agent: Any
    format_memory_prompt: Callable[[], str]
    scheduler: Any = scheduler_agent
    locator: Any = doc_locator_agent
    analyst: Any = data_analyst_agent
    answer: Any = answer_agent
    open_mcp_session: Callable[[], Any] | None = None


_runtime: Runtime | None = None


def configure_runtime(runtime: Runtime | None) -> None:
    """Explicit dependency injection for tests and local integration."""
    global _runtime
    _runtime = runtime


def _default_runtime() -> Runtime:
    from memory_agent import format_memory_prompt, memory_agent
    return Runtime(
        memory_agent=memory_agent,
        format_memory_prompt=format_memory_prompt,
        open_mcp_session=_open_mcp_session,
    )


def _get_runtime() -> Runtime:
    return _runtime or _default_runtime()


def normalize_text(value: Any) -> str:
    return clean_text(value)


def log_to_db(user_text: str, assistant_text: str) -> None:
    parent = os.path.dirname(DB_FILE)
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS chat_history ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "user_message TEXT NOT NULL, bot_response TEXT NOT NULL, "
            "created_at DATETIME DEFAULT CURRENT_TIMESTAMP)"
        )
        columns = {row[1] for row in conn.execute("PRAGMA table_info(chat_history)")}
        candidate_pairs = (
            ("user_message", "bot_response"),
            ("user_input", "bot_response"),
            ("user_text", "assistant_text"),
            ("question", "answer"),
        )
        pair = next((item for item in candidate_pairs if set(item) <= columns), None)
        if pair is None:
            raise RuntimeError("chat_history 表缺少可识别的问答列")
        conn.execute(
            f"INSERT INTO chat_history ({pair[0]}, {pair[1]}) VALUES (?, ?)",
            (normalize_text(user_text), assistant_text),
        )
        conn.commit()
    finally:
        conn.close()


@asynccontextmanager
async def _open_mcp_session() -> AsyncIterator[Any]:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    server_params = StdioServerParameters(
        command=MCP_PYTHON_BIN,
        args=[MCP_SERVER_SCRIPT],
        env=dict(os.environ),
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


def _history_messages(history: Any) -> tuple[list[dict[str, str]], str]:
    messages: list[dict[str, str]] = []
    for item in history or []:
        if isinstance(item, dict):
            role = item.get("role", "user")
            if role in {"user", "assistant"}:
                messages.append({"role": role, "content": normalize_text(item.get("content", ""))})
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            if item[0]:
                messages.append({"role": "user", "content": normalize_text(item[0])})
            if item[1]:
                messages.append({"role": "assistant", "content": normalize_text(item[1])})
    return messages, ""


def _location_failure_text(located: LocatedResult, plan: SchedulePlan) -> str:
    if located.status == "ambiguous":
        candidates = "、".join(f"《{item}》" for item in located.candidates)
        if located.doc_title:
            return f"《{located.doc_title}》里有多张可能的子表：{candidates}。请告诉我要查哪一张。"
        return f"找到多份可能的表格：{candidates}。请告诉我要查哪一份。"
    if located.status == "not_found":
        if located.doc_title and plan.sheet_hint:
            return f"在《{located.doc_title}》里没有找到你说的子表“{plan.sheet_hint}”。请确认子表名。"
        return f"在最多 100 份表格里没有找到你说的“{plan.doc_hint}”。请确认文档名或提供更完整的标题。"
    return LOCATOR_FAILURE


async def process_chat(user_input: Any, history: Any) -> str:
    runtime = _get_runtime()
    prior, _ = _history_messages(history)
    clean_user_input = normalize_text(user_input)

    try:
        mem = await runtime.memory_agent.run(clean_user_input, prior[-6:])
    except Exception:
        logging.exception("memory agent failed")
        mem = SimpleNamespace(handled=False, updated=False, reply="")

    if mem.handled:
        log_to_db(user_input, mem.reply)
        return mem.reply

    memory_note = str(mem.reply or "").strip()

    def deliver(text: str) -> str:
        body = str(text or "").strip()
        if memory_note:
            body = body.replace(memory_note, "").strip()
            body = memory_note + ("\n\n" + body if body else "")
        log_to_db(user_input, body)
        return body

    memory_text = runtime.format_memory_prompt()
    try:
        plan: SchedulePlan = await runtime.scheduler.run(clean_user_input, prior[-6:], memory_text)
    except Exception:
        logging.exception("scheduler agent failed")
        return deliver(SCHEDULER_FAILURE)

    logging.info("agent=scheduler intent=%s", plan.intent)
    if plan.intent in {"chat", "clarify"}:
        return deliver(plan.reply or SCHEDULER_FAILURE)

    opener = runtime.open_mcp_session or _open_mcp_session
    try:
        async with opener() as session:
            try:
                located: LocatedResult = await runtime.locator.run(session, plan, memory_text)
            except Exception:
                logging.exception("locator agent failed")
                return deliver(LOCATOR_FAILURE)

            logging.info("agent=locator intent=%s status=%s", plan.intent, located.status)
            if located.status != "found":
                return deliver(_location_failure_text(located, plan))

            analysis: AnalysisResult | None = None
            if plan.intent == "analyze":
                try:
                    analysis = await runtime.analyst.run(session, plan, located)
                except Exception:
                    logging.exception("analyst agent failed")
                    return deliver(ANALYST_FAILURE)

            try:
                text = await runtime.answer.run(clean_user_input, plan, located, analysis)
            except Exception:
                logging.exception("answer agent failed")
                text = ""

            if not text:
                text = ANSWER_FAILURE.format(
                    doc_title=located.doc_title,
                    sheet_title=located.sheet_title,
                )
                if plan.intent == "analyze" and analysis and analysis.output:
                    text += "\n\n" + analysis.output[:4000]
            return deliver(text)
    except Exception:
        logging.exception("MCP session failed")
        return deliver(LOCATOR_FAILURE)


def chat_interface(message, history):
    return asyncio.run(process_chat(message, history))


if __name__ == "__main__":
    from custom_ui import build_ui, custom_css

    demo = build_ui(chat_interface)
    demo.queue(default_concurrency_limit=2)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        root_path="/docs-chat",
        share=False,
        css=custom_css,
    )
