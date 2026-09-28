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

from agent_types import (
    AnalysisResult,
    LocatedResult,
    SchedulePlan,
    chat_timeout_message,
    chat_timeout_seconds,
    clean_text,
    extract_text,
    mcp_init_timeout_seconds,
)
from answer_agent import answer_agent
from data_analyst_agent import data_analyst_agent
from doc_locator_agent import doc_locator_agent
from scheduler_agent import scheduler_agent


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or not str(value).strip():
        return default
    return str(value)


DB_FILE = _env("CHAT_DB_FILE", "/home/ubuntu/tencent-docs-web/chat_history.db")
MCP_SERVER_SCRIPT = _env("MCP_SERVER_SCRIPT", "/home/ubuntu/tencent-docs-mcp/server.py")
MCP_PYTHON_BIN = _env("MCP_PYTHON_BIN", "/home/ubuntu/tencent-docs-mcp/.venv/bin/python")
SQLITE_CONNECT_TIMEOUT_SECONDS = 10.0

SCHEDULER_FAILURE = "这轮没有分清是闲聊还是查表，没有查文档。"
LOCATOR_FAILURE = "文档列表没有取到，这轮没有查表。"
ANALYST_FAILURE = "数据分析没有完成。"
ANSWER_FAILURE = "已经定位到《{doc_title}》的「{sheet_title}」，但没有组织出回答。"


class McpInitTimeout(Exception):
    """MCP 会话初始化超时。

    故意不继承 TimeoutError：避免被 process_chat 里针对整轮对话超时的
    ``except TimeoutError`` 捕获后展示错误的秒数（240 秒 vs 实际 30 秒）。
    """

    def __init__(self, seconds: float):
        super().__init__(f"MCP 会话初始化超时（超过 {seconds:g} 秒）")
        self.seconds = seconds

    def user_message(self) -> str:
        return f"连接文档服务超时（超过 {self.seconds:g} 秒），请稍后重试。"


def configure_logging() -> None:
    """配置根日志（只在 __main__ 启动时调用）。"""
    level_name = str(os.getenv("LOG_LEVEL", "INFO") or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )


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
    try:
        parent = os.path.dirname(DB_FILE)
        if parent:
            os.makedirs(parent, exist_ok=True)
        conn = sqlite3.connect(DB_FILE, timeout=SQLITE_CONNECT_TIMEOUT_SECONDS)
    except Exception:
        logging.exception("写入聊天记录失败")
        return
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
    except Exception:
        logging.exception("写入聊天记录失败")
    finally:
        try:
            conn.close()
        except Exception:
            logging.exception("关闭聊天记录连接失败")


def _safe_log_to_db(user_text: str, assistant_text: str) -> None:
    """写库失败也不丢掉正常回复。"""
    try:
        log_to_db(user_text, assistant_text)
    except Exception:
        logging.exception("写入聊天记录失败")


@asynccontextmanager
async def _open_mcp_session() -> AsyncIterator[Any]:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    server_params = StdioServerParameters(
        command=MCP_PYTHON_BIN,
        args=[MCP_SERVER_SCRIPT],
        env=dict(os.environ),
    )
    init_timeout = mcp_init_timeout_seconds()
    # 超时/取消时 async with 会自动关闭 MCP 子进程。
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            try:
                await asyncio.wait_for(session.initialize(), init_timeout)
            except (asyncio.TimeoutError, TimeoutError) as exc:
                logging.warning("MCP 会话初始化超时（超过 %s 秒）", init_timeout)
                raise McpInitTimeout(init_timeout) from exc
            yield session


def _history_messages(history: Any) -> tuple[list[dict[str, str]], str]:
    messages: list[dict[str, str]] = []
    for item in history or []:
        if isinstance(item, dict):
            role = item.get("role", "user")
            if role in {"user", "assistant"}:
                text = extract_text(item.get("content", ""))
                messages.append({"role": role, "content": normalize_text(text)})
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            if item[0] not in (None, ""):
                messages.append({"role": "user", "content": normalize_text(extract_text(item[0]))})
            if item[1] not in (None, ""):
                messages.append({"role": "assistant", "content": normalize_text(extract_text(item[1]))})
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
        available = [item for item in (located.available or []) if item][:10]
        if available:
            listing = "、".join(f"《{item}》" for item in available)
            return f"没有找到你说的“{plan.doc_hint}”。现有表格有：{listing}。请告诉我要查哪一份。"
        return f"在最多 100 份表格里没有找到你说的“{plan.doc_hint}”。请确认文档名或提供更完整的标题。"
    # error 状态：只展示友好 note，绝不展示异常原文。
    if located.note:
        return located.note
    return LOCATOR_FAILURE


async def process_chat(user_input: Any, history: Any) -> str:
    try:
        return await asyncio.wait_for(_process_chat_inner(user_input, history), chat_timeout_seconds())
    except McpInitTimeout as exc:
        logging.warning("MCP 会话初始化超时（超过 %s 秒）", exc.seconds)
        message = exc.user_message()
        prior_text = normalize_text(extract_text(user_input))
        _safe_log_to_db(prior_text, message)
        return message
    except (asyncio.TimeoutError, TimeoutError):
        logging.warning("整轮对话超时，已停止本次查询")
        prior_text = normalize_text(extract_text(user_input))
        _safe_log_to_db(prior_text, chat_timeout_message())
        return chat_timeout_message()
    except asyncio.CancelledError:
        raise
    except Exception:
        logging.exception("process_chat failed")
        return "抱歉，这次处理出错了，请稍后重试。"


async def _process_chat_inner(user_input: Any, history: Any) -> str:
    runtime = _get_runtime()
    prior, _ = _history_messages(history)
    clean_user_input = normalize_text(extract_text(user_input))

    try:
        mem = await runtime.memory_agent.run(clean_user_input, prior[-6:])
    except Exception:
        logging.exception("memory agent failed")
        mem = SimpleNamespace(handled=False, updated=False, reply="")

    if mem.handled:
        _safe_log_to_db(user_input, mem.reply)
        return mem.reply

    memory_note = str(mem.reply or "").strip()

    def deliver(text: str) -> str:
        body = str(text or "").strip()
        if memory_note:
            body = body.replace(memory_note, "").strip()
            body = memory_note + ("\n\n" + body if body else "")
        _safe_log_to_db(user_input, body)
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
                if plan.intent == "analyze" and analysis and getattr(analysis, "partial_note", ""):
                    if analysis.partial_note not in text:
                        text += "\n\n" + analysis.partial_note
            return deliver(text)
    except (asyncio.TimeoutError, TimeoutError, asyncio.CancelledError):
        # 超时/取消：让外层统一返回超时文案；async with 已关闭 MCP 子进程。
        raise
    except McpInitTimeout:
        # 初始化超时秒数与整轮超时不同，交给 process_chat 展示真实秒数。
        raise
    except Exception:
        logging.exception("MCP session failed")
        return deliver(LOCATOR_FAILURE)


def chat_interface(message, history):
    return asyncio.run(process_chat(message, history))


if __name__ == "__main__":
    from custom_ui import build_ui, custom_css

    configure_logging()
    demo = build_ui(chat_interface)
    demo.queue(default_concurrency_limit=2)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        root_path="/docs-chat",
        share=False,
        css=custom_css,
    )
