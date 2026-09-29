"""Four-agent orchestration for Tencent Docs chat.

This file contains the replacement for the old 15-turn all-tools loop. The UI hook
keeps the public ``chat_interface(message, history)`` signature.
"""

from __future__ import annotations

import ast
import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
import json
import logging
import os
import re
import secrets
import sqlite3
import threading
import time
from types import SimpleNamespace
from typing import Any, AsyncIterator, Callable

from agent_types import (
    ANALYSIS_READ_LIMIT_ROWS,
    AnalysisResult,
    CONFIRM_AFFIRMATIVES,
    FORMULA_VALUE_PREFIXES,
    LocatedResult,
    PENDING_WRITE_TTL_SECONDS,
    SchedulePlan,
    WRITER_SCOPE_REJECTION,
    WRITER_TOOL_NAMES,
    WRITE_MAX_CELLS,
    WRITE_MAX_COLS,
    WRITE_MAX_ROWS,
    WRITE_TOKEN_RE,
    WriteSpec,
    chat_timeout_message,
    chat_timeout_seconds,
    clean_text,
    col_letter,
    extract_text,
    mcp_init_timeout_seconds,
    safe_mcp_call,
)
from answer_agent import answer_agent
from data_analyst_agent import data_analyst_agent
from doc_locator_agent import doc_locator_agent
from scheduler_agent import scheduler_agent
from writer_agent import writer_agent


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

WRITE_LOCATE_FAILURE = "没能确定要修改哪一行，这次没有写入，表格保持原样。"
WRITE_NO_MATCH = "没有找到符合条件的行，这次没有做任何修改。"
WRITE_BEYOND_READ_LIMIT = (
    "表格比较长，只读取了前 {rows} 行，这里面没有找到符合条件的行。"
    "为了避免改到看不见的行，这次没有写入。请把定位条件说得更具体一些。"
)
WRITE_NOT_CONTIGUOUS = (
    "匹配到 {count} 行，但它们不连续。为避免覆盖中间无关的单元格，这次没有写入。"
    "请缩小到一行，或一段连续的区间。"
)
WRITE_TOO_MANY_ROWS = "一次要改 {count} 行，超过单次 {limit} 行的上限，这次没有写入。"
WRITE_COUNT_MISMATCH = (
    "匹配到 {count} 行，但定位结果只列出了 {listed} 行，两者对不上。"
    "为避免改错，这次没有写入。请把定位条件说得更精确一些。"
)
WRITE_OUT_OF_GRID = "目标位置 {cell} 超出了表格范围（共 {rows} 行），这次没有写入。"
WRITE_PROPOSAL_FAILURE = "没能整理出要写入的内容，这次没有写入，表格保持原样。"
WRITE_UNCLEAR = (
    "没听清要改哪一列、改成什么值，这次没有写入。"
    "请说清楚一些，例如「把 tx 表里单号 A1 那行的单价改成 88」。"
)
WRITE_COLUMN_NOT_FOUND = "在表头里找不到「{column}」这一列，为避免改错，这次没有写入。现有列：{columns}。"
WRITE_COLUMN_AMBIGUOUS = "表头里「{column}」出现了多次，无法确定改哪一列，这次没有写入。"
WRITE_VALUE_REJECTED = "要写入的值不合适（{reason}），这次没有写入。"
WRITE_VALUE_EMPTY = "新值是空的。清空单元格不在支持范围内，这次没有写入。"
WRITE_VALUE_TOO_LONG = "超过 {limit} 个字符"
WRITE_VALUE_FORMULA = "不支持写入公式，或以 {prefixes} 开头的非数字内容"
WRITE_READ_OLD_FAILURE = "没能读到目标单元格的当前值，为避免盖错内容，这次没有写入。"
WRITE_EXECUTE_FAILURE = "写入没有成功，表格保持原样。请稍后重试，或重新说要改什么。"
WRITE_VERIFY_MISMATCH = (
    "已提交写入《{doc_title}》的「{sheet_title}」{cell}，"
    "但回读到的值是「{actual}」，和预期的「{expected}」不一致，请打开表格核对。"
)
WRITE_VERIFY_UNREAD = "已提交写入《{doc_title}》的「{sheet_title}」{cell}，但没能回读确认，请打开表格核对。"

CONFIRM_EXPIRED = (
    "这次修改的确认已过期（超过 {minutes:g} 分钟），表格没有被改动。请重新说要改什么。"
)
CONFIRM_MISSING = "没有待确认的修改（可能已经执行过，或服务重启过），表格没有被再次改动。"

#: 单元格值的长度上限，防止把整篇文章塞进一个格子。
WRITE_MAX_VALUE_CHARS = 2000
#: 预览里最多逐行列出多少个单元格。
PREVIEW_MAX_LINES = 20
#: 过期确认码的墓碑数量上限。
EXPIRED_TOKEN_MAX = 256

#: 待确认的写入。服务端是唯一权威副本：客户端只能提交确认码，
#: 不能提交写入内容，所以历史消息被篡改也无法改变要写什么。
_PENDING: dict[str, WriteSpec] = {}
#: 已过期/已执行的确认码，用于区分「过期」和「从来没有过」。
_EXPIRED: set[str] = set()
_PENDING_LOCK = threading.RLock()


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
    writer: Any = writer_agent
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


# ---------------------------------------------------------------- 写入路径

#: 沙箱只被要求 print 一个不含嵌套花括号的字典，所以这样扫描就够了。
_MATCH_DICT_RE = re.compile(r"\{[^{}]*\}")
_CONFIRM_PUNCTUATION = "。．.！!？?~～、,，;； 　"


def _derive_write_calc_goal(write_goal: str) -> str:
    """把「改什么」翻译成「先只定位到哪一行」的分析目标。

    复用 analyze_sheet_pandas 是因为它能读满 1 万行，而定位阶段只预览几行；
    生产表有 2000+ 行，靠预览找不到目标行。
    """
    return (
        f"用户的修改要求是：{write_goal}\n"
        "这一步只做定位，不要计算别的，也不要修改任何东西。\n"
        "请从上面这句话里取出「定位到哪一行」的条件去筛选 df，"
        "然后只 print 一个 Python 字典字面量，格式必须是：\n"
        'print({"count": 匹配到的行数, "rows": [每个匹配行的 df 行号, ...]})\n'
        "rows 用 df 的 0 基整数索引，升序，最多列 50 个。"
        "除这一个字典外不要 print 任何内容。环境里没有 json，也不要 import。\n"
        '一行都匹配不到时，print({"count": 0, "rows": []})。'
    )


def _parse_matches(output: Any) -> tuple[int | None, list[int]]:
    """解析沙箱输出里的 {"count": N, "rows": [...]}；解析不出来返回 (None, [])。"""
    text = str(output or "")
    count: int | None = None
    rows: list[int] = []
    for candidate in _MATCH_DICT_RE.findall(text):
        try:
            value = ast.literal_eval(candidate)
        except (ValueError, SyntaxError):
            continue
        if not isinstance(value, dict) or "count" not in value or "rows" not in value:
            continue
        raw_rows = value.get("rows")
        if not isinstance(raw_rows, (list, tuple)):
            continue
        parsed_rows: list[int] = []
        for item in raw_rows:
            try:
                parsed_rows.append(int(item))
            except (TypeError, ValueError):
                continue
        try:
            parsed_count = int(value.get("count"))
        except (TypeError, ValueError):
            parsed_count = len(parsed_rows)
        # 取最后一个合法的：模型可能先 print 了调试信息。
        count, rows = parsed_count, parsed_rows
    return count, rows


def _is_plain_number(value: Any) -> bool:
    try:
        float(str(value).strip())
    except (TypeError, ValueError):
        return False
    return True


def _value_rejection(value: Any) -> str:
    """返回拒绝原因，空串表示这个值可以写。"""
    if not isinstance(value, str) or not value.strip():
        return WRITE_VALUE_EMPTY
    if len(value) > WRITE_MAX_VALUE_CHARS:
        return WRITE_VALUE_TOO_LONG.format(limit=WRITE_MAX_VALUE_CHARS)
    head = value.lstrip()[:1]
    # 负数是合法值，所以 +/- 开头只在它不是纯数字时才算公式注入。
    if head in ("=", "@") or (head in ("+", "-") and not _is_plain_number(value)):
        return WRITE_VALUE_FORMULA.format(prefixes=" ".join(FORMULA_VALUE_PREFIXES))
    return ""


def _resolve_column(column: Any, located: LocatedResult) -> tuple[int | None, str]:
    """把模型给的列名解析成 0 基列号。

    必须走 located.columns（原始表头顺序）而不是 df 的列名：沙箱会丢弃
    「表头为空且整列为空」的列，还会给重名表头加 _2 后缀，df 的列序对不上真实列序。
    """
    columns = [str(item) for item in (located.columns or [])]
    if not columns:
        return None, WRITE_LOCATE_FAILURE
    target = str(column or "").strip()
    if not target:
        return None, WRITE_UNCLEAR
    for matcher in (
        lambda name: name == target,
        lambda name: name.strip().casefold() == target.casefold(),
    ):
        hits = [index for index, name in enumerate(columns) if matcher(name)]
        if len(hits) == 1:
            return hits[0], ""
        if len(hits) > 1:
            return None, WRITE_COLUMN_AMBIGUOUS.format(column=target)
    return None, WRITE_COLUMN_NOT_FOUND.format(
        column=target, columns="、".join(columns[:30]) or "（无）"
    )


def _cell_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _extract_column_values(payload: Any, expected: int) -> list[str] | None:
    """从 read_sheet 的返回里取出单列的值，长度补齐到 expected。"""
    data = payload
    if isinstance(data, dict):
        for key in ("values", "data", "rows"):
            if key in data:
                data = data[key]
                break
    if not isinstance(data, list):
        return None
    flat: list[str] = []
    for row in data:
        if isinstance(row, (list, tuple)):
            flat.append(_cell_text(row[0]) if row else "")
        else:
            flat.append(_cell_text(row))
    flat.extend([""] * max(0, expected - len(flat)))
    return flat[:expected]


def _values_equal(actual: Any, expected: Any) -> bool:
    """数字字符串会被写成 number 类型，回读可能是 88.0，所以按数值兜底比较。"""
    if str(actual) == str(expected):
        return True
    try:
        return float(str(actual)) == float(str(expected))
    except (TypeError, ValueError):
        return False


def _safe_cell(matrix: list[list[str]], index: int) -> str:
    if index < len(matrix) and matrix[index]:
        return str(matrix[index][0])
    return ""


def _is_affirmative(text: str) -> bool:
    stripped = clean_text(text).strip(_CONFIRM_PUNCTUATION)
    if not stripped:
        return False
    if stripped in CONFIRM_AFFIRMATIVES:
        return True
    return len(stripped) <= 8 and stripped.startswith(("确认", "确定", "执行", "写吧", "就这样"))


def _pending_token_in_history(history: Any) -> str | None:
    """只认紧挨着上一轮的助手消息里的确认码。

    这样既把确认码绑定到当前会话（防止别的用户拿自己的「确认」触发别人的写入），
    又避免翻出很早以前的旧提案重放。
    """
    items = list(history or [])
    if not items:
        return None
    item = items[-1]
    if isinstance(item, dict):
        if item.get("role") != "assistant":
            return None
        text = extract_text(item.get("content"))
    elif isinstance(item, (list, tuple)) and len(item) == 2:
        text = extract_text(item[1])
    else:
        return None
    found = WRITE_TOKEN_RE.findall(text)
    return found[-1] if found else None


def _sweep_pending(now: float) -> None:
    with _PENDING_LOCK:
        stale = [token for token, spec in _PENDING.items() if spec.expired(now)]
        for token in stale:
            del _PENDING[token]
            _EXPIRED.add(token)
        while len(_EXPIRED) > EXPIRED_TOKEN_MAX:
            _EXPIRED.pop()


def _store_pending(spec: WriteSpec) -> None:
    _sweep_pending(time.time())
    with _PENDING_LOCK:
        _PENDING[spec.token] = spec


def _build_preview(spec: WriteSpec, notes: list[str]) -> str:
    """确认预览完全由代码拼装：模型不参与描述要写什么，就不会说错范围。"""
    lines = [f"将要修改《{spec.doc_title}》的「{spec.sheet_title}」：", ""]
    for index, row in enumerate(spec.row_numbers[:PREVIEW_MAX_LINES]):
        old = _safe_cell(spec.old_values, index)
        new = _safe_cell(spec.values, index)
        lines.append(f"第 {row} 行「{spec.column}」：「{old}」→「{new}」")
    if len(spec.row_numbers) > PREVIEW_MAX_LINES:
        lines.append(f"……共 {len(spec.row_numbers)} 行，都改成同一个值。")
    for note in notes:
        if note:
            lines.append("")
            lines.append(note)
    lines.append("")
    lines.append(f"确认码 {spec.token}（{PENDING_WRITE_TTL_SECONDS / 60:g} 分钟内有效）")
    lines.append("回复「确认」就写入；回复别的内容则不会改动表格。")
    return "\n".join(lines)


async def _execute_confirmed_write(spec: WriteSpec, runtime: Runtime) -> str:
    opener = runtime.open_mcp_session or _open_mcp_session
    logging.info(
        "write execute token=%s doc=%s sheet=%s range=%s rows=%s",
        spec.token, spec.doc_title, spec.sheet_title, spec.cell_range(), len(spec.row_numbers),
    )
    async with opener() as session:
        executed = await safe_mcp_call(
            session,
            "write_sheet",
            {
                "file_id": spec.file_id,
                "sheet_id": spec.sheet_id,
                "start_cell": spec.start_cell,
                "values": spec.values,
            },
            WRITER_TOOL_NAMES,
            WRITER_SCOPE_REJECTION,
        )
        if not executed.ok:
            logging.warning(
                "write failed token=%s detail=%s", spec.token, clean_text(executed.text)[:200]
            )
            return WRITE_EXECUTE_FAILURE
        verify = await safe_mcp_call(
            session,
            "read_sheet",
            {
                "file_id": spec.file_id,
                "sheet_id": spec.sheet_id,
                "cell_range": spec.cell_range(),
            },
            WRITER_TOOL_NAMES,
            WRITER_SCOPE_REJECTION,
        )

    expected = [_safe_cell(spec.values, index) for index in range(len(spec.row_numbers))]
    if not verify.ok:
        return WRITE_VERIFY_UNREAD.format(
            doc_title=spec.doc_title, sheet_title=spec.sheet_title, cell=spec.cell_range()
        )
    actual = _extract_column_values(verify.payload, len(spec.row_numbers)) or []
    bad = [
        index for index in range(len(expected))
        if not _values_equal(actual[index] if index < len(actual) else "", expected[index])
    ]
    if len(spec.row_numbers) == 1:
        if bad:
            return WRITE_VERIFY_MISMATCH.format(
                doc_title=spec.doc_title,
                sheet_title=spec.sheet_title,
                cell=spec.start_cell,
                actual=actual[0] if actual else "",
                expected=expected[0],
            )
        return (
            f"已写入《{spec.doc_title}》的「{spec.sheet_title}」{spec.start_cell}："
            f"现在是「{_safe_cell(spec.values, 0)}」。"
        )
    if bad:
        rows = "、".join(str(spec.row_numbers[index]) for index in bad[:5])
        return (
            f"已提交写入《{spec.doc_title}》的「{spec.sheet_title}」，"
            f"但第 {rows} 行回读到的值和预期不一致，请打开表格核对。"
        )
    return (
        f"已写入《{spec.doc_title}》的「{spec.sheet_title}」{spec.cell_range()}，"
        f"共 {len(spec.row_numbers)} 行。"
    )


async def _try_confirm(user_text: str, history: Any, runtime: Runtime) -> str | None:
    """确认拦截。返回 None 表示本轮不是确认，交给正常流水线。"""
    token = _pending_token_in_history(history)
    if not token:
        return None
    text = clean_text(user_text)
    if token not in text and not _is_affirmative(text):
        return None
    now = time.time()
    _sweep_pending(now)
    # pop 是原子的：重复确认或并发确认同一个码，只有一个请求拿得到 spec。
    spec = _PENDING.pop(token, None)
    minutes = PENDING_WRITE_TTL_SECONDS / 60
    if spec is None:
        if token in _EXPIRED:
            return CONFIRM_EXPIRED.format(minutes=minutes)
        return CONFIRM_MISSING
    if spec.expired(now):
        with _PENDING_LOCK:
            _EXPIRED.add(token)
        return CONFIRM_EXPIRED.format(minutes=minutes)
    logging.info("write confirm token=%s rows=%s", token, len(spec.row_numbers))
    return await _execute_confirmed_write(spec, runtime)


async def _run_write_branch(
    session: Any,
    runtime: Runtime,
    plan: SchedulePlan,
    locate_plan: SchedulePlan,
    located: LocatedResult,
) -> str:
    """第一轮：定位目标行 → 提案 → 校验 → 存待确认 → 返回预览。不写任何东西。"""
    try:
        analysis = await runtime.analyst.run(session, locate_plan, located)
    except Exception:
        logging.exception("analyst agent failed during write")
        return WRITE_LOCATE_FAILURE
    if analysis is None or analysis.status != "ok":
        logging.info("write locate failed output=%s", clean_text(getattr(analysis, "output", ""))[:200])
        return WRITE_LOCATE_FAILURE

    count, matched = _parse_matches(analysis.output)
    if count is None:
        logging.info("write locate unparsable output=%s", clean_text(analysis.output)[:200])
        return WRITE_LOCATE_FAILURE
    rows = sorted({index for index in matched if index >= 0})
    if count == 0 or not rows:
        if analysis.partial:
            # 只读了前 1 万行时的「没找到」是假阴性，绝不能当成「可以随便写」。
            return WRITE_BEYOND_READ_LIMIT.format(rows=ANALYSIS_READ_LIMIT_ROWS)
        return WRITE_NO_MATCH
    if count != len(rows):
        return WRITE_COUNT_MISMATCH.format(count=count, listed=len(rows))
    # 多行必须是连续区间，否则会盖掉中间无关的单元格。
    if len(rows) > 1 and rows[-1] - rows[0] + 1 != len(rows):
        return WRITE_NOT_CONTIGUOUS.format(count=len(rows))
    if len(rows) > WRITE_MAX_ROWS:
        return WRITE_TOO_MANY_ROWS.format(count=len(rows), limit=WRITE_MAX_ROWS)
    # df 的第 0 行是表格的第 2 行（第 1 行是表头）。
    sheet_rows = [index + 2 for index in rows]

    try:
        proposal = await runtime.writer.run(plan, located, analysis, sheet_rows)
    except Exception:
        logging.exception("writer agent failed")
        return WRITE_PROPOSAL_FAILURE
    if proposal is None or getattr(proposal, "status", "") != "ok":
        logging.info("write proposal rejected reason=%s", clean_text(getattr(proposal, "reason", ""))[:200])
        return WRITE_UNCLEAR

    column_index, rejection = _resolve_column(proposal.column, located)
    if rejection:
        return rejection
    if column_index is None or column_index >= WRITE_MAX_COLS:
        return WRITE_LOCATE_FAILURE
    value_rejection = _value_rejection(proposal.new_value)
    if value_rejection:
        return WRITE_VALUE_REJECTED.format(reason=value_rejection)
    if len(sheet_rows) > WRITE_MAX_CELLS:
        return WRITE_TOO_MANY_ROWS.format(count=len(sheet_rows), limit=WRITE_MAX_CELLS)

    letter = col_letter(column_index)
    start_cell = f"{letter}{sheet_rows[0]}"
    end_cell = f"{letter}{sheet_rows[-1]}"
    grid_rows = 0
    try:
        grid_rows = int(located.sheet_row_count or 0)
    except (TypeError, ValueError):
        grid_rows = 0
    if grid_rows and end_cell and sheet_rows[-1] > grid_rows:
        return WRITE_OUT_OF_GRID.format(cell=end_cell, rows=grid_rows)

    read = await safe_mcp_call(
        session,
        "read_sheet",
        {
            "file_id": located.file_id,
            "sheet_id": located.sheet_id,
            # 单格也必须写成 A1:A1，裸 A1 会被接口判为 Range Validate error。
            "cell_range": f"{start_cell}:{end_cell}",
        },
        WRITER_TOOL_NAMES,
        WRITER_SCOPE_REJECTION,
    )
    if not read.ok:
        logging.warning("write pre-read failed range=%s:%s", start_cell, end_cell)
        return WRITE_READ_OLD_FAILURE
    old_values = _extract_column_values(read.payload, len(sheet_rows))
    if old_values is None:
        return WRITE_READ_OLD_FAILURE

    new_value = str(proposal.new_value)
    spec = WriteSpec(
        token=f"W-{secrets.token_hex(6)}",
        file_id=located.file_id,
        sheet_id=located.sheet_id,
        doc_title=located.doc_title,
        sheet_title=located.sheet_title,
        column=str(located.columns[column_index]),
        start_cell=start_cell,
        end_cell=end_cell,
        values=[[new_value] for _ in sheet_rows],
        old_values=[[item] for item in old_values],
        row_numbers=sheet_rows,
        expires_at=time.time() + PENDING_WRITE_TTL_SECONDS,
    )
    _store_pending(spec)

    notes: list[str] = []
    if analysis.partial:
        notes.append(
            f"注意：{analysis.partial_note or '这张表没有被整张读完'}，"
            "读取范围之外可能还有符合条件的行。"
        )
    stripped_value = new_value.strip()
    if _is_plain_number(new_value) and len(stripped_value) > 1 and stripped_value.startswith("0"):
        notes.append("注意：这个值会被存成数字，前导 0 会丢失。")
    logging.info(
        "write propose token=%s doc=%s sheet=%s range=%s:%s rows=%s",
        spec.token, spec.doc_title, spec.sheet_title, start_cell, end_cell, len(sheet_rows),
    )
    return _build_preview(spec, notes)


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

    # 确认拦截放在记忆代理之前：一句「确认」不该被记忆代理的分支吃掉。
    try:
        confirmed = await _try_confirm(clean_user_input, prior, runtime)
    except (asyncio.TimeoutError, TimeoutError, McpInitTimeout, asyncio.CancelledError):
        raise
    except Exception:
        logging.exception("确认写入失败")
        confirmed = WRITE_EXECUTE_FAILURE
    if confirmed is not None:
        _safe_log_to_db(user_input, confirmed)
        return confirmed

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
    # 写入意图借用 analyze 的定位路径：定位阶段只预览几行，真正的目标行
    # 交给 analyze_sheet_pandas 读满 1 万行去找（生产表有 2000+ 行）。
    locate_plan = plan
    if plan.intent == "write":
        locate_plan = replace(
            plan, intent="analyze", calc_goal=_derive_write_calc_goal(plan.write_goal)
        )
    try:
        async with opener() as session:
            try:
                located: LocatedResult = await runtime.locator.run(session, locate_plan, memory_text)
            except Exception:
                logging.exception("locator agent failed")
                return deliver(LOCATOR_FAILURE)

            logging.info("agent=locator intent=%s status=%s", plan.intent, located.status)
            if located.status != "found":
                return deliver(_location_failure_text(located, plan))

            if plan.intent == "write":
                # 不走答复代理：预览文案由代码拼，保证和真正要执行的内容逐字一致。
                try:
                    text = await _run_write_branch(session, runtime, plan, locate_plan, located)
                except (asyncio.TimeoutError, TimeoutError, McpInitTimeout, asyncio.CancelledError):
                    raise
                except Exception:
                    logging.exception("write branch failed")
                    text = WRITE_LOCATE_FAILURE
                return deliver(text)

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
