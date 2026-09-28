"""Shared data structures and safety helpers for the four-agent pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import ast
import json
import re
from typing import Any, Iterable, Mapping, Sequence


SCHEDULER_INTENTS = frozenset({"chat", "clarify", "lookup", "analyze"})
LOCATOR_TOOL_NAMES = frozenset({
    "list_docs",
    "list_sheets",
    "read_sheet",
    "search_and_read_sheet",
})
ANALYST_TOOL_NAMES = frozenset({"analyze_sheet_pandas"})
MEMORY_TOOL_NAMES = frozenset({"update_memory_rule", "memory_decision"})

LOCATOR_SCOPE_REJECTION = "这个工具不在文档定位的范围内"
ANALYST_SCOPE_REJECTION = "这个工具不在数据分析的范围内"
INVALID_JSON_REJECTION = "工具参数不是合法 JSON"

FORBIDDEN_CODE_FRAGMENTS = (
    "import",
    "__",
    "open(",
    "exec(",
    "eval(",
    "compile(",
    "globals(",
    "locals(",
    "getattr(",
    "setattr(",
    "input(",
    "breakpoint(",
    "os",
    "sys",
    "subprocess",
    "socket",
    "pathlib",
    "shutil",
    "requests",
    "pickle",
    "ctypes",
)


@dataclass
class SchedulePlan:
    intent: str
    reply: str = ""
    question: str = ""
    doc_hint: str = ""
    sheet_hint: str = ""
    calc_goal: str = ""


@dataclass
class LocatedResult:
    status: str
    doc_title: str = ""
    file_id: str = ""
    sheet_title: str = ""
    sheet_id: str = ""
    columns: list[str] = field(default_factory=list)
    read_range: str = ""
    sheet_row_count: int | None = None
    rows: list[list[Any]] = field(default_factory=list)
    truncated: bool = False
    candidates: list[str] = field(default_factory=list)
    note: str = ""


@dataclass
class AnalysisResult:
    status: str
    doc_title: str = ""
    sheet_title: str = ""
    code: str = ""
    output: str = ""
    attempts: int = 0


@dataclass
class ToolExecution:
    name: str
    called: bool
    ok: bool
    text: str
    payload: Any = None


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def clip_text(value: Any, limit: int) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[:limit]


def history_block(prior_turns: Sequence[Mapping[str, Any]] | None) -> str:
    """Match the memory agent's last-six-messages, 200-character scale."""
    if not prior_turns:
        return "（无）"
    lines: list[str] = []
    for item in list(prior_turns)[-6:]:
        role = "用户" if item.get("role") == "user" else "助手"
        content = clean_text(item.get("content", ""))
        if len(content) > 200:
            content = content[:200] + "…"
        if content:
            lines.append(f"{role}：{content}")
    return "\n".join(lines) or "（无）"


def tool_choice(name: str) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name}}


def function_tool(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


def first_message(response: Any) -> Any:
    return response.choices[0].message


def iter_tool_calls(message: Any) -> list[Any]:
    return list(getattr(message, "tool_calls", None) or [])


def tool_call_name(call: Any) -> str:
    function = getattr(call, "function", None)
    return str(getattr(function, "name", "") or "")


def parse_tool_arguments(call: Any) -> dict[str, Any] | None:
    function = getattr(call, "function", None)
    raw = getattr(function, "arguments", None)
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def decision_arguments(message: Any, expected_name: str) -> dict[str, Any] | None:
    for call in iter_tool_calls(message):
        if tool_call_name(call) == expected_name:
            return parse_tool_arguments(call)
    content = str(getattr(message, "content", "") or "").strip()
    if content.startswith("{") and content.endswith("}"):
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def assistant_tool_message(message: Any) -> dict[str, Any]:
    if hasattr(message, "model_dump"):
        return message.model_dump(exclude_unset=True)
    tool_calls = []
    for call in iter_tool_calls(message):
        tool_calls.append({
            "id": str(getattr(call, "id", "call")),
            "type": "function",
            "function": {
                "name": tool_call_name(call),
                "arguments": getattr(getattr(call, "function", None), "arguments", "{}"),
            },
        })
    return {
        "role": "assistant",
        "content": str(getattr(message, "content", "") or ""),
        "tool_calls": tool_calls,
    }


def tool_result_message(call: Any, name: str, content: str) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": str(getattr(call, "id", "call")),
        "name": name,
        "content": content,
    }


def mcp_result_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, (dict, list)):
        return json.dumps(result, ensure_ascii=False)
    structured = getattr(result, "structuredContent", None)
    if structured is None:
        structured = getattr(result, "structured_content", None)
    if structured is not None:
        return json.dumps(structured, ensure_ascii=False)
    parts: list[str] = []
    for item in list(getattr(result, "content", None) or []):
        if getattr(item, "type", "") == "text":
            parts.append(str(getattr(item, "text", "") or ""))
    return "\n".join(parts)


def parse_json_payload(text: str) -> Any:
    value = str(text or "").strip()
    if not value:
        return {}
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        pass
    start_positions = [p for p in (value.find("{"), value.find("[")) if p >= 0]
    if not start_positions:
        return {"text": value}
    start = min(start_positions)
    for end in range(len(value), start, -1):
        try:
            return json.loads(value[start:end])
        except json.JSONDecodeError:
            continue
    return {"text": value}


async def safe_mcp_call(
    session: Any,
    name: str,
    arguments: dict[str, Any] | None,
    whitelist: frozenset[str],
    scope_rejection: str,
) -> ToolExecution:
    if name in MEMORY_TOOL_NAMES or name not in whitelist:
        return ToolExecution(name=name, called=False, ok=False, text=scope_rejection)
    if arguments is None:
        return ToolExecution(name=name, called=False, ok=False, text=INVALID_JSON_REJECTION)
    try:
        result = await session.call_tool(name, arguments)
    except Exception as exc:
        return ToolExecution(name=name, called=True, ok=False, text=f"工具调用失败：{exc}")
    text = mcp_result_text(result)
    payload = parse_json_payload(text)
    protocol_error = bool(getattr(result, "isError", False) or getattr(result, "is_error", False))
    business_error = False
    if isinstance(payload, dict):
        business_error = bool(payload.get("error")) or payload.get("success") is False
    return ToolExecution(
        name=name,
        called=True,
        ok=not protocol_error and not business_error,
        text=text,
        payload=payload,
    )


def validate_analysis_code(code: Any) -> str:
    text = str(code or "")
    lowered = text.lower()
    for fragment in FORBIDDEN_CODE_FRAGMENTS:
        if fragment in lowered:
            return f"代码包含禁止内容：{fragment}"
    try:
        tree = ast.parse(text, mode="exec")
    except SyntaxError as exc:
        return f"代码语法错误：{exc.msg}"

    has_print_call = False
    forbidden_names = {
        "open", "exec", "eval", "compile", "globals", "locals", "getattr",
        "setattr", "input", "breakpoint", "__import__",
    }
    forbidden_attrs = {
        "read_csv", "read_excel", "read_json", "read_html", "read_pickle",
        "read_parquet", "read_sql", "to_csv", "to_excel", "to_json",
        "to_pickle", "to_parquet", "to_sql",
    }
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return "代码包含禁止内容：import"
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id == "print":
                    has_print_call = True
                if node.func.id in forbidden_names:
                    return f"代码包含禁止调用：{node.func.id}"
            elif isinstance(node.func, ast.Attribute) and node.func.attr in forbidden_attrs:
                return f"代码包含禁止调用：{node.func.attr}"
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return "代码包含禁止内容：__"
    if not has_print_call:
        return "代码必须包含 print 调用"
    return ""


def parse_memory_aliases(memory_text: str) -> dict[str, str]:
    aliases: dict[str, str] = {}
    pattern = re.compile(r"^-\s*当用户提到\s+(.+?)，等同于\s+(.+?)\s*$")
    for line in str(memory_text or "").splitlines():
        match = pattern.match(line.strip())
        if match:
            aliases[clean_text(match.group(1))] = clean_text(match.group(2))
    return aliases


def normalize_rows(data: Any, row_limit: int, col_limit: int) -> tuple[list[str], list[list[Any]], bool]:
    """Normalize common 2-D and list-of-dict sheet payloads."""
    if isinstance(data, dict):
        for key in ("data", "values", "rows"):
            if key in data:
                return normalize_rows(data[key], row_limit, col_limit)
        return [], [], False
    if not isinstance(data, list) or not data:
        return [], [], False

    if all(isinstance(row, dict) for row in data):
        columns: list[str] = []
        for row in data:
            for key in row:
                name = str(key)
                if name not in columns and len(columns) < col_limit:
                    columns.append(name)
        raw_rows = [[row.get(col) for col in columns] for row in data]
        truncated = len(raw_rows) > max(0, row_limit - 1)
        return columns, raw_rows[: max(0, row_limit - 1)], truncated

    matrix = [list(row) if isinstance(row, (list, tuple)) else [row] for row in data]
    matrix = [row[:col_limit] for row in matrix]
    columns = [str(value) for value in matrix[0]]
    body = matrix[1:]
    truncated = len(matrix) > row_limit
    return columns, body[: max(0, row_limit - 1)], truncated


def public_dict(value: Any) -> dict[str, Any]:
    return asdict(value)
