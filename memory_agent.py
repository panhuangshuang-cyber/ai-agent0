"""记忆代理。

主对话把用户原话交到这里。这里只处理别名、简称这类长期规则：
要记，就改 memory.json 并给出回复；与记忆无关，就把话交还主对话。
不查腾讯文档，也不调用查表工具。
"""

import json
import logging
import os
import re
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass

import fcntl

SYSTEM_PROMPT = """你是记忆代理，只负责用户的长期说法规则（别名、简称）。
你不查询腾讯文档，不算数据，不回答表格问题。
每轮必须调用 memory_decision。
action 取值：
- ignore：这句不是在要求记住、修改或删除规则
- set：用户明确要把某个说法对应到某个名称
- delete：用户明确要忘掉某条规则
- ask：用户想记，但这句话里说法或对应名称不完整
key 是用户的说法，value 是它实际指向的名称。不要编造用户没说过的对应关系。
如果这句话里除了记规则，还有需要查文档才能回答的问题，把 continue_chat 设为 true。
ask 只在 action 为 ask 时填写，写你要问用户的那一句。"""

MEMORY_TOOL_NAMES = frozenset({"update_memory_rule", "memory_decision"})

MEMORY_DECISION_TOOL = {
    "type": "function",
    "function": {
        "name": "memory_decision",
        "description": "判断这句用户的话要不要写入、修改或删除长期记忆。",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["ignore", "set", "delete", "ask"],
                },
                "key": {"type": "string", "description": "用户的说法、别名或简称"},
                "value": {"type": "string", "description": "这个说法实际指向的名称"},
                "continue_chat": {
                    "type": "boolean",
                    "description": "这句话里是否还有需要查文档才能回答的问题",
                },
                "ask": {"type": "string", "description": "规则不完整时要问用户的一句"},
            },
            "required": ["action"],
        },
    },
}

_STRONG_CUE = re.compile(
    r"请记住|帮我记|给我记|记一下|记下来|记下|删除记忆|删掉.{0,6}记忆|不要再?记|忘掉|"
    r"(?<![我已])记住"
)
_WEAK_CUE = re.compile(r"以后我说|以后提到|就是指|指的就是|别名|简称|等同于|相当于")
_KEY_LIMIT = 80
_VALUE_LIMIT = 200


class MemoryFileError(Exception):
    pass


@dataclass
class MemoryAgentResult:
    handled: bool
    updated: bool
    reply: str


def cue_kind(text: str) -> str | None:
    """强意图才在模型失败时拦住这句话；弱意图失败后退回主对话。"""
    if _STRONG_CUE.search(text):
        return "strong"
    if _WEAK_CUE.search(text):
        return "weak"
    return None


def visible_doc_tools(tools):
    """主对话的工具列表里拿掉写记忆的工具，写入只留在记忆代理。"""
    if not tools:
        return None
    kept = [
        tool for tool in tools
        if tool.get("function", {}).get("name") not in MEMORY_TOOL_NAMES
    ]
    return kept or None


def _clean(value) -> str:
    return " ".join(str(value or "").split()).strip()


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "是"}
    return False


def _history_block(prior_turns) -> str:
    if not prior_turns:
        return "（无）"
    lines = []
    for item in list(prior_turns)[-6:]:
        role = "用户" if item.get("role") == "user" else "助手"
        content = _clean(item.get("content"))
        if len(content) > 200:
            content = content[:200] + "…"
        if content:
            lines.append(f"{role}：{content}")
    return "\n".join(lines) or "（无）"


def _render_memory(memory: dict) -> str:
    if not memory:
        return "（空）"
    return "\n".join(f"- {key} = {value}" for key, value in memory.items())


def decision_from_message(message) -> dict | None:
    tool_calls = getattr(message, "tool_calls", None) or []
    for tool_call in tool_calls:
        function = getattr(tool_call, "function", None)
        if function is None or function.name != "memory_decision":
            continue
        arguments = function.arguments
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return None
        if isinstance(arguments, dict):
            return arguments
    content = _clean(getattr(message, "content", "") or "")
    if not content or content.upper() == "SKIP":
        return {"action": "ignore"}
    fenced = re.sub(r"^```(?:json)?", "", content, flags=re.IGNORECASE).strip()
    fenced = re.sub(r"```$", "", fenced).strip()
    start = fenced.find("{")
    end = fenced.rfind("}")
    if start != -1 and end > start:
        try:
            data = json.loads(fenced[start:end + 1])
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            return data
    if content.upper().startswith("ASK"):
        ask = content[3:].lstrip(" ：:")
        return {"action": "ask", "ask": ask}
    return None


def default_memory_path() -> str:
    configured = os.getenv("MEMORY_FILE")
    if configured:
        return configured
    deployed_dir = "/home/ubuntu/tencent-docs-web"
    if os.path.isdir(deployed_dir):
        return os.path.join(deployed_dir, "memory.json")
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory.json")


class MemoryAgent:
    def __init__(self, path: str | None = None, chat=None):
        self._path = path
        self._chat = chat

    @property
    def path(self) -> str:
        return self._path or default_memory_path()

    def load(self) -> dict:
        try:
            return self._read(strict=False)
        except MemoryFileError:
            logging.error("memory file corrupt: %s", self.path)
            return {}

    def format_prompt(self) -> str:
        memory = self.load()
        if not memory:
            return ""
        lines = "\n".join(f"- 当用户提到 {key}，等同于 {value}" for key, value in memory.items())
        return (
            "\n\n【专属长期记忆】\n用户之前设定了以下规则，请在理解意图时必须遵守：\n"
            + lines
            + "\n"
        )

    async def run(self, user_text: str, prior_turns=None) -> MemoryAgentResult:
        user_text = _clean(user_text)
        kind = cue_kind(user_text)
        if not kind:
            return MemoryAgentResult(False, False, "")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"已有记忆：\n{_render_memory(self.load())}\n\n"
                    f"最近对话：\n{_history_block(prior_turns)}\n\n"
                    f"用户本轮原话：\n{user_text}"
                ),
            },
        ]
        try:
            response = await self._ask_model(messages)
            decision = decision_from_message(response.choices[0].message)
        except Exception:
            logging.exception("memory agent model call failed")
            if kind == "strong":
                return MemoryAgentResult(True, False, "记忆代理暂时没连上模型，这句话没有写入记忆。")
            return MemoryAgentResult(False, False, "")

        if decision is None:
            if kind == "strong":
                return MemoryAgentResult(
                    True,
                    False,
                    "想记住的话，请说明「哪个说法」指「什么」。例如：记住，以后我说封边条就是指封边条250424。",
                )
            return MemoryAgentResult(False, False, "")

        result = self._commit(decision)
        logging.info(
            "memory agent cue=%s action=%s updated=%s handled=%s",
            kind,
            decision.get("action"),
            result.updated,
            result.handled,
        )
        return result

    async def _ask_model(self, messages):
        tool_choice = {"type": "function", "function": {"name": "memory_decision"}}
        try:
            return await self._complete(messages, tool_choice)
        except Exception as exc:
            logging.warning("memory agent retry without tool_choice: %s", exc)
            return await self._complete(messages, None)

    async def _complete(self, messages, tool_choice):
        if self._chat is not None:
            return await self._chat(messages, [MEMORY_DECISION_TOOL], tool_choice)
        from llm_client import chat_with_fallback

        response, _model = await chat_with_fallback(
            messages,
            tools=[MEMORY_DECISION_TOOL],
            tool_choice=tool_choice,
        )
        return response

    def _commit(self, decision: dict) -> MemoryAgentResult:
        action = _clean(decision.get("action")).lower()
        if action == "ignore":
            return MemoryAgentResult(False, False, "")
        if action == "ask":
            ask = _clean(decision.get("ask")) or "想记住的话，请说明「哪个说法」指「什么」。"
            return MemoryAgentResult(True, False, ask[:300])
        if action not in {"set", "delete"}:
            return MemoryAgentResult(False, False, "")
        try:
            with self._locked():
                memory = self._read(strict=True)
                return self._apply(memory, decision, action)
        except MemoryFileError:
            return MemoryAgentResult(True, False, "记忆文件内容损坏，这次没有写入，避免盖掉原文件。")
        except OSError:
            logging.exception("memory write failed")
            return MemoryAgentResult(True, False, "记忆文件没有写成功。")

    def _apply(self, memory: dict, decision: dict, action: str) -> MemoryAgentResult:
        continue_chat = _as_bool(decision.get("continue_chat"))
        key = _clean(decision.get("key"))
        value = _clean(decision.get("value"))
        if not key:
            return MemoryAgentResult(True, False, "还差要记住的说法。请说明「哪个说法」指「什么」。")
        if len(key) > _KEY_LIMIT or len(value) > _VALUE_LIMIT:
            return MemoryAgentResult(True, False, "这条规则太长，没有写入。说法和含义请各用一个短名称。")

        if action == "delete":
            if key not in memory:
                return MemoryAgentResult(not continue_chat, False, f"记忆里没有「{key}」，没有改动。")
            old = memory.pop(key)
            self._write(memory)
            return MemoryAgentResult(not continue_chat, True, f"已忘记「{key}」（原先指「{old}」）。")

        if not value:
            return MemoryAgentResult(True, False, f"「{key}」要指什么？请补上对应的名称。")
        if key == value:
            return MemoryAgentResult(True, False, f"「{key}」和它指向的名称相同，没有写入。")
        old = memory.get(key)
        if old == value:
            reply = f"已经记得「{key}」就是「{value}」，没有改动。"
            return MemoryAgentResult(not continue_chat, False, reply)
        memory[key] = value
        self._write(memory)
        if old:
            reply = f"已更新记忆：「{key}」从「{old}」改为「{value}」。"
        else:
            reply = f"已记住：「{key}」是指「{value}」。"
        return MemoryAgentResult(not continue_chat, True, reply)

    @contextmanager
    def _locked(self):
        directory = os.path.dirname(self.path) or "."
        os.makedirs(directory, exist_ok=True)
        lock_path = self.path + ".lock"
        with open(lock_path, "a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _read(self, strict: bool) -> dict:
        path = self.path
        if not os.path.exists(path):
            return {}
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
        except json.JSONDecodeError as exc:
            if strict:
                raise MemoryFileError(path) from exc
            logging.error("memory file corrupt: %s", path)
            return {}
        except OSError:
            if strict:
                raise
            logging.exception("memory file unreadable: %s", path)
            return {}
        if not isinstance(data, dict):
            if strict:
                raise MemoryFileError(path)
            return {}
        return {
            str(key): str(value)
            for key, value in data.items()
            if isinstance(key, str) and isinstance(value, str)
        }

    def _write(self, memory: dict) -> None:
        directory = os.path.dirname(self.path) or "."
        fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(memory, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            os.replace(tmp, self.path)
        except Exception:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise


def format_memory_prompt() -> str:
    return memory_agent.format_prompt()


memory_agent = MemoryAgent()
