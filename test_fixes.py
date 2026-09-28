from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import patch

import app
import custom_ui
from agent_types import (
    AnalysisResult,
    LocatedResult,
    SchedulePlan,
    extract_text,
    normalize_rows,
)
from answer_agent import AnswerAgent
from data_analyst_agent import DataAnalystAgent, _partial_info
from doc_locator_agent import DocLocatorAgent, _contains, _contains_matches
import agent_types


class _Fn:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _ToolCall:
    def __init__(self, name, arguments, call_id="call-1"):
        self.function = _Fn(name, arguments)
        self.id = call_id


class _Message:
    def __init__(self, name=None, arguments=None, content=""):
        self.tool_calls = None if name is None else [_ToolCall(name, arguments)]
        self.content = content

    def model_dump(self, exclude_unset=True):
        calls = []
        for call in self.tool_calls or []:
            calls.append({
                "id": call.id,
                "type": "function",
                "function": {"name": call.function.name, "arguments": call.function.arguments},
            })
        return {"role": "assistant", "content": self.content, "tool_calls": calls}


class _Response:
    def __init__(self, message):
        self.choices = [SimpleNamespace(message=message)]


def _decision(name, **kwargs):
    return _Response(_Message(name, json.dumps(kwargs, ensure_ascii=False)))


class _Text:
    type = "text"

    def __init__(self, text):
        self.text = text


class _McpResult:
    def __init__(self, payload, is_error=False):
        self.content = [_Text(json.dumps(payload, ensure_ascii=False))]
        self.isError = is_error


class FakeSession:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        value = self.responses[name]
        if isinstance(value, list):
            value = value.pop(0)
        if callable(value):
            return await value(name, arguments)
        return _McpResult(value)


class GradioRegressionTests(unittest.TestCase):
    def test_extract_text_variants(self):
        self.assertEqual(extract_text("你好"), "你好")
        self.assertEqual(extract_text([{"type": "text", "text": "甲"}, {"type": "text", "text": "乙"}]), "甲乙")
        self.assertEqual(extract_text([{"text": "甲"}, "乙"]), "甲乙")
        self.assertEqual(extract_text({"type": "text", "text": "丙"}), "丙")
        self.assertEqual(extract_text(None), "")
        self.assertEqual(
            custom_ui.extract_text("记住，以后我说测试是指测试文档甲"),
            "记住，以后我说测试是指测试文档甲",
        )
        self.assertEqual(
            custom_ui.extract_text([{"type": "text", "text": "记住，以后我说测试是指测试文档甲"}]),
            "记住，以后我说测试是指测试文档甲",
        )

    def test_history_messages_list_content_not_repr(self):
        history = [
            {"role": "user", "content": [{"type": "text", "text": "查销售表"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "好的"}]},
        ]
        messages, _ = app._history_messages(history)
        self.assertEqual(messages[0]["content"], "查销售表")
        self.assertNotIn("'text'", messages[0]["content"])

    def test_to_chatbot_history_validates(self):
        from gradio.components.chatbot import ChatbotDataMessages
        legacy = [{"role": "user", "content": "记住，以后我说测试是指测试文档甲"}]
        fixed = custom_ui.to_chatbot_history(legacy)
        validated = ChatbotDataMessages.model_validate(fixed)
        self.assertTrue(validated.root)
        self.assertIsInstance(fixed[0]["content"], list)
        self.assertEqual(fixed[0]["content"][0]["text"], "记住，以后我说测试是指测试文档甲")

    def test_handlers_do_not_mutate_input(self):
        original = [{"role": "user", "content": [{"type": "text", "text": "旧问题"}]}]
        snapshot = json.loads(json.dumps(original))
        _, new_history = custom_ui.user_input_handler("新问题", original)
        self.assertEqual(original, snapshot)
        listed = list(custom_ui.bot_response(new_history, _chat_fn=lambda m, h: "回答"))
        self.assertEqual(original, snapshot)

    def test_bot_response_error_text_is_friendly(self):
        def bad_chat(message, history):
            raise RuntimeError("secret-key-xyz-123")
        yielded = list(custom_ui.bot_response(
            [{"role": "user", "content": [{"type": "text", "text": "你好"}]}],
            _chat_fn=bad_chat,
        ))
        final_text = custom_ui.extract_text(yielded[-1][-1]["content"])
        self.assertNotIn("secret-key-xyz-123", final_text)
        self.assertNotIn("发生错误", final_text)

    def test_chained_events_through_real_gradio_preprocess(self):
        def fake_chat(message, history):
            return "收到：" + message
        demo = custom_ui.build_ui(fake_chat)
        from gradio.components.chatbot import Chatbot
        chatbot = next(b for b in demo.blocks.values() if isinstance(b, Chatbot))
        user_fn, bot_fn = demo.fns[2], demo.fns[3]
        async def drive():
            from gradio.state_holder import SessionState
            state = SessionState(demo)
            prior = chatbot.postprocess([
                {"role": "user", "content": [{"type": "text", "text": "旧问题"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "旧回答"}]},
            ])
            out = await demo.process_api(
                user_fn, ["记住，以后我说测试是指测试文档甲", prior.model_dump()], state
            )
            chatbot_payload = out["data"][1]
            result = await demo.process_api(bot_fn, [chatbot_payload], state)
            self.assertTrue(result["data"])
            iterator = result.get("iterator")
            final = None
            if iterator is not None:
                async for chunk in iterator:
                    final = chunk
            self.assertIsNotNone(final)
            from gradio.components.chatbot import ChatbotDataMessages
            ChatbotDataMessages.model_validate(final)
            texts = [custom_ui.extract_text(m.get("content")) for m in final]
            self.assertTrue(any("记住" in t or "收到" in t for t in texts))
        asyncio.run(drive())

    def test_fetch_file_tree_failure_is_friendly(self):
        html = custom_ui.fetch_file_tree()
        self.assertNotIn("Traceback", html)


class TimeoutTests(unittest.IsolatedAsyncioTestCase):
    async def test_safe_mcp_call_timeout_returns_friendly_text(self):
        class SlowSession:
            async def call_tool(self, name, arguments):
                await asyncio.sleep(30)
                return _McpResult({})
        with patch.dict(os.environ, {"MCP_TIMEOUT_SECONDS": "0.05"}):
            result = await agent_types.safe_mcp_call(
                SlowSession(), "list_docs", {}, agent_types.LOCATOR_TOOL_NAMES, "scope"
            )
        self.assertFalse(result.ok)
        self.assertIn("超时", result.text)

    async def test_safe_mcp_call_error_text_is_friendly(self):
        class BoomSession:
            async def call_tool(self, name, arguments):
                raise RuntimeError("conn refused password=hunter2")
        result = await agent_types.safe_mcp_call(
            BoomSession(), "list_docs", {}, agent_types.LOCATOR_TOOL_NAMES, "scope"
        )
        self.assertFalse(result.ok)
        self.assertNotIn("hunter2", result.text)

    async def test_chat_total_timeout(self):
        class Memory:
            async def run(self, text, prior):
                await asyncio.sleep(30)
                return SimpleNamespace(handled=False, updated=False, reply="")
        tmp = tempfile.TemporaryDirectory()
        old_db = app.DB_FILE
        app.DB_FILE = os.path.join(tmp.name, "chat.db")
        app.configure_runtime(app.Runtime(memory_agent=Memory(), format_memory_prompt=lambda: ""))
        try:
            with patch.dict(os.environ, {"CHAT_TIMEOUT_SECONDS": "0.05"}):
                result = await app.process_chat("你好", [])
            self.assertIn("处理超时", result)
        finally:
            app.configure_runtime(None)
            app.DB_FILE = old_db
            tmp.cleanup()

    async def test_llm_timeout_does_not_permanently_disable(self):
        import llm_client
        async def fake_create(**kwargs):
            await asyncio.sleep(30)
            raise AssertionError("should have timed out")
        class FakeCompletions:
            create = staticmethod(fake_create)
        class FakeChat:
            completions = FakeCompletions()
        class FakeClient:
            def __init__(self, *a, **k):
                self.chat = FakeChat()
            async def close(self):
                pass
        with patch.dict(os.environ, {"LLM_TIMEOUT_SECONDS": "0.05", "LLM_TOTAL_TIMEOUT_SECONDS": "5"}):
            with patch.object(llm_client, "AsyncOpenAI", FakeClient):
                with patch.object(llm_client, "CANDIDATE_MODELS", ["m-timeout"]):
                    llm_client._DISABLED_MODELS.discard("m-timeout")
                    with self.assertRaises(Exception):
                        await llm_client.chat_with_fallback([{"role": "user", "content": "hi"}])
                    self.assertNotIn("m-timeout", llm_client._DISABLED_MODELS)


class TitleMatchingTests(unittest.TestCase):
    def test_hint_substring_of_title_only(self):
        titles = ["销售", "销售存档2025"]
        self.assertEqual(_contains_matches(titles, "销售存档"), ["销售存档2025"])
        self.assertFalse(_contains("销售", "销售存档"))

    def test_exact_wins(self):
        self.assertEqual(_contains_matches(["销售", "销售存档"], "销售"), ["销售"])
        self.assertEqual(_contains_matches(["销售存档", "销售"], "销售存档"), ["销售存档"])

    def test_ambiguous_when_several_contain_hint(self):
        matches = _contains_matches(["销售存档2025", "销售存档2024"], "销售存档")
        self.assertEqual(len(matches), 2)


class LocatorModelTests(unittest.IsolatedAsyncioTestCase):
    async def _found_session(self, titles=("订单表",), sheets=("明细",)):
        docs = {"list": [{"id": "f%d" % i, "title": t} for i, t in enumerate(titles)]}
        sheet_list = {"list": [{"id": "s%d" % i, "title": t, "row_count": 2} for i, t in enumerate(sheets)]}
        return FakeSession({
            "list_docs": docs,
            "list_sheets": sheet_list,
            "search_and_read_sheet": {"data": [["金额"], [12]], "read_range": "A1:A2"},
        })

    async def test_no_llm_on_ambiguous(self):
        async def chat(messages, tools, tool_choice):
            raise AssertionError("ambiguous must not call LLM")
        session = await self._found_session(titles=("A销售", "B销售"))
        result = await DocLocatorAgent(chat=chat).run(
            session, SchedulePlan(intent="lookup", question="查", doc_hint="销售"), ""
        )
        self.assertEqual(result.status, "ambiguous")

    async def test_no_llm_on_not_found(self):
        async def chat(messages, tools, tool_choice):
            raise AssertionError("not_found must not call LLM")
        session = await self._found_session(titles=("订单表",))
        result = await DocLocatorAgent(chat=chat).run(
            session, SchedulePlan(intent="lookup", question="查", doc_hint="不存在的表"), ""
        )
        self.assertEqual(result.status, "not_found")

    async def test_llm_failure_proceeds_with_single_candidate(self):
        async def chat(messages, tools, tool_choice):
            raise TimeoutError("llm down")
        session = await self._found_session(titles=("订单表",), sheets=("明细",))
        result = await DocLocatorAgent(chat=chat).run(
            session,
            SchedulePlan(intent="lookup", question="看订单", doc_hint="订单表", sheet_hint="明细"),
            "",
        )
        self.assertEqual(result.status, "found")
        self.assertEqual(result.doc_title, "订单表")

    async def test_model_sheet_hint_used_when_plan_empty(self):
        async def chat(messages, tools, tool_choice):
            return _decision("locate_decision", status="found", doc_title="订单表",
                             sheet_hint="二月", candidates=[])
        session = await self._found_session(titles=("订单表",), sheets=("一月", "二月"))
        result = await DocLocatorAgent(chat=chat).run(
            session,
            SchedulePlan(intent="lookup", question="查", doc_hint="订单表", sheet_hint=""),
            "",
        )
        self.assertEqual(result.status, "found")
        self.assertEqual(result.sheet_title, "二月")

    async def test_model_sheet_hint_refines_plan_hint(self):
        async def chat(messages, tools, tool_choice):
            return _decision("locate_decision", status="found", doc_title="订单表",
                             sheet_hint="销售明细2025", candidates=[])
        session = await self._found_session(titles=("订单表",), sheets=("销售明细2025", "其他"))
        result = await DocLocatorAgent(chat=chat).run(
            session,
            SchedulePlan(intent="lookup", question="查", doc_hint="订单表", sheet_hint="销售"),
            "",
        )
        self.assertEqual(result.status, "found")
        self.assertEqual(result.sheet_title, "销售明细2025")


class PartialReadTests(unittest.IsolatedAsyncioTestCase):
    def test_normalize_rows_at_cap_is_truncated(self):
        data = [["A"]] + [[i] for i in range(20)]
        columns, rows, truncated = normalize_rows(data, 21, 30)
        self.assertTrue(truncated)

    def test_normalize_rows_below_cap_not_truncated(self):
        data = [["A"], [1], [2]]
        _, _, truncated = normalize_rows(data, 21, 30)
        self.assertFalse(truncated)

    async def test_locator_marks_truncated_when_row_count_exceeds(self):
        async def chat(messages, tools, tool_choice):
            return _decision("locate_decision", status="found", doc_title="订单表",
                             sheet_hint="明细", candidates=[])
        session = FakeSession({
            "list_docs": {"list": [{"id": "f1", "title": "订单表"}]},
            "list_sheets": {"list": [{"id": "s1", "title": "明细", "row_count": 500}]},
            "search_and_read_sheet": {"data": [["金额"], [1], [2]], "read_range": "A1:A3"},
        })
        result = await DocLocatorAgent(chat=chat).run(
            session, SchedulePlan(intent="lookup", question="查", doc_hint="订单表", sheet_hint="明细"), ""
        )
        self.assertTrue(result.truncated)

    async def test_locator_capped_note_always_truncated(self):
        async def chat(messages, tools, tool_choice):
            return _decision("locate_decision", status="found", doc_title="订单表",
                             sheet_hint="明细", candidates=[])
        session = FakeSession({
            "list_docs": {"list": [{"id": "f1", "title": "订单表"}]},
            "list_sheets": {"list": [{"id": "s1", "title": "明细", "row_count": 2}]},
            "search_and_read_sheet": {"data": [["金额"], [1]], "note": "capped at 2 rows"},
        })
        result = await DocLocatorAgent(chat=chat).run(
            session, SchedulePlan(intent="lookup", question="查", doc_hint="订单表", sheet_hint="明细"), ""
        )
        self.assertTrue(result.truncated)

    def test_partial_info_truncated_payload(self):
        located = LocatedResult(status="found", doc_title="表", sheet_title="明细", sheet_row_count=15000)
        partial, note = _partial_info({"truncated": True, "rows_read": 10000, "total_rows": 15000}, located)
        self.assertTrue(partial)
        self.assertIn("注意：表格只读取了前 10000 行", note)

    def test_partial_info_row_count_exceeds_limit(self):
        located = LocatedResult(status="found", doc_title="表", sheet_title="明细", sheet_row_count=20000)
        partial, note = _partial_info({"code_output": "42"}, located)
        self.assertTrue(partial)
        self.assertIn("可能不完整", note)

    async def test_analyst_ok_with_null_error_and_partial(self):
        async def chat(messages, tools, tool_choice):
            return _decision("analyze_sheet_pandas", doc_title="表", sheet_name="明细",
                             python_code="print(df['金额'].sum())")
        session = FakeSession({
            "analyze_sheet_pandas": {"error": None, "code_output": "42",
                                     "truncated": True, "rows_read": 100, "total_rows": 500},
        })
        located = LocatedResult(status="found", doc_title="表", sheet_title="明细",
                                columns=["金额"], sheet_row_count=500)
        result = await DataAnalystAgent(chat=chat).run(
            session, SchedulePlan(intent="analyze", question="合计", calc_goal="求和"), located
        )
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.output, "42")
        self.assertTrue(result.partial)
        self.assertIn("只读取了前 100 行", result.partial_note)

    async def test_answer_appends_partial_note(self):
        async def chat(messages, tools, tool_choice):
            return _Response(_Message(content="合计是 42"))
        text = await AnswerAgent(chat=chat).run(
            "合计多少",
            SchedulePlan(intent="analyze", question="合计", calc_goal="求和"),
            LocatedResult(status="found", doc_title="表", sheet_title="明细", truncated=True,
                          note="只读取了前 3 行（共 500 行），统计结果可能不完整。"),
            AnalysisResult(status="ok", doc_title="表", sheet_title="明细", output="42",
                           partial=True, partial_note="注意：表格只读取了前 100 行（共 500 行），统计结果可能不完整。"),
        )
        self.assertIn("可能不完整", text)
        self.assertIn("只读取了前", text)


class AnalystErrorTests(unittest.IsolatedAsyncioTestCase):
    async def test_null_error_is_success(self):
        async def chat(messages, tools, tool_choice):
            return _decision("analyze_sheet_pandas", doc_title="表", sheet_name="明细",
                             python_code="print(df['金额'].sum())")
        session = FakeSession({"analyze_sheet_pandas": {"error": None, "code_output": "42"}})
        located = LocatedResult(status="found", doc_title="表", sheet_title="明细", columns=["金额"])
        result = await DataAnalystAgent(chat=chat).run(
            session, SchedulePlan(intent="analyze", question="合计", calc_goal="求和"), located
        )
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.output, "42")

    async def test_truthy_error_is_failure(self):
        replies = [
            _decision("analyze_sheet_pandas", doc_title="表", sheet_name="明细",
                      python_code="print(df['金额'].sum())"),
            _decision("analyze_sheet_pandas", doc_title="表", sheet_name="明细",
                      python_code="print(df['金额'].sum())"),
            _decision("analyze_sheet_pandas", doc_title="表", sheet_name="明细",
                      python_code="print(df['金额'].sum())"),
        ]
        async def chat(messages, tools, tool_choice):
            return replies.pop(0)
        session = FakeSession({"analyze_sheet_pandas": {"error": "boom", "code_output": ""}})
        located = LocatedResult(status="found", doc_title="表", sheet_title="明细", columns=["金额"])
        result = await DataAnalystAgent(chat=chat).run(
            session, SchedulePlan(intent="analyze", question="合计", calc_goal="求和"), located
        )
        self.assertEqual(result.status, "error")


class FriendlyErrorTests(unittest.IsolatedAsyncioTestCase):
    async def test_sqlite_failure_still_returns_reply(self):
        class Memory:
            async def run(self, text, prior):
                return SimpleNamespace(handled=True, updated=True, reply="记住了")
        tmp = tempfile.TemporaryDirectory()
        old_db = app.DB_FILE
        app.DB_FILE = tmp.name
        app.configure_runtime(app.Runtime(memory_agent=Memory(), format_memory_prompt=lambda: ""))
        try:
            result = await app.process_chat("记住规则", [])
            self.assertEqual(result, "记住了")
        finally:
            app.configure_runtime(None)
            app.DB_FILE = old_db
            tmp.cleanup()

    def test_env_override_present(self):
        with patch.dict(os.environ, {"MCP_SERVER_DIR": "/tmp/custom-mcp"}):
            self.assertEqual(os.getenv("MCP_SERVER_DIR"), "/tmp/custom-mcp")


if __name__ == "__main__":
    unittest.main()


class NoRawExceptionTests(unittest.IsolatedAsyncioTestCase):
    async def test_locator_notes_are_friendly(self):
        async def chat(messages, tools, tool_choice):
            return _decision("locate_decision", status="found", doc_title="T",
                             sheet_hint="S", candidates=[])
        class FailSession:
            async def call_tool(self, name, arguments):
                if name == "list_docs":
                    return _McpResult({"list": [{"id": "f1", "title": "T"}]})
                raise RuntimeError("secret-token-abc")
        result = await DocLocatorAgent(chat=chat).run(
            FailSession(), SchedulePlan(intent="lookup", question="q", doc_hint="T"), ""
        )
        self.assertEqual(result.status, "error")
        self.assertNotIn("secret-token-abc", result.note)

    async def test_location_error_text_never_shows_raw(self):
        located = LocatedResult(status="error", note="文档列表获取失败，请稍后重试。")
        text = app._location_failure_text(located, SchedulePlan(intent="lookup", question="q"))
        self.assertNotIn("Traceback", text)
        self.assertIn("请稍后重试", text)
