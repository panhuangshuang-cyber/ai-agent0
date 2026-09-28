from __future__ import annotations

from contextlib import asynccontextmanager
import json
import os
import tempfile
from types import SimpleNamespace
import unittest

import app
from agent_types import AnalysisResult, LocatedResult, SchedulePlan, validate_analysis_code
from answer_agent import AnswerAgent
from data_analyst_agent import DataAnalystAgent
from doc_locator_agent import DocLocatorAgent
from scheduler_agent import SchedulerAgent


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
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            })
        return {"role": "assistant", "content": self.content, "tool_calls": calls}


class _Response:
    def __init__(self, message):
        self.choices = [SimpleNamespace(message=message)]


def decision(name, **kwargs):
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
        return _McpResult(value)


class SchedulerTests(unittest.IsolatedAsyncioTestCase):
    async def test_calc_goal_forces_analyze_and_discards_reply(self):
        async def chat(messages, tools, tool_choice):
            return decision(
                "schedule_decision",
                intent="lookup",
                reply="合计是 123",
                question="合计是多少",
                doc_hint="封边条",
                sheet_hint="",
                calc_goal="计算合计",
            )

        plan = await SchedulerAgent(chat=chat).run("封边条合计是多少", [], "")
        self.assertEqual(plan.intent, "analyze")
        self.assertEqual(plan.reply, "")

    async def test_invalid_intent_is_retried_before_calc_goal_rewrite(self):
        replies = [
            decision(
                "schedule_decision",
                intent="report",
                reply="",
                question="求和",
                doc_hint="订单",
                sheet_hint="",
                calc_goal="求和",
            ),
            decision(
                "schedule_decision",
                intent="analyze",
                reply="",
                question="求和",
                doc_hint="订单",
                sheet_hint="",
                calc_goal="求和",
            ),
        ]

        async def chat(messages, tools, tool_choice):
            return replies.pop(0)

        plan = await SchedulerAgent(chat=chat).run("订单求和", [], "")
        self.assertEqual(plan.intent, "analyze")
        self.assertEqual(len(replies), 0)


class LocatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_ambiguous_titles_do_not_read_sheet(self):
        session = FakeSession({
            "list_docs": {"list": [
                {"id": "1", "title": "封边条250424-华东"},
                {"id": "2", "title": "封边条250424-华南"},
            ]},
        })

        async def chat(messages, tools, tool_choice):
            return decision(
                "locate_decision",
                status="found",
                doc_title="封边条250424-华东",
                sheet_hint="",
                candidates=[],
            )

        result = await DocLocatorAgent(chat=chat).run(
            session,
            SchedulePlan(intent="lookup", question="查表", doc_hint="封边条250424"),
            "",
        )
        self.assertEqual(result.status, "ambiguous")
        self.assertEqual(result.candidates, ["封边条250424-华东", "封边条250424-华南"])
        self.assertEqual([name for name, _ in session.calls], ["list_docs"])

    async def test_wrong_locator_tool_is_never_called(self):
        replies = [
            decision("write_sheet", file_id="x"),
            decision(
                "locate_decision",
                status="found",
                doc_title="订单表",
                sheet_hint="明细",
                candidates=[],
            ),
        ]

        async def chat(messages, tools, tool_choice):
            return replies.pop(0)

        session = FakeSession({
            "list_docs": {"list": [{"id": "f1", "title": "订单表"}]},
            "list_sheets": {"list": [{"id": "s1", "title": "明细", "row_count": 2}]},
            "search_and_read_sheet": {"data": [["金额"], [12]], "read_range": "A1:A2"},
        })
        result = await DocLocatorAgent(chat=chat).run(
            session,
            SchedulePlan(intent="lookup", question="看订单", doc_hint="订单表", sheet_hint="明细"),
            "",
        )
        self.assertEqual(result.status, "found")
        self.assertNotIn("write_sheet", [name for name, _ in session.calls])

    async def test_empty_sheet_hint_with_multiple_sheets_is_ambiguous(self):
        async def chat(messages, tools, tool_choice):
            return decision(
                "locate_decision",
                status="found",
                doc_title="订单表",
                sheet_hint="模型擅自选择",
                candidates=[],
            )

        session = FakeSession({
            "list_docs": {"list": [{"id": "f1", "title": "订单表"}]},
            "list_sheets": {"list": [
                {"id": "s1", "title": "一月"},
                {"id": "s2", "title": "二月"},
            ]},
        })
        result = await DocLocatorAgent(chat=chat).run(
            session,
            SchedulePlan(intent="lookup", question="查订单", doc_hint="订单表", sheet_hint=""),
            "",
        )
        self.assertEqual(result.status, "ambiguous")
        self.assertEqual(result.candidates, ["一月", "二月"])

    async def test_mcp_business_error_returns_locator_error(self):
        async def chat(messages, tools, tool_choice):
            raise AssertionError("model must not run when list_docs fails")

        session = FakeSession({"list_docs": {"error": "认证失效"}})
        result = await DocLocatorAgent(chat=chat).run(
            session,
            SchedulePlan(intent="lookup", question="查订单", doc_hint="订单表"),
            "",
        )
        self.assertEqual(result.status, "error")


class AnalystTests(unittest.IsolatedAsyncioTestCase):
    def test_rejects_spaced_file_open(self):
        reason = validate_analysis_code("print(open ('secret.txt').read())")
        self.assertIn("禁止调用", reason)

    async def test_overrides_model_titles(self):
        async def chat(messages, tools, tool_choice):
            return decision(
                "analyze_sheet_pandas",
                doc_title="恶意标题",
                sheet_name="恶意子表",
                python_code="print(df['金额'].sum())",
            )

        session = FakeSession({"analyze_sheet_pandas": {"code_output": "42"}})
        located = LocatedResult(
            status="found",
            doc_title="完整订单表",
            sheet_title="销售明细",
            columns=["金额"],
            rows=[[12]],
        )
        result = await DataAnalystAgent(chat=chat).run(
            session,
            SchedulePlan(intent="analyze", question="合计", calc_goal="金额合计"),
            located,
        )
        self.assertEqual(result.status, "ok")
        submitted = session.calls[0][1]
        self.assertEqual(submitted["doc_title"], "完整订单表")
        self.assertEqual(submitted["sheet_name"], "销售明细")

    async def test_rejected_code_counts_and_is_not_submitted(self):
        replies = [
            decision(
                "analyze_sheet_pandas",
                doc_title="表",
                sheet_name="明细",
                python_code="import os\nprint(1)",
            ),
            decision(
                "analyze_sheet_pandas",
                doc_title="表",
                sheet_name="明细",
                python_code="df.sum()",
            ),
            decision(
                "analyze_sheet_pandas",
                doc_title="表",
                sheet_name="明细",
                python_code="print(df['金额'].sum())",
            ),
        ]

        async def chat(messages, tools, tool_choice):
            return replies.pop(0)

        session = FakeSession({"analyze_sheet_pandas": {"code_output": "10"}})
        result = await DataAnalystAgent(chat=chat).run(
            session,
            SchedulePlan(intent="analyze", question="合计", calc_goal="求和"),
            LocatedResult(status="found", doc_title="表", sheet_title="明细", columns=["金额"]),
        )
        self.assertEqual(result.attempts, 3)
        self.assertEqual(len(session.calls), 1)
        self.assertEqual(result.output, "10")


class AnswerTests(unittest.IsolatedAsyncioTestCase):
    async def test_analysis_failure_prompt_has_error_but_no_samples(self):
        captured = {}

        async def chat(messages, tools, tool_choice):
            captured["messages"] = messages
            return _Response(_Message(content="没算出来：列不存在"))

        text = await AnswerAgent(chat=chat).run(
            "算总额",
            SchedulePlan(intent="analyze", question="算总额", calc_goal="求和"),
            LocatedResult(
                status="found",
                doc_title="订单",
                sheet_title="明细",
                columns=["金额"],
                rows=[[999], [888], [777]],
            ),
            AnalysisResult(
                status="error",
                doc_title="订单",
                sheet_title="明细",
                output="列不存在",
                attempts=3,
            ),
        )
        prompt = captured["messages"][1]["content"]
        self.assertIn("列不存在", prompt)
        self.assertNotIn("999", prompt)
        self.assertEqual(text, "没算出来：列不存在")


class OrchestrationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        app.DB_FILE = os.path.join(self.tmp.name, "chat.db")

    def tearDown(self):
        app.configure_runtime(None)
        self.tmp.cleanup()

    async def test_memory_handled_skips_scheduler(self):
        class Memory:
            async def run(self, text, prior):
                return SimpleNamespace(handled=True, updated=True, reply="记住了")

        class Scheduler:
            async def run(self, *args):
                raise AssertionError("scheduler must not run")

        app.configure_runtime(app.Runtime(
            memory_agent=Memory(),
            format_memory_prompt=lambda: "",
            scheduler=Scheduler(),
        ))
        result = await app.process_chat("记住规则", [])
        self.assertEqual(result, "记住了")

    async def test_lookup_enters_scheduler_skips_analyst_and_memory_note_once(self):
        calls = []

        class Memory:
            async def run(self, text, prior):
                return SimpleNamespace(handled=False, updated=True, reply="已记住别名。")

        class Scheduler:
            async def run(self, text, prior, memory):
                calls.append("scheduler")
                return SchedulePlan(intent="lookup", question="查订单", doc_hint="订单")

        class Locator:
            async def run(self, session, plan, memory):
                calls.append("locator")
                return LocatedResult(status="found", doc_title="订单", sheet_title="明细")

        class Analyst:
            async def run(self, *args):
                raise AssertionError("lookup must not run analyst")

        class Answer:
            async def run(self, *args):
                calls.append("answer")
                return "正文里误写：已记住别名。\n结果是 A"

        @asynccontextmanager
        async def opener():
            yield object()

        app.configure_runtime(app.Runtime(
            memory_agent=Memory(),
            format_memory_prompt=lambda: "- 当用户提到 订单，等同于 订单",
            scheduler=Scheduler(),
            locator=Locator(),
            analyst=Analyst(),
            answer=Answer(),
            open_mcp_session=opener,
        ))
        result = await app.process_chat("查订单", [])
        self.assertEqual(calls, ["scheduler", "locator", "answer"])
        self.assertTrue(result.startswith("已记住别名。"))
        self.assertEqual(result.count("已记住别名。"), 1)


if __name__ == "__main__":
    unittest.main()
