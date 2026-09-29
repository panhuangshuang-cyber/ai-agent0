"""写入路径（write 意图 + 两轮确认）的测试。

沿用 test_subagents.py 的 stub 约定：每个测试文件自带一份假 LLM / 假 MCP 会话。
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import inspect
import json
import os
import tempfile
from types import SimpleNamespace
import time
import unittest

import app
from agent_types import (
    AnalysisResult,
    LocatedResult,
    PENDING_WRITE_TTL_SECONDS,
    SchedulePlan,
    WriteProposal,
)
from writer_agent import WriterAgent


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


class SessionFactory:
    """每轮 process_chat 都会开一个新会话，这里把它们都记下来。"""

    def __init__(self, responses):
        self.responses = responses
        self.sessions = []

    @asynccontextmanager
    async def open(self):
        session = FakeSession(self.responses)
        self.sessions.append(session)
        yield session

    @property
    def calls(self):
        return [call for session in self.sessions for call in session.calls]

    def names(self):
        return [name for name, _ in self.calls]

    def args_for(self, name):
        return [arguments for tool, arguments in self.calls if tool == name]


def located_result(**overrides):
    base = dict(
        status="found",
        doc_title="tx",
        file_id="300000000$eqgxGhurREUh",
        sheet_title="工作表1",
        sheet_id="BB08J2",
        columns=["客户", "单号", "单价"],
        read_range="A1:C4",
        sheet_row_count=200,
        rows=[["客户", "单号", "单价"], ["张三", "A1", "77"]],
        truncated=False,
        note="",
    )
    base.update(overrides)
    return LocatedResult(**base)


def analysis_result(output='{"count": 1, "rows": [0]}', **overrides):
    base = dict(status="ok", output=output, partial=False, partial_note="")
    base.update(overrides)
    return AnalysisResult(**base)


def write_plan(**overrides):
    base = dict(
        intent="write",
        question="把张三那行的单价改成 88",
        doc_hint="tx",
        sheet_hint="",
        write_goal="把 客户=张三 那一行的 单价 改成 88",
    )
    base.update(overrides)
    return SchedulePlan(**base)


class _Pipeline:
    """一套可配置的假代理，默认走「找到 1 行、提案单价=88」的成功路径。"""

    def __init__(self, located=None, analysis=None, proposal=None, responses=None):
        self.calls = []
        self.located = located if located is not None else located_result()
        self.analysis = analysis if analysis is not None else analysis_result()
        self.proposal = (
            proposal if proposal is not None
            else WriteProposal(status="ok", column="单价", new_value="88")
        )
        self.factory = SessionFactory(responses if responses is not None else {
            # 第一次读是写前取旧值，第二次读是写后回读验证。
            "read_sheet": [{"values": [["77"]]}, {"values": [["88"]]}],
            "write_sheet": [{}],
        })
        pipeline = self

        class Memory:
            async def run(self, text, prior):
                pipeline.calls.append("memory")
                return SimpleNamespace(handled=False, updated=False, reply="")

        class Scheduler:
            async def run(self, text, prior, memory):
                pipeline.calls.append("scheduler")
                return write_plan()

        class Locator:
            async def run(self, session, plan, memory):
                pipeline.calls.append(("locator", plan.intent))
                return pipeline.located

        class Analyst:
            async def run(self, session, plan, located):
                pipeline.calls.append("analyst")
                return pipeline.analysis

        class Writer:
            async def run(self, plan, located, analysis, row_numbers):
                pipeline.calls.append("writer")
                pipeline.row_numbers = row_numbers
                return pipeline.proposal

        class Answer:
            async def run(self, *args):
                raise AssertionError("write 意图不应该走答复代理")

        self.runtime_kwargs = dict(
            memory_agent=Memory(),
            format_memory_prompt=lambda: "",
            scheduler=Scheduler(),
            locator=Locator(),
            analyst=Analyst(),
            answer=Answer(),
            writer=Writer(),
            open_mcp_session=self.factory.open,
        )

    def install(self):
        app.configure_runtime(app.Runtime(**self.runtime_kwargs))
        return self


class WriteFlowTestBase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        app.DB_FILE = os.path.join(self.tmp.name, "chat.db")
        app._PENDING.clear()
        app._EXPIRED.clear()

    def tearDown(self):
        app.configure_runtime(None)
        app._PENDING.clear()
        app._EXPIRED.clear()
        self.tmp.cleanup()

    def only_token(self):
        self.assertEqual(len(app._PENDING), 1)
        return next(iter(app._PENDING))

    def preview_history(self, preview):
        return [
            {"role": "user", "content": "把张三那行的单价改成 88"},
            {"role": "assistant", "content": preview},
        ]


class ParseMatchesTests(unittest.TestCase):
    def test_parses_bare_dict(self):
        self.assertEqual(app._parse_matches('{"count": 2, "rows": [3, 4]}'), (2, [3, 4]))

    def test_takes_last_dict_when_model_prints_debug_first(self):
        output = '调试 {"count": 9, "rows": [1]}\n{"count": 1, "rows": [7]}'
        self.assertEqual(app._parse_matches(output), (1, [7]))

    def test_unparsable_returns_none(self):
        self.assertEqual(app._parse_matches("我没打印字典"), (None, []))
        self.assertEqual(app._parse_matches(""), (None, []))

    def test_ignores_dicts_without_the_contract_keys(self):
        self.assertEqual(app._parse_matches('{"total": 5}'), (None, []))

    def test_non_integer_rows_are_dropped(self):
        self.assertEqual(app._parse_matches('{"count": 2, "rows": [1, "x"]}'), (2, [1]))


class ResolveColumnTests(unittest.TestCase):
    def test_exact_match(self):
        index, rejection = app._resolve_column("单价", located_result())
        self.assertEqual((index, rejection), (2, ""))

    def test_whitespace_and_case_insensitive_fallback(self):
        index, rejection = app._resolve_column(" 单价 ", located_result())
        self.assertEqual((index, rejection), (2, ""))

    def test_missing_column_is_rejected(self):
        index, rejection = app._resolve_column("数量", located_result())
        self.assertIsNone(index)
        self.assertIn("数量", rejection)

    def test_duplicate_headers_are_rejected(self):
        located = located_result(columns=["规格", "规格", "单价"])
        index, rejection = app._resolve_column("规格", located)
        self.assertIsNone(index)
        self.assertEqual(rejection, app.WRITE_COLUMN_AMBIGUOUS.format(column="规格"))

    def test_empty_column_is_rejected(self):
        index, rejection = app._resolve_column("", located_result())
        self.assertIsNone(index)
        self.assertEqual(rejection, app.WRITE_UNCLEAR)


class ValueRejectionTests(unittest.TestCase):
    def test_plain_values_are_allowed(self):
        for value in ("88", "已付款", "-5", "-12.5", "2026-09-29"):
            self.assertEqual(app._value_rejection(value), "", value)

    def test_empty_value_is_rejected(self):
        self.assertEqual(app._value_rejection("   "), app.WRITE_VALUE_EMPTY)
        self.assertEqual(app._value_rejection(None), app.WRITE_VALUE_EMPTY)

    def test_formula_prefixes_are_rejected(self):
        for value in ("=SUM(A1)", "@cmd|calc", "+cmd|calc", "-2+3+cmd|calc"):
            self.assertIn("公式", app._value_rejection(value), value)

    def test_overlong_value_is_rejected(self):
        self.assertIn("字符", app._value_rejection("x" * (app.WRITE_MAX_VALUE_CHARS + 1)))


class ConfirmationMatcherTests(unittest.TestCase):
    def test_affirmatives(self):
        for text in ("确认", "确认。", "好的", "是", "执行", "确认写入", " 确定 "):
            self.assertTrue(app._is_affirmative(text), text)

    def test_non_affirmatives(self):
        for text in ("", "算了别改", "把单价改成 99", "确认一下这个数对不对啊到底是不是"):
            self.assertFalse(app._is_affirmative(text), text)

    def test_token_only_read_from_last_assistant_message(self):
        preview = "将要修改……\n确认码 W-abc123def456\n回复「确认」就写入"
        self.assertEqual(
            app._pending_token_in_history([{"role": "assistant", "content": preview}]),
            "W-abc123def456",
        )

    def test_no_token_when_last_message_is_from_user(self):
        self.assertIsNone(app._pending_token_in_history([
            {"role": "assistant", "content": "确认码 W-abc123def456"},
            {"role": "user", "content": "确认"},
        ]))

    def test_no_token_when_history_empty_or_unrelated(self):
        self.assertIsNone(app._pending_token_in_history([]))
        self.assertIsNone(app._pending_token_in_history([{"role": "assistant", "content": "查到了"}]))

    def test_gradio_tuple_history_is_supported(self):
        self.assertEqual(
            app._pending_token_in_history([("问题", "确认码 W-abc123def456")]),
            "W-abc123def456",
        )


class WriterAgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_agent_has_no_mcp_session_parameter(self):
        """写入代理物理上拿不到 session，所以不可能自己执行写入。"""
        params = inspect.signature(WriterAgent.run).parameters
        self.assertNotIn("session", params)

    async def test_returns_proposal(self):
        async def chat(messages, tools, tool_choice):
            return decision("write_proposal", status="ok", column="单价", new_value="88", reason="用户要求")

        proposal = await WriterAgent(chat=chat).run(
            write_plan(), located_result(), analysis_result(), [2]
        )
        self.assertEqual(proposal.status, "ok")
        self.assertEqual(proposal.column, "单价")
        self.assertEqual(proposal.new_value, "88")

    async def test_unclear_status_is_preserved(self):
        async def chat(messages, tools, tool_choice):
            return decision("write_proposal", status="unclear", column="", new_value="")

        proposal = await WriterAgent(chat=chat).run(
            write_plan(), located_result(), analysis_result(), [2]
        )
        self.assertEqual(proposal.status, "unclear")

    async def test_invalid_proposal_retries_then_errors(self):
        attempts = []

        async def chat(messages, tools, tool_choice):
            attempts.append(messages)
            # status 非法 → _normalize 返回 None → 重试一次后放弃
            return decision("write_proposal", status="maybe", column="单价", new_value="88")

        proposal = await WriterAgent(chat=chat).run(
            write_plan(), located_result(), analysis_result(), [2]
        )
        self.assertEqual(proposal.status, "error")
        self.assertEqual(len(attempts), 2)

    async def test_ok_status_requires_column_and_value(self):
        async def chat(messages, tools, tool_choice):
            return decision("write_proposal", status="ok", column="", new_value="")

        proposal = await WriterAgent(chat=chat).run(
            write_plan(), located_result(), analysis_result(), [2]
        )
        self.assertEqual(proposal.status, "error")


class SchedulerWriteIntentTests(unittest.IsolatedAsyncioTestCase):
    async def test_write_intent_keeps_write_goal_and_clears_reply(self):
        from scheduler_agent import SchedulerAgent

        async def chat(messages, tools, tool_choice):
            return decision(
                "schedule_decision",
                intent="write",
                reply="已经帮你改好了",
                question="把单价改成 88",
                doc_hint="tx",
                sheet_hint="",
                calc_goal="顺便算一下合计",
                write_goal="把单价改成 88",
            )

        plan = await SchedulerAgent(chat=chat).run("把 tx 的单价改成 88", [], "")
        self.assertEqual(plan.intent, "write")
        self.assertEqual(plan.write_goal, "把单价改成 88")
        # write 绝不能被 calc_goal 劫持成 analyze，也不能自称已经改好。
        self.assertEqual(plan.calc_goal, "")
        self.assertEqual(plan.reply, "")

    async def test_write_without_goal_is_invalid(self):
        from scheduler_agent import SchedulerAgent

        async def chat(messages, tools, tool_choice):
            return decision("schedule_decision", intent="write", question="改一下", write_goal="")

        with self.assertRaises(ValueError):
            await SchedulerAgent(chat=chat).run("改一下", [], "")


class WritePreviewTests(WriteFlowTestBase):
    async def test_first_turn_previews_without_writing(self):
        pipeline = _Pipeline().install()
        preview = await app.process_chat("把张三那行的单价改成 88", [])

        self.assertNotIn("write_sheet", pipeline.factory.names())
        self.assertEqual(pipeline.calls, [
            "memory", "scheduler", ("locator", "analyze"), "analyst", "writer",
        ])
        token = self.only_token()
        spec = app._PENDING[token]
        self.assertEqual(spec.start_cell, "C2")
        self.assertEqual(spec.end_cell, "C2")
        self.assertEqual(spec.values, [["88"]])
        self.assertEqual(spec.old_values, [["77"]])
        self.assertEqual(spec.column, "单价")
        # 行号 = df 索引 + 2（表头占第 1 行）
        self.assertEqual(pipeline.row_numbers, [2])
        # 预览由代码拼装，必须逐字反映将要执行的内容
        self.assertIn("《tx》", preview)
        self.assertIn("「工作表1」", preview)
        self.assertIn("第 2 行「单价」：「77」→「88」", preview)
        self.assertIn(token, preview)
        self.assertIn("确认", preview)

    async def test_locator_receives_analyze_intent_with_derived_goal(self):
        pipeline = _Pipeline().install()
        await app.process_chat("把张三那行的单价改成 88", [])
        self.assertIn(("locator", "analyze"), pipeline.calls)

    async def test_zero_match_does_not_store_pending(self):
        _Pipeline(analysis=analysis_result('{"count": 0, "rows": []}')).install()
        result = await app.process_chat("把李四那行的单价改成 88", [])
        self.assertEqual(result, app.WRITE_NO_MATCH)
        self.assertEqual(app._PENDING, {})

    async def test_zero_match_on_truncated_read_warns_about_the_cap(self):
        _Pipeline(analysis=analysis_result(
            '{"count": 0, "rows": []}', partial=True, partial_note="只读取了前 10000 行"
        )).install()
        result = await app.process_chat("把李四那行的单价改成 88", [])
        self.assertEqual(result, app.WRITE_BEYOND_READ_LIMIT.format(rows=10000))
        self.assertEqual(app._PENDING, {})

    async def test_unparsable_analyst_output_is_rejected(self):
        _Pipeline(analysis=analysis_result("我算不出来")).install()
        result = await app.process_chat("把张三那行的单价改成 88", [])
        self.assertEqual(result, app.WRITE_LOCATE_FAILURE)
        self.assertEqual(app._PENDING, {})

    async def test_count_disagreeing_with_listed_rows_is_rejected(self):
        _Pipeline(analysis=analysis_result('{"count": 5, "rows": [1, 2]}')).install()
        result = await app.process_chat("改一批", [])
        self.assertEqual(result, app.WRITE_COUNT_MISMATCH.format(count=5, listed=2))
        self.assertEqual(app._PENDING, {})

    async def test_non_contiguous_rows_are_rejected(self):
        _Pipeline(analysis=analysis_result('{"count": 3, "rows": [0, 2, 5]}')).install()
        result = await app.process_chat("改一批", [])
        self.assertEqual(result, app.WRITE_NOT_CONTIGUOUS.format(count=3))
        self.assertEqual(app._PENDING, {})

    async def test_contiguous_rows_write_one_block(self):
        pipeline = _Pipeline(
            analysis=analysis_result('{"count": 3, "rows": [0, 1, 2]}'),
            responses={
                "read_sheet": [{"values": [["1"], ["2"], ["3"]]}, {"values": [["88"], ["88"], ["88"]]}],
                "write_sheet": [{}],
            },
        ).install()
        preview = await app.process_chat("把前三行的单价都改成 88", [])

        spec = app._PENDING[self.only_token()]
        self.assertEqual(spec.start_cell, "C2")
        self.assertEqual(spec.end_cell, "C4")
        self.assertEqual(spec.values, [["88"], ["88"], ["88"]])
        self.assertEqual(spec.row_numbers, [2, 3, 4])
        self.assertIn("第 4 行「单价」：「3」→「88」", preview)
        self.assertNotIn("write_sheet", pipeline.factory.names())

        confirm = await app.process_chat("确认", self.preview_history(preview))
        write_args = pipeline.factory.args_for("write_sheet")
        self.assertEqual(write_args, [{
            "file_id": "300000000$eqgxGhurREUh",
            "sheet_id": "BB08J2",
            "start_cell": "C2",
            "values": [["88"], ["88"], ["88"]],
        }])
        self.assertIn("共 3 行", confirm)

    async def test_formula_value_is_rejected(self):
        pipeline = _Pipeline(
            proposal=WriteProposal(status="ok", column="单价", new_value="=SUM(A1)")
        ).install()
        result = await app.process_chat("把单价改成 =SUM(A1)", [])
        self.assertIn("不合适", result)
        self.assertEqual(app._PENDING, {})
        self.assertNotIn("write_sheet", pipeline.factory.names())

    async def test_unknown_column_is_rejected(self):
        pipeline = _Pipeline(
            proposal=WriteProposal(status="ok", column="数量", new_value="88")
        ).install()
        result = await app.process_chat("把数量改成 88", [])
        self.assertIn("数量", result)
        self.assertEqual(app._PENDING, {})
        self.assertNotIn("write_sheet", pipeline.factory.names())

    async def test_unclear_proposal_asks_the_user_again(self):
        _Pipeline(proposal=WriteProposal(status="unclear")).install()
        result = await app.process_chat("把那行改一下", [])
        self.assertEqual(result, app.WRITE_UNCLEAR)
        self.assertEqual(app._PENDING, {})

    async def test_row_beyond_grid_is_rejected(self):
        _Pipeline(
            located=located_result(sheet_row_count=10),
            analysis=analysis_result('{"count": 1, "rows": [50]}'),
        ).install()
        result = await app.process_chat("改第 52 行", [])
        self.assertEqual(result, app.WRITE_OUT_OF_GRID.format(cell="C52", rows=10))
        self.assertEqual(app._PENDING, {})

    async def test_failed_preread_blocks_the_write(self):
        pipeline = _Pipeline(responses={
            "read_sheet": [{"error": "读不到"}],
            "write_sheet": [{}],
        }).install()
        result = await app.process_chat("把张三那行的单价改成 88", [])
        self.assertEqual(result, app.WRITE_READ_OLD_FAILURE)
        self.assertEqual(app._PENDING, {})
        self.assertNotIn("write_sheet", pipeline.factory.names())

    async def test_leading_zero_value_gets_a_numeric_warning(self):
        _Pipeline(
            proposal=WriteProposal(status="ok", column="单号", new_value="007")
        ).install()
        preview = await app.process_chat("把单号改成 007", [])
        self.assertIn("前导 0 会丢失", preview)

    async def test_partial_read_adds_a_caveat_but_still_proposes(self):
        _Pipeline(analysis=analysis_result(
            '{"count": 1, "rows": [0]}', partial=True, partial_note="只读取了前 10000 行"
        )).install()
        preview = await app.process_chat("把张三那行的单价改成 88", [])
        self.assertIn("只读取了前 10000 行", preview)
        self.assertEqual(len(app._PENDING), 1)

    async def test_ambiguous_location_never_reaches_the_analyst(self):
        pipeline = _Pipeline(
            located=located_result(status="ambiguous", candidates=["tx", "tx2"])
        ).install()
        result = await app.process_chat("把单价改成 88", [])
        self.assertIn("tx", result)
        self.assertNotIn("analyst", pipeline.calls)
        self.assertEqual(app._PENDING, {})


class ConfirmFlowTests(WriteFlowTestBase):
    async def propose(self, pipeline=None):
        pipeline = (pipeline or _Pipeline()).install()
        preview = await app.process_chat("把张三那行的单价改成 88", [])
        return pipeline, preview

    async def test_confirm_executes_and_reads_back(self):
        pipeline, preview = await self.propose()
        result = await app.process_chat("确认", self.preview_history(preview))

        write_args = pipeline.factory.args_for("write_sheet")
        self.assertEqual(len(write_args), 1)
        self.assertEqual(write_args[0]["start_cell"], "C2")
        self.assertEqual(write_args[0]["values"], [["88"]])
        self.assertIn("已写入《tx》的「工作表1」C2", result)
        self.assertIn("88", result)
        self.assertEqual(app._PENDING, {})

    async def test_confirm_by_token_also_works(self):
        pipeline, preview = await self.propose()
        token = self.only_token()
        result = await app.process_chat(f"执行 {token}", self.preview_history(preview))
        self.assertEqual(len(pipeline.factory.args_for("write_sheet")), 1)
        self.assertIn("已写入", result)

    async def test_second_confirm_is_a_noop(self):
        pipeline, preview = await self.propose()
        history = self.preview_history(preview)
        await app.process_chat("确认", history)
        again = await app.process_chat("确认", history)
        self.assertEqual(again, app.CONFIRM_MISSING)
        self.assertEqual(len(pipeline.factory.args_for("write_sheet")), 1)

    async def test_expired_confirmation_does_not_write(self):
        pipeline, preview = await self.propose()
        token = self.only_token()
        app._PENDING[token].expires_at = time.time() - 1
        result = await app.process_chat("确认", self.preview_history(preview))
        self.assertEqual(
            result, app.CONFIRM_EXPIRED.format(minutes=PENDING_WRITE_TTL_SECONDS / 60)
        )
        self.assertNotIn("write_sheet", pipeline.factory.names())
        self.assertEqual(app._PENDING, {})

    async def test_confirmation_without_a_preview_falls_through(self):
        class Scheduler:
            def __init__(self):
                self.ran = False

            async def run(self, text, prior, memory):
                self.ran = True
                return SchedulePlan(intent="chat", reply="在的")

        scheduler = Scheduler()
        pipeline = _Pipeline().install()
        app.configure_runtime(app.Runtime(**{**pipeline.runtime_kwargs, "scheduler": scheduler}))
        result = await app.process_chat("确认", [{"role": "user", "content": "在吗"}])
        self.assertTrue(scheduler.ran)
        self.assertEqual(result, "在的")
        self.assertEqual(pipeline.factory.names(), [])

    async def test_token_from_another_conversation_cannot_confirm(self):
        pipeline, preview = await self.propose()
        token = self.only_token()
        # 另一个浏览器标签的历史里没有这个确认码，一句「确认」不该触发写入。
        await app.process_chat("确认", [
            {"role": "user", "content": "查一下库存"},
            {"role": "assistant", "content": "库存还有 300 张。"},
        ])
        self.assertNotIn("write_sheet", pipeline.factory.names())
        self.assertIn(token, app._PENDING)

    async def test_non_affirmative_reply_leaves_the_write_pending(self):
        pipeline, preview = await self.propose()
        token = self.only_token()
        await app.process_chat("算了，先别改", self.preview_history(preview))
        self.assertNotIn("write_sheet", pipeline.factory.names())
        self.assertIn(token, app._PENDING)

    async def test_write_failure_reports_and_clears_pending(self):
        pipeline, preview = await self.propose(_Pipeline(responses={
            "read_sheet": [{"values": [["77"]]}],
            "write_sheet": [{"error": "频控"}],
        }))
        result = await app.process_chat("确认", self.preview_history(preview))
        self.assertEqual(result, app.WRITE_EXECUTE_FAILURE)
        # 失败后不自动重试：写可能已经落盘，重新入队会造成二次写入。
        self.assertEqual(app._PENDING, {})

    async def test_verify_mismatch_is_surfaced(self):
        pipeline, preview = await self.propose(_Pipeline(responses={
            "read_sheet": [{"values": [["77"]]}, {"values": [["99"]]}],
            "write_sheet": [{}],
        }))
        result = await app.process_chat("确认", self.preview_history(preview))
        self.assertIn("不一致", result)
        self.assertIn("99", result)

    async def test_numeric_readback_still_counts_as_success(self):
        pipeline, preview = await self.propose(_Pipeline(responses={
            "read_sheet": [{"values": [["77"]]}, {"values": [[88.0]]}],
            "write_sheet": [{}],
        }))
        result = await app.process_chat("确认", self.preview_history(preview))
        self.assertIn("已写入", result)
        self.assertNotIn("不一致", result)

    async def test_unreadable_verify_still_reports_the_write(self):
        pipeline, preview = await self.propose(_Pipeline(responses={
            "read_sheet": [{"values": [["77"]]}, {"error": "读不到"}],
            "write_sheet": [{}],
        }))
        result = await app.process_chat("确认", self.preview_history(preview))
        self.assertIn("已提交写入", result)
        self.assertIn("没能回读确认", result)


class ConcurrentConfirmTests(WriteFlowTestBase):
    async def test_same_token_confirmed_twice_writes_once(self):
        """两个线程各自跑 asyncio.run，_PENDING.pop 的原子性保证只写一次。"""
        pipeline = _Pipeline().install()
        preview = await app.process_chat("把张三那行的单价改成 88", [])
        history = [
            {"role": "user", "content": "把张三那行的单价改成 88"},
            {"role": "assistant", "content": preview},
        ]
        results = await asyncio.gather(
            app.process_chat("确认", history),
            app.process_chat("确认", history),
        )
        self.assertEqual(len(pipeline.factory.args_for("write_sheet")), 1)
        self.assertEqual(sum(1 for text in results if "已写入" in text), 1)
        self.assertEqual(sum(1 for text in results if text == app.CONFIRM_MISSING), 1)


if __name__ == "__main__":
    unittest.main()
