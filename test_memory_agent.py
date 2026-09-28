import json
import os
import tempfile
import unittest
from unittest.mock import patch

from memory_agent import MemoryAgent, MemoryAgentResult, cue_kind, visible_doc_tools


class _Fn:
    def __init__(self, arguments):
        self.name = "memory_decision"
        self.arguments = arguments


class _ToolCall:
    def __init__(self, arguments):
        self.function = _Fn(arguments)
        self.id = "call-1"


class _Message:
    def __init__(self, arguments=None, content=""):
        self.tool_calls = None if arguments is None else [_ToolCall(arguments)]
        self.content = content


class _Response:
    def __init__(self, message):
        self.choices = [type("Choice", (), {"message": message})()]


def _decision(**kwargs):
    return json.dumps(kwargs, ensure_ascii=False)


class MemoryAgentTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "memory.json")
        self.calls = []

    def tearDown(self):
        self.tmp.cleanup()

    def agent(self, chat):
        return MemoryAgent(path=self.path, chat=chat)

    def saved(self):
        with open(self.path, encoding="utf-8") as handle:
            return json.load(handle)

    async def test_lookup_skips_the_model(self):
        async def chat(messages, tools, tool_choice):
            raise AssertionError("查表的话不应叫醒记忆模型")

        result = await self.agent(chat).run("单价最高的是哪条")
        self.assertEqual(result, MemoryAgentResult(False, False, ""))
        self.assertFalse(os.path.exists(self.path))

    async def test_recalled_fact_is_not_a_memory_command(self):
        self.assertIsNone(cue_kind("我记住了单价最高的是哪条"))

    async def test_set_rule(self):
        async def chat(messages, tools, tool_choice):
            self.calls.append(tool_choice)
            self.assertIn("封边条", messages[-1]["content"])
            return _Response(_Message(_decision(action="set", key="封边条", value="封边条250424")))

        result = await self.agent(chat).run("记住，以后我说封边条就是指封边条250424")
        self.assertTrue(result.handled)
        self.assertTrue(result.updated)
        self.assertIn("已记住", result.reply)
        self.assertEqual(self.saved(), {"封边条": "封边条250424"})
        self.assertEqual(self.calls[0]["function"]["name"], "memory_decision")
        self.assertIn("封边条250424", self.agent(chat).format_prompt())

    async def test_continue_chat_keeps_the_document_question(self):
        async def chat(messages, tools, tool_choice):
            return _Response(_Message(_decision(
                action="set", key="封边条", value="封边条250424", continue_chat=True
            )))

        result = await self.agent(chat).run("记住，封边条就是指封边条250424，单价最高的是哪条")
        self.assertFalse(result.handled)
        self.assertTrue(result.updated)
        self.assertEqual(self.saved(), {"封边条": "封边条250424"})

    async def test_replace_and_delete(self):
        async def chat(messages, tools, tool_choice):
            return _Response(_Message(_decision(action="set", key="封边条", value="新表")))

        agent = self.agent(chat)
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump({"封边条": "旧表"}, handle)
        result = await agent.run("记住，封边条就是指新表")
        self.assertIn("从「旧表」改为「新表」", result.reply)

        async def delete(messages, tools, tool_choice):
            return _Response(_Message(_decision(action="delete", key="封边条")))

        result = await self.agent(delete).run("忘掉封边条")
        self.assertTrue(result.handled)
        self.assertEqual(self.saved(), {})
        self.assertIn("已忘记", result.reply)

    async def test_ignore_and_ask_and_reject_empty_value(self):
        async def ignore(messages, tools, tool_choice):
            return _Response(_Message(_decision(action="ignore")))

        result = await self.agent(ignore).run("以后我说的都按月汇总")
        self.assertFalse(result.handled)
        self.assertFalse(os.path.exists(self.path))

        async def ask(messages, tools, tool_choice):
            return _Response(_Message(_decision(action="ask", ask="封边条要指哪张表？")))

        result = await self.agent(ask).run("记住封边条")
        self.assertTrue(result.handled)
        self.assertEqual(result.reply, "封边条要指哪张表？")

        async def missing_value(messages, tools, tool_choice):
            return _Response(_Message(_decision(action="set", key="封边条", value="")))

        result = await self.agent(missing_value).run("记住封边条")
        self.assertIn("要指什么", result.reply)
        self.assertFalse(os.path.exists(self.path))

    async def test_same_name_and_corrupt_file_are_not_overwritten(self):
        async def chat(messages, tools, tool_choice):
            return _Response(_Message(_decision(action="set", key="封边条", value="封边条")))

        result = await self.agent(chat).run("记住，封边条就是指封边条")
        self.assertIn("没有写入", result.reply)

        with open(self.path, "w", encoding="utf-8") as handle:
            handle.write("{")

        async def chat2(messages, tools, tool_choice):
            return _Response(_Message(_decision(action="set", key="封边条", value="新表")))

        result = await self.agent(chat2).run("记住，封边条就是指新表")
        self.assertIn("损坏", result.reply)
        with open(self.path, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), "{")

    async def test_model_failure_on_a_clear_command(self):
        async def chat(messages, tools, tool_choice):
            raise RuntimeError("down")

        result = await self.agent(chat).run("请记住，封边条就是指封边条250424")
        self.assertTrue(result.handled)
        self.assertIn("没有写入", result.reply)

    async def test_weak_cue_falls_back_when_model_is_down(self):
        async def chat(messages, tools, tool_choice):
            raise RuntimeError("down")

        result = await self.agent(chat).run("封边条就是指哪张表")
        self.assertFalse(result.handled)

    async def test_json_in_content_and_retry_without_tool_choice(self):
        attempts = []

        async def chat(messages, tools, tool_choice):
            attempts.append(tool_choice)
            if tool_choice:
                raise RuntimeError("tool_choice unsupported")
            return _Response(_Message(content='{"action":"set","key":"简称A","value":"表A"}'))

        result = await self.agent(chat).run("记住，简称A就是指表A")
        self.assertTrue(result.updated)
        self.assertEqual(len(attempts), 2)
        self.assertIsNone(attempts[1])

    def test_doc_tools_hide_memory_writes(self):
        tools = [
            {"type": "function", "function": {"name": "update_memory_rule"}},
            {"type": "function", "function": {"name": "analyze_sheet_pandas"}},
        ]
        self.assertEqual(
            visible_doc_tools(tools),
            [{"type": "function", "function": {"name": "analyze_sheet_pandas"}}],
        )
        self.assertIsNone(visible_doc_tools([tools[0]]))


if __name__ == "__main__":
    unittest.main()
