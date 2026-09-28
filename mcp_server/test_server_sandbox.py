"""MCP server 沙箱/分页/匹配逻辑离线测试。

绝不发起网络请求：client 通过 monkeypatch server._client 注入 FakeClient；
腾讯接口的单次请求限制（行<=1000、列<=200、单元格<=10000）在 FakeClient 里
强制执行，用来验证 server 的分页逻辑确实遵守限制。

运行（生产目录 ~/tencent-docs-mcp 或仓库里的 mcp_server/ 均可）：
    cd /workspace/tencent-docs-mcp && \
    PYTHONDONTWRITEBYTECODE=1 /workspace/ai-agent0-venv/bin/python -m unittest test_server_sandbox -v
"""
from __future__ import annotations

import contextlib
import io
import os
import re
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:  # 生产目录里有 client.py；仓库里（mcp_server/）不提交 client.py，用离线桩代替。
    import client  # noqa: F401,E402
except ImportError:
    import types

    _stub = types.ModuleType("client")

    class TencentDocsError(Exception):
        pass

    class TencentDocsClient:  # pragma: no cover - 测试里总是被 FakeClient 替换
        def __init__(self, *args, **kwargs):
            raise TencentDocsError("离线测试桩：不应创建真实客户端")

    _stub.TencentDocsError = TencentDocsError
    _stub.TencentDocsClient = TencentDocsClient
    sys.modules["client"] = _stub

import server  # noqa: E402
from client import TencentDocsError  # noqa: E402

_RANGE_RE = re.compile(r"^([A-Z]+)(\d+):([A-Z]+)(\d+)$")


def _col_to_index(letters: str) -> int:
    n = 0
    for ch in letters:
        n = n * 26 + (ord(ch) - 64)
    return n  # 1-based


class FakeClient:
    """内存版 TencentDocsClient，行为对齐真实接口的限制。"""

    def __init__(self, docs=None, sheets=None, grid=None):
        self.docs = docs if docs is not None else {"list": []}
        self.sheets = sheets or []
        self.grid = grid or {}  # (file_id, sheet_id) -> 2D list
        self.docs_calls = 0
        self.read_calls = []

    def list_docs(self, folder_id="/", limit=20, file_type="", is_owner=0):
        self.docs_calls += 1
        return self.docs

    def list_sheets(self, file_id):
        return self.sheets

    def read_sheet_range(self, file_id, sheet_id, cell_range):
        self.read_calls.append((file_id, sheet_id, cell_range))
        m = _RANGE_RE.match(cell_range)
        if not m:
            raise TencentDocsError(f"无效范围: {cell_range}")
        start_row = int(m.group(2))
        end_col = _col_to_index(m.group(3))
        end_row = int(m.group(4))
        rows = end_row - start_row + 1
        if rows > server.SHEET_MAX_ROWS_PER_REQUEST:
            raise TencentDocsError("单次请求行数超过 1000")
        if end_col > server.SHEET_MAX_COLS_PER_REQUEST:
            raise TencentDocsError("单次请求列数超过 200")
        if rows * end_col > server.SHEET_MAX_CELLS_PER_REQUEST:
            raise TencentDocsError("单次请求单元格超过 10000")
        data = self.grid.get((file_id, sheet_id), [])
        values = []
        for row in data[start_row - 1:end_row]:
            values.append([str(v) for v in list(row)[:end_col]])
        return {"start_row": start_row - 1, "start_column": 0, "values": values}


def _sheet(sheet_id="s1", title="明细", row_count=3, col_count=2):
    return {"sheetId": sheet_id, "title": title, "rowCount": row_count, "columnCount": col_count}


def _patched(fake):
    return patch.object(server, "_client", lambda: fake)


def _grid(rows):
    return {("f1", "s1"): rows}


class TitleMatchTests(unittest.TestCase):
    def test_exact_title_wins_over_contains(self):
        fake = FakeClient(
            docs={"list": [{"id": "f1", "title": "销售"}, {"id": "f2", "title": "销售存档2025"}]},
            sheets=[_sheet()],
            grid={("f1", "s1"): [["金额"], ["1"]]},
        )
        with _patched(fake):
            result = server._search_and_read_sheet_impl("销售")
        self.assertNotIn("error", result)
        self.assertEqual(result["doc_title"], "销售")
        self.assertEqual(result["file_id"], "f1")

    def test_hint_substring_of_title_matches(self):
        fake = FakeClient(
            docs={"list": [{"id": "f1", "title": "销售"}, {"id": "f2", "title": "销售存档2025"}]},
            sheets=[_sheet()],
            grid={("f2", "s1"): [["金额"], ["1"]]},
        )
        with _patched(fake):
            result = server._search_and_read_sheet_impl("销售存档")
        self.assertEqual(result["file_id"], "f2")

    def test_ambiguous_titles_return_candidates_not_first(self):
        fake = FakeClient(docs={"list": [
            {"id": "f1", "title": "销售存档2024"},
            {"id": "f2", "title": "销售存档2025"},
        ]})
        with _patched(fake):
            result = server._search_and_read_sheet_impl("销售存档")
        self.assertTrue(result.get("error"))
        self.assertIn("多个", result["error"])
        self.assertEqual(result["candidates"], ["销售存档2024", "销售存档2025"])
        self.assertEqual(fake.read_calls, [])

    def test_not_found(self):
        fake = FakeClient(docs={"list": [{"id": "f1", "title": "销售"}]})
        with _patched(fake):
            result = server._search_and_read_sheet_impl("库存")
        self.assertTrue(result.get("error"))

    def test_ids_first_skips_doc_search(self):
        fake = FakeClient(
            docs={"list": [{"id": "other", "title": "别的"}]},
            sheets=[_sheet("s1", "明细"), _sheet("s2", "二月")],
            grid={("f9", "s2"): [["金额"], ["7"]]},
        )
        with _patched(fake):
            result = server._search_and_read_sheet_impl("", file_id="f9", sheet_id="s2")
        self.assertEqual(fake.docs_calls, 0)
        self.assertEqual(result["sheet_title"], "二月")
        self.assertEqual(result["data"], [["金额"], ["7"]])

    def test_sheet_name_not_found_is_error_no_fallback(self):
        fake = FakeClient(
            docs={"list": [{"id": "f1", "title": "销售"}]},
            sheets=[_sheet("s1", "一月"), _sheet("s2", "二月")],
            grid={("f1", "s1"): [["金额"], ["1"]]},
        )
        with _patched(fake):
            result = server._search_and_read_sheet_impl("销售", sheet_name="三月")
        self.assertTrue(result.get("error"))
        self.assertIn("三月", result["error"])
        self.assertEqual(result["candidates"], ["一月", "二月"])
        self.assertEqual(fake.read_calls, [])

    def test_sheet_name_ambiguous_returns_candidates(self):
        fake = FakeClient(
            docs={"list": [{"id": "f1", "title": "销售"}]},
            sheets=[_sheet("s1", "销售明细2024"), _sheet("s2", "销售明细2025")],
        )
        with _patched(fake):
            result = server._search_and_read_sheet_impl("销售", sheet_name="销售明细")
        self.assertTrue(result.get("error"))
        self.assertEqual(result["candidates"], ["销售明细2024", "销售明细2025"])

    def test_no_sheet_name_defaults_to_first_sheet(self):
        fake = FakeClient(
            docs={"list": [{"id": "f1", "title": "销售"}]},
            sheets=[_sheet("s1", "一月"), _sheet("s2", "二月")],
            grid={("f1", "s1"): [["金额"], ["1"]]},
        )
        with _patched(fake):
            result = server._search_and_read_sheet_impl("销售")
        self.assertEqual(result["sheet_title"], "一月")


class PaginationTests(unittest.TestCase):
    def test_paginates_within_tencent_limits(self):
        rows = [["品名", "数量", "金额", "备注"]] + [
            [f"p{i}", str(i), str(i * 2), "x"] for i in range(1, 2500)
        ]
        fake = FakeClient(
            docs={"list": [{"id": "f1", "title": "销售"}]},
            sheets=[_sheet("s1", "明细", row_count=2500, col_count=4)],
            grid={("f1", "s1"): rows},
        )
        with _patched(fake):
            result = server._search_and_read_sheet_impl("销售", max_rows=2500, max_cols=4)
        self.assertNotIn("error", result)
        self.assertEqual(result["rows_read"], 2500)
        self.assertEqual(result["total_rows"], 2500)
        self.assertEqual(result["total_cols"], 4)
        self.assertFalse(result["truncated"])
        self.assertEqual(len(fake.read_calls), 3)  # 1000 + 1000 + 500
        self.assertEqual(len(result["data"]), 2500)

    def test_caps_at_10000_rows_and_flags_truncated(self):
        rows = [["品名", "数量"]] + [[f"p{i}", str(i)] for i in range(1, 11999)]
        fake = FakeClient(
            docs={"list": [{"id": "f1", "title": "销售"}]},
            sheets=[_sheet("s1", "明细", row_count=12000, col_count=2)],
            grid={("f1", "s1"): rows},
        )
        with _patched(fake):
            result = server._search_and_read_sheet_impl("销售", max_rows=10000, max_cols=2)
        self.assertEqual(result["rows_read"], 10000)
        self.assertTrue(result["truncated"])
        self.assertIn("只读取了前 10000 行（共 12000 行）", result["note"])

    def test_small_sheet_not_truncated(self):
        fake = FakeClient(
            docs={"list": [{"id": "f1", "title": "销售"}]},
            sheets=[_sheet("s1", "明细", row_count=3, col_count=2)],
            grid=_grid([["金额", "数量"], ["1", "2"], ["3", "4"]]),
        )
        with _patched(fake):
            result = server._search_and_read_sheet_impl("销售", max_rows=21, max_cols=30)
        self.assertFalse(result["truncated"])
        self.assertNotIn("note", result)
        self.assertEqual(result["rows_read"], 3)

    def test_blank_tail_rows_stop_pagination(self):
        """rowCount 远大于真实数据行时，整批空白就该停止翻页。"""
        filled = [["品名", "数量"]] + [[f"p{i}", str(i)] for i in range(1, 30)]
        rows = filled + [["", ""] for _ in range(5000 - len(filled))]
        fake = FakeClient(
            docs={"list": [{"id": "f1", "title": "销售"}]},
            sheets=[_sheet("s1", "明细", row_count=5000, col_count=2)],
            grid={("f1", "s1"): rows},
        )
        with _patched(fake):
            result = server._search_and_read_sheet_impl("销售", max_rows=5000, max_cols=2)
        self.assertNotIn("error", result)
        self.assertEqual(result["rows_read"], 30)
        self.assertEqual(result["total_rows"], 30)
        self.assertFalse(result["truncated"])
        self.assertNotIn("note", result)
        self.assertEqual(len(result["data"]), 30)
        self.assertLessEqual(len(fake.read_calls), 2)

    def test_read_failure_is_friendly(self):
        class BoomClient(FakeClient):
            def read_sheet_range(self, file_id, sheet_id, cell_range):
                raise TencentDocsError("token=hunter2 网络错误")
        fake = BoomClient(
            docs={"list": [{"id": "f1", "title": "销售"}]},
            sheets=[_sheet()],
            grid=_grid([["金额"], ["1"]]),
        )
        with _patched(fake):
            result = server._search_and_read_sheet_impl("销售")
        self.assertTrue(result.get("error"))
        self.assertNotIn("hunter2", result["error"])


class AnalyzeToolTests(unittest.TestCase):
    def _fake(self, rows=None):
        rows = rows if rows is not None else [
            ["品名", "金额"], ["苹果", "10"], ["香蕉", ""], ["梨", "32.5"],
        ]
        return FakeClient(
            docs={"list": [{"id": "f1", "title": "销售"}]},
            sheets=[_sheet("s1", "明细", row_count=len(rows), col_count=len(rows[0]))],
            grid=_grid(rows),
        )

    def test_sum_with_blanks_pandas3(self):
        fake = self._fake()
        with _patched(fake):
            result = server.analyze_sheet_pandas("销售", "print(df['金额'].sum())")
        self.assertNotIn("error", result)
        self.assertEqual(result["code_output"], "42.5")
        self.assertEqual(result["rows_read"], 4)
        self.assertEqual(result["total_rows"], 4)
        self.assertFalse(result["truncated"])

    def test_groupby_and_sort(self):
        rows = [["类别", "数量"], ["A", "3"], ["B", "5"], ["A", "7"]]
        fake = self._fake(rows)
        with _patched(fake):
            result = server.analyze_sheet_pandas(
                "销售",
                "g = df.groupby('类别')['数量'].sum().sort_values(ascending=False)\nprint(g.to_dict())",
            )
        self.assertNotIn("error", result)
        self.assertEqual(result["code_output"], "{'A': 10, 'B': 5}")

    def test_malicious_code_rejected_without_exec(self):
        fake = self._fake()
        with _patched(fake):
            for code, fragment in [
                ("print(open('/etc/passwd').read())", "open"),
                ("print(().__class__)", "__class__"),
                ("import os\nprint(os.listdir('/'))", "import"),
                ("print(pd.read_csv('http://evil/x.csv'))", "read_csv"),
                ("df.to_csv('/tmp/x.csv')\nprint(1)", "to_csv"),
            ]:
                result = server.analyze_sheet_pandas("销售", code)
                self.assertTrue(result.get("error"), code)
                self.assertIn(fragment, result["error"])

    def test_infinite_loop_times_out_with_seconds(self):
        fake = self._fake()
        with _patched(fake):
            with patch.dict(os.environ, {"SANDBOX_EXEC_TIMEOUT_SECONDS": "0.5"}):
                result = server.analyze_sheet_pandas(
                    "销售", "print('开始')\nx = 0\nwhile True:\n    x += 1"
                )
        self.assertTrue(result.get("error"))
        self.assertIn("代码执行超时（超过 0.5 秒）", result["error"])

    def test_no_stdout_leak_and_no_errors_ignore(self):
        fake = self._fake()
        buf = io.StringIO()
        with _patched(fake):
            with contextlib.redirect_stdout(buf):
                result = server.analyze_sheet_pandas("销售", "print(df['金额'].max())")
        self.assertEqual(buf.getvalue(), "")
        self.assertEqual(result["code_output"], "32.5")
        here = os.path.dirname(os.path.abspath(__file__))
        pattern = re.compile(r"to_numeric\([^)]*errors\s*=\s*['\"]ignore")
        for name in ("server.py", "safe_pandas.py"):
            with open(os.path.join(here, name), encoding="utf-8") as fh:
                source = fh.read()
            self.assertIsNone(pattern.search(source), f"{name} 仍在使用 errors='ignore'")

    def test_insufficient_data(self):
        fake = self._fake([["品名", "金额"]])
        with _patched(fake):
            result = server.analyze_sheet_pandas("销售", "print(1)")
        self.assertTrue(result.get("error"))

    def test_truncated_analysis_carries_fields(self):
        # grid 只有 10000 行（含表头），但工作表 rowCount 是 12000 → 截断
        rows = [["类别", "数量"]] + [[str(i % 3), str(i)] for i in range(1, 10000)]
        fake = FakeClient(
            docs={"list": [{"id": "f1", "title": "销售"}]},
            sheets=[_sheet("s1", "明细", row_count=12000, col_count=2)],
            grid=_grid(rows),
        )
        with _patched(fake):
            result = server.analyze_sheet_pandas("销售", "print(df['数量'].sum())")
        self.assertTrue(result["truncated"])
        self.assertEqual(result["rows_read"], 10000)
        self.assertEqual(result["total_rows"], 12000)
        self.assertEqual(result["total_cols"], 2)
        self.assertIn("只读取了前 10000 行（共 12000 行）", result["note"])
        self.assertEqual(result["code_output"], str(sum(range(1, 10000))))
        self.assertEqual(len(fake.read_calls), 10)


class RealPayloadShapeTests(unittest.TestCase):
    """线上实测：大写 ID、rowCount=0 + rowTotal 网格、日期单元格、末尾空白行。"""

    def test_uppercase_id_rowtotal_dates_and_blank_tail(self):
        grid = [["入库日期", "品名", "数量", ""]]
        grid.append(["{'time': {'year': 2025, 'month': 10, 'day': 3, 'hour': 0, 'minute': 0, 'second': 0}}", "A", "1", ""])
        grid.append(["{'time': {'year': 2025, 'month': 10, 'day': 4, 'hour': 9, 'minute': 30, 'second': 0}}", "B", "2", ""])
        grid.extend([["", "", "", ""] for _ in range(20)])
        fake = FakeClient(
            docs={"next": 0, "list": [{"ID": "f1", "title": "板材库存"}, {"ID": "f2", "title": "tx"}]},
            sheets=[{"sheetId": "s1", "title": "工作表1", "rowCount": 0, "columnCount": 0,
                     "rowTotal": len(grid), "columnTotal": 4}],
            grid={("f1", "s1"): grid},
        )
        with _patched(fake):
            result = server._search_and_read_sheet_impl("板材库存", max_rows=200)
        self.assertNotIn("error", result)
        self.assertEqual(result["file_id"], "f1")
        self.assertEqual(result["rows_read"], 3)
        self.assertEqual(result["total_rows"], 3)
        self.assertFalse(result["truncated"])
        self.assertEqual(result["total_cols"], 4)
        self.assertEqual(result["data"][1][0], "2025-10-03")
        self.assertEqual(result["data"][2][0], "2025-10-04 09:30:00")

    def test_grid_total_note_when_row_count_zero(self):
        grid = [["金额"]] + [[str(i)] for i in range(1, 60)]
        fake = FakeClient(
            docs={"list": [{"ID": "f1", "title": "板材库存"}]},
            sheets=[{"sheetId": "s1", "title": "工作表1", "rowCount": 0, "columnCount": 0,
                     "rowTotal": 200, "columnTotal": 1}],
            grid={("f1", "s1"): grid},
        )
        with _patched(fake):
            result = server._search_and_read_sheet_impl("板材库存", max_rows=10)
        self.assertTrue(result["truncated"])
        self.assertIn("网格", result["note"])
        self.assertIn("只读取了前 10 行", result["note"])

    def test_analyze_with_real_shapes(self):
        grid = [["入库日期", "品名", "数量", ""],
                ["{'time': {'year': 2025, 'month': 10, 'day': 3, 'hour': 0, 'minute': 0, 'second': 0}}", "A", "1", ""],
                ["{'time': {'year': 2025, 'month': 11, 'day': 4, 'hour': 0, 'minute': 0, 'second': 0}}", "A", "2", ""],
                ["", "", "", ""]]
        fake = FakeClient(
            docs={"list": [{"ID": "f1", "title": "板材库存"}]},
            sheets=[{"sheetId": "s1", "title": "工作表1", "rowCount": 0, "columnCount": 0,
                     "rowTotal": 4, "columnTotal": 4}],
            grid={("f1", "s1"): grid},
        )
        code = "print(df['数量'].sum())\nprint(pd.to_datetime(df['入库日期']).dt.month.tolist())"
        with _patched(fake):
            result = server.analyze_sheet_pandas("板材库存", code)
        self.assertNotIn("error", result)
        self.assertEqual(result["code_output"], "3\n[10, 11]")
        self.assertFalse(result["truncated"])


if __name__ == "__main__":
    unittest.main()
