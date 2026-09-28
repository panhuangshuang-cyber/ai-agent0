"""safe_pandas 受限沙箱测试（离线，绝不联网，绝不调用真实 LLM/腾讯接口）。"""
from __future__ import annotations

import contextlib
import io
import os
import re
import sys
import unittest
from unittest.mock import patch

import safe_pandas
from agent_types import validate_analysis_code

ROWS = [["苹果", "3", "10.5"], ["香蕉", "", "2"], ["苹果", "7", "1.5"]]
COLUMNS = ["品名", "数量", "金额"]


class ValidateCodeTests(unittest.TestCase):
    def assert_rejected(self, code, fragment=None):
        reason = safe_pandas.validate_code(code)
        self.assertTrue(reason, f"应该拒绝: {code!r}")
        if fragment:
            self.assertIn(fragment, reason)

    def test_escapes_are_rejected(self):
        self.assert_rejected("print(().__class__)", "__class__")
        self.assert_rejected("print(df.__class__)", "__class__")
        self.assert_rejected("print(getattr(df, 'x'))", "getattr")
        self.assert_rejected("print(pd.read_csv('/etc/passwd'))", "read_csv")
        self.assert_rejected("df.to_csv('/tmp/x.csv')\nprint(1)", "to_csv")
        self.assert_rejected("print(pd.io.common.__file__)")  # io 与 __file__ 都被禁
        self.assert_rejected("print(pd.io)", "io")
        self.assert_rejected("print(open('/etc/passwd').read())", "open")
        self.assert_rejected("print(__import__('os').listdir('/'))", "__import__")
        self.assert_rejected("f = lambda: ().__class__\nprint(f())", "__class__")
        self.assert_rejected("print('{0.__class__}'.format(df))", "format")
        self.assert_rejected("print(str.mro())", "mro")
        self.assert_rejected("print(df.query('数量 > 1'))", "query")
        self.assert_rejected("print(eval('1+1'))", "eval")
        self.assert_rejected("print(exec('x=1'))", "exec")
        self.assert_rejected("import os\nprint(1)", "import")
        self.assert_rejected("from os import system\nprint(1)", "import")
        self.assert_rejected("with open('a') as f:\n    print(f)", "With")
        self.assert_rejected("def g():\n    return 1\nprint(g())", "FunctionDef")
        self.assert_rejected("try:\n    print(1)\nexcept:\n    pass", "Try")
        self.assert_rejected("print(globals())", "globals")
        self.assert_rejected("print(locals())", "locals")
        self.assert_rejected("df.sum()", "print")  # 缺 print 调用
        self.assert_rejected("x" * 5000 + "\nprint(1)", "过长")

    def test_unknown_names_rejected(self):
        self.assert_rejected("print(undefined_name)", "undefined_name")
        self.assert_rejected("print(os.environ)", "os")

    def test_positive_cases(self):
        ok_cases = [
            "print(df['金额'].sum())",
            "print(df.groupby('品名')['数量'].sum().to_dict())",
            "print(df.sort_values('金额', ascending=False)['品名'].tolist())",
            "print(df['数量'].max(), df['金额'].min())",
            "total = df['金额'].sum()\nprint(f'总计: {total}')",
            "vals = [v for v in df['数量'] if v and v > 3]\nprint(len(vals))",
            "f = lambda v: v * 2\nprint(f(21))",
            "print(df.head(2).to_string())",
            "print(round(df['金额'].mean(), 2))",
        ]
        for code in ok_cases:
            self.assertEqual(safe_pandas.validate_code(code), "", code)

    def test_web_side_delegates(self):
        self.assertEqual(validate_analysis_code("print(df['金额'].sum())"), "")
        self.assertIn("open", validate_analysis_code("print(open ('secret.txt').read())"))
        self.assertIn("禁止调用", validate_analysis_code("print(open ('secret.txt').read())"))


class RunRestrictedTests(unittest.TestCase):
    def test_positive_sum_groupby_sort_max(self):
        result = safe_pandas.run_restricted("print(df['数量'].sum())", ROWS, COLUMNS)
        self.assertEqual(result, {"code_output": "10.0\n", "error": ""})
        result = safe_pandas.run_restricted(
            "print(df.groupby('品名')['数量'].sum().to_dict())", ROWS, COLUMNS
        )
        self.assertEqual(result["error"], "")
        self.assertEqual(result["code_output"].strip(), "{'苹果': 10.0, '香蕉': 0.0}")
        result = safe_pandas.run_restricted(
            "print(df.sort_values('数量', ascending=False)['品名'].tolist())", ROWS, COLUMNS
        )
        self.assertEqual(result["code_output"].strip(), "['苹果', '苹果', '香蕉']")
        result = safe_pandas.run_restricted("print(df['金额'].max())", ROWS, COLUMNS)
        self.assertEqual(result["code_output"].strip(), "10.5")

    def test_numeric_conversion_pandas3_mixed_columns(self):
        rows = [["3", "abc", "1"], ["", "def", "2"], ["5", "ghi", ""]]
        cols = ["数字含空白", "纯文本", "数字"]
        code = (
            "print(df['数字含空白'].sum())\n"
            "print(df['纯文本'].tolist())\n"
            "print(str(df['数字'].dtype))\n"
            "print(df['数字'].sum())"
        )
        result = safe_pandas.run_restricted(code, rows, cols)
        self.assertEqual(result["error"], "")
        lines = result["code_output"].strip().splitlines()
        self.assertEqual(lines[0], "8.0")  # 空白按缺失值处理
        self.assertEqual(lines[1], "['abc', 'def', 'ghi']")  # 文本列保持文本
        self.assertEqual(lines[2], "float64")  # 含空白的数值列变 float
        self.assertEqual(lines[3], "3.0")

    def test_integer_column_stays_int(self):
        result = safe_pandas.run_restricted(
            "print(df['数量'].sum())", [["3"], ["5"], ["7"]], ["数量"]
        )
        self.assertEqual(result["code_output"].strip(), "15")

    def test_escape_attempt_via_run_restricted(self):
        result = safe_pandas.run_restricted("print(().__class__)", ROWS, COLUMNS)
        self.assertTrue(result["error"])
        self.assertEqual(result["code_output"], "")
        result = safe_pandas.run_restricted("print(open('/etc/passwd').read())", ROWS, COLUMNS)
        self.assertIn("open", result["error"])

    def test_runtime_error_reported(self):
        result = safe_pandas.run_restricted("print(df['不存在的列'].sum())", ROWS, COLUMNS)
        self.assertTrue(result["error"].startswith("执行代码出错"))

    def test_infinite_loop_stopped_by_timeout(self):
        import time
        start = time.time()
        result = safe_pandas.run_restricted(
            "print('开始')\nx = 0\nwhile True:\n    x += 1", ROWS, COLUMNS, timeout=2
        )
        elapsed = time.time() - start
        self.assertEqual(result["code_output"], "")
        self.assertIn("代码执行超时（超过 2 秒）", result["error"])
        self.assertLess(elapsed, 15)

    def test_memory_bomb_contained(self):
        result = safe_pandas.run_restricted(
            "x = 'a' * (3 * 1024 ** 3)\nprint(len(x))", ROWS, COLUMNS, timeout=30
        )
        self.assertTrue(result["error"])
        self.assertNotIn("3221225472", result["code_output"])

    def test_child_env_has_no_secrets(self):
        env = safe_pandas._child_env()
        self.assertTrue(set(env) <= {"PATH", "LANG", "HOME"})
        with patch.dict(os.environ, {
            "TENCENT_DOCS_ACCESS_TOKEN": "secret-token",
            "OPENAI_API_KEY": "secret-key",
        }):
            env = safe_pandas._child_env()
            for value in env.values():
                self.assertNotIn("secret", value)

    def test_child_cannot_see_parent_env(self):
        # 沙箱里连 os 模块都拿不到，环境变量更读不到；这里验证代码里
        # 引用 os 会在 validate 阶段被拒绝，不会进入子进程。
        result = safe_pandas.run_restricted("import os\nprint(os.environ)", ROWS, COLUMNS)
        self.assertIn("import", result["error"])

    def test_parent_stdout_untouched(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = safe_pandas.run_restricted("print(df['数量'].sum())", ROWS, COLUMNS)
        self.assertEqual(buf.getvalue(), "")
        self.assertEqual(result["code_output"].strip(), "10.0")

    def test_file_write_blocked(self):
        result = safe_pandas.run_restricted(
            "print(df.to_csv('/tmp/escape.csv'))", ROWS, COLUMNS
        )
        self.assertIn("to_csv", result["error"])

    def test_output_capped(self):
        result = safe_pandas.run_restricted("print('x' * 20000)", ROWS, COLUMNS)
        self.assertEqual(result["error"], "")
        self.assertLessEqual(len(result["code_output"]), safe_pandas.MAX_OUTPUT_CHARS + 1)


def _run_child_without_validator(code, rows=ROWS, columns=COLUMNS):
    """绕过 AST 校验直接跑子进程，单独验证运行时审计钩子这道防线。"""
    import json
    import subprocess
    boot = safe_pandas._bootstrap_source().replace(
        "safe_pandas._child_main()",
        "safe_pandas.validate_code = lambda c: ''\nsafe_pandas._child_main()",
    )
    proc = subprocess.run(
        [sys.executable, "-I", "-c", boot],
        input=json.dumps({"code": code, "rows": rows, "columns": columns}),
        capture_output=True, text=True, env=safe_pandas._child_env(),
        cwd="/tmp", timeout=60,
    )
    return json.loads(proc.stdout or "{}")


class ModuleGatewayEscapeTests(unittest.TestCase):
    """numpy/pandas 子模块上挂着 os/subprocess/ctypes/builtins 的引用
    （np.f2py.os、pd.errors.ctypes、np.ma.extras.ma.builtins），以及
    生成器帧 gi_frame.f_builtins 能拿到 __import__。"""

    GATEWAY_CODES = [
        "print(np.f2py.os.listdir('/'))",
        "print(np.f2py.subprocess.check_output(['id']))",
        "print(np.f2py.sys.modules)",
        "print(pd.errors.ctypes)",
        "print(np.ma.extras.ma.builtins.open('/etc/passwd').read())",
        "print(np.ctypeslib.load_library('c', '/lib'))",
        "g = (i for i in range(1))\nprint(g.gi_frame.f_builtins)",
        "g = (i for i in range(1))\nf = g.gi_frame\nprint(f.f_globals)",
        "g = (i for i in range(1))\nprint(g.gi_code)",
    ]

    def test_ast_rejects_gateways(self):
        for code in self.GATEWAY_CODES:
            self.assertTrue(safe_pandas.validate_code(code), code)

    def test_run_restricted_rejects_gateways(self):
        result = safe_pandas.run_restricted("print(np.f2py.os.listdir('/'))", ROWS, COLUMNS)
        self.assertEqual(result["code_output"], "")
        self.assertIn("os", result["error"])

    def test_audit_hook_blocks_sinks_even_without_validator(self):
        blocked = [
            "print(np.f2py.os.listdir('/'))",
            "print(np.f2py.os.system('id'))",
            "print(np.f2py.subprocess.check_output(['id']))",
            "print(np.f2py.os.fork())",
            "print(np.ma.extras.ma.builtins.open('/etc/passwd').read())",
            "print(np.ma.extras.ma.builtins.open('/proc/self/environ').read())",
            "print(np.ma.extras.ma.builtins.open('/tmp/sandbox-escape.txt', 'w'))",
            "print(pd.errors.ctypes.CDLL(None))",
            "b = np.ma.extras.ma.builtins\nprint(b.__import__('socket').socket())",
            "print(pd.read_csv('/etc/passwd'))",
            "print(np.load('/etc/passwd', allow_pickle=True))",
        ]
        for code in blocked:
            result = _run_child_without_validator(code)
            self.assertTrue(result.get("error"), code)
            self.assertIn("沙箱禁止", result["error"], code)
            self.assertNotIn("root:", result.get("code_output", ""), code)
        self.assertFalse(os.path.exists("/tmp/sandbox-escape.txt"))

    def test_guarded_import_is_allowlist(self):
        frame_import = "imp = (i for i in range(1)).gi_frame.f_builtins['__import__']\n"
        for mod in ("socket", "os", "gc", "marshal", "importlib"):
            result = _run_child_without_validator(frame_import + f"print(imp({mod!r}))")
            self.assertIn("沙箱禁止导入模块", result.get("error", ""), mod)
        result = _run_child_without_validator(frame_import + "print(imp('pandas').__name__)")
        self.assertEqual(result.get("error"), "")
        self.assertEqual(result["code_output"].strip(), "pandas")

    def test_common_pandas_operations_still_work_under_hook(self):
        rows = [["苹果", "3", "10.5", "2024-01-02"], ["香蕉", "", "2", "2024-02-03"],
                ["苹果", "7", "1.5", "2024-03-04"]]
        cols = ["品名", "数量", "金额", "日期"]
        code = "\n".join([
            "print(df.dtypes)",
            "print(df.describe(include='all'))",
            "print(df.groupby('品名')['数量'].agg(['sum', 'mean', 'count']))",
            "print(df.pivot_table(index='品名', values='金额', aggfunc='sum'))",
            "d = df.copy()",
            "d['日期'] = pd.to_datetime(d['日期'])",
            "print(d.set_index('日期').resample('ME')['金额'].sum())",
            "print(d['日期'].dt.strftime('%Y-%m').tolist())",
            "print(pd.crosstab(df['品名'], df['数量']))",
            "print(df['金额'].quantile(0.5), np.percentile(df['金额'], 90))",
            "print(df.corr(numeric_only=True))",
            "print(pd.Timestamp('2024-01-01', tz='Asia/Shanghai'))",
            "print('完成')",
        ])
        result = safe_pandas.run_restricted(code, rows, cols, timeout=60)
        self.assertEqual(result["error"], "")
        self.assertTrue(result["code_output"].strip().endswith("完成"))


def _mcp_copy(name):
    """MCP 服务端文件：优先仓库内 mcp_server/，其次本地/生产的 tencent-docs-mcp 目录。"""
    here = os.path.dirname(os.path.abspath(__file__))
    for base in (
        os.path.join(here, "mcp_server"),
        os.environ.get("MCP_SERVER_DIR", ""),
        "/workspace/tencent-docs-mcp",
        os.path.expanduser("~/tencent-docs-mcp"),
    ):
        if base and os.path.isfile(os.path.join(base, name)):
            return os.path.join(base, name)
    return ""


class SourceHygieneTests(unittest.TestCase):
    def test_no_errors_ignore_anywhere(self):
        pattern = re.compile(r"to_numeric\([^)]*errors\s*=\s*['\"]ignore")
        here = os.path.dirname(os.path.abspath(__file__))
        files = [
            os.path.join(here, "safe_pandas.py"),
            os.path.join(here, "agent_types.py"),
            os.path.join(here, "data_analyst_agent.py"),
            os.path.join(here, "doc_locator_agent.py"),
        ]
        server_path = _mcp_copy("server.py")
        if server_path:
            files.append(server_path)
        for path in files:
            with open(path, encoding="utf-8") as fh:
                self.assertIsNone(pattern.search(fh.read()), f"{path} 仍有 errors='ignore'")

    def test_both_copies_identical(self):
        here = os.path.dirname(os.path.abspath(__file__))
        other = _mcp_copy("safe_pandas.py")
        if not other:
            self.skipTest("找不到 MCP 服务端目录里的 safe_pandas.py")
        with open(os.path.join(here, "safe_pandas.py"), "rb") as fh:
            left = fh.read()
        with open(other, "rb") as fh:
            right = fh.read()
        self.assertEqual(left, right)


if __name__ == "__main__":
    unittest.main()
