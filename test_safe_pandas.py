"""safe_pandas 受限沙箱测试（离线，绝不联网，绝不调用真实 LLM/腾讯接口）。"""
from __future__ import annotations

import contextlib
import io
import os
import re
import sys
import tempfile
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


def _run_child_without_validator(code, rows=ROWS, columns=COLUMNS, timeout=60):
    """绕过 AST 校验直接跑子进程，单独验证运行时审计钩子这道防线。

    走 safe_pandas 的测试专用入口 _run_child_for_test；生产的 run_restricted
    永远不会跳过校验（见 PublicApiNeverSkipsValidationTests）。
    """
    return safe_pandas._run_child_for_test(
        code, rows, columns, timeout=timeout, skip_validation=True
    )


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


class ModuleAllowlistTests(unittest.TestCase):
    """pd / np / math 只能以 `模块.白名单属性` 的形式出现。"""

    ESCAPES = [
        # numpy 的磁盘读取入口：跟服务进程同一个 Unix 用户，能读 token / .env
        "print(np.genfromtxt('/etc/passwd', dtype=str))",
        "print(np.loadtxt('/etc/passwd'))",
        "print(np.fromfile('/etc/passwd'))",
        "print(np.fromregex('/etc/passwd', '(a)', dtype=str))",
        "print(np.memmap('/etc/passwd'))",
        "print(np.DataSource('/etc/passwd'))",
        "print(np.lib.recfunctions)",
        "print(np.f2py.os.listdir('/'))",
        "print(np.ctypeslib.ctypes.CDLL(None))",
        "print(np.random.default_rng())",
        "print(np.ma.extras.ma.builtins)",
        "print(np.testing.tmpdir)",
        # ctypes / 序列化桥
        "print(df.values.ctypes)",
        "print(df.values.tofile('/tmp/x.bin'))",
        "print(df.values.dump('/tmp/x.pkl'))",
        "print(pd.DataFrame.to_pickle(df, '/tmp/x.pkl'))",
        # 帧对象自省 -> 真正的 builtins -> eval/exec/__import__
        "g = df.iterrows()\nprint(g.gi_frame.f_globals['__builtins__'])",
        "g = df.iterrows()\nprint(g.gi_frame.f_builtins['__import__']('os'))",
        "c = df.iterrows()\nprint(c.cr_frame)",
        # 字符串派发
        "print(df.agg('to_json'))",
        "print(df.apply('eval', axis=1))",
        "print(df.groupby('品名')['金额'].agg('to_pickle'))",
        # 别名 / 裸模块 / 重新绑定
        "x = np\nprint(x.genfromtxt('/etc/passwd'))",
        "print(np)",
        "print(pd)",
        "print(math)",
        "print([np][0].genfromtxt('/etc/passwd'))",
        "print(np if df is not None else pd)",
        "f = pd\nprint(f.to_datetime('2024-01-01'))",
        "pd = 1\nprint(pd)",
        "np, math = 1, 2\nprint(np)",
        "for np in [1]:\n    print(np)",
        "f = lambda np: np\nprint(f(1))",
        "(np := 1)\nprint(np)",
    ]

    def test_escapes_rejected(self):
        for code in self.ESCAPES:
            reason = safe_pandas.validate_code(code)
            self.assertTrue(reason, f"应该拒绝: {code!r}")

    def test_reason_names_the_offending_attribute(self):
        cases = {
            "print(np.genfromtxt('/etc/passwd'))": "np.genfromtxt",
            "print(np.loadtxt('/etc/passwd'))": "np.loadtxt",
            "print(np.memmap('/etc/passwd'))": "np.memmap",
            "print(np.DataSource('/etc/passwd'))": "np.DataSource",
            "print(np.lib)": "np.lib",
            "print(np.f2py)": "np.f2py",
            "print(np.ctypeslib)": "np.ctypeslib",
            "print(pd.read_csv('/x'))": "pd.read_csv",
            "print(pd.io)": "pd.io",
            "print(np)": "禁止直接使用模块 np",
        }
        for code, fragment in cases.items():
            self.assertIn(fragment, safe_pandas.validate_code(code), code)

    def test_no_numpy_submodule_is_allowed(self):
        for name in ("random", "lib", "linalg", "char", "ma", "testing", "ctypeslib",
                     "f2py", "distutils", "polynomial", "fft", "strings", "rec",
                     "compat", "core", "os", "sys", "version"):
            code = f"print(np.{name})"
            self.assertTrue(safe_pandas.validate_code(code), code)

    def test_every_allowlisted_attribute_passes_validation(self):
        for module, allowed in safe_pandas.MODULE_ALLOWED_ATTRS.items():
            for attr in sorted(allowed):
                code = f"print({module}.{attr})"
                self.assertEqual(safe_pandas.validate_code(code), "", code)

    def test_module_allowlists_have_no_dunder_or_submodule(self):
        for module, allowed in safe_pandas.MODULE_ALLOWED_ATTRS.items():
            for attr in allowed:
                self.assertFalse(attr.startswith("_"), f"{module}.{attr}")
                self.assertNotIn(".", attr, f"{module}.{attr}")

    def test_math_public_only(self):
        self.assertEqual(safe_pandas.validate_code("print(math.floor(1.5))"), "")
        self.assertEqual(safe_pandas.validate_code("print(math.pi, math.e)"), "")
        self.assertTrue(safe_pandas.validate_code("print(math.__dict__)"))
        self.assertTrue(safe_pandas.validate_code("print(math.frexp.__globals__)"))


class AttributeBlocklistTests(unittest.TestCase):
    def test_introspection_prefixes_blocked(self):
        for attr in ("gi_frame", "gi_code", "cr_frame", "cr_code", "ag_frame",
                     "f_globals", "f_builtins", "f_back", "co_code", "co_consts",
                     "tb_frame", "tb_next", "im_func", "im_self", "func_code",
                     "func_globals", "__class__", "__bases__", "__subclasses__",
                     "__globals__", "__builtins__", "__import__", "__reduce__",
                     "__getattribute__", "__init_subclass__"):
            code = f"print(df.{attr})"
            self.assertTrue(safe_pandas.validate_code(code), code)
            self.assertTrue(safe_pandas._attribute_blocked(attr), attr)

    def test_disk_and_serialization_attrs_blocked(self):
        for attr in ("tofile", "fromfile", "dump", "dumps", "load", "loads", "save",
                     "savetxt", "savez", "savez_compressed", "loadtxt", "genfromtxt",
                     "fromregex", "memmap", "open_memmap", "DataSource", "ctypes",
                     "ctypeslib", "lib", "f2py", "distutils", "os", "sys", "builtins",
                     "globals", "locals", "vars", "getattr", "setattr", "delattr",
                     "eval", "exec", "compile", "query", "pipe", "style", "plot",
                     "system", "popen", "format", "format_map", "mro", "subclasses",
                     "bases", "io", "api", "options", "set_option", "compat", "core",
                     "testing", "util", "read_csv", "read_excel", "to_csv", "to_json",
                     "to_pickle", "to_excel", "to_sql"):
            self.assertTrue(safe_pandas._attribute_blocked(attr), attr)
            self.assertTrue(safe_pandas.validate_code(f"print(df.{attr})"), attr)

    def test_normal_attributes_still_allowed(self):
        for attr in ("base", "values", "index", "columns", "shape", "dtypes", "size",
                     "empty", "iloc", "loc", "str", "dt", "cat", "sum", "mean",
                     "to_string", "to_dict", "tolist", "to_numpy", "to_frame",
                     "to_datetime", "to_numeric", "to_timedelta", "groupby"):
            self.assertFalse(safe_pandas._attribute_blocked(attr), attr)
        self.assertEqual(safe_pandas.validate_code("print(df.values.base)"), "")
        self.assertEqual(safe_pandas.validate_code("print(df.shape, df.size, df.empty)"), "")


class StringDispatchTests(unittest.TestCase):
    def test_blocked_strings_rejected(self):
        for value in ("to_json", "to_pickle", "to_csv", "eval", "query", "pipe",
                      "tofile", "gi_frame", "f_globals", "read_csv", "system",
                      "popen", "open", "getattr", "compile", "exec", "globals",
                      "__class__", "__builtins__", "__import__", "os"):
            for code in (f"print(df.agg({value!r}))",
                         f"print(df.apply({value!r}, axis=1))",
                         f"print(df.groupby('品名')['金额'].agg({value!r}))"):
                self.assertTrue(safe_pandas.validate_code(code), code)

    def test_normal_and_chinese_strings_still_ok(self):
        for value in ("sum", "mean", "count", "min", "max", "std", "median",
                      "first", "last", "nunique", "size", "品名", "金额", "部门",
                      "数量", "2024-01-01", "%Y-%m", "ME", "all", "coerce",
                      "升序", "合计: {}"):
            code = f"print(df.agg({value!r}))"
            self.assertEqual(safe_pandas.validate_code(code), "", code)

    def test_chinese_column_workflows_accepted(self):
        for code in (
            "print(df['金额'].sum())",
            "print(df.groupby('部门')['金额'].sum().to_dict())",
            "print(df.sort_values('数量', ascending=False)['品名'].tolist())",
            "print(df.rename(columns={'品名': '名称'}).to_string())",
            "print(f\"合计：{df['金额'].sum():.2f} 元\")",
            "print(df[df['数量'] > 3]['品名'].tolist())",
            "print(df['品名'].str.len().sum())",
            "print(df.pivot_table(index='部门', values='金额', aggfunc='sum').to_string())",
        ):
            self.assertEqual(safe_pandas.validate_code(code), "", code)


class StringDataTests(unittest.TestCase):
    """字符串常量按上下文校验：派发位置才按属性黑名单，普通位置一律当数据放行。"""

    ALLOWED = [
        "print(df.rename(columns={'_qty': '数量'}))",
        "print(df.groupby('部门').agg(_qty=('数量','sum')).to_dict())",
        "result = df.groupby('部门').agg(_qty=('数量','sum'))\n"
        "print(result.rename(columns={'_qty':'数量'}).to_dict())",
        "print(df['_qty'] if '_qty' in df.columns else '无')",
        "print('_qty', '_备注')",
        "print(df.groupby('_secret_col')['数量'].sum().to_dict())",
    ]

    REJECTED = [
        "print(df.agg('to_json'))",
        "print(df.apply('eval'))",
        "print(df['数量'].transform('to_csv'))",
        "print(df.groupby('部门').agg(y=('数量','to_json')))",
        "print(df.agg({'数量':'to_json'}))",
        "print(df.pipe('__class__'))",
        "print('__class__')",
        "print(df.agg('_mgr'))",
        "print('_qty', 'to_json')",
        "print(df.agg('_constructor'))",
        "x = '_values'\nprint(x)",
        "print(df.agg('_data'))",
        # 间接引用（无法靠常量位置识别，统一按字符串内容校验）
        "f = 'to_json'\nprint(df.agg(f))",
        "d = {'a': 'to_json'}\nprint(df.agg(d))",
        "fs = ['sum', '_mgr']\nprint(df.agg(fs[1]))",
        "for f in ['to_json']:\n    print(df.agg(f))",
        "g = (f := 'eval')\nprint(df.apply(f))",
        "l = []\nl.append('_mgr')\nprint(df.agg(l[0]))",
    ]

    def test_data_strings_allowed(self):
        for code in self.ALLOWED:
            self.assertEqual(safe_pandas.validate_code(code), "", code)

    def test_dispatch_strings_still_rejected(self):
        for code in self.REJECTED:
            self.assertTrue(safe_pandas.validate_code(code), code)

    def test_named_agg_underscore_output_runs(self):
        rows = [["A", "3"], ["B", "5"], ["A", "2"]]
        columns = ["部门", "数量"]
        code = "g = df.groupby('部门').agg(_qty=('数量','sum'))\nprint(g['_qty'].to_dict())"
        result = safe_pandas.run_restricted(code, rows, columns, timeout=60)
        self.assertEqual(result["error"], "", result)
        self.assertEqual(result["code_output"].strip(), "{'A': 5, 'B': 5}")


class PositiveRunTests(unittest.TestCase):
    """中文列名 + 常用聚合，必须在真子进程里跑出正确结果。"""

    def run_ok(self, code, rows=ROWS, columns=COLUMNS, timeout=60):
        result = safe_pandas.run_restricted(code, rows, columns, timeout=timeout)
        self.assertEqual(result["error"], "", f"{code!r} -> {result}")
        return result["code_output"]

    def test_chinese_columns_aggregations(self):
        self.assertEqual(self.run_ok("print(df['数量'].sum())").strip(), "10.0")
        self.assertEqual(
            self.run_ok("print(df.groupby('品名')['数量'].sum().to_dict())").strip(),
            "{'苹果': 10.0, '香蕉': 0.0}",
        )
        self.assertEqual(
            self.run_ok("print(df.sort_values('数量', ascending=False)['品名'].tolist())").strip(),
            "['苹果', '苹果', '香蕉']",
        )
        self.assertEqual(self.run_ok("print(df['金额'].max())").strip(), "10.5")
        self.assertEqual(self.run_ok("print(round(df['金额'].mean(), 2))").strip(), "4.67")
        out = self.run_ok("print(df.describe().to_string())")
        self.assertIn("count", out)
        self.assertIn("金额", out)

    def test_module_allowlist_functions_work(self):
        self.assertEqual(self.run_ok("print(np.nan, pd.NA)").strip(), "nan <NA>")
        self.assertEqual(
            self.run_ok("print(pd.to_numeric(df['数量'], errors='coerce').sum())").strip(),
            "10.0",
        )
        self.assertEqual(self.run_ok("print(math.floor(df['金额'].max()))").strip(), "10")
        self.assertEqual(
            self.run_ok("print(round(math.sqrt(df['金额'].max()), 2))").strip(), "3.24"
        )
        self.assertEqual(self.run_ok("print(np.nansum(df['数量']))").strip(), "10.0")
        self.assertEqual(
            self.run_ok("print(pd.to_datetime('2024-01-02').year)").strip(), "2024"
        )
        self.assertEqual(
            self.run_ok("print(pd.Series(df['金额']).round(1).tolist())").strip(),
            "[10.5, 2.0, 1.5]",
        )

    def test_groupby_agg_with_string_list_still_works(self):
        out = self.run_ok("print(df.groupby('品名')['金额'].agg(['sum', 'mean']).to_dict())")
        self.assertIn("苹果", out)
        self.assertIn("mean", out)


class AuditHookDefenseTests(unittest.TestCase):
    """绕过 AST 校验，验证子进程里的审计钩子确实兜得住。"""

    SECRET = "SECRET-TOKEN-VALUE"

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="sandbox-test-")
        self.secret_path = os.path.join(self.tmp, "token.txt")
        with open(self.secret_path, "w", encoding="utf-8") as fh:
            fh.write(self.SECRET)
        self.builtins_line = "bi = np.ma.extras.ma.builtins\n"

    def tearDown(self):
        for name in os.listdir(self.tmp):
            os.remove(os.path.join(self.tmp, name))
        os.rmdir(self.tmp)

    def assert_blocked(self, code, fragment="沙箱禁止"):
        result = _run_child_without_validator(code)
        self.assertTrue(result.get("error"), f"应该被拦截: {code!r} -> {result}")
        if fragment:
            # fragment=None：钩子确实抛了 PermissionError，但 numpy 的 C 代码把
            # 异常吞掉后报 SystemError（例如 np.fromfile / ndarray.tofile）。
            self.assertIn(fragment, result["error"], code)
        combined = result["error"] + result.get("code_output", "")
        self.assertNotIn(self.SECRET, combined, code)
        self.assertNotIn("root:", combined, code)
        return result

    def test_open_secret_file_blocked(self):
        self.assert_blocked(self.builtins_line + f"print(bi.open({self.secret_path!r}).read())")
        self.assert_blocked(f"print(pd.read_csv({self.secret_path!r}))")
        self.assert_blocked(f"print(np.genfromtxt({self.secret_path!r}, dtype=str))")
        self.assert_blocked(f"print(np.loadtxt({self.secret_path!r}, dtype=str))")
        self.assert_blocked(f"print(np.fromfile({self.secret_path!r}, dtype='S1'))", None)
        self.assert_blocked(self.builtins_line + "print(bi.open('/etc/passwd').read())")
        self.assert_blocked(self.builtins_line + "print(bi.open('/proc/self/environ').read())")

    def test_write_blocked(self):
        target = os.path.join(self.tmp, "escape.txt")
        self.assert_blocked(self.builtins_line + f"print(bi.open({target!r}, 'w'))")
        self.assert_blocked(f"print(df.values.tofile({target!r}))", None)
        self.assert_blocked(f"print(df.values.dump({target!r}))", None)
        self.assertFalse(os.path.exists(target))

    def test_socket_blocked(self):
        self.assert_blocked("import socket\nprint(socket.socket())", "沙箱禁止导入模块：socket")
        self.assert_blocked(self.builtins_line + "print(bi.__import__('socket').socket())")

    def test_os_and_subprocess_blocked(self):
        self.assert_blocked(self.builtins_line + "print(bi.__import__('os').listdir('/'))")
        self.assert_blocked(self.builtins_line + "print(bi.__import__('os').system('id'))")
        self.assert_blocked(
            self.builtins_line + "print(bi.__import__('subprocess').check_output(['id']))"
        )
        self.assert_blocked(self.builtins_line + "print(bi.__import__('ctypes').CDLL(None))")

    def test_second_exec_and_compile_blocked(self):
        self.assert_blocked(self.builtins_line + "print(bi.exec('print(1)'))")
        self.assert_blocked(self.builtins_line + "print(bi.eval('1+1'))")
        self.assert_blocked(
            self.builtins_line + "print(bi.compile('print(1)', '<string>', 'exec'))"
        )

    def test_module_mutation_rejected_by_validator(self):
        for code in (
            "print(object.__setattr__(pd, 'NA', 1))",
            "setattr(pd, 'NA', 1)\nprint(pd.NA)",
            "print(pd.__dict__)",
            "print(np.__dict__)",
            "print(df.__class__.__bases__)",
        ):
            self.assertTrue(safe_pandas.validate_code(code), code)

    def test_normal_pandas_work_still_succeeds_with_hook_armed(self):
        rows = [["苹果", "3", "10.5", "2024-01-02"], ["香蕉", "", "2", "2024-02-03"],
                ["苹果", "7", "1.5", "2024-03-04"]]
        cols = ["品名", "数量", "金额", "日期"]
        code = "\n".join([
            "print(df.dtypes.to_dict()['金额'])",
            "print(df.groupby('品名')['数量'].sum().to_dict())",
            "print(df.sort_values('金额', ascending=False)['品名'].tolist())",
            "print(round(df['金额'].mean(), 2))",
            "d = df.copy()",
            "d['日期'] = pd.to_datetime(d['日期'])",
            "print(d['日期'].dt.strftime('%Y-%m').tolist())",
            "print(np.nansum(df['数量']))",
            "print(math.floor(df['金额'].max()))",
            "print(df.describe().shape)",
            "print('完成')",
        ])
        result = safe_pandas.run_restricted(code, rows, cols, timeout=60)
        self.assertEqual(result["error"], "")
        lines = result["code_output"].strip().splitlines()
        self.assertEqual(lines[-1], "完成")
        self.assertEqual(lines[1], "{'苹果': 10.0, '香蕉': 0.0}")
        self.assertEqual(lines[3], "4.67")


class PublicApiNeverSkipsValidationTests(unittest.TestCase):
    def test_run_restricted_source_has_no_skip_flag(self):
        import inspect
        source = inspect.getsource(safe_pandas.run_restricted)
        self.assertNotIn("skip_validation", source)
        self.assertNotIn("_run_child_for_test", source)

    def test_child_revalidates_even_if_parent_validator_patched(self):
        original = safe_pandas.validate_code
        safe_pandas.validate_code = lambda code: ""
        try:
            result = safe_pandas.run_restricted(
                "print(open('/etc/passwd').read())", ROWS, COLUMNS
            )
        finally:
            safe_pandas.validate_code = original
        self.assertTrue(result["error"])
        self.assertEqual(result["code_output"], "")
        self.assertIn("open", result["error"])

    def test_test_helper_is_private_and_opt_in(self):
        self.assertTrue(hasattr(safe_pandas, "_run_child_for_test"))
        # 默认不跳过校验：即使走测试入口，危险代码也会被拒绝
        result = safe_pandas._run_child_for_test(
            "print(open('/etc/passwd').read())", ROWS, COLUMNS, timeout=30
        )
        self.assertIn("open", result["error"])


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



class InstanceAttrStringTests(unittest.TestCase):
    """实例属性（不在类 dir() 里）当字符串用也要拒绝，例如 df.agg('_grouper')。"""

    def test_instance_only_underscore_attrs_rejected(self):
        for name in ("_attrs", "_flags", "_grouper", "_mgr"):
            code = f"print(df.agg('{name}'))"
            self.assertTrue(safe_pandas.validate_code(code), code)

    def test_plain_underscore_data_still_allowed(self):
        self.assertEqual(safe_pandas.validate_code("print(df.rename(columns={'_qty': '数量'}))"), "")


if __name__ == "__main__":
    unittest.main()
