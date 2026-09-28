"""受限 pandas 代码沙箱（AST 白名单 + 独立子进程执行）。

本模块同时被 web 端（仓库根目录）和 MCP 服务端（仓库 mcp_server/，部署到
~/tencent-docs-mcp）使用，两个副本必须保持逐字节一致。

安全模型：
- validate_code: 纯 AST 白名单校验（允许节点类型、允许名称、pd/np/math 属性白名单、
  通用属性黑名单、字符串常量黑名单、必须 print、长度上限），在调用 MCP / 启动子
  进程之前给模型反馈。
- run_restricted: 在独立子进程里执行（sys.executable -I，隔离环境变量与
  当前目录），最小化 env（不继承任何父进程密钥），resource 限制
  CPU / 地址空间 / 文件大小 / 进程数 / fd 数，超时用 killpg 杀掉整个
  进程组。子进程通过 stdin 收 JSON、stdout 回 JSON；父进程绝不向真实
  stdout 打印任何东西（MCP server 的 stdout 是 JSON-RPC 通道）。
- 子进程里还会先预热 pandas/numpy 的惰性导入，然后装上 sys.addaudithook
  作为运行时兜底：即使 AST 校验被绕过，文件访问 / 进程 / 网络 / ctypes /
  动态 exec 也会在审计事件处被拒绝。
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import logging
import os
import signal
import subprocess
import sys
import tempfile

SANDBOX_TIMEOUT_ENV = "SANDBOX_EXEC_TIMEOUT_SECONDS"
DEFAULT_SANDBOX_TIMEOUT_SECONDS = 20.0
MAX_CODE_LENGTH = 4000
MAX_OUTPUT_CHARS = 8000

#: 代码里可以直接 Load 的基础名称（不含后面动态收集的赋值名）。
#: pd / np / math 不在这里：它们只能以 `pd.属性` 的形式出现（见 MODULE_ALLOWED_ATTRS）。
ALLOWED_BASE_NAMES = frozenset({"df"})

#: 提供给沙箱 __builtins__ 的白名单内置函数/类型。
ALLOWED_BUILTIN_NAMES = (
    "print", "len", "range", "sum", "min", "max", "abs", "round", "sorted",
    "enumerate", "zip", "list", "dict", "set", "tuple", "str", "int", "float",
    "bool", "any", "all", "isinstance", "reversed", "map", "filter",
)

ALLOWED_NAMES = ALLOWED_BASE_NAMES | frozenset(ALLOWED_BUILTIN_NAMES)

#: pandas 白名单：只有这些属性可以通过 `pd.X` 访问（不含任何子模块）。
PD_ALLOWED_ATTRS = frozenset({
    "DataFrame", "Series", "Index", "Categorical", "Interval", "Period",
    "Timestamp", "Timedelta", "NA", "NaT",
    "to_numeric", "to_datetime", "to_timedelta",
    "concat", "merge", "merge_asof", "melt", "wide_to_long", "pivot_table",
    "crosstab", "get_dummies", "cut", "qcut", "factorize", "json_normalize",
    "isna", "isnull", "notna", "notnull", "unique",
    "date_range", "bdate_range", "period_range", "timedelta_range",
})

#: numpy 白名单：只允许顶层函数/常量/标量类型，任何子模块（np.random、np.lib、
#: np.ctypeslib、np.f2py、np.ma...）一律禁止，因为子模块上挂着 os/sys/ctypes/
#: builtins 以及 genfromtxt/loadtxt/memmap 这类能读磁盘的函数。
NP_ALLOWED_ATTRS = frozenset({
    "nan", "inf", "pi", "e", "bool_", "str_", "object_", "dtype",
    "int64", "int32", "float64", "float32", "datetime64", "timedelta64",
    "isnan", "isfinite", "isinf", "nansum", "nanmean", "nanmedian", "nanmax",
    "nanmin", "nanstd", "sum", "mean", "median", "std", "var", "min", "max",
    "amin", "amax", "abs", "absolute", "round", "around", "floor", "ceil",
    "sqrt", "log", "log10", "log2", "exp", "power", "sign", "mod",
    "where", "select", "clip", "array", "asarray", "arange", "linspace",
    "zeros", "ones", "full", "zeros_like", "ones_like", "full_like",
    "cumsum", "cumprod", "diff", "sort", "argsort", "argmax", "argmin",
    "unique", "percentile", "quantile", "average", "corrcoef", "cov",
    "histogram", "bincount", "digitize", "count_nonzero", "any", "all",
    "maximum", "minimum", "divide", "multiply", "add", "subtract",
    "floor_divide", "true_divide", "prod", "isin", "in1d", "intersect1d",
    "union1d", "setdiff1d", "concatenate", "stack", "vstack", "hstack",
    "reshape", "transpose", "dot",
})

#: math 白名单：只允许公开的函数与常量。
MATH_ALLOWED_ATTRS = frozenset({
    "e", "pi", "tau", "inf", "nan", "isnan", "isinf", "isfinite",
    "fabs", "ceil", "floor", "trunc", "fmod", "remainder", "gcd", "lcm",
    "exp", "expm1", "log", "log1p", "log2", "log10", "pow", "sqrt", "isqrt",
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "hypot", "dist",
    "degrees", "radians", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
    "erf", "erfc", "gamma", "lgamma", "copysign", "comb", "perm", "prod",
    "frexp", "ldexp", "modf", "nextafter", "ulp",
})

#: 模块名 -> 允许访问的属性集合。这些名字只能作为属性访问的目标出现
#: （`pd.to_numeric(...)`），不能裸用、不能赋值给别的名字、不能被重新绑定。
MODULE_ALLOWED_ATTRS: dict[str, frozenset[str]] = {
    "pd": PD_ALLOWED_ATTRS,
    "np": NP_ALLOWED_ATTRS,
    "math": MATH_ALLOWED_ATTRS,
}

#: 明确放行的 to_* 属性。
ATTRIBUTE_ALLOWLIST = frozenset({
    "to_string", "to_dict", "to_list", "tolist", "to_numpy", "to_frame",
    "to_datetime", "to_numeric", "to_timedelta",
})

#: 属性名黑名单前缀：帧/代码对象/生成器/协程/方法的内部属性一律禁止
#: （gi_frame.f_globals 能拿到真正的 builtins，从而拿到 eval/exec/__import__）。
ATTRIBUTE_BLOCKED_PREFIXES = (
    "_", "gi_", "cr_", "ag_", "f_", "co_", "tb_", "im_", "func_",
)

#: 属性黑名单（另外所有 "_" 开头的属性一律禁止；read_*、to_* 默认禁止）。
ATTRIBUTE_BLOCKLIST = frozenset({
    "eval", "exec", "compile", "query", "pipe", "style", "plot", "hist",
    "boxplot", "io", "api", "options", "set_option", "compat", "core",
    "testing", "util", "system", "popen", "open", "format", "format_map",
    "mro", "subclasses", "bases", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "import_module", "modules",
    # 磁盘读写 / 序列化：能从同一个 Unix 用户读到 token、.env 等敏感文件
    "load", "loads", "save", "dump", "dumps", "loadtxt", "genfromtxt",
    "fromfile", "fromregex", "frombuffer", "fromstring", "tofile",
    "savetxt", "savez", "savez_compressed", "memmap", "open_memmap",
    "DataSource", "pickle",
    # numpy/pandas 子模块里挂着 os/sys/subprocess/ctypes/builtins 等模块引用
    # （例如 np.f2py.os、pd.errors.ctypes、np.ma.extras.ma.builtins）。
    # 真正的边界是子进程里的审计钩子，这里只是尽早给模型反馈。
    "f2py", "ctypeslib", "ctypes", "lib", "distutils", "errors", "ma",
    "random", "linalg", "char", "rec", "fft", "polynomial", "strings",
    "builtins", "os", "sys", "subprocess", "socket", "shutil", "pathlib",
    "importlib", "inspect", "gc", "signal", "threading", "multiprocessing",
    "code", "types", "collections", "functools", "itertools", "warnings",
    "getframe", "settrace", "setprofile", "addaudithook",
})

_ALLOWED_NODES = frozenset({
    ast.Module, ast.Expr, ast.Assign, ast.AugAssign, ast.AnnAssign,
    ast.If, ast.For, ast.While, ast.Break, ast.Continue, ast.Pass,
    ast.BoolOp, ast.NamedExpr, ast.BinOp, ast.UnaryOp, ast.Lambda,
    ast.IfExp, ast.Dict, ast.Set, ast.List, ast.Tuple,
    ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
    ast.Compare, ast.Call, ast.keyword, ast.Attribute, ast.Subscript,
    ast.Starred, ast.Name, ast.Load, ast.Store, ast.Constant,
    ast.JoinedStr, ast.FormattedValue, ast.Slice, ast.comprehension,
    ast.arguments, ast.arg,
    ast.And, ast.Or, ast.Not, ast.Invert, ast.UAdd, ast.USub,
    ast.Add, ast.Sub, ast.Mult, ast.MatMult, ast.Div, ast.Mod, ast.Pow,
    ast.LShift, ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd, ast.FloorDiv,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.Is, ast.IsNot, ast.In, ast.NotIn,
})


def _attribute_blocked(attr: str) -> bool:
    """属性名是否被禁止（同时用于字符串常量，防 df.agg('to_json') 这类字符串派发）。"""
    text = str(attr)
    if text.startswith(ATTRIBUTE_BLOCKED_PREFIXES):
        return True
    if text in ATTRIBUTE_ALLOWLIST:
        return False
    if text in ATTRIBUTE_BLOCKLIST:
        return True
    if text.startswith("read_") or text.startswith("to_"):
        return True
    return False


def _collect_bound_names(tree: ast.AST) -> set[str]:
    """收集代码里所有被绑定的名称（赋值、for、推导式、walrus、lambda 参数）。"""
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
    return bound


def _module_attr_target_ids(tree: ast.AST) -> set[int]:
    """收集「作为属性访问目标」出现的模块 Name 节点 id（即 pd.X 里的那个 pd）。"""
    return {
        id(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id in MODULE_ALLOWED_ATTRS
    }


def _string_constant_reason(value: str) -> str:
    """字符串常量校验：防 df.agg('to_json') / df.apply('eval') 这类字符串派发绕过。"""
    if "__" in value:
        return "代码包含禁止的字符串（含有双下划线）"
    if value.isidentifier() and _attribute_blocked(value):
        return f"代码包含禁止的字符串：{value}"
    return ""


def validate_code(code: object) -> str:
    """校验模型生成的 pandas 代码。合法返回 ""，否则返回中文原因。"""
    text = str(code or "")
    if not text.strip():
        return "代码为空"
    if len(text) > MAX_CODE_LENGTH:
        return f"代码过长（超过 {MAX_CODE_LENGTH} 字符），请精简后重试"
    try:
        tree = ast.parse(text, mode="exec")
    except SyntaxError as exc:
        return f"代码语法错误：{exc.msg}"

    nodes = list(ast.walk(tree))
    for node in nodes:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return "代码包含禁止内容：import"
        if type(node) not in _ALLOWED_NODES:
            return f"代码包含不允许的语法：{type(node).__name__}"

    # pd / np / math 不能被重新绑定（`pd = ...`、`for np in ...`、lambda 参数等）。
    for node in nodes:
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            if node.id in MODULE_ALLOWED_ATTRS:
                return f"禁止给模块名重新赋值：{node.id}"
        elif isinstance(node, ast.arg) and node.arg in MODULE_ALLOWED_ATTRS:
            return f"禁止把模块名用作参数名：{node.arg}"

    # 属性访问：模块属性走各自的白名单，其余对象走通用黑名单。
    for node in nodes:
        if not isinstance(node, ast.Attribute):
            continue
        if isinstance(node.value, ast.Name) and node.value.id in MODULE_ALLOWED_ATTRS:
            if node.attr not in MODULE_ALLOWED_ATTRS[node.value.id]:
                return f"代码包含禁止访问的属性：{node.value.id}.{node.attr}"
        elif _attribute_blocked(node.attr):
            return f"代码包含禁止访问的属性：{node.attr}"

    bound_names = _collect_bound_names(tree)
    call_func_ids = {
        node.func.id
        for node in nodes
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    module_targets = _module_attr_target_ids(tree)
    has_print = False
    for node in nodes:
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id in MODULE_ALLOWED_ATTRS:
                # 只允许 pd.X 这种形式；x = np / [np][0] / f(pd) 一律拒绝，
                # 否则拿到模块对象就能取到 genfromtxt、ctypeslib 等危险成员。
                if id(node) not in module_targets:
                    return f"禁止直接使用模块 {node.id}，只能写成 {node.id}.白名单属性"
                continue
            if node.id in ALLOWED_NAMES or node.id in bound_names:
                continue
            if node.id in call_func_ids:
                return f"代码包含禁止调用：{node.id}"
            return f"代码包含禁止使用的名称：{node.id}"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
            has_print = True

    for node in nodes:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            reason = _string_constant_reason(node.value)
            if reason:
                return reason
    if not has_print:
        return "代码必须包含 print 调用"
    return ""


def sandbox_exec_timeout_seconds() -> float:
    raw = os.environ.get(SANDBOX_TIMEOUT_ENV, "")
    try:
        value = float(raw) if str(raw).strip() else DEFAULT_SANDBOX_TIMEOUT_SECONDS
    except (TypeError, ValueError):
        value = DEFAULT_SANDBOX_TIMEOUT_SECONDS
    return value if value > 0 else DEFAULT_SANDBOX_TIMEOUT_SECONDS


# ---------------------------------------------------------------------------
# 子进程侧（在沙箱里运行）
# ---------------------------------------------------------------------------

#: pandas/numpy 惰性导入时会从当前帧的 builtins 里取 __import__；直接删掉会让
#: df.dtypes 之类的操作抛 KeyError。这里提供一个受控版本作为兜底：即使校验被
#: 绕过，危险顶层模块也无法导入（用户代码本身引用不到 __import__ 这个名称）。
_IMPORT_DENY_TOP = frozenset({
    "os", "sys", "subprocess", "socket", "shutil", "pathlib", "ctypes",
    "importlib", "builtins", "code", "codeop", "pickle", "signal",
    "threading", "multiprocessing", "http", "urllib", "requests",
    "resource", "tempfile", "webbrowser", "sqlite3", "asyncio",
    "concurrent", "io", "posix", "nt", "pty", "fcntl", "grp", "pwd",
    "marshal", "gc", "inspect", "runpy", "pkgutil", "zipimport", "site",
    "sysconfig", "platform", "faulthandler", "tracemalloc", "linecache",
    "mmap", "select", "selectors", "ssl", "_io", "_thread", "_socket",
    "_ctypes", "_posixsubprocess", "_signal", "_imp", "_frozen_importlib",
    "_frozen_importlib_external", "posixpath", "genericpath", "glob",
})


#: 允许惰性导入（含新子模块）的顶层包。
_IMPORT_ALLOW_TOP = frozenset({"numpy", "pandas", "dateutil", "pytz", "tzdata"})
#: 进入沙箱时已加载的模块名快照（在 _child_main 里填充）。
_PRELOADED_MODULES: frozenset = frozenset()


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    """白名单式 __import__：只允许 numpy/pandas 等包，以及进入沙箱前已经加载的
    非危险模块（pandas 函数体内的 `import warnings` 之类）。相对导入只在
    白名单包内部发生。危险顶层模块即使已加载也拒绝。"""
    text = str(name or "")
    top = text.split(".")[0]
    if level == 0:
        if top in _IMPORT_DENY_TOP:
            raise ImportError(f"沙箱禁止导入模块：{top}")
        if top not in _IMPORT_ALLOW_TOP and text not in _PRELOADED_MODULES:
            raise ImportError(f"沙箱禁止导入模块：{text}")
    else:
        package = (globals or {}).get("__package__") or ""
        if str(package).split(".")[0] not in _IMPORT_ALLOW_TOP:
            raise ImportError("沙箱禁止相对导入")
    return __import__(name, globals, locals, fromlist, level)


def _sandbox_read_roots() -> tuple[str, ...]:
    """允许只读打开 / 列目录的根目录：Python 标准库和 site-packages（惰性导入需要）。"""
    import sysconfig
    roots = set()
    for key in ("stdlib", "platstdlib", "purelib", "platlib"):
        path = sysconfig.get_paths().get(key)
        if path:
            roots.add(os.path.realpath(path))
    for name in ("numpy", "pandas"):
        mod = sys.modules.get(name)
        if mod is not None and getattr(mod, "__file__", None):
            roots.add(os.path.realpath(os.path.dirname(os.path.dirname(mod.__file__))))
    roots.add("/usr/share/zoneinfo")
    return tuple(sorted(r.rstrip("/") + "/" for r in roots))


_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND

#: 用户代码在沙箱里编译时用的文件名（审计钩子靠它区分“用户代码触发”与
#: “pandas/numpy 库内部触发”）。
_SANDBOX_FILENAME = "<sandbox>"

#: 只允许库内部使用的 sys 级事件（用户代码无论如何都拿不到 sys，这里是兜底）。
_BLOCKED_SYS_EVENTS = frozenset({
    "sys._getframe", "sys.settrace", "sys.setprofile", "sys.addaudithook",
    "sys.excepthook", "sys.unraisablehook", "sys.remote_exec",
    "sys.set_asyncgen_hooks", "sys.set_coroutine_origin_tracking_depth",
})

#: 审计钩子内部状态：inside 防止钩子自身调用 sys._getframe 时递归拒绝；
#: user_exec_used 保证用户代码只被父进程 exec 一次。
_HOOK_STATE = {"inside": False, "user_exec_used": False}


def _user_initiated(max_depth: int = 8) -> bool:
    """当前审计事件是否由沙箱里的用户代码直接触发。

    库内部的惰性 import / compile（pandas 里 `import warnings`、functools 编译
    lru_cache 源码等）必须放行，否则正常的 df.groupby(...).sum() 都会失败；
    从用户代码帧（co_filename == "<sandbox>"）直接发起的一律按危险处理。
    """
    try:
        frame = sys._getframe(1)
    except ValueError:
        return False
    depth = 0
    while frame is not None and depth < max_depth:
        name = frame.f_code.co_filename
        if name == _SANDBOX_FILENAME:
            return True
        if not name.endswith("safe_pandas.py"):
            return False
        frame = frame.f_back
        depth += 1
    return False


def _install_audit_hook() -> None:
    """运行时兜底：无论用户代码经由哪条属性链拿到 os/subprocess/ctypes/socket，
    危险操作都会在审计事件处被拒绝。

    - open / io.open_code：只允许只读打开 Python 库目录下的文件（pandas/numpy
      惰性导入需要），其余路径（含 /proc、应用目录下的 .env / token）一律拒绝，
      写模式一律拒绝；
    - import：用户代码触发的危险顶层模块一律拒绝（库内部惰性导入放行）；
    - exec / compile：只放行父进程那一次 "<sandbox>" 的 exec，用户代码再想
      exec/compile 就断掉（拿不到第二次动态执行代码的机会）；
    - os.listdir/os.scandir：只允许库目录；其余 os.* 事件（system/exec/spawn/fork/
      kill/remove/rename/chmod/putenv...）全部拒绝；
    - subprocess/socket/ctypes/mmap/resource/signal/pty/webbrowser 等一律拒绝；
    - object.__setattr__/__delattr__：用户代码不许改模块或类的属性；
    - sys._getframe/settrace/setprofile/addaudithook 一律拒绝。
    审计钩子一旦安装无法移除。
    """
    import types

    roots = _sandbox_read_roots()

    def _under_roots(path) -> bool:
        if isinstance(path, int):
            return False
        if isinstance(path, bytes):
            path = path.decode("utf-8", "replace")
        if not isinstance(path, str):
            return False
        real = os.path.realpath(path)
        return any((real + "/").startswith(root) for root in roots)

    blocked_prefixes = (
        "subprocess.", "socket.", "ctypes.", "mmap.", "resource.", "signal.",
        "pty.", "fcntl.", "webbrowser.", "urllib.", "http.", "ftplib.",
        "smtplib.", "imaplib.", "poplib.", "nntplib.", "telnetlib.",
        "sqlite3.", "shutil.", "glob.", "winreg.", "msvcrt.", "_winapi.",
        "syslog.", "ensurepip.", "sys.remote_exec", "sys.addaudithook",
        "cpython.remote_debugger",
    )

    def _check(event: str, args: tuple) -> None:
        if event in ("open", "io.open_code"):
            path = args[0] if args else None
            mode = args[1] if len(args) > 1 else None
            flags = args[2] if len(args) > 2 else 0
            writable = False
            if isinstance(mode, str) and any(ch in mode for ch in "wax+"):
                writable = True
            if isinstance(flags, int) and flags & _WRITE_FLAGS:
                writable = True
            if writable or not _under_roots(path):
                raise PermissionError("沙箱禁止访问文件")
            return
        if event == "import":
            top = str(args[0] if args else "").split(".")[0]
            if top in _IMPORT_DENY_TOP and _user_initiated():
                raise PermissionError(f"沙箱禁止导入模块：{top}")
            return
        if event == "exec":
            filename = getattr(args[0] if args else None, "co_filename", "")
            if filename == _SANDBOX_FILENAME:
                if _HOOK_STATE["user_exec_used"]:
                    raise PermissionError("沙箱禁止操作：exec")
                _HOOK_STATE["user_exec_used"] = True
                return
            if _user_initiated():
                raise PermissionError("沙箱禁止操作：exec")
            return
        if event == "compile":
            filename = args[1] if len(args) > 1 else ""
            if filename == _SANDBOX_FILENAME or _user_initiated():
                raise PermissionError("沙箱禁止操作：compile")
            return
        if event in ("object.__setattr__", "object.__delattr__"):
            target = args[0] if args else None
            if isinstance(target, (types.ModuleType, type)) and _user_initiated():
                raise PermissionError("沙箱禁止修改模块或类型")
            return
        if event in _BLOCKED_SYS_EVENTS:
            raise PermissionError(f"沙箱禁止操作：{event}")
        if event in ("os.listdir", "os.scandir"):
            if args and args[0] is not None and _under_roots(args[0]):
                return
            raise PermissionError("沙箱禁止列目录")
        if event.startswith("os.") or event.startswith("posix."):
            raise PermissionError(f"沙箱禁止操作：{event}")
        if event.startswith(blocked_prefixes):
            raise PermissionError(f"沙箱禁止操作：{event}")

    def _hook(event: str, args: tuple) -> None:
        if _HOOK_STATE["inside"]:
            return
        _HOOK_STATE["inside"] = True
        try:
            _check(event, args)
        finally:
            _HOOK_STATE["inside"] = False

    sys.addaudithook(_hook)


def _warm_up() -> None:
    """在装上审计钩子之前跑一遍常用操作，让 pandas/numpy 的惰性 import 都完成。

    钩子只允许 Python 库目录内的文件访问，库代码自己的 import 不受影响；这里
    预热主要是减少沙箱内触发惰性导入的次数，顺带把可能的告警提前消化掉。
    任何一步失败都忽略（不同 pandas 版本 API 略有差异）。
    """
    import warnings

    import numpy as np
    import pandas as pd

    frame = pd.DataFrame({"数量": [1.0, 2.0, None], "品名": ["x", "y", "z"]})
    dated = pd.DataFrame({
        "数量": [1.0, 2.0, 3.0],
        "日期": pd.to_datetime(["2024-01-02", "2024-02-03", "2024-03-04"]),
    })
    ops = (
        lambda: frame.dtypes,
        lambda: frame.shape,
        lambda: frame.columns.tolist(),
        lambda: frame.describe(),
        lambda: frame.describe(include="all"),
        lambda: frame.head(2),
        lambda: frame.to_string(),
        lambda: frame.to_dict(),
        lambda: frame.to_numpy(),
        lambda: frame.copy(),
        lambda: frame.reset_index(),
        lambda: frame.rename(columns={"品名": "名称"}),
        lambda: frame.fillna(0),
        lambda: frame.dropna(),
        lambda: frame.drop_duplicates(),
        lambda: frame.astype({"数量": "float64"}),
        lambda: frame.T,
        lambda: frame.iloc[0],
        lambda: frame["数量"].sum(),
        lambda: frame["数量"].mean(),
        lambda: frame["数量"].std(),
        lambda: frame["数量"].median(),
        lambda: frame["数量"].quantile(0.5),
        lambda: frame["数量"].max(),
        lambda: frame["数量"].cumsum(),
        lambda: frame["数量"].diff(),
        lambda: frame["数量"].pct_change(),
        lambda: frame["数量"].rank(),
        lambda: frame["数量"].round(2),
        lambda: frame["数量"].rolling(2).sum(),
        lambda: frame["数量"].isin([1.0]),
        lambda: frame["品名"].value_counts(),
        lambda: frame["品名"].str.upper(),
        lambda: frame["品名"].astype("category"),
        lambda: frame["品名"].unique(),
        lambda: frame.groupby("品名")["数量"].sum(),
        lambda: frame.groupby("品名")["数量"].agg(["sum", "mean", "count"]),
        lambda: frame.groupby("品名").agg({"数量": ["sum", "mean"]}),
        lambda: frame.sort_values("数量"),
        lambda: frame.sort_values("数量", ascending=False),
        lambda: frame.nlargest(1, "数量"),
        lambda: frame.apply(lambda row: row["数量"], axis=1),
        lambda: frame.corr(numeric_only=True),
        lambda: pd.concat([frame, frame]),
        lambda: pd.merge(frame, frame, on="品名"),
        lambda: pd.crosstab(frame["品名"], frame["数量"]),
        lambda: pd.pivot_table(frame, index="品名", values="数量", aggfunc="sum"),
        lambda: pd.to_numeric(frame["品名"], errors="coerce"),
        lambda: pd.to_datetime(["2024-01-01"]),
        lambda: pd.to_timedelta(["1 days"]),
        lambda: pd.Timestamp("2024-01-01", tz="Asia/Shanghai"),
        lambda: pd.Timedelta("1 days"),
        lambda: pd.date_range("2024-01-01", periods=3),
        lambda: pd.cut(frame["数量"], 2),
        lambda: pd.factorize(frame["品名"]),
        lambda: pd.get_dummies(frame["品名"]),
        lambda: pd.isna(frame),
        lambda: pd.notna(frame),
        lambda: pd.Series([1, 2, 3]),
        lambda: pd.Index(["a", "b"]),
        lambda: dated.set_index("日期").resample("ME")["数量"].sum(),
        lambda: dated["日期"].dt.strftime("%Y-%m"),
        lambda: dated["日期"].dt.year,
        lambda: np.percentile([1.0, 2.0], 90),
        lambda: np.isnan([1.0, np.nan]),
        lambda: np.where(True, 1, 2),
        lambda: np.array([1, 2]),
        lambda: np.nansum([1.0, np.nan]),
        lambda: np.arange(3),
        lambda: np.sqrt(4),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for op in ops:
            try:
                op()
            except Exception:
                pass


def _safe_builtins() -> dict:
    import builtins
    safe = {name: getattr(builtins, name) for name in ALLOWED_BUILTIN_NAMES}
    safe["__import__"] = _guarded_import
    return safe


def _maybe_numeric(series):
    """整列可解析为数值时返回数值 Series，否则返回 None（保持原文本）。

    兼容 pandas 3：to_numeric 已没有 ignore 错误模式（传了会抛 ValueError），
    这里逐列 try/except；空白单元格视作缺失值；混有非数值文本的列不转换。
    """
    import pandas as pd
    cleaned = []
    for value in series.tolist():
        if value is None:
            cleaned.append(None)
            continue
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            cleaned.append(value)
            continue
        text = str(value).strip()
        if text == "":
            cleaned.append(None)
            continue
        cleaned.append(text)
    try:
        return pd.to_numeric(pd.Series(cleaned, dtype="object"), errors="raise")
    except (ValueError, TypeError):
        return None


def build_frame(rows, columns):
    """由二维数据构建 DataFrame，并尽量把纯数值列转成数值类型。"""
    import pandas as pd
    rows = list(rows or [])
    cols = [str(c if c is not None else "").strip() for c in (columns or [])]
    if not cols:
        cols = [str(i) for i in range(max((len(r) for r in rows), default=0))]
    data = []
    for row in rows:
        values = list(row)
        if len(values) < len(cols):
            values.extend([None] * (len(cols) - len(values)))
        data.append(values[: len(cols)])

    def _blank(value) -> bool:
        return value is None or str(value).strip() == ""

    # 去掉“表头为空且整列为空”的列（腾讯表格网格常带大量空白列）
    keep = [
        i for i, name in enumerate(cols)
        if name or any(not _blank(row[i]) for row in data)
    ]
    cols = [cols[i] for i in keep]
    data = [[row[i] for i in keep] for row in data]
    # 表头为空补名、重复表头加后缀，保证 df[列名] 总是 Series
    seen: dict[str, int] = {}
    unique_cols = []
    for index, name in enumerate(cols):
        base = name or f"列{index + 1}"
        count = seen.get(base, 0)
        seen[base] = count + 1
        unique_cols.append(base if count == 0 else f"{base}_{count + 1}")
    df = pd.DataFrame(data, columns=unique_cols)
    for col in df.columns:
        converted = _maybe_numeric(df[col])
        if converted is not None:
            df[col] = converted
    return df


#: 子进程启动时捕获的真实 stdout（用户代码的 print 会被重定向走，
#: 结果 JSON 必须始终写到这里，父进程才能解析）。
_CHILD_STDOUT = sys.stdout


def _emit(obj: dict) -> None:
    _CHILD_STDOUT.write(json.dumps(obj, ensure_ascii=False))
    _CHILD_STDOUT.flush()


def _child_main() -> None:
    global _CHILD_STDOUT
    _CHILD_STDOUT = sys.stdout
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except Exception:
        _emit({"code_output": "", "error": "沙箱输入解析失败"})
        return
    code = str(payload.get("code", "") or "")
    rows = payload.get("rows") or []
    columns = payload.get("columns") or []
    # skip_validation 只可能由父进程的测试专用入口 _run_child_for_test 传进来，
    # 用于单独验证运行时审计钩子这道防线；run_restricted 永远不会传。
    skip_validation = bool(payload.get("skip_validation"))

    if not skip_validation:
        reason = validate_code(code)
        if reason:
            _emit({"code_output": "", "error": reason})
            return
    try:
        df = build_frame(rows, columns)
    except Exception as exc:
        _emit({"code_output": "", "error": f"构建 DataFrame 失败: {type(exc).__name__}"})
        return

    import math
    import numpy as np
    import pandas as pd

    # safe_pandas 已导入，不再需要它所在目录（可能与 token 文件同目录）在 sys.path 上。
    module_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path if os.path.realpath(p or ".") != os.path.realpath(module_dir)]
    global _PRELOADED_MODULES
    _PRELOADED_MODULES = frozenset(sys.modules)
    try:
        _warm_up()
        compiled = compile(code, _SANDBOX_FILENAME, "exec")
        _install_audit_hook()
    except Exception as exc:
        _emit({"code_output": "", "error": f"沙箱初始化失败: {type(exc).__name__}"})
        return

    buffer = io.StringIO()
    sandbox_globals = {
        "__builtins__": _safe_builtins(),
        "pd": pd,
        "np": np,
        "math": math,
        "df": df,
    }
    try:
        with contextlib.redirect_stdout(buffer):
            exec(compiled, sandbox_globals)
        output = buffer.getvalue()
        error = ""
    except Exception as exc:
        output = buffer.getvalue()
        error = f"执行代码出错: {type(exc).__name__}: {exc}"[:500]
    _emit({
        "code_output": output[:MAX_OUTPUT_CHARS],
        "error": error,
    })


# ---------------------------------------------------------------------------
# 父进程侧
# ---------------------------------------------------------------------------

def _child_env() -> dict[str, str]:
    """最小化环境变量：绝不把父进程的密钥传给沙箱子进程。"""
    return {
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "LANG": "C.UTF-8",
        "HOME": "/tmp",
    }


def _make_preexec(cpu_seconds: int):
    import resource

    def _apply_limits() -> None:
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
        except (ValueError, OSError):
            pass
        try:
            resource.setrlimit(resource.RLIMIT_AS, (1536 * 1024 * 1024, 1536 * 1024 * 1024))
        except (ValueError, OSError):
            pass
        try:
            resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
        except (ValueError, OSError):
            pass
        try:
            soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
            target = 4096
            if soft != resource.RLIM_INFINITY and soft < target:
                target = soft
            resource.setrlimit(resource.RLIMIT_NPROC, (target, hard))
        except (ValueError, OSError, AttributeError):
            pass
        try:
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            target = 256
            if soft != resource.RLIM_INFINITY and soft < target:
                target = soft
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
        except (ValueError, OSError):
            pass

    return _apply_limits


def _bootstrap_source() -> str:
    module_dir = os.path.dirname(os.path.abspath(__file__))
    return (
        "import sys\n"
        f"sys.path.insert(0, {module_dir!r})\n"
        "import safe_pandas\n"
        "safe_pandas._child_main()\n"
    )


def _resolve_timeout(timeout: float | None) -> float:
    if timeout is None:
        timeout = sandbox_exec_timeout_seconds()
    try:
        return max(0.5, float(timeout))
    except (TypeError, ValueError):
        return DEFAULT_SANDBOX_TIMEOUT_SECONDS


def _serialize_payload(code: str, rows, columns, skip_validation: bool = False) -> str:
    body: dict = {"code": code, "rows": rows or [], "columns": columns or []}
    if skip_validation:
        # 只有测试入口会传 skip_validation=True；生产路径永远不带这个键。
        body["skip_validation"] = True
    return json.dumps(body, ensure_ascii=False, default=str)


def _spawn_and_wait(payload: str, timeout: float) -> dict:
    """启动受限子进程、把 payload 喂进去并等结果。绝不向父进程 stdout 写任何东西。"""
    cpu_seconds = max(1, int(timeout) + 2)
    try:
        with tempfile.TemporaryDirectory(prefix="pandas-sandbox-") as cwd:
            try:
                proc = subprocess.Popen(
                    [sys.executable, "-I", "-B", "-c", _bootstrap_source()],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=_child_env(),
                    cwd=cwd,
                    start_new_session=True,
                    preexec_fn=_make_preexec(cpu_seconds),
                )
            except OSError:
                logging.exception("沙箱子进程启动失败")
                return {"code_output": "", "error": "代码沙箱启动失败，请稍后重试。"}
            try:
                out, err = proc.communicate(payload, timeout=timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except OSError:
                    proc.kill()
                try:
                    out, err = proc.communicate(timeout=5)
                except Exception:
                    out, err = "", ""
                logging.warning("沙箱代码执行超时（超过 %s 秒）", timeout)
                return {"code_output": "", "error": f"代码执行超时（超过 {timeout:g} 秒）"}
    except Exception:
        logging.exception("沙箱执行环境异常")
        return {"code_output": "", "error": "代码沙箱执行失败，请稍后重试。"}

    if proc.returncode != 0:
        logging.warning(
            "沙箱子进程异常退出 rc=%s stderr=%s",
            proc.returncode,
            (err or "")[-300:],
        )
        return {"code_output": "", "error": "代码执行失败（进程异常退出，可能超出内存或时间限制）。"}
    try:
        result = json.loads(out or "{}")
    except json.JSONDecodeError:
        logging.warning("沙箱输出不是合法 JSON: %s", (out or "")[-300:])
        return {"code_output": "", "error": "代码沙箱返回结果解析失败。"}
    return {
        "code_output": str(result.get("code_output", "") or "")[:MAX_OUTPUT_CHARS],
        "error": str(result.get("error", "") or ""),
    }


def run_restricted(code: object, rows, columns, timeout: float | None = None) -> dict:
    """在受限子进程里执行代码，返回 {"code_output": str, "error": str}。"""
    text = str(code or "")
    reason = validate_code(text)
    if reason:
        return {"code_output": "", "error": reason}
    try:
        payload = _serialize_payload(text, rows, columns)
    except (TypeError, ValueError):
        return {"code_output": "", "error": "表格数据无法传入沙箱（序列化失败）"}
    return _spawn_and_wait(payload, _resolve_timeout(timeout))


def _run_child_for_test(
    code: object,
    rows,
    columns,
    timeout: float | None = None,
    skip_validation: bool = False,
) -> dict:
    """测试专用入口：可以绕过 AST 校验直接跑子进程，用来验证运行时审计钩子。

    只有测试会调用它；run_restricted 永远不会跳过校验。
    """
    text = str(code or "")
    try:
        payload = _serialize_payload(text, rows, columns, skip_validation=skip_validation)
    except (TypeError, ValueError):
        return {"code_output": "", "error": "表格数据无法传入沙箱（序列化失败）"}
    return _spawn_and_wait(payload, _resolve_timeout(timeout))
