"""受限 pandas 代码沙箱（AST 白名单 + 独立子进程执行）。

本模块同时被 web 端（仓库根目录）和 MCP 服务端（仓库 mcp_server/，部署到
~/tencent-docs-mcp）使用，两个副本必须保持逐字节一致。

安全模型：
- validate_code: 纯 AST 白名单校验（允许节点类型、允许名称、属性黑名单、
  必须 print、长度上限），在调用 MCP / 启动子进程之前给模型反馈。
- run_restricted: 在独立子进程里执行（sys.executable -I，隔离环境变量与
  当前目录），最小化 env（不继承任何父进程密钥），resource 限制
  CPU / 地址空间 / 文件大小 / 进程数 / fd 数，超时用 killpg 杀掉整个
  进程组。子进程通过 stdin 收 JSON、stdout 回 JSON；父进程绝不向真实
  stdout 打印任何东西（MCP server 的 stdout 是 JSON-RPC 通道）。
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
ALLOWED_BASE_NAMES = frozenset({"df", "pd", "np", "math"})

#: 提供给沙箱 __builtins__ 的白名单内置函数/类型。
ALLOWED_BUILTIN_NAMES = (
    "print", "len", "range", "sum", "min", "max", "abs", "round", "sorted",
    "enumerate", "zip", "list", "dict", "set", "tuple", "str", "int", "float",
    "bool", "any", "all", "isinstance", "reversed", "map", "filter",
)

ALLOWED_NAMES = ALLOWED_BASE_NAMES | frozenset(ALLOWED_BUILTIN_NAMES)

#: 明确放行的 to_* 属性。
ATTRIBUTE_ALLOWLIST = frozenset({
    "to_string", "to_dict", "to_list", "tolist", "to_numpy", "to_frame",
    "to_datetime", "to_numeric", "to_timedelta",
})

#: 属性黑名单（另外所有 "_" 开头的属性一律禁止；read_*、to_* 默认禁止）。
ATTRIBUTE_BLOCKLIST = frozenset({
    "eval", "query", "pipe", "style", "plot", "hist", "boxplot",
    "io", "api", "options", "set_option", "compat", "core", "testing",
    "util", "system", "popen", "open", "load", "save", "loads", "dumps",
    "format", "format_map", "mro", "subclasses", "bases",
    # numpy/pandas 子模块里挂着 os/sys/subprocess/ctypes/builtins 等模块引用
    # （例如 np.f2py.os、pd.errors.ctypes、np.ma.extras.ma.builtins），
    # 以及生成器/帧对象的内部属性（gi_frame.f_builtins 可拿到 __import__）。
    # 真正的边界是子进程里的审计钩子，这里只是尽早给模型反馈。
    "f2py", "ctypeslib", "ctypes", "errors", "builtins", "os", "sys",
    "subprocess", "modules", "exec", "globals", "import_module",
    "gi_frame", "gi_code", "gi_yieldfrom", "cr_frame", "cr_code",
    "ag_frame", "ag_code", "tb_frame", "tb_next", "f_builtins",
    "f_globals", "f_locals", "f_back", "f_code",
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
    if attr.startswith("_"):
        return True
    if attr in ATTRIBUTE_ALLOWLIST:
        return False
    if attr in ATTRIBUTE_BLOCKLIST:
        return True
    if attr.startswith("read_") or attr.startswith("to_"):
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

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return "代码包含禁止内容：import"
        if type(node) not in _ALLOWED_NODES:
            return f"代码包含不允许的语法：{type(node).__name__}"

    bound_names = _collect_bound_names(tree)
    call_func_ids = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    has_print = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id in ALLOWED_NAMES or node.id in bound_names:
                continue
            if node.id in call_func_ids:
                return f"代码包含禁止调用：{node.id}"
            return f"代码包含禁止使用的名称：{node.id}"
        if isinstance(node, ast.Attribute) and _attribute_blocked(node.attr):
            return f"代码包含禁止访问的属性：{node.attr}"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
            has_print = True
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


def _install_audit_hook() -> None:
    """运行时兜底：无论用户代码经由哪条属性链拿到 os/subprocess/ctypes/socket，
    危险操作都会在审计事件处被拒绝。

    - open：只允许只读打开 Python 库目录下的文件（pandas/numpy 惰性导入需要），
      其余路径（含 /proc、应用目录下的 .env / token）一律拒绝，写模式一律拒绝；
    - os.listdir/os.scandir：只允许库目录；其余 os.* 事件（system/exec/spawn/fork/
      kill/remove/rename/chmod/putenv...）全部拒绝；
    - subprocess/socket/ctypes/mmap/resource/signal/pty/webbrowser 等一律拒绝。
    审计钩子一旦安装无法移除。
    """
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

    def _hook(event: str, args: tuple) -> None:
        if event == "open":
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
        if event in ("os.listdir", "os.scandir"):
            if args and args[0] is not None and _under_roots(args[0]):
                return
            raise PermissionError("沙箱禁止列目录")
        if event.startswith("os.") or event.startswith("posix."):
            raise PermissionError(f"沙箱禁止操作：{event}")
        if event.startswith(blocked_prefixes):
            raise PermissionError(f"沙箱禁止操作：{event}")

    sys.addaudithook(_hook)


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
    cols = [str(c) for c in (columns or [])]
    if not cols:
        cols = [str(i) for i in range(max((len(r) for r in rows), default=0))]
    data = []
    for row in rows or []:
        values = list(row)
        if len(values) < len(cols):
            values.extend([None] * (len(cols) - len(values)))
        data.append(values[: len(cols)])
    df = pd.DataFrame(data, columns=cols)
    for col in df.columns:
        converted = _maybe_numeric(df[col])
        if converted is not None:
            df[col] = converted
    return df


def _emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False))
    sys.stdout.flush()


def _child_main() -> None:
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
        compiled = compile(code, "<sandbox>", "exec")
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


def run_restricted(code: object, rows, columns, timeout: float | None = None) -> dict:
    """在受限子进程里执行代码，返回 {"code_output": str, "error": str}。"""
    text = str(code or "")
    reason = validate_code(text)
    if reason:
        return {"code_output": "", "error": reason}

    if timeout is None:
        timeout = sandbox_exec_timeout_seconds()
    try:
        timeout = max(0.5, float(timeout))
    except (TypeError, ValueError):
        timeout = DEFAULT_SANDBOX_TIMEOUT_SECONDS

    try:
        payload = json.dumps(
            {"code": text, "rows": rows or [], "columns": columns or []},
            ensure_ascii=False,
            default=str,
        )
    except (TypeError, ValueError):
        return {"code_output": "", "error": "表格数据无法传入沙箱（序列化失败）"}

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
