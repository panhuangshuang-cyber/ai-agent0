"""腾讯文档 MCP Server。

启动方式(STDIO,给 codebuddycli 等 MCP host 使用):
    python server.py

依赖 mcp 2.x(FastMCP 已更名为 MCPServer,见
https://py.sdk.modelcontextprotocol.io/v2/migration/#fastmcp-renamed-to-mcpserver)

工具列表:
    get_auth_url        生成授权链接
    exchange_code       用授权码换 token(整个流程只需一次)
    get_user_info       验证授权是否生效
    list_docs           拉取文档列表
    create_doc          新建文档
    export_doc          异步导出文档(每天限 9 次)
    get_export_progress 查询导出进度/下载链接
    get_doc_content     读取文档内容(内部走导出流程,返回下载链接)
    list_sheets         查询表格的所有工作表(sheetId/标题)
    read_sheet          读取工作表范围(A1 表示法,返回二维数组)
    write_sheet         写入数据到工作表(A1 起点单元格 + 二维数组)
"""
import logging
import os
import time

from mcp.server.mcpserver import MCPServer

import safe_pandas
from client import TencentDocsClient, TencentDocsError

logger = logging.getLogger(__name__)

mcp = MCPServer("tencent-docs")


def _client() -> TencentDocsClient:
    client_id = os.environ.get("TENCENT_DOCS_CLIENT_ID", "")
    client_secret = os.environ.get("TENCENT_DOCS_CLIENT_SECRET", "")
    if not client_id or not client_secret:
        # 只有直接注入 token 时才允许缺 client_id/secret
        if not (os.environ.get("TENCENT_DOCS_ACCESS_TOKEN") and os.environ.get("TENCENT_DOCS_OPEN_ID")):
            raise TencentDocsError(
                "缺少 TENCENT_DOCS_CLIENT_ID / TENCENT_DOCS_CLIENT_SECRET,"
                "请在 .mcp.json 的 env 里配置"
            )
    redirect_uri = os.environ.get("TENCENT_DOCS_REDIRECT_URI", "https://docs.qq.com")
    return TencentDocsClient(client_id, client_secret, redirect_uri)


@mcp.tool()
def get_auth_url() -> str:
    """生成腾讯文档 OAuth 授权链接。用浏览器打开、扫码同意后,
    从跳转回来的地址栏里复制 code 参数(5 分钟内有效),再调用 exchange_code。"""
    return _client().build_auth_url()


@mcp.tool()
def exchange_code(code: str) -> str:
    """用授权码换取并保存 access_token。code 来自授权回调地址的 ?code= 参数,
    只能用一次。成功后返回 open_id。"""
    result = _client().exchange_code(code)
    return f"授权成功!open_id: {result['open_id']}。token 已保存,其他工具可用了。"


@mcp.tool()
def get_user_info() -> dict:
    """获取当前授权用户信息(昵称、open_id 等),可用来验证授权是否生效。"""
    return _client().get_user_info()


@mcp.tool()
def list_docs(
    folder_id: str = "/", limit: int = 20, file_type: str = "", is_owner: int = 0
) -> dict:
    """拉取腾讯文档列表。folder_id 默认 '/'(根目录);
    file_type 可选 'doc'/'sheet'/'slide';is_owner=1 只看自己创建的。"""
    return _client().list_docs(folder_id, limit, file_type, is_owner)


@mcp.tool()
def create_doc(title: str, doc_type: str = "doc") -> dict:
    """新建腾讯文档,返回文档 ID 和链接。doc_type: doc(在线文档)/ sheet(表格)/ slide / folder。"""
    return _client().create_doc(title, doc_type)


@mcp.tool()
def export_doc(file_id: str, export_type: str = "") -> dict:
    """发起异步导出文档,返回 operationID。注意:每用户每天限 9 次。
    export_type 留空导出原格式;doc/slide 可传 'pdf'。"""
    return _client().async_export(file_id, export_type)


@mcp.tool()
def get_export_progress(file_id: str, operation_id: str) -> dict:
    """查询导出进度。progress=100 时 data.url 为下载链接(30 分钟有效)。"""
    return _client().export_progress(file_id, operation_id)


@mcp.tool()
def get_doc_content(file_id: str, timeout_seconds: int = 60) -> dict:
    """读取文档内容:内部走异步导出,轮询直到拿到下载链接。
    受导出频控限制(每用户每天 9 次),不适合频繁调用。
    返回的 url 是导出文件(docx/xlsx 等)的临时下载链接。"""
    c = _client()
    op = c.async_export(file_id)
    operation_id = op.get("operationID", "")
    if not operation_id:
        raise TencentDocsError(f"导出失败,未返回 operationID: {op}")
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        result = c.export_progress(file_id, operation_id)
        if result.get("progress") == 100:
            return {"url": result.get("url"), "operation_id": operation_id}
        time.sleep(2)
    return {
        "operation_id": operation_id,
        "note": "超时未完成,稍后用 get_export_progress 查询",
    }


@mcp.tool()
def list_sheets(file_id: str) -> list[dict]:
    """查询在线表格的所有工作表,返回每个工作表的 sheetId、标题、行列数。
    file_id 可从 list_docs 返回或文档 URL(docs.qq.com/sheet/DZxxxx?tab=BB08J2 中
    DZxxxx 是 file_id,tab 参数是 sheetId)。"""
    return _client().list_sheets(file_id)


@mcp.tool()
def read_sheet(file_id: str, sheet_id: str, cell_range: str) -> dict:
    """读取工作表范围数据。cell_range 用 A1 表示法,如 'A1:D11'。
    限制:行<=1000、列<=200、总单元格<=10000。
    返回 {start_row, start_column(0-based), values(二维文本数组)}。"""
    return _client().read_sheet_range(file_id, sheet_id, cell_range)


@mcp.tool()
def write_sheet(
    file_id: str, sheet_id: str, start_cell: str, values: list[list[str]]
) -> dict:
    """写入数据到工作表。start_cell 为 A1 起点单元格(如 'B3'),
    values 为二维字符串数组,如 [['姓名','数量'], ['张三','10']]。
    数字字符串自动按 number 写入,其余按 text。频控 50次/分钟。"""
    return _client().write_sheet_range(file_id, sheet_id, start_cell, values)



def get_col_name(n: int) -> str:
    res = ""
    while n >= 0:
        res = chr(n % 26 + 65) + res
        n = n // 26 - 1
    return res


# 腾讯表格接口的单次请求硬限制
SHEET_MAX_ROWS_PER_REQUEST = 1000
SHEET_MAX_COLS_PER_REQUEST = 200
SHEET_MAX_CELLS_PER_REQUEST = 10000
# 一次分析/阅读最多累计读取的数据行数
SHEET_MAX_TOTAL_ROWS = 10000


def _norm_title(value) -> str:
    return " ".join(str(value or "").split()).casefold()


def _as_int(value):
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _sheet_id_of(sheet: dict) -> str:
    return str(sheet.get("sheetId") or sheet.get("id") or "")


def _pick_doc(c, doc_title: str) -> tuple[str, str, dict | None]:
    """按标题定位文档。返回 (file_id, real_title, error_dict)。

    优先精确匹配（去空白、casefold），否则取“标题包含关键词”的文档；
    多个匹配且没有精确命中时返回候选列表而不是擅自挑第一个。
    """
    try:
        docs = c.list_docs(limit=100, file_type="sheet").get("list", []) or []
    except Exception:
        logger.exception("拉取文档列表失败")
        return "", "", {"error": "文档列表获取失败，请稍后重试。"}
    hint = _norm_title(doc_title)
    if not hint:
        return "", "", {"error": "请提供文档标题（或 file_id）。"}
    exact = [d for d in docs if _norm_title(d.get("title")) == hint]
    matches = exact or [d for d in docs if hint in _norm_title(d.get("title"))]
    if not matches:
        return "", "", {"error": f"找不到标题包含 {doc_title} 的表格文档，请确认名称是否正确。"}
    if len(matches) > 1:
        return "", "", {
            "error": f"“{doc_title}”匹配到多个文档，请提供更完整的标题，或直接传 file_id。",
            "candidates": [str(d.get("title") or "") for d in matches],
        }
    doc = matches[0]
    return str(doc.get("id") or ""), str(doc.get("title") or doc_title), {}


def _pick_sheet(c, file_id: str, real_title: str, sheet_name: str, sheet_id: str) -> tuple[dict | None, dict | None]:
    """定位工作表。返回 (sheet_props, error_dict)。

    sheet_id 优先；sheet_name 给出时精确匹配优先、其次包含匹配，
    找不到就报错（绝不静默退回第一张子表）。
    """
    try:
        sheets = c.list_sheets(file_id) or []
    except Exception:
        logger.exception("拉取子表列表失败: %s", file_id)
        return None, {"error": "子表列表获取失败，请稍后重试。"}
    if not sheets:
        return None, {"error": "该文档内没有可用的工作表。"}
    if sheet_id:
        for sheet in sheets:
            if _sheet_id_of(sheet) == sheet_id:
                return sheet, None
        return None, {"error": f"文档《{real_title}》里没有找到 sheetId 为 {sheet_id} 的工作表。"}
    if sheet_name:
        hint = _norm_title(sheet_name)
        exact = [s for s in sheets if _norm_title(s.get("title")) == hint]
        matches = exact or [s for s in sheets if hint and hint in _norm_title(s.get("title"))]
        all_titles = [str(s.get("title") or "") for s in sheets]
        if not matches:
            return None, {
                "error": f"文档《{real_title}》里没有找到子表“{sheet_name}”，请确认子表名。",
                "candidates": all_titles,
            }
        if len(matches) > 1:
            return None, {
                "error": f"子表“{sheet_name}”匹配到多个工作表，请提供更完整的子表名，或直接传 sheet_id。",
                "candidates": [str(s.get("title") or "") for s in matches],
            }
        return matches[0], None
    return sheets[0], None


def _read_rows_paginated(c, file_id: str, sheet_id: str, rows_to_read: int, cols: int) -> list[list[str]]:
    """分页读取，遵守腾讯接口单次 行<=1000、单元格<=10000 的限制。"""
    per_request = max(1, min(SHEET_MAX_ROWS_PER_REQUEST, SHEET_MAX_CELLS_PER_REQUEST // max(1, cols)))
    end_col = get_col_name(cols - 1)
    values: list[list[str]] = []
    start = 1
    while len(values) < rows_to_read:
        batch = min(per_request, rows_to_read - len(values))
        cell_range = f"A{start}:{end_col}{start + batch - 1}"
        chunk = (c.read_sheet_range(file_id, sheet_id, cell_range) or {}).get("values") or []
        if not chunk:
            break
        values.extend(chunk)
        if len(chunk) < batch:
            # 返回行数不足，说明已到表格末尾
            break
        start += batch
    return values


def _search_and_read_sheet_impl(
    doc_title: str, sheet_name: str = "", max_rows: int = 200, max_cols: int = 30,
    file_id: str = "", sheet_id: str = "",
) -> dict:
    c = _client()
    file_id = str(file_id or "").strip()
    sheet_id = str(sheet_id or "").strip()
    real_title = str(doc_title or "")

    if not file_id:
        file_id, real_title, err = _pick_doc(c, doc_title)
        if err:
            return err

    sheet, err = _pick_sheet(c, file_id, real_title, sheet_name, sheet_id)
    if err:
        return err

    resolved_sheet_id = _sheet_id_of(sheet)
    sheet_title = str(sheet.get("title") or "")
    total_rows = _as_int(sheet.get("rowCount"))
    column_count = _as_int(sheet.get("columnCount", sheet.get("colCount")))

    cols = _as_int(max_cols) or 30
    cols = max(1, min(cols, SHEET_MAX_COLS_PER_REQUEST))
    if column_count:
        cols = min(cols, column_count)

    rows_to_read = _as_int(max_rows) or 200
    rows_to_read = max(1, min(rows_to_read, SHEET_MAX_TOTAL_ROWS))
    if total_rows and total_rows > 0:
        rows_to_read = min(rows_to_read, total_rows)

    try:
        values = _read_rows_paginated(c, file_id, resolved_sheet_id, rows_to_read, cols)
    except Exception:
        logger.exception("读取表格数据失败: %s/%s", file_id, resolved_sheet_id)
        return {"error": "读取表格数据失败，请稍后重试。"}

    rows_read = len(values)
    end_col = get_col_name(cols - 1)
    truncated = rows_read >= rows_to_read and (total_rows is None or total_rows > rows_read)
    result = {
        "doc_title": real_title,
        "sheet_title": sheet_title,
        "file_id": file_id,
        "sheet_id": resolved_sheet_id,
        "read_range": f"A1:{end_col}{rows_read}" if rows_read else "",
        "data": values,
        "rows_read": rows_read,
        "total_rows": total_rows,
        "total_cols": cols,
        "truncated": truncated,
    }
    if truncated:
        if total_rows:
            result["note"] = f"只读取了前 {rows_read} 行（共 {total_rows} 行），统计结果可能不完整。"
        else:
            result["note"] = f"只读取了前 {rows_read} 行，统计结果可能不完整。"
    return result


@mcp.tool()
def search_and_read_sheet(
    doc_title: str, sheet_name: str = "", max_rows: int = 200, max_cols: int = 30,
    file_id: str = "", sheet_id: str = "",
) -> dict:
    """
    一键智能阅读工具：输入文档标题（或直接用 file_id + sheet_id），自动定位并分页读取表格数据
    （默认前 200 行 30 列，最多 10000 行；自动遵守腾讯接口单次请求限制）。
    如果不传 sheet_name / sheet_id，默认读取第一个子表；sheet_name 找不到时直接报错，不会退回第一张。
    标题或子表名有多个匹配且没有精确命中时，返回 error + candidates，请让用户澄清。
    本工具自带超大表格截断防护，是最安全、快捷的表格阅读方式！优先使用此工具！
    返回 {doc_title, sheet_title, file_id, sheet_id, read_range, data(带表头的二维数组),
          rows_read, total_rows, total_cols, truncated}
    """
    return _search_and_read_sheet_impl(doc_title, sheet_name, max_rows, max_cols, file_id, sheet_id)


# ================= 长期记忆管理员 =================
MEMORY_FILE = "/home/ubuntu/tencent-docs-web/memory.json"
import json
import os

def _load_memory():
    if not os.path.exists(MEMORY_FILE):
        return {}
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

def _save_memory(data):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

@mcp.tool()
def update_memory_rule(keyword: str, exact_meaning: str) -> str:
    """
    长期记忆管理员：当用户指定某个词的具体含义（如“封边条”特指“封边条250424”），
    调用此工具将规则存入长期记忆。下次对话系统将自动附带此记忆，实现越用越聪明的定制化。
    """
    mem = _load_memory()
    mem[keyword] = exact_meaning
    _save_memory(mem)
    return f"已成功将 {keyword} -> {exact_meaning} 写入长期记忆。"

# ================= 数据分析专员 (代码解释器) =================
@mcp.tool()
def analyze_sheet_pandas(
    doc_title: str, python_code: str, sheet_name: str = "", file_id: str = "", sheet_id: str = "",
) -> dict:
    """
    数据分析专员 (代码解释器)：需要求最值、计算总和、排序或复杂统计时，使用此工具。
    它会自动定位表格（可传 file_id + sheet_id 精确定位）并转为 pandas DataFrame，变量名为 df。
    请在 python_code 中编写纯 Python 逻辑（不用写 markdown，只写代码；不要 import、不要访问
    下划线开头的属性），并务必使用 print() 输出结果。代码在受限沙箱子进程里执行，
    只允许使用 df / pd / np / math 和常用内置函数。
    示例:
    print(df.head())
    print("总计:", df["总价"].sum())
    返回 {doc_title, sheet_title, code_output, rows_read, total_rows, total_cols, truncated,
          [error], [note]}
    """
    data_res = _search_and_read_sheet_impl(
        doc_title, sheet_name,
        max_rows=SHEET_MAX_TOTAL_ROWS, max_cols=SHEET_MAX_COLS_PER_REQUEST,
        file_id=file_id, sheet_id=sheet_id,
    )
    if data_res.get("error"):
        return data_res

    values = data_res.get("data") or []
    if len(values) < 2:
        return {"error": "数据不足，无法生成 DataFrame（至少需要表头和一行数据）。"}

    columns = [str(cell) for cell in values[0]]
    rows = values[1:]
    exec_res = safe_pandas.run_restricted(python_code, rows, columns)
    output = str(exec_res.get("code_output") or "")
    error = str(exec_res.get("error") or "")

    result = {
        "doc_title": data_res.get("doc_title"),
        "sheet_title": data_res.get("sheet_title"),
        "file_id": data_res.get("file_id"),
        "sheet_id": data_res.get("sheet_id"),
        "rows_read": data_res.get("rows_read"),
        "total_rows": data_res.get("total_rows"),
        "total_cols": data_res.get("total_cols"),
        "truncated": bool(data_res.get("truncated")),
        "code_output": output.strip() or error or "代码执行成功，但没有打印任何内容。",
    }
    if error:
        result["error"] = error
    if data_res.get("note"):
        result["note"] = data_res["note"]
    return result

if __name__ == "__main__":
    mcp.run()
