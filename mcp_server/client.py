"""腾讯文档 OpenAPI 客户端。

封装 OAuth 授权与 OpenAPI 请求,token 自动持久化到 ~/.tencent-docs-mcp/token.json,
过期时用 refresh_token 自动续期。

API 文档: https://docs.qq.com/open/document/app/openapi/v2/
"""

import json
import time
from pathlib import Path
from urllib.parse import urlencode

import httpx

OAUTH_BASE = "https://docs.qq.com/oauth/v2"
OPENAPI_BASE = "https://docs.qq.com/openapi"

TOKEN_PATH = Path.home() / ".tencent-docs-mcp" / "token.json"


class TencentDocsError(Exception):
    """业务错误:ret != 0 或网络/配置问题。"""


class TokenNotFound(TencentDocsError):
    """本地没有 token,需要先走授权流程。"""


class TencentDocsClient:
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        redirect_uri: str = "https://docs.qq.com",
        token_path: Path = TOKEN_PATH,
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.token_path = token_path
        self._token: dict | None = None  # {access_token, refresh_token, open_id, expires_at}

        # 允许用环境变量/配置直接注入 token(适合个人开发者入口拿到的长期 token)
        import os
        env_token = os.environ.get("TENCENT_DOCS_ACCESS_TOKEN", "")
        env_open_id = os.environ.get("TENCENT_DOCS_OPEN_ID", "")
        if env_token and env_open_id:
            self._token = {
                "access_token": env_token,
                "refresh_token": "",
                "open_id": env_open_id,
                "expires_at": 0,  # 0 表示未知,不做本地过期判断
            }
        else:
            self._token = self._load_token()

    # ---------- token 持久化 ----------

    def _load_token(self) -> dict | None:
        if not self.token_path.exists():
            return None
        try:
            return json.loads(self.token_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def _save_token(self, token: dict) -> None:
        self._token = token
        self.token_path.parent.mkdir(parents=True, exist_ok=True)
        self.token_path.write_text(
            json.dumps(token, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        # token 文件包含敏感凭证,收紧权限
        self.token_path.chmod(0o600)

    # ---------- OAuth ----------

    def build_auth_url(self, state: str = "mcp") -> str:
        """拼接用户授权链接,浏览器打开、扫码同意后从回调地址里复制 code。"""
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "new_login": 1,
            "response_type": "code",
            "scope": "all",
            "state": state,
        }
        return f"https://docs.qq.com/oauth/v2/authorize?{urlencode(params)}"

    def exchange_code(self, code: str) -> dict:
        """用授权码换 token(code 5 分钟有效、只能用一次)。"""
        resp = httpx.get(
            f"{OAUTH_BASE}/token",
            params={
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "redirect_uri": self.redirect_uri,
                "grant_type": "authorization_code",
                "code": code,
            },
            timeout=15,
        )
        data = resp.json()
        if resp.status_code != 200 or "access_token" not in data:
            raise TencentDocsError(f"换取 token 失败: HTTP {resp.status_code} {data}")
        self._save_token(
            {
                "access_token": data["access_token"],
                "refresh_token": data.get("refresh_token", ""),
                "open_id": data.get("user_id", ""),
                "expires_at": int(time.time()) + int(data.get("expires_in", 2592000)),
            }
        )
        return {"open_id": data.get("user_id", ""), "scope": data.get("scope", "")}

    def refresh(self) -> None:
        """用 refresh_token 换新的 access_token。"""
        token = self._load_token()
        if not token or not token.get("refresh_token"):
            raise TokenNotFound("没有 refresh_token,请重新走授权流程")
        resp = httpx.get(
            f"{OAUTH_BASE}/token",
            params={
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "grant_type": "refresh_token",
                "refresh_token": token["refresh_token"],
            },
            timeout=15,
        )
        data = resp.json()
        if resp.status_code != 200 or "access_token" not in data:
            raise TencentDocsError(f"刷新 token 失败: HTTP {resp.status_code} {data}")
        self._save_token(
            {
                "access_token": data["access_token"],
                "refresh_token": data.get("refresh_token", token["refresh_token"]),
                "open_id": data.get("user_id", token.get("open_id", "")),
                "expires_at": int(time.time()) + int(data.get("expires_in", 2592000)),
            }
        )

    # ---------- 请求底层 ----------

    def _ensure_token(self) -> dict:
        if not self._token:
            self._token = self._load_token()
        if not self._token:
            raise TokenNotFound("尚未授权:请先调用 get_auth_url + exchange_code,或在环境变量里直接配置 token")
        # expires_at == 0 表示外部注入的 token,跳过本地过期判断
        if self._token.get("expires_at", 0) and time.time() > self._token["expires_at"] - 60:
            self.refresh()
            self._token = self._load_token()
        return self._token

    def _openapi_request(
        self,
        method: str,
        path: str,
        params: dict | None = None,
        data: dict | None = None,
        json_body: dict | None = None,
    ) -> dict:
        """调 OpenAPI(自动带 Access-Token / Client-Id / Open-Id 三个鉴权头)。

        data 为 form 编码,v3 接口用 json_body 传 JSON 请求体。
        兼容 v2(ret/msg)与 v3(code/message)两种错误格式。
        """
        token = self._ensure_token()
        headers = {
            "Access-Token": token["access_token"],
            "Client-Id": self.client_id,
            "Open-Id": token.get("open_id", ""),
            "Accept": "application/json",
        }
        resp = httpx.request(
            method,
            f"{OPENAPI_BASE}{path}",
            headers=headers,
            params=params,
            data=data,
            json=json_body,
            timeout=30,
        )
        try:
            body = resp.json()
        except ValueError:
            raise TencentDocsError(f"HTTP {resp.status_code}, 响应不是 JSON: {resp.text[:200]}")
        # v2 返回 ret/msg,v3 返回 code/message,两种都检查
        ret = body.get("ret", body.get("code", 0))
        if ret != 0:
            msg = body.get("msg") or body.get("message")
            raise TencentDocsError(f"业务错误 ret={ret}: {msg} (HTTP {resp.status_code})")
        return body

    # ---------- OpenAPI 封装 ----------

    def get_user_info(self) -> dict:
        """获取当前授权用户信息。"""
        token = self._ensure_token()
        resp = httpx.get(
            f"{OAUTH_BASE}/userinfo",
            headers={
                "Access-Token": token["access_token"],
                "Client-Id": self.client_id,
                "Open-Id": token.get("open_id", ""),
            },
            timeout=15,
        )
        data = resp.json()
        if resp.status_code != 200:
            raise TencentDocsError(f"获取用户信息失败: HTTP {resp.status_code} {data}")
        return data

    def list_docs(
        self,
        folder_id: str = "/",
        limit: int = 20,
        file_type: str = "",
        is_owner: int = 0,
    ) -> dict:
        """拉取文档列表。folder_id 为 '/' 表示根目录;file_type 如 'doc'/'sheet'。"""
        params: dict = {"folderID": folder_id, "limit": min(limit, 20)}
        if file_type:
            params["fileType"] = file_type
        if is_owner:
            params["isOwner"] = is_owner
        return self._openapi_request("GET", "/drive/v2/filter", params=params).get("data", {})

    def create_doc(self, title: str, doc_type: str = "doc") -> dict:
        """新建文档。doc_type: doc / sheet / slide / folder。"""
        return self._openapi_request(
            "POST", "/drive/v2/files", data={"title": title, "type": doc_type}
        ).get("data", {})

    def async_export(self, file_id: str, export_type: str = "") -> dict:
        """发起异步导出,返回 operationID(注意:每用户每天限 9 次)。"""
        data = {"exportType": export_type} if export_type else None
        return self._openapi_request(
            "POST", f"/drive/v2/files/{file_id}/async-export", data=data
        ).get("data", {})

    def export_progress(self, file_id: str, operation_id: str) -> dict:
        """查询导出进度,成功时返回 30 分钟有效的下载链接。"""
        return self._openapi_request(
            "GET",
            f"/drive/v2/files/{file_id}/export-progress",
            params={"operationID": operation_id},
        ).get("data", {})

    # ---------- Sheet v3 ----------

    def list_sheets(self, file_id: str) -> list[dict]:
        """查询在线表格的所有工作表(sheetId、标题、行列数)。

        注意:该接口的响应 properties 在顶层,没有 data 包裹(与其他接口不同)。
        """
        body = self._openapi_request(
            "GET", f"/spreadsheet/v3/files/{file_id}", params={"concise": 1}
        )
        return body.get("properties") or body.get("data", {}).get("properties", [])

    def read_sheet_range(self, file_id: str, sheet_id: str, cell_range: str) -> dict:
        """读取工作表范围数据。cell_range 用 A1 表示法,如 'A1:D11' 或 'A1:C5'。

        限制:行<=1000、列<=200、总单元格<=10000。
        返回简化后的二维文本数组(原始结构中的 cellValue 提取为字符串)。
        """
        body = self._openapi_request(
            "GET", f"/spreadsheet/v3/files/{file_id}/{sheet_id}/{cell_range}"
        )
        # 注意:v3 实际响应 gridData 在顶层,无 data 包裹(与官方文档示例不符)
        grid = body.get("gridData") or body.get("data", {}).get("gridData", {})
        rows = []
        for row in grid.get("rows", []):
            cells = []
            for cell in row.get("values", []):
                cells.append(_cell_to_text(cell.get("cellValue")))
            rows.append(cells)
        return {
            "start_row": grid.get("startRow", 0),  # 0-based
            "start_column": grid.get("startColumn", 0),  # 0-based
            "values": rows,
        }

    def write_sheet_range(
        self, file_id: str, sheet_id: str, start_cell: str, values: list[list[str]]
    ) -> dict:
        """写入数据到工作表,start_cell 为 A1 表示法起点(如 'B3'),values 为二维字符串数组。

        限制:行<=1000、列<=200、总单元格<=10000;频控 50次/分钟。
        单元格只支持 text / number / link,这里统一按 text 写入。
        """
        if not values or not values[0]:
            raise TencentDocsError("values 不能为空")
        start_row, start_col = _parse_a1(start_cell)
        rows_payload = [
            {
                "values": [
                    ({"cellValue": {"number": v}} if _is_number(v) else {"cellValue": {"text": v}})
                    for v in row
                ]
            }
            for row in values
        ]
        body = self._openapi_request(
            "POST",
            f"/spreadsheet/v3/files/{file_id}/batchUpdate",
            json_body={
                "requests": [
                    {
                        "updateRangeRequest": {
                            "sheetId": sheet_id,
                            "gridData": {
                                "startRow": start_row,
                                "startColumn": start_col,
                                "rows": rows_payload,
                            },
                        }
                    }
                ]
            },
        )
        return body.get("data", {})


# ---------- A1 表示法工具 ----------

def _parse_a1(cell: str) -> tuple[int, int]:
    """'B3' -> (row=2, col=1),0-based。"""
    import re

    m = re.fullmatch(r"([A-Za-z]+)(\d+)", cell.strip())
    if not m:
        raise TencentDocsError(f"无效的 A1 单元格: {cell}(示例: 'B3')")
    col = 0
    for ch in m.group(1).upper():
        col = col * 26 + (ord(ch) - ord("A") + 1)
    return int(m.group(2)) - 1, col - 1


def _cell_to_text(cell_value: dict | None) -> str:
    """把 cellValue(text/number/link/location 等)统一转成字符串。"""
    if not cell_value:
        return ""
    if "text" in cell_value:
        return str(cell_value["text"])
    if "number" in cell_value:
        return str(cell_value["number"])
    if "link" in cell_value:
        link = cell_value["link"]
        return str(link.get("text") or link.get("url", ""))
    if "location" in cell_value:
        return str(cell_value["location"].get("name", ""))
    return str(cell_value)


def _is_number(v: str) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False
