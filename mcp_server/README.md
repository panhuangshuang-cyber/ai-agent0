# tencent-docs MCP 服务端

生产部署目录：`~/tencent-docs-mcp`（有独立 venv；web 端通过 stdio 启动它，路径可用环境变量
`MCP_SERVER_SCRIPT` / `MCP_PYTHON_BIN` 覆盖）。

- `server.py`：MCP 服务（工具名/参数向后兼容）。
- `safe_pandas.py`：受限 pandas 沙箱，必须与仓库根目录的 `safe_pandas.py` **逐字节一致**。
- `client.py`：腾讯文档 OpenAPI 客户端，必须与生产目录的 `client.py` **逐字节一致**。
  文件里没有凭证：token 走构造函数参数、环境变量 `TENCENT_DOCS_ACCESS_TOKEN` +
  `TENCENT_DOCS_OPEN_ID`，或 `~/.tencent-docs-mcp/token.json`。
- `test_server_sandbox.py`：离线测试（不联网，FakeClient 注入）。

token 文件（`~/.tencent-docs-mcp/token.json`）只存在于生产目录，**不提交到仓库**。

部署：把 `server.py`、`safe_pandas.py`、`client.py` 复制到 `~/tencent-docs-mcp/`，然后重启 web 服务。

测试：

```bash
cd mcp_server && PYTHONDONTWRITEBYTECODE=1 python -m unittest test_server_sandbox -v
```
