# tencent-docs MCP 服务端

生产部署目录：`~/tencent-docs-mcp`（有独立 venv；web 端通过 stdio 启动它，路径可用环境变量
`MCP_SERVER_SCRIPT` / `MCP_PYTHON_BIN` 覆盖）。

- `server.py`：MCP 服务（工具名/参数向后兼容）。
- `safe_pandas.py`：受限 pandas 沙箱，必须与仓库根目录的 `safe_pandas.py` **逐字节一致**。
- `test_server_sandbox.py`：离线测试（不联网，FakeClient 注入）。

`client.py`（腾讯文档 OpenAPI 客户端）以及 token 文件只存在于生产目录，**不提交到仓库**；
测试在缺少 `client.py` 时会自动使用离线桩。

部署：把 `server.py`、`safe_pandas.py` 复制到 `~/tencent-docs-mcp/`，然后重启 web 服务。

测试：

```bash
cd mcp_server && PYTHONDONTWRITEBYTECODE=1 python -m unittest test_server_sandbox -v
```
