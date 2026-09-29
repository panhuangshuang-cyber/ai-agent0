"""校验仓库副本与生产目录的 MCP 代码逐字节一致。

README 要求 server.py / safe_pandas.py / client.py 三份文件在仓库和生产目录
（~/tencent-docs-mcp）之间保持一致，但此前没有任何自动校验，分叉不会报警。

生产目录不存在时跳过（例如在别的机器或 CI 上跑），不算失败。
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MCP_SERVER_DIR = Path(os.getenv("MCP_SERVER_DIR", "/home/ubuntu/tencent-docs-mcp"))

# (仓库内路径, 生产目录内文件名)
MIRRORED = [
    (REPO_ROOT / "mcp_server" / "server.py", "server.py"),
    (REPO_ROOT / "mcp_server" / "client.py", "client.py"),
    (REPO_ROOT / "mcp_server" / "safe_pandas.py", "safe_pandas.py"),
]


class VendoredCopySyncTests(unittest.TestCase):
    def test_safe_pandas_copies_match_inside_the_repo(self):
        """仓库内两份 safe_pandas.py（根目录给 web 用，mcp_server/ 给 MCP 用）必须一致。"""
        root_copy = REPO_ROOT / "safe_pandas.py"
        vendored = REPO_ROOT / "mcp_server" / "safe_pandas.py"
        self.assertEqual(
            root_copy.read_bytes(), vendored.read_bytes(),
            f"{vendored} 与 {root_copy} 不一致，把改过的那份复制到另一份",
        )

    @unittest.skipUnless(MCP_SERVER_DIR.is_dir(), f"生产目录不存在：{MCP_SERVER_DIR}")
    def test_repo_copies_match_the_deployed_directory(self):
        drifted = []
        for repo_path, name in MIRRORED:
            live_path = MCP_SERVER_DIR / name
            if not live_path.is_file():
                drifted.append(f"{name}: 生产目录里没有这个文件")
                continue
            if repo_path.read_bytes() != live_path.read_bytes():
                drifted.append(f"{name}: {repo_path} 与 {live_path} 内容不同")
        self.assertEqual(
            drifted, [],
            "仓库副本与生产目录分叉了：\n  " + "\n  ".join(drifted)
            + f"\n部署方向是仓库 -> 生产：把 mcp_server/ 下的文件复制到 {MCP_SERVER_DIR}/ 再重启 web 服务",
        )


if __name__ == "__main__":
    unittest.main()
