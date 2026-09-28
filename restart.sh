#!/bin/bash
# App now runs under systemd (tencent-docs-web.service); the old tmux variant is in ~/tencent-docs-web.bak-20260928-225155/restart.sh
sudo systemctl restart tencent-docs-web.service
sudo systemctl --no-pager status tencent-docs-web.service | head -5
