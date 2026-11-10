#!/usr/bin/env bash
set -e

echo "=== Starting DeepSeek Server on Colab ==="
nohup python3 deepseek_app.py > /tmp/deepseek.log 2>&1 &
echo "Process launched in background. Monitoring /tmp/deepseek.log..."
sleep 5
tail -f /tmp/deepseek.log
