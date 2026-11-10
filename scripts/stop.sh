#!/usr/bin/env bash
echo "Stopping all active model and tunnel processes..."
pkill -f "python3.*(qwen27b_app|deepseek_app|colab_keepalive)" || true
pkill -f "cloudflared tunnel" || true
echo "All processes stopped cleanly."
