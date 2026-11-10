#!/usr/bin/env python3
"""
Colab Keepalive Heartbeat Service
Prevents Google Colab from pruning or disconnecting the GPU runtime due to idle timeout.
"""
import time
import urllib.request
import sys

def main():
    target_url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000/health"
    print(f"=== Starting Colab Keepalive Heartbeat Worker ===", flush=True)
    print(f"Target health probe: {target_url}", flush=True)

    iteration = 0
    while True:
        status_msg = "UNKNOWN"
        try:
            req = urllib.request.Request(target_url, headers={"User-Agent": "ColabKeepalive/1.0"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                status_msg = f"HTTP {resp.status}"
        except Exception as e:
            status_msg = f"Error: {e}"

        if iteration % 10 == 0:
            print(f"[Keepalive #{iteration}] Server health: {status_msg}", flush=True)

        iteration += 1
        time.sleep(15)

if __name__ == "__main__":
    main()
