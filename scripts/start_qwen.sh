#!/usr/bin/env bash
set -e

echo "=== Starting Qwen 27B Q5_K on Colab L4 ==="

# Configure CUDA dynamic linker
python3 -c "
import glob, subprocess
dirs = glob.glob('/usr/local/lib/python3.*/dist-packages/nvidia/*/lib')
if dirs:
    with open('/etc/ld.so.conf.d/nvidia-python.conf', 'w') as f:
        for d in dirs:
            f.write(d + '\n')
    subprocess.run(['ldconfig'])
"

# Launch in background with nohup
nohup python3 qwen27b_app.py > /tmp/qwen27b.log 2>&1 &
echo "Process launched in background. Monitoring /tmp/qwen27b.log..."
sleep 5
tail -f /tmp/qwen27b.log
