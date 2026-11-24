#!/usr/bin/env bash
set -euo pipefail
root=/root/autodl-tmp
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
cd "$root/.deep-report-run"
exec "$root/.venv-vllm/bin/vllm" serve "$root/models/Qwen3-32B" \
    --served-model-name qwen3-32b --host 127.0.0.1 --port 8000 \
    --dtype bfloat16 --max-model-len 16384 --max-num-seqs 8 \
    --gpu-memory-utilization 0.92 --enforce-eager
