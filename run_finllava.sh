#!/bin/bash
# 启动 FinLLaVA-AWQ 服务（多模态，chart_only / multimodal 实验用）
# 显存需求：~10GB（RTX 4090 24GB 足够）

MODEL_PATH=${1:-~/models/finllava-13b-awq}
PORT=${2:-8001}

vllm serve "$MODEL_PATH" \
    --port $PORT \
    --max-model-len 4096 \
    --quantization awq \
    --gpu-memory-utilization 0.5

# 跑实验命令（新终端）：
# python demo/run_vlm_multimodel_batch.py \
#     --model-key finllava-13b-awq --provider finllava_local \
#     --api-base http://localhost:8001/v1 \
#     --modes text_only,chart_only,multimodal --end-date 20230630
#
# python demo/run_multi_model_chart_only.py \
#     --model-specs "finllava_local:finllava-13b-awq" \
#     --api-base http://localhost:8001/v1 \
#     --end-date 20230630 --output-root results_multi_model_chart_only
