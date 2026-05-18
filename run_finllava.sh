#!/bin/bash
# 启动 FinLLaVA-8B 服务（多模态，chart_only / multimodal 实验用）
# 显存需求：bf16 ~16GB（RTX 4090 24GB 足够）
# 用法：bash run_finllava.sh [模型路径] [端口]

MODEL_PATH=${1:-~/models/finllava-8b}
PORT=${2:-8001}

vllm serve "$MODEL_PATH" \
    --port $PORT \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.75

# ============================================================
# 跑实验（新终端，在本机或云服务器上执行）：
#
# python demo/run_vlm_multimodel_batch.py \
#     --model-key finllava-8b --provider finllava_local \
#     --api-base http://localhost:8001/v1 \
#     --modes text_only,chart_only,multimodal --end-date 20230630
#
# python demo/run_multi_model_chart_only.py \
#     --model-specs "finllava_local:finllava-8b" \
#     --api-base http://localhost:8001/v1 \
#     --end-date 20230630 --output-root results_multi_model_chart_only
# ============================================================
