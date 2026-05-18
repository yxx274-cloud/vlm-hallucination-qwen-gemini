#!/bin/bash
# 启动 XuanYuan-FinX1-8B 服务（纯文本，text_only 实验用）
# 显存需求：~16GB（RTX 4090 24GB 足够）

MODEL_PATH=${1:-~/models/xuanyuan-finx1-8b}
PORT=${2:-8000}

vllm serve "$MODEL_PATH" \
    --port $PORT \
    --max-model-len 4096 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.85

# 跑实验命令（新终端）：
# python demo/run_vlm_multimodel_batch.py \
#     --model-key xuanyuan-finx1-8b --provider xuanyuan_local \
#     --api-base http://localhost:8000/v1 \
#     --modes text_only --end-date 20230630
