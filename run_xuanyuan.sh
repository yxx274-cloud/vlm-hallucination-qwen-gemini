#!/bin/bash
# 启动 XuanYuan-13B-Chat 服务（纯文本，text_only 实验用）
# 显存需求：fp16 ~26GB（RTX 4090 24GB 不够，需要 4-bit 量化）
# 用法：bash run_xuanyuan.sh [模型路径] [端口]

MODEL_PATH=${1:-~/models/xuanyuan-13b}
PORT=${2:-8000}

# 4-bit 量化（显存 ~13GB，RTX 4090 24GB 足够）
pip install bitsandbytes -q 2>/dev/null || true

vllm serve "$MODEL_PATH" \
    --port $PORT \
    --max-model-len 4096 \
    --quantization bitsandbytes \
    --load-format bitsandbytes \
    --gpu-memory-utilization 0.85

# ============================================================
# 跑实验（新终端，在本机或云服务器上执行）：
#
# python demo/run_vlm_multimodel_batch.py \
#     --model-key xuanyuan-13b --provider xuanyuan_local \
#     --api-base http://localhost:8000/v1 \
#     --modes text_only --end-date 20230630
# ============================================================
