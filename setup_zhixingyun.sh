#!/bin/bash
# 智星云 RTX 4090 一键环境配置
# 在云服务器上运行一次即可

set -e

echo "=== 安装依赖 ==="
pip install vllm autoawq huggingface_hub -q

echo "=== 下载 XuanYuan-FinX1-8B ==="
huggingface-cli download Duxiaoman-FL/XuanYuan-FinX1-8B-Instruct \
    --local-dir ~/models/xuanyuan-finx1-8b \
    --exclude "*.bin"   # 只下 safetensors

echo "=== 下载 FinLLaVA-AWQ ==="
huggingface-cli download SALT-NLP/FinLLaVA-AWQ \
    --local-dir ~/models/finllava-13b-awq

echo "=== 完成 ==="
echo "接下来运行 run_xuanyuan.sh 或 run_finllava.sh 启动服务"
