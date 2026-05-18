#!/bin/bash
# 智星云 RTX 4090 一键环境配置
# 在云服务器上运行一次即可：bash setup_zhixingyun.sh

set -e

echo "=== 安装 vLLM ==="
pip install vllm huggingface_hub -q

echo "=== 登录 HuggingFace（FinLLaVA 需要同意协议） ==="
echo "请先在 https://huggingface.co/TheFinAI/FinLLaVA 点击 Agree，然后输入 token："
huggingface-cli login

echo "=== 下载 FinLLaVA ==="
huggingface-cli download TheFinAI/FinLLaVA \
    --local-dir ~/models/finllava-8b

echo "=== 下载 XuanYuan-13B-Chat ==="
huggingface-cli download Duxiaoman-DI/XuanYuan-13B-Chat \
    --local-dir ~/models/xuanyuan-13b

echo "=== 完成 ==="
echo "接下来运行 run_finllava.sh 或 run_xuanyuan.sh 启动服务"
