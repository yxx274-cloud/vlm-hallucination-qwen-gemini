# 复制此文件为 config.local.py，填入你的 API keys
# cp config.local.example.py config.local.py

# ── 必填 ──────────────────────────────────────────────────────────
TUSHARE_TOKEN = "your_tushare_token_here"   # https://tushare.pro/register

# ── 按需填写（只填你要跑的模型对应的 key）────────────────────────
ZHIPU_API_KEY = ""          # GLM-4V-Flash：https://open.bigmodel.cn/
NVIDIA_API_KEY = ""         # Gemma / DeepSeek：https://build.nvidia.com/
YUNWU_API_KEY = ""          # Gemini / Claude / GPT：https://yunwu.ai/

# ── Qwen3-VL 本地部署（vLLM）────────────────────────────────────
# 本地部署不需要 key，留空即可
# 启动命令示例：
#   vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8000 --max-model-len 8192
# 运行实验时加 --api-base http://localhost:8000/v1
QWEN_LOCAL_API_KEY = ""

OPENROUTER_API_KEY = ""
OPENAI_API_KEY = ""
