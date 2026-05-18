"""
公共配置：所有 API key 通过环境变量读取，仓库内不硬编码任何机密。

跑批前请先在 PowerShell 中：
  $env:TUSHARE_TOKEN  = "..."
  $env:ZHIPU_API_KEY  = "..."   # 智谱 GLM
  $env:NVIDIA_API_KEY = "..."   # NVIDIA Build

或在 demo/config.local.py 中按 demo/config.example.py 的格式填好后再运行；
config.local.py 已加入 .gitignore，不会随仓库提交。
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

_CONFIG_DIR = Path(__file__).resolve().parent
_LOCAL_FILE = _CONFIG_DIR / "config.local.py"


def _load_local_overrides() -> None:
    """如果 demo/config.local.py 存在，把其中的 KEY 注入环境变量（已设置的不覆盖）。"""
    if not _LOCAL_FILE.is_file():
        return
    try:
        spec = importlib.util.spec_from_file_location("_config_local", _LOCAL_FILE)
        if spec is None or spec.loader is None:
            return
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception:
        return
    for name in ("TUSHARE_TOKEN", "ZHIPU_API_KEY", "NVIDIA_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY", "QWEN_LOCAL_API_KEY", "YUNWU_API_KEY"):
        val = getattr(mod, name, None)
        if isinstance(val, str) and val.strip() and not os.environ.get(name):
            os.environ[name] = val.strip()


_load_local_overrides()

TUSHARE_TOKEN = os.environ.get("TUSHARE_TOKEN", "")

LLM_API_KEY = os.environ.get("ZHIPU_API_KEY", "")
LLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
QWEN_LOCAL_API_KEY = os.environ.get("QWEN_LOCAL_API_KEY", "")
YUNWU_API_KEY = os.environ.get("YUNWU_API_KEY", "")

MODELS = {
    "glm-4v-flash": "glm-4v-flash",
}

HS300_INDEX = "399300.SZ"
DATA_YEAR = "2023"
WINDOW_SIZE = 30

CHART_DIR = "charts"
OUTPUT_DIR = "outputs"
