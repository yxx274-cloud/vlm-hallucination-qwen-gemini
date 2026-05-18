# VLM Hallucination Detection — Qwen3-VL & Gemini Experiments

本仓库包含运行 Qwen3-VL（2B/4B/8B）和 Gemini 系列模型实验所需的全部代码。

---

## 环境要求

```bash
pip install openai tushare mplfinance tqdm pandas numpy pyarrow
```

---

## 配置 API Key

在 `demo/` 目录下新建 `config.local.py`（已在 .gitignore 中，不会被提交）：

```python
# demo/config.local.py

# 智谱 GLM（如需对比）
ZHIPU_API_KEY = ""

# 云雾 AI 中转（Gemini / Claude / GPT）
# 申请地址：https://yunwu.ai/
YUNWU_API_KEY = "your-yunwu-key-here"

# Tushare（拉取 A 股 OHLCV 数据）
# 申请地址：https://tushare.pro/register
TUSHARE_TOKEN = "your-tushare-token-here"
```

---

## 运行 Qwen3-VL（本地部署）

### 第一步：启动本地模型服务

```bash
pip install vllm

# 8B（显存 ≥ 20GB）
vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8000 --max-model-len 8192

# 4B（显存 ≥ 12GB）
vllm serve Qwen/Qwen3-VL-4B-Instruct --port 8000 --max-model-len 8192

# 2B（显存 ≥ 8GB）
vllm serve Qwen/Qwen3-VL-2B-Instruct --port 8000 --max-model-len 8192
```

### 第二步：运行实验（新开终端）

```bash
# 冒烟测试（先跑 5 条验证环境）
python demo/run_vlm_multimodel_batch.py \
    --model-key qwen3-vl-8b --provider qwen_local \
    --api-base http://localhost:8000/v1 \
    --modes text_only,chart_only,multimodal \
    --end-date 20230630 --limit 5

# 全量（去掉 --limit）
python demo/run_vlm_multimodel_batch.py \
    --model-key qwen3-vl-8b --provider qwen_local \
    --api-base http://localhost:8000/v1 \
    --modes text_only,chart_only,multimodal \
    --end-date 20230630

# chart-only × 4 方法（Baseline / Abstain / VtA v2 / EF-TTC）
python demo/run_multi_model_chart_only.py \
    --model-specs "qwen_local:qwen3-vl-8b" \
    --api-base http://localhost:8000/v1 \
    --end-date 20230630 \
    --output-root results_multi_model_chart_only
```

对 4B / 2B 只需把 `qwen3-vl-8b` 替换为 `qwen3-vl-4b` / `qwen3-vl-2b`，重启 vllm 服务换对应模型。

---

## 运行 Gemini / Claude / GPT（云雾 AI 中转）

需要先在 `demo/config.local.py` 中填写 `YUNWU_API_KEY`。

```bash
# Gemini 2.5 Pro 冒烟测试
python demo/run_vlm_multimodel_batch.py \
    --model-key gemini-2.5-pro --provider yunwu \
    --modes text_only,chart_only,multimodal \
    --end-date 20230630 --limit 5

# 全量
python demo/run_vlm_multimodel_batch.py \
    --model-key gemini-2.5-pro --provider yunwu \
    --modes text_only,chart_only,multimodal \
    --end-date 20230630

# chart-only × 4 方法
python demo/run_multi_model_chart_only.py \
    --model-specs "yunwu:gemini-2.5-pro" \
    --end-date 20230630 \
    --output-root results_multi_model_chart_only
```

可用模型名（在 `vlm_agent_v2.py` 的 `yunwu` 部分）：
- `gemini-2.5-pro` / `gemini-2.5-flash` / `gemini-2.0-flash`
- `claude-opus-4-7` / `claude-sonnet-4-5`
- `gpt-4o` / `gpt-4o-mini` / `o3`

---

## 结果目录

| 脚本 | 输出目录 |
|---|---|
| `run_vlm_multimodel_batch.py` | `demo/outputs/multimodel/` |
| `run_multi_model_chart_only.py` | `results_multi_model_chart_only/` |

详细实验口径见 `docs/rerun_2026-05.md`。
