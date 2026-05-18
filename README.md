# VLM Hallucination in Financial Technical Analysis

## 环境

```bash
pip install openai tushare mplfinance tqdm pandas numpy pyarrow vllm
```

## 配置

把 `demo/config.local.example.py` 复制为 `demo/config.local.py`，填入 key。

## Qwen3-VL（本地）

```bash
# 启动服务（以 8B 为例，4B/2B 同理）
vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8000 --max-model-len 8192

# 跑实验（新终端）
python demo/run_vlm_multimodel_batch.py \
    --model-key qwen3-vl-8b --provider qwen_local \
    --api-base http://localhost:8000/v1 \
    --modes text_only,chart_only,multimodal --end-date 20230630

python demo/run_multi_model_chart_only.py \
    --model-specs "qwen_local:qwen3-vl-8b" \
    --api-base http://localhost:8000/v1 \
    --end-date 20230630 --output-root results_multi_model_chart_only
```

4B/2B 把 `qwen3-vl-8b` 换成 `qwen3-vl-4b` / `qwen3-vl-2b`，重启服务即可。

## Gemini / Claude / GPT（yunwu）

填好 `YUNWU_API_KEY` 后：

```bash
python demo/run_vlm_multimodel_batch.py \
    --model-key gemini-2.5-pro --provider yunwu \
    --modes text_only,chart_only,multimodal --end-date 20230630
```

可用模型：`gemini-2.5-pro` `gemini-2.5-flash` `claude-opus-4-7` `gpt-4o` `o3`

## 结果

- `demo/outputs/multimodel/` — trimodal baseline
- `results_multi_model_chart_only/` — chart-only × 4 方法
