# 金融专用模型实验指南（智星云 RTX 4090）

两个模型：
- **XuanYuan-FinX1-8B**（度小满，纯文本金融大模型）— text_only 实验
- **FinLLaVA-13B-AWQ**（Open-FinLLMs，金融多模态）— 三模态全跑

---

## 第一步：租机器

智星云 https://www.zhixingyun.com/ 选 RTX 4090 × 1，镜像选 PyTorch 2.x + CUDA 12.x。

---

## 第二步：上传代码

```bash
git clone https://github.com/yxx274-cloud/vlm-hallucination-qwen-gemini
cd vlm-hallucination-qwen-gemini
cp demo/config.local.example.py demo/config.local.py
# config.local.py 里 TUSHARE_TOKEN 已填好，不用改
```

---

## 第三步：安装环境 + 下载模型

```bash
bash setup_zhixingyun.sh
```

下载时间约 30-60 分钟（取决于网速）。

---

## 第四步：跑 XuanYuan（text_only）

终端 1：
```bash
bash run_xuanyuan.sh
# 等出现 "Application startup complete" 再开终端 2
```

终端 2：
```bash
python demo/run_vlm_multimodel_batch.py \
    --model-key xuanyuan-finx1-8b --provider xuanyuan_local \
    --api-base http://localhost:8000/v1 \
    --modes text_only --end-date 20230630
```

---

## 第五步：跑 FinLLaVA（三模态）

关掉 XuanYuan 服务，终端 1：
```bash
bash run_finllava.sh
```

终端 2：
```bash
# 三模态 baseline
python demo/run_vlm_multimodel_batch.py \
    --model-key finllava-13b-awq --provider finllava_local \
    --api-base http://localhost:8001/v1 \
    --modes text_only,chart_only,multimodal --end-date 20230630

# chart-only × 4 方法
python demo/run_multi_model_chart_only.py \
    --model-specs "finllava_local:finllava-13b-awq" \
    --api-base http://localhost:8001/v1 \
    --end-date 20230630 --output-root results_multi_model_chart_only
```

---

## 结果打包

```bash
tar -czf results_finance_models.tar.gz \
    demo/outputs/multimodel/ \
    results_multi_model_chart_only/
```

下载到本地后发给我分析。
