"""
VLM Agent v2：多 provider 支持 + 自定义 user_prompt 支持

相比 v1 的改动：
1. 支持通过 provider 参数切换 API 后端：
     - bigmodel:     GLM-4V-Flash / GLM-4V-Plus（原有）
     - openrouter:   Qwen2.5-VL-72B / GPT-4o 等（多模型扩展用）
     - openai:       GPT-4o / GPT-4o-mini 直连
2. 新增 custom_user_prompt 参数：VtA 重新生成时传入带 failed_claims 反馈的 prompt
3. 新增 custom_system_prompt 参数：Abstain / VtA / Blind control 等实验统一走这里
4. 保持与 v1 的 query_vlm() 参数向后兼容，老代码可以直接 import

环境变量（按 provider 取一个即可）：
    ZHIPU_API_KEY       -> bigmodel（保持原有）
    OPENROUTER_API_KEY  -> openrouter
    OPENAI_API_KEY      -> openai
"""
import base64
import os
import random
import re
import time

from openai import OpenAI

try:
    from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

    _OPENAI_TRANSIENT_TYPES = (APIConnectionError, APITimeoutError, RateLimitError)
except ImportError:
    APIStatusError = None  # type: ignore
    _OPENAI_TRANSIENT_TYPES = ()


# ================================================================
# Provider 配置
# ================================================================

VLM_PROVIDERS = {
    "bigmodel": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "env_key": "ZHIPU_API_KEY",
        "model_aliases": {
            "glm-4v-flash": "glm-4v-flash",
            "glm-4v-plus": "glm-4v-plus",
            "glm-4-flash": "glm-4-flash",   # 纯文本 LLM，仅 text_only 模式
        },
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "env_key": "OPENROUTER_API_KEY",
        "model_aliases": {
            "gpt-4o":           "openai/gpt-4o",
            "gpt-4o-mini":      "openai/gpt-4o-mini",
            "qwen2.5-vl-7b":    "qwen/qwen2.5-vl-7b-instruct",
            "qwen2.5-vl-72b":   "qwen/qwen2.5-vl-72b-instruct",
            "claude-sonnet-45": "anthropic/claude-sonnet-4.5",
        },
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "env_key": "OPENAI_API_KEY",
        "model_aliases": {
            "gpt-4o":      "gpt-4o",
            "gpt-4o-mini": "gpt-4o-mini",
        },
    },
    "nvidia": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "env_key": "NVIDIA_API_KEY",
        "model_aliases": {
            "nemotron-nano-vl":         "nvidia/nemotron-nano-12b-v2-vl",
            "llama-3.2-11b-vision":     "meta/llama-3.2-11b-vision-instruct",
            "llama-3.2-90b-vision":     "meta/llama-3.2-90b-vision-instruct",
            "gemma-4-31b-it":           "google/gemma-4-31b-it",
            "gemma-3n-e4b-it":          "google/gemma-3n-e4b-it",
            "gemma-3n-e2b-it":          "google/gemma-3n-e2b-it",
            "cosmos-reason2-8b":        "nvidia/cosmos-reason2-8b",
            "phi-4-multimodal":         "microsoft/phi-4-multimodal-instruct",
            "deepseek-v4-flash":        "deepseek-ai/deepseek-v4-flash",
            "deepseek-v4-pro":          "deepseek-ai/deepseek-v4-pro",
        },
    },
    # 导师本地部署（vLLM / LMDeploy 等 OpenAI 兼容服务）
    # base_url 占位，实际由 runner --api-base 参数覆盖
    "qwen_local": {
        "base_url": "http://localhost:8000/v1",
        "env_key": "QWEN_LOCAL_API_KEY",
        "model_aliases": {
            "qwen3-vl-2b": "Qwen/Qwen3-VL-2B-Instruct",
            "qwen3-vl-4b": "Qwen/Qwen3-VL-4B-Instruct",
            "qwen3-vl-8b": "Qwen/Qwen3-VL-8B-Instruct",
        },
    },
    # 云雾 AI 中转（OpenAI 兼容，支持 Gemini / Claude / GPT 等）
    # base_url: https://yunwu.ai/v1
    # 申请 key: https://yunwu.ai/
    "yunwu": {
        "base_url": "https://yunwu.ai/v1",
        "env_key": "YUNWU_API_KEY",
        "model_aliases": {
            # Gemini 最新系列
            "gemini-2.5-pro":        "gemini-2.5-pro",
            "gemini-2.5-flash":      "gemini-2.5-flash",
            "gemini-2.0-flash":      "gemini-2.0-flash",
            # Claude 最新系列
            "claude-opus-4-7":       "claude-opus-4-7",
            "claude-sonnet-4-5":     "claude-sonnet-4-5",
            # GPT 最新系列
            "gpt-4o":                "gpt-4o",
            "gpt-4o-mini":           "gpt-4o-mini",
            "o3":                    "o3",
        },
    },
}


def _get_client(provider, api_key=None, api_base=None):
    if provider not in VLM_PROVIDERS:
        raise ValueError(f"Unsupported provider: {provider}")
    cfg = VLM_PROVIDERS[provider]
    key = api_key or os.environ.get(cfg["env_key"])
    if not key:
        # 兼容老代码：如果用户没设 ZHIPU_API_KEY 但有 config.py 里的 LLM_API_KEY
        if provider == "bigmodel":
            try:
                from config import LLM_API_KEY
                key = LLM_API_KEY
            except ImportError:
                pass
        # qwen_local 等本地部署可以不鉴权，用占位 key
        if not key:
            if provider in ("qwen_local",):
                key = "EMPTY"
            else:
                raise ValueError(
                    f"API key required for {provider}: set {cfg['env_key']} or pass api_key="
                )
    # api_base 参数可覆盖 provider 默认 base_url（用于 qwen_local 等本地部署）
    base_url = api_base or cfg["base_url"]
    # 读超时拉长，减少 chart 大 payload 下「Connection error」假失败
    # 注意：若系统代理（如 Clash）对 open.bigmodel.cn 做了 TLS 中间人，会导致 SSLEOFError。
    # 解决方法：在代理工具里把 bigmodel.cn 加入直连规则（DIRECT），而非修改此处代码。
    try:
        import httpx
        http_client = httpx.Client(timeout=httpx.Timeout(180.0))
        return OpenAI(api_key=key, base_url=base_url, http_client=http_client)
    except Exception:
        return OpenAI(api_key=key, base_url=base_url, timeout=180.0)


def _resolve_model(provider, model_key):
    """把统一的 model_key 映射到具体 provider 的 model 名"""
    cfg = VLM_PROVIDERS[provider]
    return cfg["model_aliases"].get(model_key, model_key)


# ================================================================
# 三种标准 system prompt（保持与 v1 兼容）
# ================================================================

SYSTEM_PROMPT_BASELINE = """你是一个专业的A股技术分析师。请根据提供的股票数据进行技术分析，并给出交易建议。

**重要**：只描述你在图表或数据中能直接观察到的内容，不要罗列你无法确认的信号。

请严格按照以下格式输出：

【技术分析】
用 3-5 句话描述这只股票近期的技术走势。只写你能从图表或数据中直接看到的内容，不要使用固定模板或逐项列举。

【推理理由】
基于上述技术分析，给出你的综合判断理由。

【交易建议】
动作: BUY / SELL / HOLD（只能选一个）
置信度: 1-5（1=非常不确定, 5=非常确定）

请使用简体中文回答。"""


SYSTEM_PROMPT_ABSTAIN = """你是一个专业的A股技术分析师。你需要根据提供的股票数据进行技术分析，并给出交易建议。

**核心原则**：在证据不足时，承认"无法判断"比强行给出建议更专业。

**何时应该输出 UNCERTAIN**：
- 技术信号不明确（如横盘震荡、无明显趋势）
- 技术信号相互矛盾（如量价背离、均线交叉混乱）
- 数据不足以支持任何方向性判断
- 你对判断没有信心

**重要**：输出 UNCERTAIN 不是失败，而是专业的表现。强行给出没有依据的建议才是错误的。

请严格按照以下格式输出（每部分保持简洁）：

【技术分析】
简明扼要地描述关键技术信号（2-3句话）。如果信号不明确或相互矛盾，请如实说明。

【推理理由】
基于上述技术分析，简要给出你的综合判断理由（2-3句话）。如果证据不足，请明确指出。

【交易建议】
动作: BUY / SELL / HOLD / UNCERTAIN
- 只有在信号明确且有充分证据时，才选择 BUY / SELL / HOLD
- 如果信号不足、不明确、相互矛盾，请选择 UNCERTAIN
置信度: 1-5（仅在选择 BUY/SELL/HOLD 时需要；选择 UNCERTAIN 时不需要置信度）

请使用简体中文回答。"""


# 新增：VtA verification prompt（强制方向性输出，不允许 UNCERTAIN）
SYSTEM_PROMPT_VTA_REGEN = """你是一个专业的A股技术分析师。你之前对某只股票的技术分析存在问题，现在需要你基于反馈重新分析。

**修正原则**：
1. 严格依据实际 OHLCV 数据和 K 线图中可观察到的信号
2. 不得重复之前的错误断言（下文会明确列出）
3. 每条技术信号必须有图表或数据支撑，不确定的信号不要提
4. 仍然必须给出方向性建议（BUY/SELL/HOLD 之一），不可拒绝回答

请严格按照以下格式输出：

【技术分析】
基于实际数据重新描述技术信号，避免之前的错误断言。

【推理理由】
说明修正后的判断依据。

【交易建议】
动作: BUY / SELL / HOLD（必须选一个，不得输出 UNCERTAIN）
置信度: 1-5（根据修正后的证据强度合理给出）

请使用简体中文回答。"""


SYSTEM_PROMPT_EF_TTC_EVIDENCE = """You are a strict technical analyst. Your ONLY task is to fill in a structured evidence form based on the chart image.

OUTPUT FORMAT: You MUST output ONLY a valid JSON object. No markdown, no explanation, no disclaimer. Just the JSON.

Required JSON fields (choose ONLY from the given options):
{
  "ma_cross": "golden_cross|death_cross|no_cross|uncertain",
  "ma_alignment": "bullish|bearish|mixed|uncertain",
  "volume_change": "heavy_volume|light_volume|normal_volume|uncertain",
  "price_breakout": "breakout_high|breakout_low|within_range|uncertain",
  "trend_direction": "uptrend|downtrend|sideways|uncertain",
  "price_position": "high_zone|mid_zone|low_zone|uncertain",
  "short_momentum": "accelerating_up|accelerating_down|decelerating_up|decelerating_down|flat_momentum|uncertain",
  "volume_price_divergence": "volume_price_sync_up|volume_price_sync_down|bullish_divergence|bearish_divergence|no_divergence|uncertain"
}

Example output (bullish chart):
{"ma_cross":"golden_cross","ma_alignment":"bullish","volume_change":"heavy_volume","price_breakout":"breakout_high","trend_direction":"uptrend","price_position":"high_zone","short_momentum":"accelerating_up","volume_price_divergence":"volume_price_sync_up"}

Example output (unclear signals):
{"ma_cross":"uncertain","ma_alignment":"mixed","volume_change":"normal_volume","price_breakout":"within_range","trend_direction":"sideways","price_position":"mid_zone","short_momentum":"flat_momentum","volume_price_divergence":"no_divergence"}

IMPORTANT: Output ONLY the JSON object. No text before or after it."""


SYSTEM_PROMPT_EF_TTC_DECISION = """你是一个专业且克制的A股技术分析师。现在你只能基于“已通过核验的证据”生成结论，不得使用任何未通过核验或不确定的信号。

请严格按照以下格式输出：

【技术分析】
仅描述已通过核验的技术信号；如果有效证据不足，请明确指出信号不足。

【推理理由】
说明这些已验证证据如何支持你的判断；如果证据不足，请明确说明为什么只能保守处理。

【交易建议】
动作: BUY / SELL / HOLD / UNCERTAIN
- 若已验证证据不足以支持明确方向，请优先选择 HOLD 或 UNCERTAIN
- 不得引用未通过核验的信号
置信度: 1-5（若动作为 UNCERTAIN 可省略）"""


SYSTEM_PROMPTS = {
    "baseline":   SYSTEM_PROMPT_BASELINE,
    "abstain":    SYSTEM_PROMPT_ABSTAIN,
    "vta_regen":  SYSTEM_PROMPT_VTA_REGEN,
    "ef_ttc_evidence": SYSTEM_PROMPT_EF_TTC_EVIDENCE,
    "ef_ttc_decision": SYSTEM_PROMPT_EF_TTC_DECISION,
}


# ================================================================
# 核心调用
# ================================================================

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _build_default_user_text(ts_code, end_date, ohlcv_text=None, has_chart=False, anonymize=True):
    """构造默认的 user prompt 文本部分。
    anonymize=True（默认）时不把真实 ts_code 写进 prompt，避免模型靠记忆默写。
    """
    if anonymize:
        intro = f"请分析这只股票（数据截至 {end_date}），请进行技术分析并给出交易建议。"
    else:
        intro = f"请分析股票 {ts_code}（数据截至 {end_date}），请进行技术分析并给出交易建议。"
    parts = [intro]
    if ohlcv_text:
        parts.append(f"以下是最近30个交易日OHLCV数据：\n{ohlcv_text}")
    return "\n\n".join(parts)


def query_vlm(
    model_key,
    ohlcv_text=None,
    chart_path=None,
    ts_code="",
    end_date="",
    allow_abstain=False,
    # 新增参数
    provider="bigmodel",
    system_prompt_key=None,     # "baseline" | "abstain" | "vta_regen"
    custom_system_prompt=None,  # 自定义 system，覆盖上面
    custom_user_prompt=None,    # 自定义 user，覆盖默认文本部分
    temperature=0.0,            # 固定 0 保证可复现（旧默认 0.1 已改）
    seed=20230630,              # 固定 seed 保证可复现
    max_tokens=1024,
    api_key=None,
    api_base=None,              # 覆盖 provider base_url，qwen_local 等本地部署必传
    anonymize=True,             # True 时不把 ts_code 写进 user prompt
):
    """
    调用 VLM 进行分析（多 provider + 自定义 prompt 版本）

    向后兼容：
      - 不传 provider 默认 bigmodel（走原 GLM-4V-Flash 路径）
      - allow_abstain=True 时等价于 system_prompt_key="abstain"
      - 老代码 query_vlm("glm-4v-flash", ohlcv_text=..., chart_path=...) 仍可用

    自定义扩展：
      - provider="openrouter", model_key="qwen2.5-vl-72b" 即可切多模型
      - system_prompt_key="vta_regen" + custom_user_prompt=反馈文本 即可做 VtA
      - custom_system_prompt / custom_user_prompt 任意覆盖
      - api_base 覆盖 provider 默认 base_url（qwen_local 本地部署用）

    Returns:
        {"model", "model_name", "provider", "mode", "response", "usage", "temperature", "seed"}
    """
    client = _get_client(provider, api_key=api_key, api_base=api_base)
    model_name = _resolve_model(provider, model_key)

    # 1. 决定 system prompt
    if custom_system_prompt is not None:
        system_prompt = custom_system_prompt
    elif system_prompt_key is not None:
        system_prompt = SYSTEM_PROMPTS.get(system_prompt_key, SYSTEM_PROMPT_BASELINE)
    elif allow_abstain:
        system_prompt = SYSTEM_PROMPT_ABSTAIN
    else:
        system_prompt = SYSTEM_PROMPT_BASELINE

    # 2. 决定 user prompt 文本
    if custom_user_prompt is not None:
        user_text = custom_user_prompt
    else:
        user_text = _build_default_user_text(
            ts_code, end_date, ohlcv_text=ohlcv_text,
            has_chart=chart_path is not None, anonymize=anonymize
        )

    # 3. 构造 messages（含图则用 list 格式）
    mode_desc = []
    if ohlcv_text:
        mode_desc.append("文本数据")
    if chart_path:
        mode_desc.append("K线图")

    if chart_path:
        b64 = encode_image(chart_path)
        user_content = [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]
    else:
        user_content = user_text

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    # 4. 调 API（网络/限流/网关抖动：多轮退避 + 抖动，便于批量跑满 N=323）
    def _retryable(exc: BaseException) -> bool:
        if _OPENAI_TRANSIENT_TYPES and isinstance(exc, _OPENAI_TRANSIENT_TYPES):
            return True
        if APIStatusError is not None and isinstance(exc, APIStatusError):
            code = getattr(exc, "status_code", None)
            if code is not None and int(code) in (408, 429, 500, 502, 503, 504):
                return True
        msg = str(exc).lower()
        return any(
            s in msg
            for s in (
                "connection",
                "timeout",
                "timed out",
                "429",
                "503",
                "502",
                "504",
                "500",
                "rate limit",
                "overload",
                "temporarily unavailable",
                "remote end closed",
                "broken pipe",
                "reset by peer",
                "ssl",
                "winerror",
            )
        )

    max_attempts = 10
    resp = None
    for attempt in range(max_attempts):
        try:
            create_kwargs = dict(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )
            # seed 参数：OpenAI 兼容协议均支持；部分 provider 忽略但不报错
            if seed is not None:
                create_kwargs["seed"] = seed
            resp = client.chat.completions.create(**create_kwargs)
            break
        except Exception as exc:
            if attempt < max_attempts - 1 and _retryable(exc):
                # 指数退避，上限约 60s，加少量抖动避免同一时刻齐刷刷重试
                base = min(60.0, 2.0 * (2**attempt))
                time.sleep(base + random.uniform(0.15, 1.35))
                continue
            raise
    assert resp is not None

    content = resp.choices[0].message.content or ""
    usage = resp.usage
    prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
    completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

    return {
        "model": model_key,
        "model_name": model_name,
        "provider": provider,
        "mode": "+".join(mode_desc) if mode_desc else "none",
        "response": content,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
        "temperature": temperature,
        "seed": seed,
    }


# ================================================================
# 解析函数（从 v1 原样复制，保持兼容）
# ================================================================

def strip_thinking(text):
    """移除 <think>...</think> 标签"""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def parse_action(response_text):
    """从模型输出中提取动作和置信度"""
    text = strip_thinking(response_text)
    action = None
    confidence = None
    text_upper = text.upper()

    if re.search(r"动作\s*[：:]\s*UNCERTAIN", text_upper):
        action = "UNCERTAIN"
    elif "UNCERTAIN" in text_upper and "动作" in text:
        action = "UNCERTAIN"

    if action is None:
        for keyword in ["BUY", "SELL", "HOLD"]:
            if re.search(rf"动作\s*[：:]\s*{keyword}", text_upper):
                action = keyword
                break
        if action is None:
            for keyword in ["BUY", "SELL", "HOLD"]:
                if keyword in text_upper:
                    action = keyword
                    break

    conf_match = re.search(r"置信度\s*[：:]\s*(\d)", text)
    if conf_match:
        confidence = int(conf_match.group(1))

    if action == "UNCERTAIN":
        confidence = None

    return {"action": action, "confidence": confidence}


def get_analysis_text(response_text):
    return strip_thinking(response_text)


# ================================================================
# 烟雾测试
# ================================================================

if __name__ == "__main__":
    print("=== VLM Agent v2 烟雾测试 ===\n")

    # 测试 1: 保持 v1 兼容路径（GLM-4V-Flash via bigmodel）
    print("--- Test 1: bigmodel/glm-4v-flash baseline (v1 兼容) ---")
    try:
        from data_pipeline import create_sample
        sample = create_sample("600519.SH", "20230630")
        if sample:
            r = query_vlm(
                "glm-4v-flash",
                ohlcv_text=None,
                chart_path=sample["chart_path"],
                ts_code=sample["ts_code"],
                end_date=sample["end_date"],
                provider="bigmodel",
            )
            print(f"  provider={r['provider']}  model={r['model_name']}")
            print(f"  usage={r['usage']}")
            print(f"  response[:150]: {r['response'][:150]}")
    except Exception as e:
        print(f"  跳过（{e}）")

    # 测试 2: VtA verification prompt
    print("\n--- Test 2: VtA regen prompt（带 failed claims 反馈）---")
    failed_claims_feedback = """你之前对股票 600519.SH 的分析中，以下断言无法从数据支持：
- 均线金叉: 实际事实标签为 no_cross（近 5 日未发生交叉）
- 放量突破: 实际成交量 ratio=0.8（实为缩量，不是放量）

请基于上述反馈重新分析。"""

    try:
        sample = create_sample("600519.SH", "20230630")
        if sample:
            r = query_vlm(
                "glm-4v-flash",
                chart_path=sample["chart_path"],
                ts_code=sample["ts_code"],
                end_date=sample["end_date"],
                provider="bigmodel",
                system_prompt_key="vta_regen",
                custom_user_prompt=failed_claims_feedback,
            )
            print(f"  response[:200]: {r['response'][:200]}")
            parsed = parse_action(r["response"])
            print(f"  parsed: {parsed}")
            print(f"  期望: action != UNCERTAIN（强制方向性）")
    except Exception as e:
        print(f"  跳过（{e}）")

    # 测试 3: OpenRouter 路径（需要 OPENROUTER_API_KEY）
    print("\n--- Test 3: openrouter/qwen2.5-vl-72b （需 OPENROUTER_API_KEY） ---")
    if os.environ.get("OPENROUTER_API_KEY"):
        try:
            sample = create_sample("600519.SH", "20230630")
            if sample:
                r = query_vlm(
                    "qwen2.5-vl-72b",
                    chart_path=sample["chart_path"],
                    ts_code=sample["ts_code"],
                    end_date=sample["end_date"],
                    provider="openrouter",
                )
                print(f"  provider={r['provider']}  model={r['model_name']}")
                print(f"  usage={r['usage']}")
                print(f"  response[:150]: {r['response'][:150]}")
        except Exception as e:
            print(f"  调用失败: {e}")
    else:
        print("  跳过（OPENROUTER_API_KEY 未设置）")
