"""
Self-Check Agent（单轮自检）

两步流程：
1. 直接 baseline 调用得 action + response（与 Forced-Decision 相同）
2. 用同一模型 + 同一 chart + 原始 response 追问：
   "Based ONLY on the visible chart evidence, is the reasoning above well-supported?
    Answer with YES or NO on the first line, then briefly explain."
   若回答 NO → 改 action 为 HOLD，final_confidence = 2

这代表"只靠模型自审，无外部 deterministic verification"的基线，
与 VtA v2（外部规则核验）和 EF-TTC（结构化证据 + 核验 + 约束生成）形成对照。
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT_DIR = Path(__file__).resolve().parent
_DEMO_DIR = _ROOT_DIR / "demo"
if str(_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMO_DIR))

from hallucination_detector import detect_all_hallucinations
from vlm_agent_v2 import parse_action, query_vlm

SELF_CHECK_PROMPT = """You have just produced the technical analysis above for this stock chart.

Now, review your own reasoning critically:
- Look ONLY at what is directly visible in the chart image.
- Do NOT rely on general market knowledge or assumptions.

Question: Is the trading recommendation above well-supported by concrete, visible evidence in the chart?

Answer format (strictly follow this):
Line 1: YES or NO
Line 2: One sentence explaining the key evidence (if YES) or the key gap (if NO)."""


class SelfCheckAgent:
    def __init__(self, provider: str = "bigmodel"):
        self.provider = provider

    def generate_with_self_check(
        self,
        ts_code: str,
        end_date: str,
        ohlcv_text: str | None,
        chart_path: str,
        model_key: str,
        df_window=None,
    ) -> dict:
        """
        Step 1: 直接 baseline 调用
        Step 2: 追问自检
        若自检回答 NO → action 改为 HOLD
        """
        # ── Step 1: baseline 调用 ──────────────────────────────────────
        step1 = query_vlm(
            model_key,
            ohlcv_text=ohlcv_text,
            chart_path=chart_path,
            ts_code=ts_code,
            end_date=end_date,
            provider=self.provider,
            system_prompt_key="baseline",
        )
        parsed1 = parse_action(step1["response"])
        original_action = parsed1["action"]
        original_confidence = parsed1["confidence"]

        # ── Step 2: 自检追问 ──────────────────────────────────────────
        # 把原始 response 拼入 user message，让模型自审
        self_check_user_msg = (
            f"[Previous analysis for {ts_code} on {end_date}]\n\n"
            f"{step1['response']}\n\n"
            f"---\n{SELF_CHECK_PROMPT}"
        )
        step2 = query_vlm(
            model_key,
            ohlcv_text=None,          # 不再传 OHLCV，只看图
            chart_path=chart_path,
            ts_code=ts_code,
            end_date=end_date,
            provider=self.provider,
            system_prompt_key="baseline",   # 用 baseline system prompt 保持一致
            custom_user_prompt=self_check_user_msg,
        )
        sc_response = step2["response"].strip()

        # 解析自检结果：第一行是 YES/NO
        first_line = sc_response.splitlines()[0].strip().upper() if sc_response else ""
        self_check_passed = first_line.startswith("YES")
        self_check_triggered = not self_check_passed  # NO → 触发降级

        # ── 最终 action ───────────────────────────────────────────────
        if self_check_triggered:
            final_action = "HOLD"
            final_confidence = 2
        else:
            final_action = original_action
            final_confidence = original_confidence

        # ── 幻觉检测（df_window 可为 None：与 chart-only baseline 一致用占位 facts）──
        detection = detect_all_hallucinations(
            step1["response"],
            final_action,
            final_confidence,
            df_window,
        )

        total_usage = {
            "prompt_tokens": step1["usage"]["prompt_tokens"] + step2["usage"]["prompt_tokens"],
            "completion_tokens": step1["usage"]["completion_tokens"] + step2["usage"]["completion_tokens"],
        }

        return {
            "method": "self_check",
            "response": step1["response"],
            "step1_response": step1["response"],
            "step1_action": original_action,
            "step1_confidence": original_confidence,
            "self_check_response": sc_response,
            "self_check_passed": self_check_passed,
            "self_check_triggered": self_check_triggered,
            "action": final_action,
            "confidence": final_confidence,
            "detection": detection,
            "final_ucr": detection["type1_ucc"]["ucr"],
            "type2_inconsistent": detection["type2_rai"]["inconsistent"],
            "type3_overclaim": detection["type3_ieo"]["overclaim"],
            "total_usage": total_usage,
        }
