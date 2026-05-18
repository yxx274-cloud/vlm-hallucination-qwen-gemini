"""
Verify-then-Act Agent v2

修复 v1 的致命 bug：
  v1 的 generate_with_verification_prompt() 直接调用 allow_abstain=True 的 Abstain prompt，
  这导致 "重新生成" 步骤实际上只是在切换到 Abstain 模式 —— VtA 就成了
  "Baseline + 高 UCR 时 fallback 到 Abstain" 的级联，不是真正的 CoVe。

v2 的正确做法：
  1. 生成初始分析（baseline prompt）
  2. 正则 + 可选 LLM-judge 提取 failed claims
  3. 构造真正的 verification user prompt：把 failed claims 和正确 fact labels 明确告诉模型
  4. 用 SYSTEM_PROMPT_VTA_REGEN（强制方向性、不允许 UNCERTAIN）重新生成
  5. 再次验证新输出的 UCR，作为最终 UCR

这样 VtA 才能和 Abstain 形成真正的对照：
  - Abstain = 期望层面：允许模型说"不知道"
  - VtA     = 结构层面：必须给出方向性建议，但先修正错误断言
"""
import json
import sys
from pathlib import Path

_ROOT_DIR = Path(__file__).resolve().parent
_DEMO_DIR = _ROOT_DIR / "demo"
if str(_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMO_DIR))

from hallucination_detector import detect_type1_ucc
from fact_labels import compute_all_facts
from vlm_agent_v2 import query_vlm, parse_action


# ================================================================
# 工具函数：把 failed claims 和事实标签转成人类可读的反馈文本
# ================================================================

def _fact_to_human(fact_dict):
    """把 fact dict 转成一句话描述（用于反馈给模型）"""
    label = fact_dict.get("label", "unknown")
    detail = fact_dict.get("detail", "")

    # 常见标签的中文说明
    label_map = {
        "no_cross":           "近 5 日均线未发生金叉或死叉",
        "golden_cross":       "近 5 日 MA5 上穿 MA10（金叉）",
        "death_cross":        "近 5 日 MA5 下穿 MA10（死叉）",
        "bullish":            "MA5 > MA10 > MA20（多头排列）",
        "bearish":            "MA5 < MA10 < MA20（空头排列）",
        "mixed":              "均线无明确排列",
        "breakout_high":      "突破 20 日新高",
        "breakout_low":       "跌破 20 日新低",
        "in_range":           "价格处于 20 日区间内",
        "heavy_volume":       "当日放量（量 > 20 日均量 × 1.5）",
        "light_volume":       "当日缩量",
        "normal_volume":      "成交量正常",
        "uptrend":            "处于上升趋势",
        "downtrend":          "处于下降趋势",
        "sideways":           "横盘震荡，无明确方向",
        "high_zone":          "处于近 20 日高位区间（≥80 分位）",
        "low_zone":           "处于近 20 日低位区间（≤20 分位）",
        "mid_zone":           "处于近 20 日中位区间",
        "volume_price_sync_up":   "量价齐升",
        "volume_price_sync_down": "量价齐跌",
        "bullish_divergence":     "价跌量增（看涨背离）",
        "bearish_divergence":     "价涨量缩（看跌背离）",
        "accelerating_up":    "动量加速上行",
        "accelerating_down":  "动量加速下行",
        "decelerating":       "动量减速",
        "stable":             "动量平稳",
    }
    return label_map.get(label, f"{label}（{detail}）" if detail else label)


def build_verification_feedback(ts_code, end_date, failed_claims, facts):
    """
    构造反馈给模型的 user prompt

    这是 VtA v2 的核心：明确告诉模型哪些断言错了、正确的事实是什么。
    """
    if not failed_claims:
        # 没有 failed claims 时走一般提示
        return (
            f"请重新分析股票 {ts_code}（数据截至 {end_date}）。\n"
            f"仅基于图表和数据中能直接观察到的信号陈述，避免推测性描述。"
        )

    lines = [
        f"你之前对股票 {ts_code}（数据截至 {end_date}）的技术分析存在以下问题：",
        "",
        "**无法被数据支持的断言（必须避免重复）：**",
    ]

    for c in failed_claims[:6]:  # 最多列 6 条，避免 prompt 过长
        claim_desc = c.get("claim_desc", "unknown")
        fact_key = None
        for key in ["ma_cross", "ma_alignment", "price_breakout", "volume_change",
                    "trend_direction", "volume_price_divergence",
                    "price_position", "short_momentum"]:
            if key in c.get("detail", {}):
                fact_key = key
                break
        fact_dict = c.get("detail", {})
        fact_human = _fact_to_human(fact_dict) if fact_dict else "（事实标签不明）"
        lines.append(f"- 你说了 \"{claim_desc}\"，但实际事实是：{fact_human}")

    lines.extend([
        "",
        "**修正要求：**",
        "1. 不得重复上述无依据断言",
        "2. 仅基于实际 OHLCV 数据和 K 线图中可观察到的信号陈述",
        "3. 每条技术信号需有直接数据支撑",
        "4. 必须给出 BUY / SELL / HOLD 之一的方向性建议，不可拒绝回答",
        "5. 置信度需与修正后的证据强度匹配",
        "",
        f"请重新对 {ts_code} 给出技术分析和交易建议。",
    ])

    return "\n".join(lines)


# ================================================================
# VtA Agent v2
# ================================================================

class VerifyThenActAgentV2:
    """
    真正的 Verify-then-Act：
      Step 1: baseline 生成初始分析
      Step 2: 正则 + 可选 LLM-judge 提取 claims
      Step 3: 对 failed claims 构造 verification feedback
      Step 4: 用 vta_regen system prompt + feedback 重新生成（强制方向性）
      Step 5: 对新输出再次验证，记录 final UCR
    """

    def __init__(self,
                 provider="bigmodel",
                 ucr_threshold_high=0.5,
                 ucr_threshold_medium=0.3,
                 use_llm_judge=False,
                 judge=None):
        """
        Args:
            provider: VLM provider（bigmodel/openrouter/openai）
            ucr_threshold_high: UCR > 此值触发重新生成
            ucr_threshold_medium: UCR > 此值触发降低置信度
            use_llm_judge: 是否用 LLM-as-judge 做 Type 1 精确过滤
            judge: 若 use_llm_judge=True，传入 LLMJudge 实例
        """
        self.provider = provider
        self.ucr_high = ucr_threshold_high
        self.ucr_medium = ucr_threshold_medium
        self.use_llm_judge = use_llm_judge
        self.judge = judge

        if use_llm_judge and judge is None:
            raise ValueError("use_llm_judge=True 时必须传入 judge 实例")

    # --------------------------------------------------------------
    # 各步骤
    # --------------------------------------------------------------

    def _verify(self, response_text, df_window):
        """统一的验证接口，可用规则或 hybrid（规则+LLM）"""
        facts = compute_all_facts(df_window)

        if self.use_llm_judge:
            from llm_judge import judge_type1_hybrid
            result = judge_type1_hybrid(response_text, facts, self.judge)
        else:
            result = detect_type1_ucc(response_text, facts)

        return {
            "ucr": result["ucr"],
            "total_claims": result["total_claims"],
            "unsupported_count": result["unsupported_count"],
            "failed_claims": result["unsupported_claims"],  # list of full dict
            "facts": facts,
            "detector": result.get("detector", "regex_only"),
        }

    def _generate_initial(self, model_key, ohlcv_text, chart_path, ts_code, end_date):
        """Step 1: baseline 生成"""
        result = query_vlm(
            model_key,
            ohlcv_text=ohlcv_text,
            chart_path=chart_path,
            ts_code=ts_code,
            end_date=end_date,
            provider=self.provider,
            system_prompt_key="baseline",  # 明确指定 baseline，不允许 UNCERTAIN
        )
        parsed = parse_action(result["response"])
        return {
            "response": result["response"],
            "action": parsed["action"],
            "confidence": parsed["confidence"],
            "usage": result["usage"],
        }

    def _regenerate_with_feedback(self, model_key, ohlcv_text, chart_path,
                                  ts_code, end_date, failed_claims, facts):
        """Step 4: 带反馈的重新生成（VtA 核心修复）"""
        feedback_prompt = build_verification_feedback(
            ts_code, end_date, failed_claims, facts
        )

        result = query_vlm(
            model_key,
            ohlcv_text=ohlcv_text,
            chart_path=chart_path,
            ts_code=ts_code,
            end_date=end_date,
            provider=self.provider,
            system_prompt_key="vta_regen",       # 强制方向性，不许 UNCERTAIN
            custom_user_prompt=feedback_prompt,  # 带 failed claims 的反馈
        )
        parsed = parse_action(result["response"])
        return {
            "response": result["response"],
            "action": parsed["action"],
            "confidence": parsed["confidence"],
            "usage": result["usage"],
            "feedback_prompt": feedback_prompt,
        }

    # --------------------------------------------------------------
    # 完整流程
    # --------------------------------------------------------------

    def generate_with_vta(self, model_key, ohlcv_text, chart_path,
                          ts_code, end_date, df_window, verbose=False):
        """
        完整 VtA 流程

        Returns:
            {
                "response": str,       # 最终输出文本
                "action": str,
                "confidence": int,
                "strategy": str,       # "passed" / "confidence_reduced" / "regenerated"
                "initial_ucr": float,
                "final_ucr": float,
                "initial_response": str,
                "initial_action": str,
                "initial_confidence": int,
                "verification": dict,
                "total_usage": dict,
            }
        """
        if verbose:
            print(f"VtA-v2: {ts_code} @ {end_date}")

        # Step 1: 初始生成
        if verbose:
            print("  [1] baseline 生成")
        initial = self._generate_initial(
            model_key, ohlcv_text, chart_path, ts_code, end_date
        )

        # Step 2-3: 验证
        if verbose:
            print("  [2-3] 验证断言")
        verif = self._verify(initial["response"], df_window)
        initial_ucr = verif["ucr"]

        total_usage = {
            "prompt_tokens": initial["usage"]["prompt_tokens"],
            "completion_tokens": initial["usage"]["completion_tokens"],
        }

        # Step 4: 决策
        if initial_ucr > self.ucr_high:
            # 高 UCR → 重新生成
            if verbose:
                print(f"  [4a] UCR={initial_ucr:.1%} > {self.ucr_high:.0%}, 重新生成...")

            regen = self._regenerate_with_feedback(
                model_key, ohlcv_text, chart_path, ts_code, end_date,
                verif["failed_claims"], verif["facts"],
            )

            # 对新输出再次验证
            verif_final = self._verify(regen["response"], df_window)
            total_usage["prompt_tokens"] += regen["usage"]["prompt_tokens"]
            total_usage["completion_tokens"] += regen["usage"]["completion_tokens"]

            return {
                "strategy": "regenerated",
                "response": regen["response"],
                "action": regen["action"],
                "confidence": regen["confidence"],
                "initial_response": initial["response"],
                "initial_action": initial["action"],
                "initial_confidence": initial["confidence"],
                "initial_ucr": initial_ucr,
                "final_ucr": verif_final["ucr"],
                "verification_initial": verif,
                "verification_final": verif_final,
                "feedback_prompt": regen["feedback_prompt"],
                "total_usage": total_usage,
            }

        elif initial_ucr > self.ucr_medium:
            # 中等 UCR → 降低置信度
            if verbose:
                print(f"  [4b] UCR={initial_ucr:.1%} > {self.ucr_medium:.0%}, 降低置信度")

            original_conf = initial["confidence"]
            reduced_conf = max(1, original_conf - 2) if original_conf else None

            return {
                "strategy": "confidence_reduced",
                "response": initial["response"],
                "action": initial["action"],
                "confidence": reduced_conf,
                "initial_response": initial["response"],
                "initial_action": initial["action"],
                "initial_confidence": original_conf,
                "initial_ucr": initial_ucr,
                "final_ucr": initial_ucr,
                "verification_initial": verif,
                "verification_final": verif,
                "total_usage": total_usage,
            }

        else:
            # 低 UCR → 通过
            if verbose:
                print(f"  [4c] UCR={initial_ucr:.1%} < {self.ucr_medium:.0%}, 通过")

            return {
                "strategy": "passed",
                "response": initial["response"],
                "action": initial["action"],
                "confidence": initial["confidence"],
                "initial_response": initial["response"],
                "initial_action": initial["action"],
                "initial_confidence": initial["confidence"],
                "initial_ucr": initial_ucr,
                "final_ucr": initial_ucr,
                "verification_initial": verif,
                "verification_final": verif,
                "total_usage": total_usage,
            }


# ================================================================
# 测试入口
# ================================================================

if __name__ == "__main__":
    from data_pipeline import load_stock_data

    print("=== VtA v2 测试 ===\n")

    ts_code = "600519.SH"
    end_date = "20230630"
    chart_path = f"charts/{ts_code}_{end_date}.png"

    df = load_stock_data(ts_code, end_date, window=30)
    if df is None or len(df) < 30:
        print("数据加载失败")
        exit()

    agent = VerifyThenActAgentV2(
        provider="bigmodel",
        ucr_threshold_high=0.5,
        ucr_threshold_medium=0.3,
        use_llm_judge=False,  # 先不用 LLM-judge，纯规则即可验证 prompt 修复
    )

    result = agent.generate_with_vta(
        model_key="glm-4v-flash",
        ohlcv_text=None,
        chart_path=chart_path,
        ts_code=ts_code,
        end_date=end_date,
        df_window=df,
        verbose=True,
    )

    print("\n" + "="*60)
    print(f"策略: {result['strategy']}")
    print(f"初始 UCR: {result['initial_ucr']:.1%}")
    print(f"最终 UCR: {result['final_ucr']:.1%}")
    print(f"初始 action: {result['initial_action']} (confidence={result['initial_confidence']})")
    print(f"最终 action: {result['action']} (confidence={result['confidence']})")

    if result["strategy"] == "regenerated":
        print(f"\n--- 反馈 prompt ---")
        print(result["feedback_prompt"][:400])
        print(f"\n--- 初始输出 [:200] ---")
        print(result["initial_response"][:200])
        print(f"\n--- 重新生成 [:200] ---")
        print(result["response"][:200])
        print(f"\n关键验证：最终 action 应 != UNCERTAIN（VtA 强制方向性）")
        assert result["action"] != "UNCERTAIN", "❌ VtA 不应输出 UNCERTAIN！"
        print("✓ 通过")

    print(f"\n总 token 使用: {result['total_usage']}")
