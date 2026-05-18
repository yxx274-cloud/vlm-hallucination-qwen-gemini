"""
Evidence-First Test-Time Control (EF-TTC)

最小可运行版流程：
1. 先让 VLM 逐项填写结构化证据表（只允许离散标签）
2. 使用 OHLCV facts 做 deterministic verification
3. 仅把“通过核验”的证据反馈给模型生成最终分析与动作
4. 若通过核验的方向性证据不足，则鼓励 HOLD / UNCERTAIN
"""
import json
import re
import sys
from pathlib import Path

_ROOT_DIR = Path(__file__).resolve().parent
_DEMO_DIR = _ROOT_DIR / "demo"
if str(_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMO_DIR))

from fact_labels import compute_all_facts
from hallucination_detector import detect_all_hallucinations
from vlm_agent_v2 import parse_action, query_vlm

EVIDENCE_FIELDS = [
    "ma_cross",
    "ma_alignment",
    "volume_change",
    "price_breakout",
    "trend_direction",
    "price_position",
    "short_momentum",
    "volume_price_divergence",
]

FIELD_LABELS_ZH = {
    "ma_cross": "均线交叉",
    "ma_alignment": "均线排列",
    "volume_change": "成交量变化",
    "price_breakout": "价格突破",
    "trend_direction": "趋势方向",
    "price_position": "价格区间位置",
    "short_momentum": "短期动量",
    "volume_price_divergence": "量价关系",
}


def _extract_json_object(text: str):
    text_orig = text.strip()
    text_lower = text_orig.lower()

    # 1. 先尝试从 markdown 代码块中提取（```json ... ``` 或 ``` ... ```）
    md_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text_orig)
    if md_match:
        try:
            return json.loads(md_match.group(1))
        except json.JSONDecodeError:
            pass

    # 2. 尝试直接找最外层 JSON 对象
    brace_matches = list(re.finditer(r"\{", text_orig))
    for start_m in brace_matches:
        start = start_m.start()
        depth = 0
        for i, ch in enumerate(text_orig[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text_orig[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break

    # 3. 从自然语言文本中提取字段值（处理 Llama 的 **Field:** value 格式）
    # 字段别名 -> 标准字段名
    FIELD_ALIASES = {
        "ma_cross": ["ma cross", "ma_cross", "moving average cross", "golden cross", "death cross"],
        "ma_alignment": ["ma alignment", "ma_alignment", "moving average alignment", "ma trend"],
        "volume_change": ["volume change", "volume_change", "volume"],
        "price_breakout": ["price breakout", "price_breakout", "breakout"],
        "trend_direction": ["trend direction", "trend_direction", "trend"],
        "price_position": ["price position", "price_position", "price zone", "price level"],
        "short_momentum": ["short momentum", "short_momentum", "momentum", "rsi", "macd"],
        "volume_price_divergence": ["volume.price divergence", "volume_price_divergence",
                                    "volume-price divergence", "volume price divergence",
                                    "volume and price"],
    }
    # 值关键词 -> 标准值
    VALUE_MAP = [
        (["golden cross", "bullish cross", "crossed above"], "golden_cross"),
        (["death cross", "bearish cross", "crossed below"], "death_cross"),
        (["no cross", "not crossed", "has not crossed", "no clear cross"], "no_cross"),
        (["bullish alignment", "bullish", "all above", "upward alignment"], "bullish"),
        (["bearish alignment", "bearish", "all below", "downward alignment"], "bearish"),
        (["mixed", "neutral alignment", "mixed alignment"], "mixed"),
        (["heavy volume", "high volume", "increased volume", "significant volume"], "heavy_volume"),
        (["light volume", "low volume", "decreased volume", "below average"], "light_volume"),
        (["normal volume", "average volume", "consistent", "relatively consistent"], "normal_volume"),
        (["breakout high", "broken out", "above resistance", "new high"], "breakout_high"),
        (["breakout low", "below support", "new low"], "breakout_low"),
        (["within range", "within its range", "no breakout", "consolidating"], "within_range"),
        (["uptrend", "upward trend", "upward", "rising"], "uptrend"),
        (["downtrend", "downward trend", "downward", "falling", "declining"], "downtrend"),
        (["sideways", "lateral", "ranging", "consolidation", "flat"], "sideways"),
        (["high zone", "upper zone", "near resistance", "overbought"], "high_zone"),
        (["mid zone", "middle zone", "mid-range", "middle range"], "mid_zone"),
        (["low zone", "lower zone", "near support", "oversold"], "low_zone"),
        (["accelerating up", "accelerating upward", "strong momentum up"], "accelerating_up"),
        (["accelerating down", "accelerating downward", "strong momentum down"], "accelerating_down"),
        (["decelerating up", "slowing up"], "decelerating_up"),
        (["decelerating down", "slowing down"], "decelerating_down"),
        (["flat momentum", "flat", "no momentum", "weak momentum"], "flat_momentum"),
        (["sync up", "volume price sync up", "bullish sync"], "volume_price_sync_up"),
        (["sync down", "volume price sync down", "bearish sync"], "volume_price_sync_down"),
        (["bullish divergence"], "bullish_divergence"),
        (["bearish divergence"], "bearish_divergence"),
        (["no divergence", "no significant divergence", "aligned"], "no_divergence"),
        (["uncertain", "unclear", "cannot determine", "difficult to determine",
          "not enough", "insufficient", "n/a", "unknown", "hard to"], "uncertain"),
    ]

    def map_value(val_text: str) -> str:
        val_lower = val_text.lower()
        for keywords, std_val in VALUE_MAP:
            if any(kw in val_lower for kw in keywords):
                return std_val
        return "uncertain"

    rebuilt = {}
    # 匹配 **Field:** 后跟内容（同行或下一行）
    for field, aliases in FIELD_ALIASES.items():
        for alias in aliases:
            # 格式1: **MA Cross:** value（同行）
            pat1 = rf'\*{{0,2}}{re.escape(alias)}\*{{0,2}}\s*:+\s*(.+?)(?:\n|$)'
            m = re.search(pat1, text_lower)
            if m:
                rebuilt[field] = map_value(m.group(1))
                break
            # 格式2: MA Cross:\nThe ... (下一行是描述)
            pat2 = rf'{re.escape(alias)}\s*:+\s*\n\s*(.+?)(?:\n|$)'
            m = re.search(pat2, text_lower)
            if m:
                rebuilt[field] = map_value(m.group(1))
                break

    if len(rebuilt) >= 3:
        return rebuilt

    raise ValueError("EF-TTC evidence step 未返回 JSON")


def _normalize_prediction(pred: dict):
    normalized = {}
    for key in EVIDENCE_FIELDS:
        value = pred.get(key, "uncertain")
        if value is None:
            value = "uncertain"
        normalized[key] = str(value).strip().lower()
    return normalized


def verify_evidence_prediction(predicted: dict, facts: dict):
    verified = []
    rejected = []

    for key in EVIDENCE_FIELDS:
        pred_label = predicted.get(key, "uncertain")
        fact_label = str(facts.get(key, {}).get("label", "unknown")).strip().lower()

        if pred_label == "uncertain":
            rejected.append({
                "field": key,
                "predicted": pred_label,
                "actual": fact_label,
                "reason": "model_uncertain",
            })
            continue

        if pred_label == fact_label:
            verified.append({
                "field": key,
                "predicted": pred_label,
                "actual": fact_label,
            })
        else:
            rejected.append({
                "field": key,
                "predicted": pred_label,
                "actual": fact_label,
                "reason": "label_mismatch",
            })

    directional_fields = {
        "ma_cross": {"golden_cross", "death_cross"},
        "price_breakout": {"breakout_high", "breakout_low"},
        "trend_direction": {"uptrend", "downtrend"},
        "short_momentum": {"accelerating_up", "accelerating_down"},
        "volume_price_divergence": {
            "volume_price_sync_up",
            "volume_price_sync_down",
            "bullish_divergence",
            "bearish_divergence",
        },
    }

    directional_verified = [
        item for item in verified
        if item["field"] in directional_fields and item["actual"] in directional_fields[item["field"]]
    ]

    return {
        "verified": verified,
        "rejected": rejected,
        "verified_count": len(verified),
        "rejected_count": len(rejected),
        "directional_verified_count": len(directional_verified),
        "directional_verified": directional_verified,
    }


def build_verified_evidence_prompt(ts_code: str, end_date: str, verification: dict):
    verified = verification["verified"]
    rejected = verification["rejected"]

    lines = [
        f"请基于已通过核验的证据重新分析股票 {ts_code}（数据截至 {end_date}）。",
        "",
        "【已通过核验的证据】",
    ]

    if verified:
        for item in verified:
            field = item["field"]
            zh = FIELD_LABELS_ZH.get(field, field)
            lines.append(f"- {zh}: {item['actual']}")
    else:
        lines.append("- 无。当前没有足够的已验证技术信号支持方向性判断。")

    lines.extend(["", "【未通过核验或不确定的证据】"])
    if rejected:
        for item in rejected:
            zh = FIELD_LABELS_ZH.get(item["field"], item["field"])
            lines.append(f"- {zh}: 模型填写为 {item['predicted']}，实际为 {item['actual']}，不得作为结论依据")
    else:
        lines.append("- 无")

    lines.extend([
        "",
        "要求：",
        "1. 只能依据已通过核验的证据撰写分析",
        "2. 不得再次使用未通过核验或不确定的信号",
        "3. 若方向性证据不足，请输出 HOLD 或 UNCERTAIN",
        "4. 置信度必须与已验证证据强度匹配",
    ])

    return "\n".join(lines)


class EvidenceFirstTTCAgent:
    def __init__(self, provider="bigmodel"):
        self.provider = provider

    def _extract_evidence_table(self, model_key, ohlcv_text, chart_path, ts_code, end_date):
        result = query_vlm(
            model_key,
            ohlcv_text=ohlcv_text,
            chart_path=chart_path,
            ts_code=ts_code,
            end_date=end_date,
            provider=self.provider,
            system_prompt_key="ef_ttc_evidence",
            temperature=0.0,
            max_tokens=512,
        )
        parsed = _extract_json_object(result["response"])
        normalized = _normalize_prediction(parsed)
        return {
            "raw_response": result["response"],
            "predicted_evidence": normalized,
            "usage": result["usage"],
        }

    def _generate_decision(self, model_key, ohlcv_text, chart_path, ts_code, end_date, verification):
        decision_prompt = build_verified_evidence_prompt(ts_code, end_date, verification)
        result = query_vlm(
            model_key,
            ohlcv_text=ohlcv_text,
            chart_path=chart_path,
            ts_code=ts_code,
            end_date=end_date,
            provider=self.provider,
            system_prompt_key="ef_ttc_decision",
            custom_user_prompt=decision_prompt,
            temperature=0.1,
            max_tokens=1024,
        )
        parsed = parse_action(result["response"])
        return {
            "decision_prompt": decision_prompt,
            "response": result["response"],
            "action": parsed["action"],
            "confidence": parsed["confidence"],
            "usage": result["usage"],
        }

    def generate_with_ef_ttc(self, model_key, ohlcv_text, chart_path, ts_code, end_date, df_window):
        facts = compute_all_facts(df_window)
        evidence_step = self._extract_evidence_table(
            model_key, ohlcv_text, chart_path, ts_code, end_date
        )
        verification = verify_evidence_prediction(evidence_step["predicted_evidence"], facts)
        decision_step = self._generate_decision(
            model_key, ohlcv_text, chart_path, ts_code, end_date, verification
        )
        detection = detect_all_hallucinations(
            decision_step["response"],
            decision_step["action"],
            decision_step["confidence"],
            df_window,
        )

        total_usage = {
            "prompt_tokens": evidence_step["usage"]["prompt_tokens"] + decision_step["usage"]["prompt_tokens"],
            "completion_tokens": evidence_step["usage"]["completion_tokens"] + decision_step["usage"]["completion_tokens"],
        }

        return {
            "method": "ef_ttc",
            "evidence_raw_response": evidence_step["raw_response"],
            "predicted_evidence": evidence_step["predicted_evidence"],
            "verification": verification,
            "facts": facts,
            "decision_prompt": decision_step["decision_prompt"],
            "response": decision_step["response"],
            "action": decision_step["action"],
            "confidence": decision_step["confidence"],
            "detection": detection,
            "final_ucr": detection["type1_ucc"]["ucr"],
            "type2_inconsistent": detection["type2_rai"]["inconsistent"],
            "type3_overclaim": detection["type3_ieo"]["overclaim"],
            "total_usage": total_usage,
        }

    def generate_with_ef_ttc_ac(self, model_key, ohlcv_text, chart_path, ts_code, end_date, df_window):
        """
        EF-TTC + Action Consistency Check (EF-TTC+AC)

        在 EF-TTC 三步流程基础上，增加第四步：
        用结构化证据表中已通过核验的字段聚合方向（bullish/bearish/neutral），
        与最终 action 比对；若方向相反则降为 HOLD。

        聚合规则：
          bullish 信号：golden_cross, bullish, heavy_volume, breakout_high,
                        uptrend, high_zone, accelerating_up, volume_price_sync_up
          bearish 信号：death_cross, bearish, breakout_low, downtrend,
                        low_zone, accelerating_down, volume_price_sync_down
          其余（uncertain / mixed / normal / within_range 等）：neutral，不计分

        聚合方向 = bullish_count > bearish_count → "bullish"
                 = bearish_count > bullish_count → "bearish"
                 = 否则 → "neutral"

        不一致条件：
          - 聚合方向 bullish 但 action == SELL
          - 聚合方向 bearish 但 action == BUY
        触发时：action → HOLD，confidence → 2
        """
        # 先跑标准 EF-TTC
        base = self.generate_with_ef_ttc(model_key, ohlcv_text, chart_path, ts_code, end_date, df_window)

        # ── 聚合证据方向 ──────────────────────────────────────────────
        BULLISH_VALUES = {
            "golden_cross", "bullish", "heavy_volume", "breakout_high",
            "uptrend", "high_zone", "accelerating_up", "volume_price_sync_up",
            "bullish_divergence",
        }
        BEARISH_VALUES = {
            "death_cross", "bearish", "breakout_low", "downtrend",
            "low_zone", "accelerating_down", "volume_price_sync_down",
            "bearish_divergence",
        }

        # 只用通过核验的字段
        verified_fields = {item["field"]: item["actual"]
                           for item in base["verification"].get("verified", [])}
        # 若无通过核验字段，退而使用预测证据
        evidence_to_use = verified_fields if verified_fields else base["predicted_evidence"]

        bullish_count = sum(1 for v in evidence_to_use.values() if v in BULLISH_VALUES)
        bearish_count = sum(1 for v in evidence_to_use.values() if v in BEARISH_VALUES)

        if bullish_count > bearish_count:
            agg_direction = "bullish"
        elif bearish_count > bullish_count:
            agg_direction = "bearish"
        else:
            agg_direction = "neutral"

        # ── 一致性检查 ────────────────────────────────────────────────
        action = base["action"]
        ac_triggered = (
            (agg_direction == "bullish" and action == "SELL") or
            (agg_direction == "bearish" and action == "BUY")
        )

        if ac_triggered:
            final_action = "HOLD"
            final_confidence = 2
        else:
            final_action = action
            final_confidence = base["confidence"]

        # 重新跑幻觉检测（用修正后的 action）
        detection = detect_all_hallucinations(
            base["response"],
            final_action,
            final_confidence,
            df_window,
        )

        return {
            **base,
            "method": "ef_ttc_ac",
            "ac_agg_direction": agg_direction,
            "ac_bullish_count": bullish_count,
            "ac_bearish_count": bearish_count,
            "ac_triggered": ac_triggered,
            "action": final_action,
            "confidence": final_confidence,
            "detection": detection,
            "final_ucr": detection["type1_ucc"]["ucr"],
            "type2_inconsistent": detection["type2_rai"]["inconsistent"],
            "type3_overclaim": detection["type3_ieo"]["overclaim"],
        }
