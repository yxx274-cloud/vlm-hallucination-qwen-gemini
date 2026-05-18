"""
三类幻觉自动检测系统（v3.1）

改进依据：
  - ChartHal (2025): 扩展 chart-absent information 检测维度
  - HALoGEN (ACL 2025 Outstanding Paper): 否定句极性过滤，避免误报
  - HalluLens (ACL 2025): Type2 内部矛盾强度分级

三类幻觉维度（方案A命名）：
  Type 1 — Fact Grounding Rate (FGR)：事实依据率
           原名 Unsupported Chart Claim (UCC)
           度量模型技术断言中缺乏 OHLCV 事实支撑的比例
  Type 2 — Reasoning-Action Consistency Index (RCI)：推理-动作一致性指数
           原名 Rationale-Action Inconsistency (RAI)
           检测推理文本情感方向与最终交易动作之间的矛盾
  Type 3 — Evidence Calibration Index (ECI)：证据校准指数
           原名 Insufficient-Evidence Overclaim (IEO)
           检测在技术信号不足时模型仍以高置信度给出方向性建议的过度断言行为

向后兼容：旧字段名 type1_ucc / type2_rai / type3_ieo 保留为别名。
"""
import re
from fact_labels import compute_all_facts, facts_to_readable


# ================================================================
# 语言检测（防止英文响应让中文 CLAIM_PATTERNS 全部失效）
# ================================================================

def detect_response_language(text: str) -> str:
    """检测响应语言。中文字符占比 < 10% 视为非中文（non_zh）。"""
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    total = max(len(text.strip()), 1)
    return "zh" if chinese_chars / total >= 0.10 else "non_zh"


# ================================================================
# 否定词列表（HALoGEN 原子断言极性过滤思路）
# ================================================================

NEGATION_WORDS = [r"未", r"没有", r"不存在", r"并未", r"尚未", r"无明显", r"没出现",
                  r"不构成", r"未形成", r"未见", r"未发现", r"并不", r"不太"]


def is_negated(text, match_start, window=12):
    """检查 match_start 前 window 字符内是否含否定词"""
    context = text[max(0, match_start - window):match_start]
    return any(re.search(p, context) for p in NEGATION_WORDS)


# ================================================================
# Type 1 — Fact Grounding Rate (FGR)：事实依据率
# 扩展自 10 → 16 个 claim 类型（+量价背离、高低位、动量相关）
# ================================================================

CLAIM_PATTERNS = {
    # --- 原有 10 类 ---
    "ma_bullish": {
        "patterns": [r"多头排列", r"均线向上发散", r"短期均线在.{0,4}上方", r"均线多头"],
        "fact_key": "ma_alignment",
        "valid_labels": ["bullish"],
        "desc": "均线多头排列",
    },
    "ma_bearish": {
        "patterns": [r"空头排列", r"均线向下", r"短期均线在.{0,4}下方", r"均线空头"],
        "fact_key": "ma_alignment",
        "valid_labels": ["bearish"],
        "desc": "均线空头排列",
    },
    "golden_cross": {
        "patterns": [r"金叉", r"金交叉", r"短期均线上穿", r"MA5.*上穿.*MA10", r"5日.*上穿.*10日"],
        "fact_key": "ma_cross",
        "valid_labels": ["golden_cross"],
        "desc": "均线金叉",
    },
    "death_cross": {
        "patterns": [r"死叉", r"死交叉", r"短期均线下穿", r"MA5.*下穿.*MA10", r"5日.*下穿.*10日"],
        "fact_key": "ma_cross",
        "valid_labels": ["death_cross"],
        "desc": "均线死叉",
    },
    "heavy_volume": {
        "patterns": [r"放量", r"成交量放大", r"量能放大", r"成交量显著增加", r"量能增加", r"量能充足"],
        "fact_key": "volume_change",
        "valid_labels": ["heavy_volume"],
        "desc": "放量",
    },
    "light_volume": {
        "patterns": [r"缩量", r"成交量萎缩", r"量能不足", r"成交量减少", r"量能萎缩", r"成交清淡"],
        "fact_key": "volume_change",
        "valid_labels": ["light_volume"],
        "desc": "缩量",
    },
    "breakout_high": {
        "patterns": [r"突破.{0,4}高点", r"突破.{0,4}阻力", r"向上突破", r"创.{0,3}新高", r"突破前高"],
        "fact_key": "price_breakout",
        "valid_labels": ["breakout_high"],
        "desc": "价格突破高点",
    },
    "breakout_low": {
        "patterns": [r"跌破.{0,4}低点", r"跌破.{0,4}支撑", r"向下突破", r"创.{0,3}新低", r"跌破前低"],
        "fact_key": "price_breakout",
        "valid_labels": ["breakout_low"],
        "desc": "价格跌破低点",
    },
    "uptrend": {
        "patterns": [r"上升趋势", r"上涨趋势", r"趋势向上", r"走势偏强", r"处于上行通道", r"上行趋势"],
        "fact_key": "trend_direction",
        "valid_labels": ["uptrend"],
        "desc": "上涨趋势",
    },
    "downtrend": {
        "patterns": [r"下降趋势", r"下跌趋势", r"趋势向下", r"走势偏弱", r"处于下行通道", r"下行趋势"],
        "fact_key": "trend_direction",
        "valid_labels": ["downtrend"],
        "desc": "下跌趋势",
    },
    # --- 新增 6 类（ChartHal 扩展维度）---
    "volume_price_sync_up": {
        "patterns": [r"量价齐升", r"放量上涨", r"成交量配合.{0,4}上涨", r"量价配合良好.*上"],
        "fact_key": "volume_price_divergence",
        "valid_labels": ["volume_price_sync_up"],
        "desc": "量价齐升",
    },
    "volume_price_sync_down": {
        "patterns": [r"量价齐跌", r"放量下跌", r"成交量配合.{0,4}下跌", r"量价配合.*下"],
        "fact_key": "volume_price_divergence",
        "valid_labels": ["volume_price_sync_down"],
        "desc": "量价齐跌",
    },
    "bullish_divergence": {
        "patterns": [r"价涨量缩", r"量价背离.*看涨", r"缩量上涨"],
        "fact_key": "volume_price_divergence",
        "valid_labels": ["bullish_divergence"],
        "desc": "看涨量价背离",
    },
    "high_zone": {
        "patterns": [r"高位.{0,4}区间", r"处于高位", r"价格偏高", r"近期高位", r"高位震荡", r"在高位"],
        "fact_key": "price_position",
        "valid_labels": ["high_zone"],
        "desc": "价格高位区间",
    },
    "low_zone": {
        "patterns": [r"低位.{0,4}区间", r"处于低位", r"价格偏低", r"近期低位", r"低位整理", r"在低位"],
        "fact_key": "price_position",
        "valid_labels": ["low_zone"],
        "desc": "价格低位区间",
    },
    "accelerating_up": {
        "patterns": [r"加速上涨", r"动量增强", r"上涨加速", r"涨势加快", r"快速拉升"],
        "fact_key": "short_momentum",
        "valid_labels": ["accelerating_up"],
        "desc": "动量加速上行",
    },
}


def detect_type1_fgr(response_text, facts):
    """
    Type 1 — Fact Grounding Rate (FGR)：事实依据率
    检测模型技术断言中缺乏 OHLCV 事实支撑的比例。
    原名 detect_type1_ucc (Unsupported Chart Claim)，保留旧名为别名。
    改进：加入否定句过滤（HALoGEN 极性判断思路），避免 "未出现金叉" 被误计
    """
    claims_found = []

    for claim_id, cfg in CLAIM_PATTERNS.items():
        fact_key = cfg["fact_key"]
        if fact_key not in facts:
            continue
        for pattern in cfg["patterns"]:
            m = re.search(pattern, response_text)
            if m:
                # 否定句过滤：若匹配位置前有否定词，跳过此 claim
                if is_negated(response_text, m.start()):
                    break
                fact = facts[fact_key]
                actual_label = fact["label"]
                is_supported = actual_label in cfg["valid_labels"]
                claims_found.append({
                    "claim_id": claim_id,
                    "claim_desc": cfg["desc"],
                    "matched_pattern": pattern,
                    "fact_label": actual_label,
                    "supported": is_supported,
                    "detail": fact,
                })
                break

    total_claims = len(claims_found)
    unsupported = [c for c in claims_found if not c["supported"]]
    ucr = len(unsupported) / total_claims if total_claims > 0 else 0.0

    return {
        "type": "Type1_FGR",
        "total_claims": total_claims,
        "unsupported_count": len(unsupported),
        "fgr": round(ucr, 3),
        "ucr": round(ucr, 3),          # 向后兼容旧字段名
        "claims": claims_found,
        "unsupported_claims": unsupported,
    }


# ================================================================
# Type 2 — Reasoning-Action Consistency Index (RCI)：推理-动作一致性指数
# 依据 HalluLens(ACL2025) intrinsic hallucination 分级思路
# ================================================================

BULLISH_KEYWORDS = [
    r"看多", r"看涨", r"偏强", r"上涨", r"向好", r"积极", r"乐观",
    r"买入信号", r"多头", r"突破", r"金叉", r"放量上攻", r"企稳回升",
    r"强势", r"反弹", r"上行", r"利好", r"超跌反弹", r"筑底",
]

BEARISH_KEYWORDS = [
    r"看空", r"看跌", r"偏弱", r"下跌", r"风险", r"谨慎", r"悲观",
    r"卖出信号", r"空头", r"跌破", r"死叉", r"放量下跌", r"破位",
    r"弱势", r"回调", r"下行", r"利空", r"承压", r"高位风险",
]

NEUTRAL_KEYWORDS = [
    r"震荡", r"横盘", r"观望", r"不确定", r"方向不明", r"等待",
    r"区间", r"盘整", r"犹豫", r"谨慎操作",
]

# 否定修饰词：这些词出现在关键词前，应将该关键词极性翻转
NEGATION_MODIFIER = re.compile(r"(未|没有|不|并非|并不|尚未|无明显).{0,6}$")


def extract_sentiment_with_negation(text):
    """
    情感分析（含否定修正）。
    遍历关键词时，检查其前缀是否有否定词；若有则不计入对应方向。
    """
    bullish_count = 0
    bearish_count = 0
    neutral_count = 0

    for p in BULLISH_KEYWORDS:
        for m in re.finditer(p, text):
            context_before = text[max(0, m.start() - 8): m.start()]
            if NEGATION_MODIFIER.search(context_before):
                bearish_count += 0.5  # 否定看多 → 轻微看空
            else:
                bullish_count += 1

    for p in BEARISH_KEYWORDS:
        for m in re.finditer(p, text):
            context_before = text[max(0, m.start() - 8): m.start()]
            if NEGATION_MODIFIER.search(context_before):
                bullish_count += 0.5  # 否定看空 → 轻微看多
            else:
                bearish_count += 1

    for p in NEUTRAL_KEYWORDS:
        if re.search(p, text):
            neutral_count += 1

    if bullish_count == 0 and bearish_count == 0 and neutral_count == 0:
        return "unknown", {"bullish": 0, "bearish": 0, "neutral": 0}

    if bullish_count > bearish_count and bullish_count > neutral_count:
        sentiment = "bullish"
    elif bearish_count > bullish_count and bearish_count > neutral_count:
        sentiment = "bearish"
    else:
        sentiment = "neutral"

    return sentiment, {
        "bullish": round(bullish_count, 1),
        "bearish": round(bearish_count, 1),
        "neutral": neutral_count,
    }


def detect_type2_rci(response_text, action):
    """
    Type 2 — Reasoning-Action Consistency Index (RCI)：推理-动作一致性指数
    检测推理文本情感方向与最终交易动作之间的矛盾（含否定修正 + 强度分级）。
    原名 detect_type2_rai，保留旧名为别名。
    """
    rationale_section = response_text
    for marker in ["【交易建议】", "【推理理由】"]:
        if marker in response_text:
            parts = response_text.split(marker)
            rationale_section = parts[0] if marker == "【交易建议】" else parts[1].split("【")[0]

    sentiment, scores = extract_sentiment_with_negation(rationale_section)

    inconsistent = False
    inconsistency_level = "none"
    explanation = ""

    gap = abs(scores["bullish"] - scores["bearish"])

    if sentiment == "bearish" and action == "BUY":
        inconsistent = True
        inconsistency_level = "strong" if gap >= 3 else "mild"
        explanation = f"推理倾向看空(bearish={scores['bearish']:.1f}), 但动作为BUY [强度:{inconsistency_level}]"
    elif sentiment == "bullish" and action == "SELL":
        inconsistent = True
        inconsistency_level = "strong" if gap >= 3 else "mild"
        explanation = f"推理倾向看多(bullish={scores['bullish']:.1f}), 但动作为SELL [强度:{inconsistency_level}]"

    return {
        "type": "Type2_RCI",
        "sentiment": sentiment,
        "sentiment_scores": scores,
        "action": action,
        "inconsistent": inconsistent,
        "inconsistency_level": inconsistency_level,
        "explanation": explanation if inconsistent else "推理方向与动作一致",
    }


# ================================================================
# Type 3: 信息不足时过度断言检测
# ================================================================

def compute_evidence_strength(facts):
    """
    基于明确技术信号的数量计算证据强度
    不直接使用"横盘"、"低波动"等市场状态标签，避免循环定义

    明确信号包括：
    1. 均线交叉（金叉/死叉）
    2. 价格突破（突破高位/低位）
    3. 成交量异常（放量/缩量）
    4. 强趋势（斜率绝对值 > 0.3%）
    5. 量价背离
    6. 价格极值位置（高位/低位）
    7. 动量加速
    """
    strong_signals = []

    # 1. 均线信号
    ma_cross = facts.get("ma_cross", {})
    if ma_cross.get("label") in ["golden_cross", "death_cross"]:
        strong_signals.append("均线交叉")

    # 2. 突破信号
    price_breakout = facts.get("price_breakout", {})
    if price_breakout.get("label") in ["breakout_high", "breakout_low"]:
        strong_signals.append("价格突破")

    # 3. 放量/缩量信号
    volume_change = facts.get("volume_change", {})
    if volume_change.get("label") in ["heavy_volume", "light_volume"]:
        strong_signals.append("成交量异常")

    # 4. 强趋势信号（斜率绝对值 > 0.3%/天）
    # fact_labels.label_trend_direction 返回 slope_pct 字段（非 slope）
    trend = facts.get("trend_direction", {})
    trend_slope = trend.get("slope_pct") or 0
    if abs(trend_slope) > 0.3:  # 日均变化>0.3%
        strong_signals.append("明确趋势")

    # 5. 量价背离信号
    volume_price_div = facts.get("volume_price_divergence", {})
    if volume_price_div.get("label") in ["bullish_divergence", "bearish_divergence"]:
        strong_signals.append("量价背离")

    # 6. 价格位置信号（高位/低位）
    price_position = facts.get("price_position", {})
    if price_position.get("label") in ["high_zone", "low_zone"]:
        strong_signals.append("价格极值位置")

    # 7. 动量加速信号
    momentum = facts.get("short_momentum", {})
    momentum_label = momentum.get("label", "")
    if "accelerating" in momentum_label:
        strong_signals.append("动量加速")

    signal_count = len(strong_signals)

    # 根据信号数量判断证据强度
    if signal_count >= 3:
        strength = "strong"
    elif signal_count >= 1:
        strength = "medium"
    else:
        strength = "weak"

    return {
        "signal_count": signal_count,
        "signals": strong_signals,
        "strength": strength
    }


def detect_type3_eci(response_text, action, confidence, facts):
    """
    Type 3 — Evidence Calibration Index (ECI)：证据校准指数
    检测在技术信号不足时模型仍以高置信度给出方向性建议的过度断言行为。
    原名 detect_type3_ieo (Insufficient-Evidence Overclaim)，保留旧名为别名。

    关键设计：
    - 不直接使用"横盘"、"低波动"等市场状态标签作为 low_evidence 的判断依据
    - 基于明确技术信号的数量来定义证据强度
    - 市场状态（横盘/上涨/下跌）作为条件变量用于后续分析，不是定义的一部分
    """
    evidence = compute_evidence_strength(facts)

    is_low_evidence = evidence["strength"] == "weak"  # 信号数 < 1
    is_directional = action in ("BUY", "SELL")
    is_high_confidence = confidence is not None and confidence >= 4

    overclaim = is_low_evidence and is_directional and is_high_confidence

    return {
        "type": "Type3_ECI",
        "evidence_strength": evidence["strength"],
        "signal_count": evidence["signal_count"],
        "signals": evidence["signals"],
        "is_directional": is_directional,
        "is_high_confidence": is_high_confidence,
        "action": action,
        "confidence": confidence,
        "overclaim": overclaim,
        "explanation": (
            f"证据不足(仅{evidence['signal_count']}个明确信号)，"
            f"但模型仍以置信度{confidence}给出{action}建议"
            if overclaim
            else f"证据充分({evidence['signal_count']}个信号: {', '.join(evidence['signals'])})"
        ),
    }


# ================================================================
# 综合检测
# ================================================================

def detect_all_hallucinations(response_text, action, confidence, df_window):
    """运行全部三类幻觉检测（v3.1，方案A命名）"""
    facts = compute_all_facts(df_window)

    type1 = detect_type1_fgr(response_text, facts)
    type2 = detect_type2_rci(response_text, action)
    type3 = detect_type3_eci(response_text, action, confidence, facts)

    has_hallucination = (
        type1["unsupported_count"] > 0
        or type2["inconsistent"]
        or type3["overclaim"]
    )

    return {
        "has_hallucination": has_hallucination,
        "language": detect_response_language(response_text),
        "facts": facts,
        "facts_readable": facts_to_readable(facts),
        # 新字段名（方案A）
        "type1_fgr": type1,
        "type2_rci": type2,
        "type3_eci": type3,
        # 向后兼容旧字段名（分析脚本不需要改）
        "type1_ucc": type1,
        "type2_rai": type2,
        "type3_ieo": type3,
    }


def format_detection_report(result):
    """格式化检测报告"""
    lines = ["=" * 60, "幻觉检测报告（v3.1）", "=" * 60]
    lines.append(f"\n综合结论: {'⚠️ 检测到幻觉' if result['has_hallucination'] else '✅ 未检测到幻觉'}")
    lines.append(f"\n--- 事实标签 (Ground Truth) ---\n{result['facts_readable']}")

    t1 = result["type1_fgr"]
    lines.append(f"\n--- Type 1 FGR: 事实依据率 (FGR={t1['fgr']}) ---")
    lines.append(f"共提取 {t1['total_claims']} 个技术断言, {t1['unsupported_count']} 个无依据")
    for c in t1["claims"]:
        status = "❌ 不支持" if not c["supported"] else "✅ 支持"
        lines.append(f"  [{status}] {c['claim_desc']} → 实际: {c['fact_label']}")

    t2 = result["type2_rci"]
    status = "❌ 不一致" if t2["inconsistent"] else "✅ 一致"
    lines.append(f"\n--- Type 2 RCI: 推理-动作一致性 [{status}] ---")
    lines.append(f"推理情感: {t2['sentiment']} {t2['sentiment_scores']} 强度:{t2['inconsistency_level']}")
    lines.append(f"动作: {t2['action']}")
    lines.append(f"说明: {t2['explanation']}")

    t3 = result["type3_eci"]
    status = "❌ 过度断言" if t3["overclaim"] else "✅ 正常"
    lines.append(f"\n--- Type 3 ECI: 证据校准指数 [{status}] ---")
    lines.append(f"信号数: {t3['signal_count']}  信号: {t3.get('signals', [])}")
    lines.append(f"动作: {t3['action']}, 置信度: {t3['confidence']}")
    lines.append(f"说明: {t3['explanation']}")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


# ================================================================
# 向后兼容别名（旧函数名 → 新函数名）
# ================================================================
detect_type1_ucc = detect_type1_fgr
detect_type2_rai = detect_type2_rci
detect_type3_ieo = detect_type3_eci
