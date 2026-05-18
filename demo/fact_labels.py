"""
事实标签自动计算系统：基于 OHLCV 数据生成可验证的金融事实标签
"""
import numpy as np
import pandas as pd


def compute_ma(df, periods=(5, 10, 20)):
    """计算移动均线"""
    result = {}
    for p in periods:
        col = f"MA{p}"
        result[col] = df["Close"].rolling(p).mean()
    return pd.DataFrame(result, index=df.index)


def label_ma_alignment(df):
    """
    均线排列判定
    返回: bullish(多头) / bearish(空头) / mixed(交叉)
    """
    ma = compute_ma(df)
    last = ma.iloc[-1]
    ma5, ma10, ma20 = last["MA5"], last["MA10"], last["MA20"]

    if pd.isna(ma5) or pd.isna(ma10) or pd.isna(ma20):
        return {"label": "insufficient_data", "ma5": None, "ma10": None, "ma20": None}

    if ma5 > ma10 > ma20:
        alignment = "bullish"
    elif ma5 < ma10 < ma20:
        alignment = "bearish"
    else:
        alignment = "mixed"

    return {
        "label": alignment,
        "ma5": round(ma5, 2),
        "ma10": round(ma10, 2),
        "ma20": round(ma20, 2),
    }


def label_ma_cross(df):
    """
    金叉/死叉判定（MA5 与 MA10 在最近5日内是否发生交叉）
    """
    ma = compute_ma(df)
    if len(ma) < 6 or ma["MA5"].isna().any() or ma["MA10"].isna().any():
        return {"label": "no_cross", "detail": "insufficient_data"}

    recent = ma.iloc[-5:]
    diff = recent["MA5"] - recent["MA10"]

    for i in range(1, len(diff)):
        if diff.iloc[i - 1] < 0 and diff.iloc[i] > 0:
            return {"label": "golden_cross", "detail": "MA5上穿MA10"}
        if diff.iloc[i - 1] > 0 and diff.iloc[i] < 0:
            return {"label": "death_cross", "detail": "MA5下穿MA10"}

    return {"label": "no_cross", "detail": "近5日无交叉"}


def label_volume_change(df, lookback=20):
    """
    放量/缩量判定
    当日成交量 vs 过去 N 日均量
    """
    if len(df) < lookback + 1:
        return {"label": "insufficient_data", "ratio": None}

    recent_vol = df["Volume"].iloc[-1]
    avg_vol = df["Volume"].iloc[-(lookback + 1):-1].mean()

    if avg_vol == 0:
        return {"label": "no_volume", "ratio": None}

    ratio = recent_vol / avg_vol

    if ratio > 1.5:
        label = "heavy_volume"
    elif ratio < 0.7:
        label = "light_volume"
    else:
        label = "normal_volume"

    return {"label": label, "ratio": round(ratio, 2), "last_vol": recent_vol, "avg_vol": round(avg_vol, 0)}


def label_price_breakout(df, lookback=20):
    """
    价格突破/跌破判定
    收盘价 vs 近 N 日高低点
    """
    if len(df) < lookback + 1:
        return {"label": "insufficient_data"}

    current_close = df["Close"].iloc[-1]
    prev = df.iloc[-(lookback + 1):-1]
    high_n = prev["High"].max()
    low_n = prev["Low"].min()

    if current_close > high_n:
        label = "breakout_high"
    elif current_close < low_n:
        label = "breakout_low"
    else:
        pos = (current_close - low_n) / (high_n - low_n) if high_n != low_n else 0.5
        label = "within_range"
        return {"label": label, "position": round(pos, 2), "high_n": round(high_n, 2), "low_n": round(low_n, 2)}

    return {"label": label, "close": round(current_close, 2), "high_n": round(high_n, 2), "low_n": round(low_n, 2)}


def label_trend_direction(df, lookback=20):
    """
    趋势方向判定：线性回归斜率
    """
    if len(df) < lookback:
        return {"label": "insufficient_data", "slope": None}

    closes = df["Close"].iloc[-lookback:].values
    x = np.arange(len(closes))
    slope, intercept = np.polyfit(x, closes, 1)

    pct_slope = slope / closes.mean() * 100

    if pct_slope > 0.15:
        label = "uptrend"
    elif pct_slope < -0.15:
        label = "downtrend"
    else:
        label = "sideways"

    return {"label": label, "slope_pct": round(pct_slope, 3)}


def label_volatility(df, lookback=20):
    """
    波动率判定
    """
    if len(df) < lookback:
        return {"label": "insufficient_data", "volatility": None}

    returns = df["Close"].iloc[-lookback:].pct_change().dropna()
    vol = returns.std() * np.sqrt(252)

    if vol > 0.4:
        label = "high_volatility"
    elif vol < 0.15:
        label = "low_volatility"
    else:
        label = "medium_volatility"

    return {"label": label, "annualized_vol": round(vol, 4)}


def label_recent_change(df, days=5):
    """
    近 N 日涨跌幅
    """
    if len(df) < days + 1:
        return {"label": "insufficient_data", "change_pct": None}

    old_close = df["Close"].iloc[-(days + 1)]
    new_close = df["Close"].iloc[-1]
    change_pct = (new_close - old_close) / old_close * 100

    if change_pct > 3:
        label = "significant_rise"
    elif change_pct < -3:
        label = "significant_drop"
    else:
        label = "flat"

    return {"label": label, "change_pct": round(change_pct, 2)}


def label_volume_price_divergence(df, lookback=10):
    """
    量价背离检测（Volume-Price Divergence）。
    参考 ChartHal(2025) 对 chart-absent information 的定义：
    价涨量缩 = 看涨背离信号，价跌量缩 = 看跌背离信号（情绪衰减）。
    """
    if len(df) < lookback + 1:
        return {"label": "insufficient_data", "price_change": None, "vol_change": None}
    recent_price_change = (df["Close"].iloc[-1] - df["Close"].iloc[-lookback]) / df["Close"].iloc[-lookback]
    recent_vol_change = (df["Volume"].iloc[-1] - df["Volume"].iloc[-lookback:].mean()) / (df["Volume"].iloc[-lookback:].mean() + 1e-9)
    price_up = recent_price_change > 0.01
    price_down = recent_price_change < -0.01
    vol_shrink = recent_vol_change < -0.2
    vol_expand = recent_vol_change > 0.2
    if price_up and vol_shrink:
        label = "bullish_divergence"
    elif price_down and vol_shrink:
        label = "bearish_divergence"
    elif price_up and vol_expand:
        label = "volume_price_sync_up"
    elif price_down and vol_expand:
        label = "volume_price_sync_down"
    else:
        label = "no_divergence"
    return {
        "label": label,
        "price_change_pct": round(recent_price_change * 100, 2),
        "vol_change_pct": round(recent_vol_change * 100, 2),
    }


def label_price_position(df, lookback=20):
    """
    价格区间位置（Relative Price Position）。
    当前收盘价在近 N 日高低区间的百分位，判断高位/中位/低位。
    """
    if len(df) < lookback:
        return {"label": "insufficient_data", "percentile": None}
    window = df.iloc[-lookback:]
    high = window["High"].max()
    low = window["Low"].min()
    current = df["Close"].iloc[-1]
    if high == low:
        percentile = 0.5
    else:
        percentile = (current - low) / (high - low)
    if percentile >= 0.8:
        label = "high_zone"
    elif percentile <= 0.2:
        label = "low_zone"
    else:
        label = "mid_zone"
    return {"label": label, "percentile": round(percentile, 3), "high": round(high, 2), "low": round(low, 2)}


def label_short_momentum(df, days=3):
    """
    短期动量检测（Short-term Momentum）。
    近 3 日价格变化速度（加速上涨/加速下跌/减速/平稳）。
    参考 HALoGEN(ACL2025) 原子事实分解思路：将动量断言分解为可验证单元。
    """
    if len(df) < days + 2:
        return {"label": "insufficient_data", "momentum": None}
    closes = df["Close"].iloc[-(days + 1):].values
    daily_changes = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    if len(daily_changes) < 2:
        return {"label": "insufficient_data", "momentum": None}
    avg_change = sum(daily_changes) / len(daily_changes)
    recent_change = daily_changes[-1]
    pct = avg_change / (closes[0] + 1e-9) * 100
    if abs(avg_change) < closes[0] * 0.003:
        label = "flat_momentum"
    elif avg_change > 0 and recent_change > avg_change:
        label = "accelerating_up"
    elif avg_change > 0:
        label = "decelerating_up"
    elif avg_change < 0 and recent_change < avg_change:
        label = "accelerating_down"
    else:
        label = "decelerating_down"
    return {"label": label, "avg_daily_change_pct": round(pct, 3)}


def empty_facts_unknown(reason="no_ohlcv"):
    """
    无有效 OHLCV 窗口时的占位事实（与 verify / Type1–3 检测兼容）。
    label 统一为 unknown，避免 compute_all_facts(None) 触发下游 NoneType。
    """
    return {
        "ma_alignment": {"label": "unknown", "ma5": None, "ma10": None, "ma20": None},
        "ma_cross": {"label": "unknown", "detail": reason},
        "volume_change": {"label": "unknown", "ratio": None},
        "price_breakout": {"label": "unknown"},
        "trend_direction": {"label": "unknown", "slope": 0.0, "slope_pct": None},
        "volatility": {"label": "unknown", "annualized_vol": None},
        "recent_5d_change": {"label": "unknown", "change_pct": None},
        "volume_price_divergence": {
            "label": "unknown",
            "price_change_pct": None,
            "vol_change_pct": None,
        },
        "price_position": {"label": "unknown", "percentile": None},
        "short_momentum": {"label": "unknown", "avg_daily_change_pct": None},
    }


def compute_all_facts(df_window):
    """计算全部事实标签"""
    if df_window is None or getattr(df_window, "empty", True):
        return empty_facts_unknown()
    _need = {"Open", "High", "Low", "Close", "Volume"}
    if not _need.issubset(set(df_window.columns)):
        return empty_facts_unknown("bad_columns")
    try:
        return {
            "ma_alignment": label_ma_alignment(df_window),
            "ma_cross": label_ma_cross(df_window),
            "volume_change": label_volume_change(df_window),
            "price_breakout": label_price_breakout(df_window),
            "trend_direction": label_trend_direction(df_window),
            "volatility": label_volatility(df_window),
            "recent_5d_change": label_recent_change(df_window, days=5),
            "volume_price_divergence": label_volume_price_divergence(df_window),
            "price_position": label_price_position(df_window),
            "short_momentum": label_short_momentum(df_window),
        }
    except Exception:
        return empty_facts_unknown("compute_error")


def facts_to_readable(facts):
    """将事实标签转为可读的中文描述"""
    lines = []

    ma = facts["ma_alignment"]
    ma_map = {"bullish": "多头排列(MA5>MA10>MA20)", "bearish": "空头排列(MA5<MA10<MA20)", "mixed": "交叉排列"}
    lines.append(f"均线排列: {ma_map.get(ma['label'], ma['label'])} "
                 f"(MA5={ma.get('ma5')}, MA10={ma.get('ma10')}, MA20={ma.get('ma20')})")

    cross = facts["ma_cross"]
    cross_map = {"golden_cross": "金叉", "death_cross": "死叉", "no_cross": "无交叉"}
    lines.append(f"均线交叉: {cross_map.get(cross['label'], cross['label'])} ({cross.get('detail', '')})")

    vol = facts["volume_change"]
    vol_map = {"heavy_volume": "放量", "light_volume": "缩量", "normal_volume": "正常"}
    lines.append(f"成交量: {vol_map.get(vol['label'], vol['label'])} "
                 f"(当日/20日均量={vol.get('ratio', 'N/A')}倍)")

    bp = facts["price_breakout"]
    bp_map = {"breakout_high": "突破近20日高点", "breakout_low": "跌破近20日低点", "within_range": "区间内运行"}
    detail = f"(位置={bp.get('position', 'N/A')})" if bp["label"] == "within_range" else ""
    lines.append(f"价格位置: {bp_map.get(bp['label'], bp['label'])} {detail}")

    trend = facts["trend_direction"]
    trend_map = {"uptrend": "上涨趋势", "downtrend": "下跌趋势", "sideways": "横盘震荡"}
    lines.append(f"趋势方向: {trend_map.get(trend['label'], trend['label'])} "
                 f"(斜率={trend.get('slope_pct', 'N/A')}%/天)")

    volatility = facts["volatility"]
    vol_level_map = {"high_volatility": "高波动", "medium_volatility": "中等波动", "low_volatility": "低波动"}
    lines.append(f"波动率: {vol_level_map.get(volatility['label'], volatility['label'])} "
                 f"(年化={volatility.get('annualized_vol', 'N/A')})")

    rc = facts["recent_5d_change"]
    lines.append(f"近5日涨跌: {rc.get('change_pct', 'N/A')}% ({rc['label']})")

    vpd = facts["volume_price_divergence"]
    vpd_map = {
        "bullish_divergence": "价涨量缩(看涨背离)",
        "bearish_divergence": "价跌量缩(看跌背离)",
        "volume_price_sync_up": "量价齐升",
        "volume_price_sync_down": "量价齐跌",
        "no_divergence": "无背离",
    }
    lines.append(f"量价关系: {vpd_map.get(vpd['label'], vpd['label'])} "
                 f"(价格变动={vpd.get('price_change_pct', 'N/A')}%, 量变={vpd.get('vol_change_pct', 'N/A')}%)")

    pp = facts["price_position"]
    pp_map = {"high_zone": "高位区间(≥80%)", "mid_zone": "中位区间(20-80%)", "low_zone": "低位区间(≤20%)"}
    lines.append(f"价格高低位: {pp_map.get(pp['label'], pp['label'])} "
                 f"(分位={pp.get('percentile', 'N/A')})")

    sm = facts["short_momentum"]
    sm_map = {
        "accelerating_up": "动量加速上行",
        "decelerating_up": "动量减速上行",
        "flat_momentum": "动量平稳",
        "decelerating_down": "动量减速下行",
        "accelerating_down": "动量加速下行",
    }
    lines.append(f"短期动量: {sm_map.get(sm['label'], sm['label'])} "
                 f"(均日变动={sm.get('avg_daily_change_pct', 'N/A')}%)")

    return "\n".join(lines)


if __name__ == "__main__":
    from data_pipeline import create_sample

    print("=== 测试事实标签系统 ===\n")
    sample = create_sample("600519.SH", "20230630")
    if sample:
        facts = compute_all_facts(sample["df_window"])
        print(f"股票: {sample['ts_code']}  截至: {sample['end_date']}\n")
        print(facts_to_readable(facts))
        print("\n--- 原始标签 ---")
        for k, v in facts.items():
            print(f"  {k}: {v}")
