"""
数据管道：Tushare 拉取沪深300数据 + mplfinance 生成 K 线图
"""
from __future__ import annotations

import os
import pathlib
import tushare as ts
import pandas as pd
import mplfinance as mpf
from config import TUSHARE_TOKEN, HS300_INDEX, DATA_YEAR, WINDOW_SIZE, CHART_DIR

pro = ts.pro_api(TUSHARE_TOKEN)

# ================================================================
# 本地 parquet 缓存（避免每次实时拉 tushare，防止限流导致 df_window_unavailable）
# 缓存目录：demo/data_cache/  文件名：{ts_code}_{start}_{end}.parquet
# ================================================================
_CACHE_DIR = pathlib.Path(__file__).parent / "data_cache"
_CACHE_DIR.mkdir(exist_ok=True)


def _cache_path(ts_code: str, start_date: str, end_date: str) -> pathlib.Path:
    key = f"{ts_code}_{start_date}_{end_date}.parquet"
    return _CACHE_DIR / key


def get_hs300_stocks(date="20231229"):
    """获取沪深300成分股列表"""
    df = pro.index_weight(index_code=HS300_INDEX, end_date=date)
    df = df.drop_duplicates(subset="con_code")
    return df["con_code"].tolist()


def _normalize_ohlcv_df(df: pd.DataFrame) -> pd.DataFrame | None:
    if df is None or df.empty or "trade_date" not in df.columns:
        return None
    df = df.sort_values("trade_date").reset_index(drop=True)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.set_index("trade_date")
    df = df.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "vol": "Volume",
    })
    return df


def get_daily_data(ts_code, start_date, end_date):
    """获取单只股票日线行情（优先前复权 pro_bar；失败时用 pro.daily 兜底）。
    结果自动缓存到 data_cache/ 目录，避免重复拉取触发 tushare 限流。
    """
    cache = _cache_path(ts_code, start_date, end_date)
    if cache.exists():
        try:
            return pd.read_parquet(cache)
        except Exception:
            cache.unlink(missing_ok=True)

    result = None
    try:
        df = ts.pro_bar(
            ts_code=ts_code,
            adj="qfq",
            start_date=start_date,
            end_date=end_date,
            api=pro,
        )
        result = _normalize_ohlcv_df(df)
    except Exception:
        pass

    if result is None:
        try:
            df = pro.daily(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
            )
            result = _normalize_ohlcv_df(df)
        except Exception:
            return None

    if result is not None:
        try:
            result.to_parquet(cache)
        except Exception:
            pass
    return result


def generate_chart(df_window, ts_code, end_date, save_dir=CHART_DIR):
    """生成标准 K 线图（含 MA5/10/20 + 成交量）"""
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{ts_code}_{end_date}.png"
    filepath = os.path.join(save_dir, filename)

    mc = mpf.make_marketcolors(up="r", down="g", inherit=True)
    s = mpf.make_mpf_style(marketcolors=mc, gridstyle="-", gridcolor="#e6e6e6")

    mpf.plot(
        df_window[["Open", "High", "Low", "Close", "Volume"]],
        type="candle",
        style=s,
        volume=True,
        mav=(5, 10, 20),
        title=f"{ts_code} (ending {end_date})",
        figsize=(12, 7),
        savefig=filepath,
    )
    return filepath


def build_ohlcv_text(df_window, ts_code):
    """将 OHLCV 数据转为文本格式，供 Text-only 模式使用"""
    lines = [f"股票代码: {ts_code}", f"数据窗口: 最近 {len(df_window)} 个交易日", ""]
    lines.append("日期 | 开盘 | 最高 | 最低 | 收盘 | 成交量(手)")
    lines.append("-" * 60)
    for date, row in df_window.iterrows():
        d = date.strftime("%Y-%m-%d")
        lines.append(
            f"{d} | {row['Open']:.2f} | {row['High']:.2f} | "
            f"{row['Low']:.2f} | {row['Close']:.2f} | {row['Volume']:.0f}"
        )
    return "\n".join(lines)


def create_sample(
    ts_code,
    end_date_str,
    start_date="20220101",
    pro_end_date="20251231",
    chart_save_dir=None,
):
    """
    创建一个完整样本：
    返回 {ts_code, end_date, df_window, chart_path, ohlcv_text}

    Args:
        ts_code: 股票代码
        end_date_str: 目标截止日 (YYYYMMDD)
        start_date: Tushare 拉取行情的起始日（需足够早以覆盖长窗口）
        pro_end_date: Tushare 拉取行情的截止日，必须 >= end_date_str
        chart_save_dir: 若指定，K 线图保存到该目录（否则用默认 CHART_DIR）
    """
    df = get_daily_data(ts_code, start_date, pro_end_date)
    if df is None:
        return None

    target_date = pd.Timestamp(end_date_str)
    mask = df.index <= target_date
    df_up_to = df[mask]
    if len(df_up_to) < WINDOW_SIZE:
        return None

    df_window = df_up_to.iloc[-WINDOW_SIZE:]
    actual_end = df_window.index[-1].strftime("%Y%m%d")
    # 截面批处理时文件名用请求的 end_date_str，便于与 run_vta --end-date 对齐
    chart_label = end_date_str if chart_save_dir is not None else actual_end

    if chart_save_dir is not None:
        chart_path = generate_chart(
            df_window, ts_code, chart_label, save_dir=chart_save_dir
        )
    else:
        chart_path = generate_chart(df_window, ts_code, chart_label)
    ohlcv_text = build_ohlcv_text(df_window, ts_code)

    return {
        "ts_code": ts_code,
        "end_date": actual_end,
        "end_date_requested": end_date_str,
        "df_window": df_window,
        "chart_path": chart_path,
        "ohlcv_text": ohlcv_text,
    }


if __name__ == "__main__":
    print("获取沪深300成分股...")
    stocks = get_hs300_stocks()
    print(f"共 {len(stocks)} 只成分股")
    print(f"前5只: {stocks[:5]}")

    print("\n创建示例样本: 贵州茅台 600519.SH, 截至 2023-06-30")
    sample = create_sample("600519.SH", "20230630")
    if sample:
        print(f"K线图已保存: {sample['chart_path']}")
        print(f"OHLCV 文本前5行:")
        print("\n".join(sample["ohlcv_text"].split("\n")[:8]))
    else:
        print("样本创建失败")
