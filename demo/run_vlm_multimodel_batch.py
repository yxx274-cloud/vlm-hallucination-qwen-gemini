"""
多模型 / 多 provider VLM 批量实验（基于 vlm_agent_v2）

与 run_batch.py 输出字段对齐，便于 unified_analysis 汇总。
支持：bigmodel / openrouter / openai，断点续跑，按截面与模态过滤。

用法:
  cd demo
  export ZHIPU_API_KEY=...
  python3 run_vlm_multimodel_batch.py --end-date 20230630 --model-key glm-4v-flash --provider bigmodel --chart-dir charts

  export OPENROUTER_API_KEY=...
  python3 run_vlm_multimodel_batch.py --end-date 20230630 --model-key qwen2.5-vl-72b --provider openrouter \\
      --chart-dir charts --output-root outputs/multimodel --modes chart_only --limit 50
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
DEMO_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))


def inject_api_keys_from_demo_config() -> None:
    """按文件路径加载 demo/config.py，避免与其它路径上的 `config` 包重名冲突。"""
    cfg_file = DEMO_DIR / "config.py"
    if not cfg_file.is_file():
        return
    try:
        spec = importlib.util.spec_from_file_location("_stock_demo_secrets", cfg_file)
        if spec is None or spec.loader is None:
            return
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        nk = getattr(mod, "NVIDIA_API_KEY", None) or ""
        zk = getattr(mod, "LLM_API_KEY", None) or ""
        if nk.strip() and not os.environ.get("NVIDIA_API_KEY"):
            os.environ["NVIDIA_API_KEY"] = nk.strip()
        if zk.strip() and not os.environ.get("ZHIPU_API_KEY"):
            os.environ["ZHIPU_API_KEY"] = zk.strip()
    except Exception:
        pass


inject_api_keys_from_demo_config()

from tqdm import tqdm

from data_pipeline import build_ohlcv_text, create_sample, get_daily_data
from hallucination_detector import detect_all_hallucinations
from vlm_agent_v2 import VLM_PROVIDERS, get_analysis_text, parse_action, query_vlm

import pandas as pd

from stocks_hs300 import STOCKS_HS300

WINDOW_SIZE = 30
START_DATE = "20220101"


def load_window(ts_code: str, end_date: str):
    pro_end = max(end_date, "20251231")
    df = get_daily_data(ts_code, START_DATE, pro_end)
    if df is None or df.empty:
        return None
    target = pd.Timestamp(end_date)
    df_up = df[df.index <= target]
    if len(df_up) < WINDOW_SIZE:
        return None
    return df_up.iloc[-WINDOW_SIZE:]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--end-date", required=True, help="截面日 YYYYMMDD，与图文件名后缀一致")
    p.add_argument("--chart-dir", default="charts", help="含 {ts}_{end_date}.png 的目录")
    p.add_argument(
        "--output-root",
        default="outputs/multimodel",
        help="输出根目录，实际写入 {output-root}/{end-date}/batch/",
    )
    p.add_argument("--model-key", required=True, help="vlm_agent_v2 中的 model_key")
    p.add_argument("--provider", default="bigmodel")
    p.add_argument(
        "--modes",
        default="chart_only,text_only,multimodal",
        help="逗号分隔: chart_only | text_only | multimodal",
    )
    p.add_argument("--limit", type=int, default=0, help="最多股票数，0=全部")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--sleep", type=float, default=1.0)
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--stocks-file", default="", help="可选：每行一个 ts_code，否则用 STOCKS_HS300")
    p.add_argument("--api-base", default=None, help="覆盖 provider base_url（qwen_local 等本地部署必传，例如 http://localhost:8000/v1）")
    p.add_argument("--api-key", default=None, help="覆盖 provider 默认 API key（用于 yunwu 等中转服务）")
    p.add_argument("--seed", type=int, default=20230630, help="随机种子，透传给 API")
    return p.parse_args()


def task_key(ts_code: str, model_key: str, mode: str) -> str:
    return f"{ts_code}_{model_key}_{mode}"


def progress_path(out_batch: Path, model_key: str) -> Path:
    safe = model_key.replace("/", "_").replace(":", "_")
    return out_batch / f"progress_{safe}.json"


def load_progress(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"completed": [], "failed": []}


def save_progress(path: Path, data: dict):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def make_blank_png(path: Path, size=(800, 500)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(255, 255, 255)).save(path)


def _build_sample(
    ts_code: str, end_date: str, chart_dir: str
) -> Optional[Dict[str, Any]]:
    """优先使用已存在的 K 线图；若日线不可得，也用占位文本继续跑。"""
    try:
        p = Path(chart_dir) / f"{ts_code}_{end_date}.png"
        if not p.exists():
            sample = create_sample(
                ts_code,
                end_date,
                start_date=START_DATE,
                pro_end_date=max(end_date, "20251231"),
                chart_save_dir=chart_dir,
            )
            if sample is not None:
                return sample
            make_blank_png(p)
        df_window = load_window(ts_code, end_date)
        if df_window is not None:
            ohlcv = build_ohlcv_text(df_window, ts_code)
            actual_end = df_window.index[-1].strftime("%Y%m%d")
        else:
            actual_end = end_date
            ohlcv = f"Stock: {ts_code}  Date: {end_date}  [OHLCV data temporarily unavailable]"
        return {
            "ts_code": ts_code,
            "end_date": actual_end,
            "df_window": df_window,
            "chart_path": str(p),
            "ohlcv_text": ohlcv,
        }
    except Exception:
        return None


def run_one(
    ts_code: str,
    end_date: str,
    chart_dir: str,
    model_key: str,
    provider: str,
    mode_name: str,
    mode_cfg: dict,
    sample_cache: dict,
    max_retries: int,
    api_base: str = None,
    seed: int = 20230630,
) -> Tuple[Optional[Dict], Optional[str]]:
    try:
        if ts_code not in sample_cache:
            sample = _build_sample(ts_code, end_date, chart_dir)
            if sample is None:
                return None, "create_sample_failed"
            sample_cache[ts_code] = sample
        sample = sample_cache[ts_code]
        df_window = sample["df_window"]
    except Exception as e:
        return None, f"build_sample:{str(e)[:200]}"

    for attempt in range(max_retries):
        try:
            # 严格按模态传参：chart_only 不传 ohlcv_text，text_only 不传 chart_path
            ohlcv_text = sample["ohlcv_text"] if mode_cfg["use_text"] else None
            chart_path = sample["chart_path"] if mode_cfg["use_chart"] else None
            result = query_vlm(
                model_key,
                ohlcv_text=ohlcv_text,
                chart_path=chart_path,
                ts_code=ts_code,
                end_date=sample["end_date"],
                provider=provider,
                api_base=api_base,
                seed=seed,
            )
            analysis_text = get_analysis_text(result["response"])
            parsed = parse_action(result["response"])
            if df_window is None:
                # OHLCV 不可得时不编造检测结果，直接报错让 retry-only 重跑
                return None, "ohlcv_unavailable"
            detection = detect_all_hallucinations(
                analysis_text,
                parsed["action"],
                parsed["confidence"],
                df_window,
            )
            ed = sample["end_date"]
            out = {
                "ts_code": ts_code,
                "end_date": ed,
                "model": model_key,
                "provider": provider,
                "mode": mode_name,
                "parsed_action": parsed,
                "detection_summary": {
                    "has_hallucination": detection["has_hallucination"],
                    "type1_fgr": detection["type1_fgr"]["fgr"],
                    "type1_ucr": detection["type1_fgr"]["ucr"],          # 向后兼容
                    "type1_unsupported": detection["type1_fgr"]["unsupported_count"],
                    "type1_total_claims": detection["type1_fgr"]["total_claims"],
                    "type2_inconsistent": detection["type2_rci"]["inconsistent"],
                    "type2_sentiment": detection["type2_rci"]["sentiment"],
                    "type3_overclaim": detection["type3_eci"]["overclaim"],
                    "response_language": detection.get("language", "unknown"),
                },
                "prompt_version": "v4-2026-05",
                "detector_version": "v3.1",
                "sampling": {"temperature": 0.0, "seed": 20230630},
                "facts_readable": detection["facts_readable"],
                "usage": result["usage"],
                "vlm_response": result["response"],
            }
            return out, None
        except Exception as e:
            err = str(e)[:300]
            is_rl = "429" in err or "rate" in err.lower()
            is_auth = "403" in err or "401" in err
            if is_auth:
                return None, err
            if attempt < max_retries - 1:
                backoff = (2 ** attempt) + random.uniform(0, 1.5)
                if is_rl:
                    backoff *= 2
                time.sleep(backoff)
            else:
                return None, err
    return None, "max_retries"


def main():
    args = parse_args()
    inject_api_keys_from_demo_config()
    if args.provider in VLM_PROVIDERS:
        # qwen_local 本地部署不需要 key；yunwu 等在运行时传 --api-key 也可以
        if args.provider not in ("qwen_local",):
            env_k = VLM_PROVIDERS[args.provider]["env_key"]
            if not os.environ.get(env_k) and not args.api_key:
                raise SystemExit(
                    f"缺少 {env_k}（provider={args.provider}）。"
                    f"请先 export/set 该变量，或在本脚本同目录的 config.py 中配置对应密钥。"
                )
    modes_raw = [m.strip() for m in args.modes.split(",") if m.strip()]
    mode_cfg_map = {
        "text_only": {"use_text": True, "use_chart": False},
        "chart_only": {"use_text": False, "use_chart": True},
        "multimodal": {"use_text": True, "use_chart": True},
    }
    for m in modes_raw:
        if m not in mode_cfg_map:
            raise SystemExit(f"未知模态: {m}")

    if args.stocks_file:
        stock_list = [
            line.strip()
            for line in Path(args.stocks_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        stock_list = STOCKS_HS300
    if args.limit > 0:
        stock_list = stock_list[: args.limit]

    out_batch = Path(args.output_root) / args.end_date / "batch"
    out_batch.mkdir(parents=True, exist_ok=True)

    prog = load_progress(progress_path(out_batch, args.model_key))
    completed = set(prog.get("completed", []))

    def _is_valid_result(path: Path) -> bool:
        """JSON 存在、可解析、且含有效 parsed_action 字段才算真正完成。"""
        if not path.exists():
            return False
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
            pa = d.get("parsed_action") or {}
            return bool(pa.get("action"))
        except Exception:
            return False

    tasks = []
    stale_keys: set = set()
    for ts_code in stock_list:
        for mode_name in modes_raw:
            key = task_key(ts_code, args.model_key, mode_name)
            detail = out_batch / f"{ts_code}_{args.end_date}_{args.model_key}_{mode_name}.json"
            if args.resume and key in completed:
                if _is_valid_result(detail):
                    continue
                # completed 里有记录但文件无效 → 剔除并重跑
                stale_keys.add(key)
            tasks.append((ts_code, mode_name, mode_cfg_map[mode_name]))

    if stale_keys:
        completed -= stale_keys
        prog["completed"] = sorted(completed)
        save_progress(progress_path(out_batch, args.model_key), prog)
        print(f"[resume] 发现 {len(stale_keys)} 条 completed 记录无效，已剔除并加入重跑队列")

    print(
        f"=== 多模型批跑 end_date={args.end_date} model={args.model_key} "
        f"provider={args.provider} ==="
    )
    print(f"待运行: {len(tasks)}  output: {out_batch}")

    sample_cache: dict = {}
    for ts_code, mode_name, mc in tqdm(tasks, desc="vlm"):
        key = task_key(ts_code, args.model_key, mode_name)
        try:
            out, err = run_one(
                ts_code,
                args.end_date,
                args.chart_dir,
                args.model_key,
                args.provider,
                mode_name,
                mc,
                sample_cache,
                args.max_retries,
                api_base=args.api_base,
                seed=args.seed,
            )
        except Exception as exc:
            out, err = None, f"run_one_crash:{str(exc)[:200]}"
        detail = out_batch / f"{ts_code}_{args.end_date}_{args.model_key}_{mode_name}.json"
        if out:
            with open(detail, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2, default=str)
            if key not in completed:
                completed.add(key)
                prog["completed"] = sorted(completed)
                save_progress(progress_path(out_batch, args.model_key), prog)
        else:
            failed_list = prog.setdefault("failed", [])
            failed_list.append({"key": key, "error": err})
            # 去重并截断，避免 progress 文件无限膨胀
            seen_fail: dict = {}
            for item in failed_list:
                seen_fail[item["key"]] = item
            prog["failed"] = list(seen_fail.values())[-500:]
            save_progress(progress_path(out_batch, args.model_key), prog)
        time.sleep(args.sleep)

    print("完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
