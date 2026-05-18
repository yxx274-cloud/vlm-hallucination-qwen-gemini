"""
多模型 chart-only 四组核心实验：
- forced-decision baseline
- Abstain
- VtA v2
- EF-TTC

设计目标：
1. 统一输出字段，便于后续汇总 coverage-risk 指标
2. 支持 bigmodel / openrouter / openai / nvidia
3. 支持断点续跑和 limit 样本冒烟
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

from PIL import Image

import pandas as pd
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
DEMO_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))


def inject_api_keys_from_demo_config() -> None:
    """按路径加载 demo/config.py，避免与其它 `config` 包重名。"""
    cfg_file = DEMO_DIR / "config.py"
    if not cfg_file.is_file():
        return
    try:
        spec = importlib.util.spec_from_file_location("_stock_demo_cfg_multi", cfg_file)
        if spec is None or spec.loader is None:
            return
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        nk = (getattr(mod, "NVIDIA_API_KEY", None) or "").strip()
        zk = (getattr(mod, "LLM_API_KEY", None) or "").strip()
        if nk and not os.environ.get("NVIDIA_API_KEY"):
            os.environ["NVIDIA_API_KEY"] = nk
        if zk and not os.environ.get("ZHIPU_API_KEY"):
            os.environ["ZHIPU_API_KEY"] = zk
    except Exception:
        pass


inject_api_keys_from_demo_config()

from data_pipeline import create_sample, get_daily_data
from hallucination_detector import detect_all_hallucinations
from stocks_hs300 import STOCKS_HS300
from vlm_agent_v2 import VLM_PROVIDERS, parse_action, query_vlm
from vta_agent_v2 import VerifyThenActAgentV2
from ef_ttc_agent import EvidenceFirstTTCAgent
from self_check_agent import SelfCheckAgent

START_DATE = "20230101"
WINDOW_SIZE = 30
METHODS = ("baseline", "abstain", "vta_v2", "ef_ttc", "self_check", "ef_ttc_ac")


DEFAULT_MODELS = [
    {"model_key": "glm-4v-flash",        "provider": "bigmodel"},
    {"model_key": "nemotron-nano-vl",    "provider": "nvidia"},
    {"model_key": "llama-3.2-11b-vision","provider": "nvidia"},
    {"model_key": "gemma-4-31b-it",      "provider": "nvidia"},
    {"model_key": "cosmos-reason2-8b",   "provider": "nvidia"},
]


def has_provider_credentials(provider: str):
    # 所有 *_local provider 本地部署不需要 key，直接放行
    if provider.endswith("_local"):
        return True
    env_key = VLM_PROVIDERS.get(provider, {}).get("env_key")
    if not env_key:
        return False
    return bool(os.environ.get(env_key))


def load_window(ts_code: str, end_date: str, window: int = WINDOW_SIZE):
    df = get_daily_data(ts_code, START_DATE, end_date)
    if df is None or df.empty:
        return None
    target_date = pd.Timestamp(end_date)
    df_up_to = df[df.index <= target_date]
    if len(df_up_to) < window:
        return None
    return df_up_to.iloc[-window:]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--end-date", default="20230630")
    parser.add_argument("--chart-dir", default="charts")
    parser.add_argument("--output-root", default="results_multi_model_chart_only")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.9,
        help="每只股票每个方法完成后的间隔（秒）；多轮方法易触限流，默认 0.9；仍失败可调大到 1.2–2.0",
    )
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument(
        "--model-specs",
        default="",
        help="逗号分隔 provider:model_key，例如 bigmodel:glm-4v-flash,nvidia:nemotron-nano-vl",
    )
    parser.add_argument(
        "--retry-errors-only",
        action="store_true",
        help="仅重跑各方法输出 JSON 中带 error 字段的股票（用于补全 API 失败样本）",
    )
    parser.add_argument(
        "--api-base",
        default=None,
        help="覆盖 provider base_url（qwen_local 等本地部署必传，例如 http://localhost:8000/v1）",
    )
    parser.add_argument("--seed", type=int, default=20230630, help="随机种子，透传给 API")
    parser.add_argument("--api-key", default=None, help="覆盖 provider 默认 API key（用于 yunwu 等中转服务）")
    return parser.parse_args()


def build_model_specs(raw: str):
    if not raw.strip():
        return DEFAULT_MODELS

    specs = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        provider, model_key = item.split(":", 1)
        specs.append({"provider": provider.strip(), "model_key": model_key.strip()})
    return specs


def normalize_methods(raw: str):
    methods = [m.strip() for m in raw.split(",") if m.strip()]
    for method in methods:
        if method not in METHODS:
            raise ValueError(f"未知方法: {method}")
    return methods


def make_blank_png(path: Path, size=(800, 500)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(255, 255, 255)).save(path)


def ensure_chart_exists(chart_dir: str, ts_code: str, end_date: str):
    chart_path = Path(chart_dir) / f"{ts_code}_{end_date}.png"
    if chart_path.exists():
        return chart_path

    try:
        sample = create_sample(ts_code, end_date, chart_save_dir=chart_dir)
    except Exception as exc:
        print(f"[WARN] create_sample failed for {ts_code}: {exc}")
        sample = None
    if sample is not None:
        return Path(sample["chart_path"])
    # 兜底：至少保证图像存在，后续方法可以继续跑并产出结果
    make_blank_png(chart_path)
    return chart_path


def run_direct(model_key, provider, chart_path, ts_code, end_date, allow_abstain=False, api_base=None, seed=20230630):
    result = query_vlm(
        model_key,
        ohlcv_text=None,
        chart_path=chart_path,
        ts_code=ts_code,
        end_date=end_date,
        provider=provider,
        system_prompt_key="abstain" if allow_abstain else "baseline",
        api_base=api_base,
        seed=seed,
    )
    parsed = parse_action(result["response"])
    return {
        "response": result["response"],
        "action": parsed["action"],
        "confidence": parsed["confidence"],
        "usage": result["usage"],
    }


def summarize_common(ts_code: str, end_date: str, provider: str, model_key: str, method: str, raw: Dict[str, Any], detection: Dict[str, Any], seed: int = 20230630):
    return {
        "ts_code": ts_code,
        "end_date": end_date,
        "provider": provider,
        "model": model_key,
        "mode": "chart_only",
        "method": method,
        "prompt_version": "v4-2026-05",
        "detector_version": "v3.1",
        "sampling": {
            "temperature": 0.0,
            "seed": seed,
        },
        "action": raw.get("action"),
        "confidence": raw.get("confidence"),
        "response": raw.get("response"),
        "usage": raw.get("usage") or raw.get("total_usage"),
        "metrics": {
            "answered": raw.get("action") in {"BUY", "SELL", "HOLD"},
            "uncertain": raw.get("action") == "UNCERTAIN",
            "type1_fgr": detection["type1_fgr"]["fgr"],
            "type1_ucr": detection["type1_fgr"]["ucr"],          # 向后兼容
            "type1_total_claims": detection["type1_fgr"]["total_claims"],
            "type1_unsupported": detection["type1_fgr"]["unsupported_count"],
            "type2_inconsistent": detection["type2_rci"]["inconsistent"],
            "type2_sentiment": detection["type2_rci"]["sentiment"],
            "type3_overclaim": detection["type3_eci"]["overclaim"],
            "signal_count": detection["type3_eci"]["signal_count"],
            "response_language": detection.get("language", "unknown"),
        },
        "detection_summary": {
            "type1": detection["type1_fgr"],
            "type2": detection["type2_rci"],
            "type3": detection["type3_eci"],
        },
    }


def main():
    args = parse_args()
    inject_api_keys_from_demo_config()
    methods = normalize_methods(args.methods)
    model_specs = build_model_specs(args.model_specs)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    stocks = list(STOCKS_HS300)
    if args.limit > 0:
        stocks = stocks[:args.limit]

    model_specs = [spec for spec in model_specs if has_provider_credentials(spec["provider"])]
    if not model_specs:
        raise RuntimeError("没有可用 provider 凭据；请设置 ZHIPU_API_KEY / OPENROUTER_API_KEY / OPENAI_API_KEY / NVIDIA_API_KEY")
    for spec in model_specs:
        pk = spec["provider"]
        if pk in VLM_PROVIDERS:
            env_k = VLM_PROVIDERS[pk]["env_key"]
            if not os.environ.get(env_k):
                raise RuntimeError(
                    f"缺少 {env_k}（provider={pk}）。请在环境变量或 demo/config.py 中配置后重试。"
                )

    if args.retry_errors_only:
        first = model_specs[0]
        probe_dir = output_root / first["provider"] / first["model_key"] / args.end_date / "chart_only"
        err_codes = []
        for method in methods:
            for p in probe_dir.glob(f"*_{method}.json"):
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    continue
                # 1) 显式 error
                needs = "error" in data
                # 2) detection_summary 缺失 / 为空 dict（旧版本残留：没跑检测器就落盘了）
                if not needs:
                    ds = data.get("detection_summary")
                    if not isinstance(ds, dict) or len(ds) == 0:
                        needs = True
                if not needs:
                    continue
                ts = data.get("ts_code") or p.stem.rsplit("_", 1)[0]
                if ts:
                    err_codes.append(ts)
        stocks = sorted(set(err_codes))
        if not stocks:
            print("[INFO] --retry-errors-only：未发现带 error 或检测缺失的落盘，退出")
            return
        print(f"[INFO] --retry-errors-only：将重试 {len(stocks)} 只股票")

    vta_agents = {}
    ef_ttc_agents = {}
    self_check_agents = {}

    for spec in model_specs:
        key = (spec["provider"], spec["model_key"])
        vta_agents[key] = VerifyThenActAgentV2(provider=spec["provider"])
        ef_ttc_agents[key] = EvidenceFirstTTCAgent(provider=spec["provider"])
        self_check_agents[key] = SelfCheckAgent(provider=spec["provider"])

    for spec in model_specs:
        provider = spec["provider"]
        model_key = spec["model_key"]
        model_dir = output_root / provider / model_key / args.end_date / "chart_only"
        model_dir.mkdir(parents=True, exist_ok=True)

        for ts_code in tqdm(stocks, desc=f"{provider}/{model_key}"):
            chart_path = ensure_chart_exists(args.chart_dir, ts_code, args.end_date)
            if chart_path is None or not chart_path.exists():
                continue

            try:
                df_window = load_window(ts_code, args.end_date)
            except Exception as exc:
                print(f"[WARN] load_window failed for {ts_code}: {exc}")
                df_window = None

            for method in methods:
                output_file = model_dir / f"{ts_code}_{method}.json"
                if output_file.exists():
                    try:
                        existing = json.loads(output_file.read_text(encoding="utf-8"))
                        ds = existing.get("detection_summary")
                        # 跳过条件：(a) 没有 error 字段；(b) detection_summary 是有内容的 dict
                        if "error" not in existing and isinstance(ds, dict) and len(ds) > 0:
                            continue
                    except Exception:
                        pass

                # df_window 不可得：直接落盘 error，不参与汇总（避免污染 UCR 均值）
                if df_window is None:
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "ts_code": ts_code,
                                "end_date": args.end_date,
                                "provider": provider,
                                "model": model_key,
                                "method": method,
                                "error": "df_window_unavailable",
                            },
                            f,
                            ensure_ascii=False,
                            indent=2,
                        )
                    continue

                try:
                    if method == "baseline":
                        direct = run_direct(model_key, provider, str(chart_path), ts_code, args.end_date, allow_abstain=False, api_base=args.api_base, seed=args.seed)
                        detection = detect_all_hallucinations(direct["response"], direct["action"], direct["confidence"], df_window)
                        record = summarize_common(ts_code, args.end_date, provider, model_key, method, direct, detection, seed=args.seed)

                    elif method == "abstain":
                        direct = run_direct(model_key, provider, str(chart_path), ts_code, args.end_date, allow_abstain=True, api_base=args.api_base, seed=args.seed)
                        detection = detect_all_hallucinations(direct["response"], direct["action"], direct["confidence"], df_window)
                        record = summarize_common(ts_code, args.end_date, provider, model_key, method, direct, detection, seed=args.seed)

                    elif method == "vta_v2":
                        agent = vta_agents[(provider, model_key)]
                        result = agent.generate_with_vta(
                            model_key=model_key,
                            ohlcv_text=None,
                            chart_path=str(chart_path),
                            ts_code=ts_code,
                            end_date=args.end_date,
                            df_window=df_window,
                            verbose=False,
                        )
                        detection = detect_all_hallucinations(result["response"], result["action"], result["confidence"], df_window)
                        record = summarize_common(ts_code, args.end_date, provider, model_key, method, result, detection, seed=args.seed)
                        record["vta"] = {
                            "strategy": result["strategy"],
                            "initial_ucr": result["initial_ucr"],
                            "final_ucr": result["final_ucr"],
                            "initial_action": result["initial_action"],
                            "initial_confidence": result["initial_confidence"],
                            "feedback_prompt": result.get("feedback_prompt"),
                        }

                    elif method == "ef_ttc":
                        agent = ef_ttc_agents[(provider, model_key)]
                        result = agent.generate_with_ef_ttc(
                            model_key=model_key,
                            ohlcv_text=None,
                            chart_path=str(chart_path),
                            ts_code=ts_code,
                            end_date=args.end_date,
                            df_window=df_window,
                        )
                        detection = result["detection"]
                        record = summarize_common(ts_code, args.end_date, provider, model_key, method, result, detection, seed=args.seed)
                        record["ef_ttc"] = {
                            "predicted_evidence": result["predicted_evidence"],
                            "verification": result["verification"],
                            "decision_prompt": result["decision_prompt"],
                            "evidence_raw_response": result["evidence_raw_response"],
                        }

                    elif method == "self_check":
                        agent = self_check_agents[(provider, model_key)]
                        result = agent.generate_with_self_check(
                            ts_code=ts_code,
                            end_date=args.end_date,
                            ohlcv_text=None,
                            chart_path=str(chart_path),
                            model_key=model_key,
                            df_window=df_window,
                        )
                        detection = result["detection"]
                        record = summarize_common(ts_code, args.end_date, provider, model_key, method, result, detection, seed=args.seed)
                        record["self_check"] = {
                            "self_check_passed": result["self_check_passed"],
                            "self_check_triggered": result["self_check_triggered"],
                            "self_check_response": result["self_check_response"][:400],
                            "step1_action": result["step1_action"],
                        }

                    else:  # ef_ttc_ac
                        agent = ef_ttc_agents[(provider, model_key)]
                        result = agent.generate_with_ef_ttc_ac(
                            model_key=model_key,
                            ohlcv_text=None,
                            chart_path=str(chart_path),
                            ts_code=ts_code,
                            end_date=args.end_date,
                            df_window=df_window,
                        )
                        detection = result["detection"]
                        record = summarize_common(ts_code, args.end_date, provider, model_key, method, result, detection, seed=args.seed)
                        record["ef_ttc_ac"] = {
                            "predicted_evidence": result["predicted_evidence"],
                            "verification": result["verification"],
                            "ac_agg_direction": result["ac_agg_direction"],
                            "ac_triggered": result["ac_triggered"],
                            "ac_bullish_count": result["ac_bullish_count"],
                            "ac_bearish_count": result["ac_bearish_count"],
                        }

                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(record, f, ensure_ascii=False, indent=2)

                except Exception as exc:
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "ts_code": ts_code,
                                "end_date": args.end_date,
                                "provider": provider,
                                "model": model_key,
                                "method": method,
                                "error": str(exc),
                            },
                            f,
                            ensure_ascii=False,
                            indent=2,
                        )

                time.sleep(args.sleep)


if __name__ == "__main__":
    main()
