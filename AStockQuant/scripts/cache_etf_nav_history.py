# -*- coding: utf-8 -*-
"""
Cache historical ETF NAV and purchase/redeem status from AkShare.

This creates a point-in-time usable historical external feature set:
- unit NAV and cumulative NAV
- NAV daily growth
- purchase/redeem status
- market close from local data_cache
- premium_rate = close / unit_nav - 1

Unlike current external snapshots, these rows are dated and can be used by
walk-forward research without backfilling today's information into the past.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data_cache"
OUTPUT_DIR = ROOT / "external_cache" / "nav_history"
COMBINED_PATH = ROOT / "external_cache" / "etf_nav_history.csv"
META_PATH = ROOT / "external_cache" / "etf_nav_history_metadata.json"


def load_local_universe(data_dir: Path, max_codes: int, min_rows: int) -> List[str]:
    rows = []
    for path in data_dir.glob("*.csv"):
        code = path.stem
        if not code.isdigit():
            continue
        try:
            df = pd.read_csv(path, usecols=["date", "close", "volume"], parse_dates=["date"])
        except Exception:
            continue
        if len(df) < min_rows:
            continue
        amount = float((pd.to_numeric(df["close"], errors="coerce") * pd.to_numeric(df["volume"], errors="coerce")).tail(250).median())
        rows.append((code, amount))
    rows.sort(key=lambda item: item[1], reverse=True)
    return [code for code, _ in rows[:max_codes]]


def normalize_nav_history(code: str, nav_df: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    if nav_df.empty:
        return pd.DataFrame()
    out = nav_df.copy()
    out = out.rename(
        columns={
            "净值日期": "date",
            "单位净值": "unit_nav",
            "累计净值": "cum_nav",
            "日增长率": "nav_growth_pct",
            "申购状态": "purchase_status",
            "赎回状态": "redeem_status",
        }
    )
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["code"] = code
    for col in ["unit_nav", "cum_nav", "nav_growth_pct"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["nav_growth_rate"] = out["nav_growth_pct"] / 100.0
    out["purchase_allowed"] = out["purchase_status"].astype(str).str.contains("买入|申购|开放|场内", regex=True)
    out["redeem_allowed"] = out["redeem_status"].astype(str).str.contains("卖出|赎回|开放|场内", regex=True)

    price_path = data_dir / f"{code}.csv"
    if price_path.exists():
        try:
            price_df = pd.read_csv(price_path, usecols=["date", "close"], parse_dates=["date"])
            price_df = price_df.rename(columns={"close": "market_close"})
            out = out.merge(price_df, on="date", how="left")
            out["premium_rate"] = out["market_close"] / out["unit_nav"] - 1.0
        except Exception:
            out["market_close"] = pd.NA
            out["premium_rate"] = pd.NA
    else:
        out["market_close"] = pd.NA
        out["premium_rate"] = pd.NA

    keep = [
        "date",
        "code",
        "unit_nav",
        "cum_nav",
        "nav_growth_rate",
        "purchase_status",
        "redeem_status",
        "purchase_allowed",
        "redeem_allowed",
        "market_close",
        "premium_rate",
    ]
    return out[keep].dropna(subset=["date", "unit_nav"]).sort_values("date")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache historical ETF NAV data")
    parser.add_argument("--codes", default="", help="comma-separated ETF codes; overrides liquidity selection")
    parser.add_argument("--max-codes", type=int, default=40)
    parser.add_argument("--min-rows", type=int, default=500)
    parser.add_argument("--start", default="20190101")
    parser.add_argument("--end", default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--sleep", type=float, default=0.15)
    parser.add_argument("--force", action="store_true", help="refetch even if a per-code cache exists")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    COMBINED_PATH.parent.mkdir(parents=True, exist_ok=True)

    if args.codes.strip():
        codes = [code.strip() for code in args.codes.split(",") if code.strip()]
    else:
        codes = load_local_universe(DATA_DIR, args.max_codes, args.min_rows)

    try:
        import akshare as ak
    except Exception as exc:
        print(f"Failed to import akshare: {exc}")
        return 2

    errors: Dict[str, str] = {}
    frames: List[pd.DataFrame] = []
    fetched = 0
    reused = 0

    for idx, code in enumerate(codes, start=1):
        cache_path = OUTPUT_DIR / f"{code}.csv"
        if cache_path.exists() and not args.force:
            try:
                df = pd.read_csv(cache_path, parse_dates=["date"], dtype={"code": str})
                frames.append(df)
                reused += 1
                print(f"[{idx}/{len(codes)}] reuse {code}: {len(df)} rows")
                continue
            except Exception:
                pass

        try:
            raw = ak.fund_etf_fund_info_em(fund=code, start_date=args.start, end_date=args.end)
            df = normalize_nav_history(code, raw, DATA_DIR)
            if df.empty:
                errors[code] = "empty normalized dataframe"
                print(f"[{idx}/{len(codes)}] empty {code}")
                continue
            df.to_csv(cache_path, index=False, encoding="utf-8-sig")
            frames.append(df)
            fetched += 1
            print(f"[{idx}/{len(codes)}] fetched {code}: {len(df)} rows")
        except Exception as exc:
            errors[code] = f"{type(exc).__name__}: {exc}"
            print(f"[{idx}/{len(codes)}] error {code}: {errors[code]}")
        time.sleep(args.sleep)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not combined.empty:
        combined = combined.sort_values(["date", "code"])
        combined.to_csv(COMBINED_PATH, index=False, encoding="utf-8-sig")
    else:
        COMBINED_PATH.write_text("", encoding="utf-8")

    metadata = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "codes_requested": len(codes),
        "fetched": fetched,
        "reused": reused,
        "combined_rows": int(len(combined)),
        "start": args.start,
        "end": args.end,
        "output_dir": str(OUTPUT_DIR),
        "combined_path": str(COMBINED_PATH),
        "errors": errors,
        "warning": "Historical NAV rows are dated and can be used as external features; still verify publication lag before live execution.",
    }
    META_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
