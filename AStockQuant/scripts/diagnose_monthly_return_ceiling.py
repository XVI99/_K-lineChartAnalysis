# -*- coding: utf-8 -*-
"""
Diagnose the hindsight monthly return ceiling from local ETF price caches.

This is not a tradable strategy. It answers a narrower feasibility question:
if we could know in hindsight which local ETF had the best close-to-close
monthly return, how often did even that best ETF reach a target return?
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data_cache"
ETF_LIST_PATH = ROOT / "all_etf_list.json"
REPORT_DIR = ROOT / "reports"
ETF_PREFIXES = ("510", "511", "512", "513", "515", "516", "517", "518", "560", "561", "562", "563", "588", "159")


def load_name_map(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    try:
        items = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {str(item.get("code", "")).strip().zfill(6): str(item.get("name", "")).strip() for item in items}


def monthly_returns_for_code(path: Path, start: pd.Timestamp, end: pd.Timestamp, min_rows: int) -> pd.DataFrame:
    code = path.stem
    if not code.startswith(ETF_PREFIXES):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, usecols=["date", "close"], parse_dates=["date"])
    except Exception:
        return pd.DataFrame()
    if len(df) < min_rows:
        return pd.DataFrame()
    df = df.sort_values("date")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "close"])
    df = df[(df["date"] >= start) & (df["date"] <= end) & (df["close"] > 0)]
    if df.empty:
        return pd.DataFrame()

    monthly = df.set_index("date")["close"].resample("ME").agg(["first", "last", "count"])
    monthly = monthly.dropna(subset=["first", "last"])
    if monthly.empty:
        return pd.DataFrame()
    out = monthly.reset_index().rename(columns={"date": "month_end", "count": "trading_days"})
    out["month"] = out["month_end"].dt.strftime("%Y-%m")
    out["code"] = code
    out["monthly_return"] = out["last"] / out["first"] - 1.0
    return out[["month", "month_end", "code", "first", "last", "trading_days", "monthly_return"]]


def build_ceiling(
    data_dir: Path,
    name_map: Dict[str, str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    min_rows: int,
    target_return: float,
) -> tuple[pd.DataFrame, Dict[str, object]]:
    frames: List[pd.DataFrame] = []
    for path in data_dir.glob("*.csv"):
        item = monthly_returns_for_code(path, start, end, min_rows)
        if not item.empty:
            frames.append(item)
    if not frames:
        raise RuntimeError("No local ETF monthly returns were available for the requested period.")

    all_monthly = pd.concat(frames, ignore_index=True)
    all_monthly["name"] = all_monthly["code"].map(name_map).fillna("")
    best = (
        all_monthly.sort_values(["month", "monthly_return"], ascending=[True, False])
        .groupby("month", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    best["best_ge_target"] = best["monthly_return"] >= target_return

    best_for_json = best.copy()
    best_for_json["month_end"] = best_for_json["month_end"].astype(str)
    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "script": str(Path(__file__).relative_to(ROOT)),
        "assumption": "hindsight monthly best ETF from local data_cache; not tradable and not point-in-time universe safe",
        "start": str(start.date()),
        "end": str(end.date()),
        "min_rows": int(min_rows),
        "target_return": float(target_return),
        "months": int(len(best)),
        "months_best_ge_target": int(best["best_ge_target"].sum()),
        "months_best_ge_target_rate": float(best["best_ge_target"].mean()) if len(best) else 0.0,
        "best_monthly_return": float(best["monthly_return"].max()),
        "weakest_monthly_best_return": float(best["monthly_return"].min()),
        "monthly_best_sample": best_for_json.to_dict(orient="records"),
    }
    return best, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose hindsight ETF monthly return ceiling")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-05-29")
    parser.add_argument("--min-rows", type=int, default=120)
    parser.add_argument("--target-return", type=float, default=0.30)
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--output-csv", default=str(REPORT_DIR / "monthly_return_ceiling.csv"))
    parser.add_argument("--output-summary", default=str(REPORT_DIR / "monthly_return_ceiling_summary.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_csv = Path(args.output_csv)
    output_summary = Path(args.output_summary)
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)

    best, summary = build_ceiling(
        data_dir=data_dir,
        name_map=load_name_map(ETF_LIST_PATH),
        start=start,
        end=end,
        min_rows=args.min_rows,
        target_return=args.target_return,
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    best.to_csv(output_csv, index=False, encoding="utf-8-sig")
    output_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print("Monthly ETF return ceiling diagnostic")
    print("=" * 72)
    print(f"Period: {start.date()} -> {end.date()} | target: {args.target_return * 100:.1f}%")
    print(
        f"Months best ETF >= target: {summary['months_best_ge_target']}/{summary['months']} "
        f"({summary['months_best_ge_target_rate'] * 100:.1f}%)"
    )
    print(f"Best monthly return: {summary['best_monthly_return'] * 100:.2f}%")
    print(f"Weakest monthly best return: {summary['weakest_monthly_best_return'] * 100:.2f}%")
    print("Outputs:")
    print(f"  {output_csv}")
    print(f"  {output_summary}")


if __name__ == "__main__":
    main()
