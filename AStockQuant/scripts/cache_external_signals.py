# -*- coding: utf-8 -*-
"""
Cache external ETF and concept-flow signals from AkShare.

The output is a point-in-time snapshot for current research and daily signal
review. Do not backfill historical backtests with today's snapshot.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "external_cache"
ETF_LIST_PATH = ROOT / "all_etf_list.json"

THEME_KEYWORDS = {
    "cash_bond": ("货币", "现金", "日利", "添益", "快钱", "快线", "保证金", "国债", "债", "短融", "利率债"),
    "qdii_us": ("纳指", "纳斯达克", "美国", "标普", "道琼斯", "德国", "法国", "日经"),
    "communication": ("5G", "通信", "光通信", "CPO", "光纤", "共封装光学"),
    "ai": ("AI", "人工智能", "机器人", "AIDC", "云", "数据", "软件"),
    "chip": ("芯片", "半导体", "集成", "电子", "存储芯片", "中芯国际", "光刻机", "国家大基金"),
    "hk_tech": ("恒生科技", "恒科", "港股通科技"),
    "china_internet": ("互联网", "中概", "港股通互联网"),
    "healthcare": ("创新药", "医药", "医疗", "生物", "疫苗"),
    "broker": ("证券", "券商", "融资融券", "互联金融"),
    "defense": ("军工", "国防", "航天", "军民融合"),
    "ev": ("新能源车", "电池", "汽车", "智能车", "光伏", "新能源", "储能", "锂电"),
    "energy_materials": ("有色", "稀土", "煤炭", "能源", "钢铁", "化工"),
    "consumer": ("消费", "酒", "食品", "家电", "旅游"),
    "finance": ("银行", "保险", "金融", "地产"),
    "gold": ("黄金", "贵金属"),
    "growth": ("创业板", "双创", "科创", "科创板", "专精特新"),
}

CODE_THEME_OVERRIDES = {
    "511010": "cash_bond",
    "511260": "cash_bond",
    "511880": "cash_bond",
    "511990": "cash_bond",
    "518880": "gold",
    "159915": "growth",
    "510300": "broad_300",
    "510500": "small_mid",
    "512000": "broker",
    "512480": "chip",
    "512760": "chip",
    "513090": "hk_broker",
    "513120": "healthcare",
    "513130": "hk_tech",
    "513180": "hk_tech",
    "513330": "china_internet",
    "513870": "qdii_us",
    "159577": "qdii_us",
    "159660": "qdii_us",
    "159811": "communication",
    "159941": "qdii_us",
    "515000": "technology",
    "515030": "ev",
    "515050": "technology",
    "515980": "ai",
}


def load_etf_codes() -> set[str]:
    if not ETF_LIST_PATH.exists():
        return set()
    try:
        items = json.loads(ETF_LIST_PATH.read_text(encoding="utf-8"))
    except Exception:
        return set()
    return {str(item.get("code", "")).strip() for item in items if str(item.get("code", "")).strip()}


def normalize_code(value: object) -> str:
    text = str(value).strip()
    match = re.search(r"(\d{6})", text)
    return match.group(1) if match else text


def to_float(value: object) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().replace(",", "")
    if not text or text in {"-", "--", "nan", "None"}:
        return None
    is_percent = text.endswith("%")
    if is_percent:
        text = text[:-1]
    try:
        number = float(text)
    except ValueError:
        return None
    return number / 100.0 if is_percent else number


def find_column(columns: Iterable[str], include: Iterable[str], exclude: Iterable[str] = ()) -> Optional[str]:
    include = tuple(include)
    exclude = tuple(exclude)
    for column in columns:
        name = str(column)
        if all(key in name for key in include) and not any(key in name for key in exclude):
            return name
    return None


def infer_theme_from_text(text: object, code: str = "") -> str:
    if code in CODE_THEME_OVERRIDES:
        return CODE_THEME_OVERRIDES[code]
    value = str(text or "")
    for theme, keywords in THEME_KEYWORDS.items():
        if any(keyword in value for keyword in keywords):
            return theme
    return "other"


def call_source(name: str, fn: Callable[[], pd.DataFrame], errors: Dict[str, str]) -> pd.DataFrame:
    try:
        df = fn()
    except Exception as exc:
        errors[name] = f"{type(exc).__name__}: {exc}"
        return pd.DataFrame()
    if not isinstance(df, pd.DataFrame):
        errors[name] = f"unexpected result type: {type(df).__name__}"
        return pd.DataFrame()
    if df.empty:
        errors[name] = "empty dataframe"
    return df


def normalize_etf_daily(df: pd.DataFrame, etf_codes: set[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    code_col = find_column(df.columns, ("基金代码",)) or find_column(df.columns, ("代码",))
    name_col = find_column(df.columns, ("基金简称",)) or find_column(df.columns, ("名称",))
    price_col = find_column(df.columns, ("市价",))
    discount_col = find_column(df.columns, ("折价率",))
    growth_col = find_column(df.columns, ("增长率",))
    latest_nav_col = find_column(df.columns, ("单位净值",), ("累计",))
    latest_cum_nav_col = find_column(df.columns, ("累计净值",))

    if code_col is None:
        return pd.DataFrame()

    out = pd.DataFrame()
    out["code"] = df[code_col].map(normalize_code)
    out["name"] = df[name_col].astype(str) if name_col else ""
    out["market_price"] = df[price_col].map(to_float) if price_col else None
    out["discount_rate"] = df[discount_col].map(to_float) if discount_col else None
    out["nav_growth_rate"] = df[growth_col].map(to_float) if growth_col else None
    out["unit_nav"] = df[latest_nav_col].map(to_float) if latest_nav_col else None
    out["cum_nav"] = df[latest_cum_nav_col].map(to_float) if latest_cum_nav_col else None
    out["theme"] = [infer_theme_from_text(name, code) for name, code in zip(out["name"], out["code"])]
    if etf_codes:
        out = out[out["code"].isin(etf_codes)]
    out = out.drop_duplicates("code", keep="first").sort_values("code")
    return out


def normalize_purchase_status(df: pd.DataFrame, etf_codes: set[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    code_col = find_column(df.columns, ("基金代码",)) or find_column(df.columns, ("代码",))
    if code_col is None:
        return pd.DataFrame()
    out = pd.DataFrame()
    out["code"] = df[code_col].map(normalize_code)
    for source, target in [
        ("基金简称", "purchase_name"),
        ("基金类型", "fund_type"),
        ("申购状态", "purchase_status"),
        ("赎回状态", "redeem_status"),
        ("日累计限定金额", "daily_purchase_limit"),
        ("手续费", "fee"),
    ]:
        col = find_column(df.columns, (source,))
        if col is not None:
            out[target] = df[col]
    if etf_codes:
        out = out[out["code"].isin(etf_codes)]
    if "daily_purchase_limit" in out.columns:
        out["daily_purchase_limit"] = out["daily_purchase_limit"].map(to_float)
    if "fee" in out.columns:
        out["fee"] = out["fee"].map(to_float)
    return out.drop_duplicates("code", keep="first").sort_values("code")


def normalize_concept_flow(df: pd.DataFrame, source: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    name_col = find_column(df.columns, ("名称",)) or find_column(df.columns, ("行业",))
    net_col = find_column(df.columns, ("净额",))
    pct_col = find_column(df.columns, ("涨跌幅",))
    leader_col = find_column(df.columns, ("领涨股",)) or find_column(df.columns, ("最大股",))
    if name_col is None:
        return pd.DataFrame()
    out = pd.DataFrame(index=df.index)
    out["concept"] = df[name_col].astype(str)
    out["source"] = source
    out["theme"] = out["concept"].map(infer_theme_from_text)
    out["net_flow"] = df[net_col].map(to_float) if net_col else None
    out["change_pct"] = df[pct_col].map(to_float) if pct_col else None
    if "change_pct" in out.columns and out["change_pct"].notna().any() and out["change_pct"].abs().max() > 1.0:
        out["change_pct"] = out["change_pct"] / 100.0
    out["leader"] = df[leader_col].astype(str) if leader_col else ""
    out["rank"] = range(1, len(out) + 1)
    return out


def aggregate_theme_flow(concept_flow: pd.DataFrame) -> pd.DataFrame:
    if concept_flow.empty:
        return pd.DataFrame()
    usable = concept_flow[concept_flow["theme"] != "other"].copy()
    if usable.empty:
        return pd.DataFrame()
    agg = (
        usable.groupby("theme", as_index=False)
        .agg(
            concept_count=("concept", "count"),
            net_flow_sum=("net_flow", "sum"),
            net_flow_median=("net_flow", "median"),
            change_pct_median=("change_pct", "median"),
            best_rank=("rank", "min"),
        )
        .sort_values(["net_flow_sum", "change_pct_median"], ascending=[False, False])
    )
    return agg


def latest_scale_change(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = df.copy()
    for col in ["期间申购", "期间赎回", "期末总份额", "期末净资产"]:
        if col in out.columns:
            out[col] = out[col].map(to_float)
    if "截止日期" in out.columns:
        out["截止日期"] = pd.to_datetime(out["截止日期"], errors="coerce")
        out = out.sort_values("截止日期", ascending=False)
    return out.head(8)


def save_df(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        path.write_text("", encoding="utf-8")
    else:
        df.to_csv(path, index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache external ETF and concept-flow signals")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--skip-akshare", action="store_true", help="only validate file wiring; do not call network sources")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    errors: Dict[str, str] = {}
    etf_codes = load_etf_codes()

    if args.skip_akshare:
        metadata = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "skip-akshare",
            "etf_code_count": len(etf_codes),
            "errors": {},
            "outputs": {},
        }
        (output_dir / "external_signals_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return 0

    try:
        import akshare as ak
    except Exception as exc:
        print(f"Failed to import akshare: {exc}", file=sys.stderr)
        return 2

    etf_daily_raw = call_source("fund_etf_fund_daily_em", ak.fund_etf_fund_daily_em, errors)
    purchase_raw = call_source("fund_purchase_em", ak.fund_purchase_em, errors)
    scale_raw = call_source("fund_scale_change_em", ak.fund_scale_change_em, errors)
    concept_flow_raw = call_source(
        "stock_fund_flow_concept_instant",
        lambda: ak.stock_fund_flow_concept(symbol="即时"),
        errors,
    )
    concept_rank_today_raw = call_source(
        "stock_sector_fund_flow_rank_concept_today",
        lambda: ak.stock_sector_fund_flow_rank(indicator="今日", sector_type="概念资金流"),
        errors,
    )

    etf_daily = normalize_etf_daily(etf_daily_raw, etf_codes)
    purchase = normalize_purchase_status(purchase_raw, etf_codes)
    etf_snapshot = etf_daily.merge(purchase, on="code", how="left") if not etf_daily.empty else purchase

    concept_flow = pd.concat(
        [
            normalize_concept_flow(concept_flow_raw, "ths_instant"),
            normalize_concept_flow(concept_rank_today_raw, "em_today_rank"),
        ],
        ignore_index=True,
    )
    theme_flow = aggregate_theme_flow(concept_flow)
    scale_latest = latest_scale_change(scale_raw)

    outputs = {
        "etf_external_snapshot": output_dir / "etf_external_snapshot.csv",
        "concept_flow_snapshot": output_dir / "concept_flow_snapshot.csv",
        "theme_flow_snapshot": output_dir / "theme_flow_snapshot.csv",
        "fund_scale_change_recent": output_dir / "fund_scale_change_recent.csv",
    }
    save_df(etf_snapshot, outputs["etf_external_snapshot"])
    save_df(concept_flow, outputs["concept_flow_snapshot"])
    save_df(theme_flow, outputs["theme_flow_snapshot"])
    save_df(scale_latest, outputs["fund_scale_change_recent"])

    metadata = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "live-akshare",
        "etf_code_count": len(etf_codes),
        "rows": {
            "etf_daily_raw": int(len(etf_daily_raw)),
            "purchase_raw": int(len(purchase_raw)),
            "scale_raw": int(len(scale_raw)),
            "concept_flow_raw": int(len(concept_flow_raw)),
            "concept_rank_today_raw": int(len(concept_rank_today_raw)),
            "etf_external_snapshot": int(len(etf_snapshot)),
            "concept_flow_snapshot": int(len(concept_flow)),
            "theme_flow_snapshot": int(len(theme_flow)),
            "fund_scale_change_recent": int(len(scale_latest)),
        },
        "errors": errors,
        "outputs": {key: str(path) for key, path in outputs.items()},
        "warning": "These are current snapshots. Use them for live review or future point-in-time logs, not for historical backtest backfill.",
    }
    (output_dir / "external_signals_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
