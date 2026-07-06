# -*- coding: utf-8 -*-
"""
Aggressive ETF walk-forward research harness.

This script is intentionally stricter than the older in-sample momentum demos:
- train-window parameter selection, then out-of-sample test-window validation
- T+1 open execution from close-based signals
- ETF 100-share lot sizing
- explicit fee/slippage cost
- monthly return target diagnostics

It is a research tool, not a live-trading promise.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# 把 AStockQuant 根目录加入 sys.path, 确保 from models.* / from backtest.* 可导入
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    # 可选依赖: 仅在 --use-ai 启用时需要
    from backtest.ai_signal_bridge import AISignalBridge, inject_ai_prob
except Exception:  # pragma: no cover - import 失败时延迟到 --use-ai 才报错
    AISignalBridge = None  # type: ignore
    inject_ai_prob = None  # type: ignore


warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data_cache"
REPORT_DIR = ROOT / "reports"
ETF_LIST_PATH = ROOT / "all_etf_list.json"
NAV_HISTORY_PATH = ROOT / "external_cache" / "etf_nav_history.csv"

ETF_PREFIXES = ("510", "511", "512", "513", "515", "516", "517", "518", "560", "561", "562", "563", "588", "159")
MONEY_OR_BOND_TERMS = ("保证金", "快线", "快钱", "货币", "现金", "短融", "国债", "债")
DEFENSIVE_CANDIDATES = ("518880", "511880", "511010", "511260", "159001", "159003", "159005")
MONTHLY_FOR_100_CAGR = (2.0 ** (1.0 / 12.0)) - 1.0

CODE_THEME_OVERRIDES = {
    "159915": "growth",
    "159919": "broad_300",
    "510050": "broad_50",
    "510300": "broad_300",
    "510500": "small_mid",
    "512000": "broker",
    "512010": "healthcare",
    "512100": "defense",
    "512200": "real_estate",
    "512480": "chip",
    "512690": "consumer",
    "512760": "chip",
    "513050": "china_internet",
    "513090": "hk_broker",
    "513120": "healthcare",
    "513130": "hk_tech",
    "513180": "hk_tech",
    "513330": "china_internet",
    "513870": "qdii_us",
    "515000": "technology",
    "515030": "ev",
    "515050": "technology",
    "515980": "ai",
    "518880": "gold",
    "159577": "qdii_us",
    "159660": "qdii_us",
    "159811": "communication",
    "159941": "qdii_us",
}

THEME_KEYWORDS = [
    ("cash_bond", ("保证金", "快线", "快钱", "货币", "现金", "短融", "国债", "债")),
    ("qdii_us", ("纳指", "纳斯达克", "美国", "标普", "道琼斯", "德国", "法国", "日经")),
    ("communication", ("5G", "通信", "光通信", "CPO", "光纤", "共封装光学")),
    ("ai", ("AI", "人工智能", "机器人", "AIDC", "云", "数据", "软件")),
    ("chip", ("芯片", "半导体", "集成", "电子", "中韩半导体")),
    ("hk_tech", ("恒生科技", "恒科", "港股通科技")),
    ("china_internet", ("互联网", "中概", "港股通互联网")),
    ("healthcare", ("创新药", "医药", "医疗", "生物", "疫苗")),
    ("broker", ("证券", "券商")),
    ("defense", ("军工", "国防", "航天")),
    ("ev", ("新能源车", "电池", "汽车", "智能车", "光伏", "新能源")),
    ("energy_materials", ("有色", "稀土", "煤炭", "能源", "钢铁", "化工")),
    ("consumer", ("消费", "酒", "食品", "家电", "旅游")),
    ("finance", ("银行", "保险", "金融", "地产")),
    ("gold", ("黄金", "金")),
    ("broad_50", ("上证50", "50ETF")),
    ("broad_300", ("沪深300", "300")),
    ("small_mid", ("中证500", "500", "中证1000", "1000", "2000")),
    ("growth", ("创业板", "双创", "科创", "科创板")),
]


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    profile: str
    top_k: int
    rebalance_days: int
    stop_loss: float
    trailing_stop: float
    min_score: float
    min_ret20: float
    max_rsi: float
    cash_reserve: float
    risk_off: bool
    ai_weight: float = 0.0   # 0 = 不用 AI; >0 时 ai_prob 作为打分项


@dataclass
class BacktestResult:
    config: StrategyConfig
    metrics: Dict[str, float]
    equity: pd.DataFrame
    trades: pd.DataFrame


def load_name_map() -> Dict[str, str]:
    if not ETF_LIST_PATH.exists():
        return {}
    try:
        items = json.loads(ETF_LIST_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {str(item.get("code", "")).strip(): str(item.get("name", "")).strip() for item in items}


def is_etf_code(code: str) -> bool:
    return code.startswith(ETF_PREFIXES)


def is_money_or_bond_name(name: str) -> bool:
    return any(term in name for term in MONEY_OR_BOND_TERMS)


def infer_theme(code: str, name: str) -> str:
    if code in CODE_THEME_OVERRIDES:
        return CODE_THEME_OVERRIDES[code]
    clean_name = str(name or "")
    for theme, keywords in THEME_KEYWORDS:
        if any(keyword in clean_name for keyword in keywords):
            return theme
    return "other"


def positive_rate(values: pd.Series) -> float:
    clean = values.dropna()
    if clean.empty:
        return np.nan
    return float((clean > 0.0).mean())


def read_one_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path, parse_dates=["date"])
    except Exception:
        return None
    required = {"date", "open", "close", "high", "low", "volume"}
    if not required.issubset(df.columns):
        return None
    df = df[list(required)].copy()
    df = df.sort_values("date").drop_duplicates("date", keep="last")
    for col in ["open", "close", "high", "low", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date", "open", "close"])
    df = df[(df["open"] > 0) & (df["close"] > 0)]
    if df.empty:
        return None
    return df


def compute_features(df: pd.DataFrame, code: str, name: str, theme: str) -> pd.DataFrame:
    out = df.copy()
    ret1 = out["close"].pct_change()
    out["ret1"] = ret1
    for window in (2, 3, 5, 10, 20, 60, 120):
        out[f"ret{window}"] = out["close"].pct_change(window)
    for window in (20, 50, 120, 200):
        out[f"ma{window}"] = out["close"].rolling(window).mean()
    out["vol20"] = ret1.rolling(20).std()
    out["vol60"] = ret1.rolling(60).std()
    out["amount1"] = out["close"] * out["volume"]
    out["amount3"] = out["amount1"].rolling(3).median()
    out["amount5"] = (out["close"] * out["volume"]).rolling(5).median()
    out["amount20"] = (out["close"] * out["volume"]).rolling(20).median()
    out["amount_surge3"] = out["amount3"] / out["amount20"]
    out["amount_surge"] = out["amount5"] / out["amount20"]
    out["breakout20"] = (out["close"] / out["high"].rolling(20).max()) - 1.0
    out["breakout60"] = (out["close"] / out["high"].rolling(60).max()) - 1.0
    out["breakout120"] = (out["close"] / out["close"].rolling(120).max()) - 1.0
    out["trend_ma50"] = (out["close"] / out["ma50"]) - 1.0
    out["trend_ma120"] = (out["ma50"] / out["ma120"]) - 1.0
    out["overnight_gap"] = (out["open"] / out["close"].shift(1)) - 1.0
    out["intraday_ret"] = (out["close"] / out["open"]) - 1.0
    out["range_pct"] = (out["high"] / out["low"]) - 1.0
    price_range = (out["high"] - out["low"]).replace(0.0, np.nan)
    out["close_pos"] = (out["close"] - out["low"]) / price_range
    out["accel_3_20"] = out["ret3"] - (out["ret20"] / 6.0)
    out["accel_5_20"] = out["ret5"] - (out["ret20"] / 4.0)

    delta = out["close"].diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / loss.replace(0.0, 1e-10)
    out["rsi14"] = 100.0 - (100.0 / (1.0 + rs))

    out["code"] = code
    out["name"] = name
    out["theme"] = theme
    return out


def add_theme_features(panel: pd.DataFrame) -> pd.DataFrame:
    theme_stats = (
        panel.groupby(["date", "theme"], dropna=False)
        .agg(
            theme_ret3=("ret3", "median"),
            theme_ret5=("ret5", "median"),
            theme_ret10=("ret10", "median"),
            theme_ret20=("ret20", "median"),
            theme_ret60=("ret60", "median"),
            theme_breadth5=("ret5", positive_rate),
            theme_breadth20=("ret20", positive_rate),
            theme_amount_surge3=("amount_surge3", "median"),
            theme_amount_surge=("amount_surge", "median"),
            theme_members=("code", "nunique"),
        )
        .reset_index()
    )
    heat = (
        0.25 * theme_stats["theme_ret5"].fillna(0.0)
        + 0.40 * theme_stats["theme_ret20"].fillna(0.0)
        + 0.15 * theme_stats["theme_ret60"].fillna(0.0)
        + 0.10 * (theme_stats["theme_breadth20"].fillna(0.5) - 0.5)
        + 0.10 * (theme_stats["theme_amount_surge"].fillna(1.0) - 1.0).clip(-1.0, 2.0)
    )
    theme_stats["theme_heat"] = heat
    theme_stats["theme_rank"] = theme_stats.groupby("date")["theme_heat"].rank(method="average", pct=True)
    event_heat = (
        0.30 * theme_stats["theme_ret3"].fillna(0.0)
        + 0.25 * theme_stats["theme_ret5"].fillna(0.0)
        + 0.20 * theme_stats["theme_ret10"].fillna(0.0)
        + 0.10 * (theme_stats["theme_breadth5"].fillna(0.5) - 0.5)
        + 0.15 * (theme_stats["theme_amount_surge3"].fillna(1.0) - 1.0).clip(-1.0, 3.0)
    )
    theme_stats["theme_event_heat"] = event_heat
    theme_stats["theme_event_rank"] = theme_stats.groupby("date")["theme_event_heat"].rank(method="average", pct=True)

    out = panel.merge(theme_stats, on=["date", "theme"], how="left")
    out["theme_leader_rank3"] = out.groupby(["date", "theme"])["ret3"].rank(method="average", pct=True)
    out["theme_leader_rank10"] = out.groupby(["date", "theme"])["ret10"].rank(method="average", pct=True)
    out["theme_leader_rank20"] = out.groupby(["date", "theme"])["ret20"].rank(method="average", pct=True)
    out["theme_leader_rank60"] = out.groupby(["date", "theme"])["ret60"].rank(method="average", pct=True)
    return out


def load_nav_history(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        nav = pd.read_csv(path, parse_dates=["date"], dtype={"code": str})
    except Exception:
        return pd.DataFrame()
    required = {"date", "code", "unit_nav"}
    if not required.issubset(nav.columns):
        return pd.DataFrame()
    keep = [
        col
        for col in [
            "date",
            "code",
            "unit_nav",
            "cum_nav",
            "nav_growth_rate",
            "purchase_status",
            "redeem_status",
            "purchase_allowed",
            "redeem_allowed",
            "premium_rate",
        ]
        if col in nav.columns
    ]
    nav = nav[keep].dropna(subset=["date", "code", "unit_nav"])
    nav["code"] = nav["code"].astype(str).str.zfill(6)
    return nav.sort_values(["date", "code"]).drop_duplicates(["date", "code"], keep="last")


def add_nav_features(panel: pd.DataFrame, nav_history: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    if nav_history.empty:
        out["nav_available"] = False
        out["unit_nav"] = np.nan
        out["nav_growth_rate"] = np.nan
        out["premium_rate"] = np.nan
        out["premium_change5"] = np.nan
        out["premium_z60"] = np.nan
        out["purchase_allowed"] = True
        return out

    out = out.merge(nav_history, on=["date", "code"], how="left")
    out["nav_available"] = out["unit_nav"].notna()
    missing_premium = out["premium_rate"].isna() & out["unit_nav"].notna() & (out["unit_nav"] > 0)
    out.loc[missing_premium, "premium_rate"] = out.loc[missing_premium, "close"] / out.loc[missing_premium, "unit_nav"] - 1.0
    out["purchase_allowed"] = out["purchase_allowed"].fillna(True).astype(bool)
    out["nav_growth_rate"] = pd.to_numeric(out["nav_growth_rate"], errors="coerce")
    out["premium_rate"] = pd.to_numeric(out["premium_rate"], errors="coerce")
    out = out.sort_values(["code", "date"])
    out["premium_change5"] = out.groupby("code")["premium_rate"].diff(5)
    premium_mean60 = out.groupby("code")["premium_rate"].transform(lambda x: x.rolling(60, min_periods=20).mean())
    premium_std60 = out.groupby("code")["premium_rate"].transform(lambda x: x.rolling(60, min_periods=20).std())
    out["premium_z60"] = (out["premium_rate"] - premium_mean60) / premium_std60.replace(0, np.nan)
    return out.sort_values(["date", "code"])


def load_feature_panel(
    data_dir: Path,
    name_map: Dict[str, str],
    max_etfs: int,
    min_rows: int,
    min_amount: float,
    nav_history_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[Dict[str, object]], Dict[str, pd.DataFrame]]:
    records: List[Tuple[str, float, int, str, str]] = []
    raw: Dict[str, pd.DataFrame] = {}

    for path in data_dir.glob("*.csv"):
        code = path.stem
        if not is_etf_code(code):
            continue
        df = read_one_csv(path)
        if df is None or len(df) < min_rows:
            continue
        name = name_map.get(code, "")
        amount = float((df["close"] * df["volume"]).tail(250).median())
        if amount < min_amount:
            continue
        raw[code] = df
        records.append((code, amount, len(df), str(df["date"].iloc[0].date()), str(df["date"].iloc[-1].date())))

    records.sort(key=lambda item: item[1], reverse=True)

    selected_codes: List[str] = []
    for code, _, _, _, _ in records:
        name = name_map.get(code, "")
        if is_money_or_bond_name(name) and code not in DEFENSIVE_CANDIDATES:
            continue
        selected_codes.append(code)
        if len(selected_codes) >= max_etfs:
            break

    for code in DEFENSIVE_CANDIDATES + ("510300", "159915"):
        if code in raw and code not in selected_codes:
            selected_codes.append(code)

    frames = []
    universe_rows = []
    for code in selected_codes:
        df = raw[code]
        name = name_map.get(code, "")
        theme = infer_theme(code, name)
        frames.append(compute_features(df, code, name, theme))
        universe_rows.append(
            {
                "code": code,
                "name": name,
                "theme": theme,
                "rows": int(len(df)),
                "start": str(df["date"].iloc[0].date()),
                "end": str(df["date"].iloc[-1].date()),
                "median_amount_250d": float((df["close"] * df["volume"]).tail(250).median()),
            }
        )

    if not frames:
        raise RuntimeError("No ETF data matched the universe filters.")

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["date", "code"])
    panel = add_theme_features(panel)
    nav_history = load_nav_history(nav_history_path) if nav_history_path else pd.DataFrame()
    panel = add_nav_features(panel, nav_history)
    open_wide = panel.pivot(index="date", columns="code", values="open").sort_index()
    close_wide = panel.pivot(index="date", columns="code", values="close").sort_index()
    close_wide = close_wide.ffill()
    panel = panel.set_index(["date", "code"]).sort_index()
    # stock_data_map: code -> 原始 OHLCV DataFrame, 供 AISignalBridge 训练/预测
    stock_data_map: Dict[str, pd.DataFrame] = {code: raw[code] for code in selected_codes}
    return panel, open_wide, close_wide, universe_rows, stock_data_map


def build_configs(grid: str, ai_weights: Sequence[float] = (0.0,)) -> List[StrategyConfig]:
    profiles = ["burst", "swing", "trend", "theme_burst", "theme_swing", "nav_theme"]
    top_ks = [1, 2, 3]
    rebalances = [3, 5, 10]
    stops = [0.06, 0.08]
    risk_flags = [True, False] if grid == "full" else [True]
    if grid == "quick":
        profiles = ["burst", "swing", "theme_burst", "nav_theme"]
        rebalances = [3, 5]
    elif grid == "event":
        profiles = ["event_burst", "theme_event"]
        top_ks = [1, 2, 3]
        rebalances = [1, 2, 3]
        stops = [0.04, 0.06, 0.08]
        risk_flags = [True, False]
    # 规范化 ai_weights (去重保序, 至少含 0.0 作为 baseline)
    weights = tuple(float(w) for w in ai_weights) if ai_weights else (0.0,)
    configs: List[StrategyConfig] = []
    for profile in profiles:
        for top_k in top_ks:
            for rebalance in rebalances:
                for stop in stops:
                    for risk_off in risk_flags:
                        for ai_weight in weights:
                            ai_suffix = f"_ai{int(ai_weight * 100)}" if ai_weight > 0 else ""
                            configs.append(
                                StrategyConfig(
                                    name=f"{profile}_k{top_k}_r{rebalance}_s{int(stop*100)}_{'ro' if risk_off else 'ri'}{ai_suffix}",
                                    profile=profile,
                                    top_k=top_k,
                                    rebalance_days=rebalance,
                                    stop_loss=stop,
                                    trailing_stop=0.12,
                                    min_score=0.0,
                                    min_ret20=0.0,
                                    max_rsi=82.0,
                                    cash_reserve=0.05,
                                    risk_off=risk_off,
                                    ai_weight=ai_weight,
                                )
                            )
    return configs


def score_candidates(day_panel: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    df = day_panel.copy()
    if cfg.profile == "burst":
        score = 0.15 * df["ret5"] + 0.55 * df["ret20"] + 0.25 * df["ret60"] + 0.05 * df["ret120"] - 0.40 * df["vol20"]
    elif cfg.profile == "event_burst":
        amount_impulse = (df["amount_surge3"].fillna(1.0) - 1.0).clip(-0.8, 4.0)
        close_strength = (df["close_pos"].fillna(0.5) - 0.5).clip(-0.5, 0.5)
        score = (
            0.22 * df["ret2"].fillna(0.0)
            + 0.28 * df["ret3"].fillna(0.0)
            + 0.22 * df["ret5"].fillna(0.0)
            + 0.10 * df["ret10"].fillna(0.0)
            + 0.12 * df["accel_3_20"].fillna(0.0)
            + 0.08 * df["intraday_ret"].fillna(0.0)
            + 0.05 * amount_impulse
            + 0.04 * close_strength
            + 0.06 * df["breakout20"].fillna(0.0).clip(-0.15, 0.0)
        )
    elif cfg.profile == "theme_event":
        amount_impulse = (df["amount_surge3"].fillna(1.0) - 1.0).clip(-0.8, 4.0)
        close_strength = (df["close_pos"].fillna(0.5) - 0.5).clip(-0.5, 0.5)
        score = (
            0.16 * df["ret2"].fillna(0.0)
            + 0.22 * df["ret3"].fillna(0.0)
            + 0.18 * df["ret5"].fillna(0.0)
            + 0.10 * df["ret10"].fillna(0.0)
            + 0.08 * df["accel_3_20"].fillna(0.0)
            + 0.08 * df["intraday_ret"].fillna(0.0)
            + 0.04 * amount_impulse
            + 0.04 * close_strength
            + 0.28 * df["theme_event_heat"].fillna(0.0)
            + 0.08 * (df["theme_event_rank"].fillna(0.5) - 0.5)
            + 0.06 * (df["theme_leader_rank3"].fillna(0.5) - 0.5)
        )
    elif cfg.profile == "theme_burst":
        score = (
            0.10 * df["ret5"]
            + 0.35 * df["ret20"]
            + 0.18 * df["ret60"]
            - 0.25 * df["vol20"]
            + 0.35 * df["theme_heat"].fillna(0.0)
            + 0.10 * (df["theme_rank"].fillna(0.5) - 0.5)
            + 0.06 * (df["theme_leader_rank20"].fillna(0.5) - 0.5)
        )
    elif cfg.profile == "theme_swing":
        score = (
            0.08 * df["ret5"]
            + 0.25 * df["ret20"]
            + 0.28 * df["ret60"]
            + 0.08 * df["ret120"]
            - 0.18 * df["vol20"]
            + 0.30 * df["theme_heat"].fillna(0.0)
            + 0.08 * (df["theme_rank"].fillna(0.5) - 0.5)
            + 0.06 * (df["theme_leader_rank60"].fillna(0.5) - 0.5)
        )
    elif cfg.profile == "nav_theme":
        premium_penalty = df["premium_rate"].fillna(0.0).clip(lower=0.0, upper=0.15)
        premium_momentum = df["premium_change5"].fillna(0.0).clip(-0.08, 0.08)
        premium_z = df["premium_z60"].fillna(0.0).clip(-3.0, 3.0)
        score = (
            0.08 * df["ret5"]
            + 0.24 * df["ret20"]
            + 0.20 * df["ret60"]
            - 0.20 * df["vol20"]
            + 0.25 * df["theme_heat"].fillna(0.0)
            + 0.16 * df["nav_growth_rate"].fillna(0.0).clip(-0.10, 0.10)
            + 0.08 * premium_momentum
            + 0.03 * premium_z
            - 0.18 * premium_penalty
        )
    elif cfg.profile == "trend":
        score = 0.05 * df["ret5"] + 0.20 * df["ret20"] + 0.45 * df["ret60"] + 0.30 * df["ret120"] - 0.18 * df["vol20"]
    else:
        score = 0.08 * df["ret5"] + 0.35 * df["ret20"] + 0.42 * df["ret60"] + 0.15 * df["ret120"] - 0.25 * df["vol20"]
    score = score + 0.05 * df["trend_ma50"].clip(-0.2, 0.2) + 0.05 * df["trend_ma120"].clip(-0.2, 0.2)
    score = score + 0.04 * df["breakout120"].fillna(0.0).clip(-0.3, 0.0)
    # AI 信号注入: ai_weight>0 且 panel 含 ai_prob 列时, 作为打分项 (减 0.5 保持中性)
    if cfg.ai_weight > 0 and "ai_prob" in df.columns:
        score = score + cfg.ai_weight * (df["ai_prob"].fillna(0.5) - 0.5)
    df["score"] = score
    valid = df["score"].notna() & (df["score"] > cfg.min_score) & (df["rsi14"] < cfg.max_rsi) & (df["amount20"].notna())
    if cfg.profile in ("event_burst", "theme_event"):
        valid = (
            valid
            & (df["ret3"] > 0.0)
            & (df["ret10"] > -0.03)
            & (df["close"] > df["ma20"] * 0.98)
            & (df["amount_surge3"].fillna(0.0) > 0.9)
            & (df["close_pos"].fillna(0.0) > 0.45)
        )
    else:
        valid = valid & (df["ret20"] > cfg.min_ret20) & (df["close"] > df["ma20"])
    if cfg.profile.startswith("theme_"):
        valid = valid & (df["theme_members"].fillna(0) >= 2) & df["theme_heat"].notna()
    if cfg.profile == "theme_event":
        valid = valid & (df["theme_members"].fillna(0) >= 2) & df["theme_event_heat"].notna()
    if cfg.profile == "nav_theme":
        valid = (
            valid
            & df["nav_available"].fillna(False)
            & df["nav_growth_rate"].notna()
            & df["premium_rate"].notna()
            & df["purchase_allowed"].fillna(True)
            & (df["theme_members"].fillna(0) >= 1)
        )
    return df.loc[valid].sort_values("score", ascending=False)


def market_is_risk_off(day_panel: pd.DataFrame) -> bool:
    if "510300" not in day_panel.index:
        return False
    row = day_panel.loc["510300"]
    if row[["close", "ma120", "ma200", "ret20"]].isna().any():
        return False
    return bool((row["close"] < row["ma120"] and row["ret20"] < 0.0) or (row["close"] < row["ma200"] and row["ret60"] < 0.0))


def pick_defensive(day_panel: pd.DataFrame) -> List[str]:
    choices = []
    for code in DEFENSIVE_CANDIDATES:
        if code not in day_panel.index:
            continue
        row = day_panel.loc[code]
        if pd.isna(row.get("ret20", np.nan)) or pd.isna(row.get("ma50", np.nan)):
            continue
        score = float(row.get("ret20", 0.0)) - float(row.get("vol20", 0.0))
        if row["close"] > row["ma50"] or code in ("159001", "159003", "159005"):
            choices.append((code, score))
    choices.sort(key=lambda item: item[1], reverse=True)
    return [choices[0][0]] if choices else []


def select_targets(panel: pd.DataFrame, signal_date: pd.Timestamp, cfg: StrategyConfig) -> Tuple[List[str], str]:
    try:
        day_panel = panel.loc[signal_date]
    except KeyError:
        return [], "no_panel"

    if cfg.risk_off and market_is_risk_off(day_panel):
        defensive = pick_defensive(day_panel)
        return defensive, "risk_off_defensive" if defensive else "risk_off_cash"

    candidates = score_candidates(day_panel, cfg)
    if candidates.empty:
        return [], "no_candidates"
    if cfg.profile == "nav_theme":
        reason = "nav_theme"
    else:
        reason = "theme_momentum" if cfg.profile.startswith("theme_") else "momentum"
    return list(candidates.head(cfg.top_k).index), reason


def price_for(codes: Iterable[str], prices: pd.Series) -> Dict[str, float]:
    out = {}
    for code in codes:
        value = prices.get(code, np.nan)
        if pd.notna(value) and value > 0:
            out[code] = float(value)
    return out


def execute_target(
    date: pd.Timestamp,
    target_codes: Sequence[str],
    positions: Dict[str, Dict[str, object]],
    cash: float,
    open_wide: pd.DataFrame,
    close_wide: pd.DataFrame,
    fee_rate: float,
    cash_reserve: float,
    trades: List[Dict[str, object]],
    reason: str,
) -> Tuple[float, Dict[str, Dict[str, object]]]:
    open_row = open_wide.loc[date] if date in open_wide.index else pd.Series(dtype=float)
    close_row = close_wide.loc[date] if date in close_wide.index else pd.Series(dtype=float)
    target_set = set(target_codes)

    current_codes = list(positions.keys())
    sell_codes = [code for code in current_codes if code not in target_set]
    for code in sell_codes:
        price = open_row.get(code, np.nan)
        if pd.isna(price) or price <= 0:
            price = close_row.get(code, np.nan)
        if pd.isna(price) or price <= 0:
            continue
        qty = int(positions[code]["qty"])
        gross = qty * float(price)
        fee = gross * fee_rate
        cash += gross - fee
        trades.append({"date": date, "code": code, "action": "SELL", "qty": qty, "price": float(price), "fee": fee, "reason": reason})
        del positions[code]

    if not target_codes:
        return cash, positions

    priced_codes = [code for code in target_codes if pd.notna(open_row.get(code, np.nan)) and open_row.get(code, np.nan) > 0]
    if not priced_codes:
        return cash, positions

    equity_open = cash
    for code, pos in positions.items():
        price = open_row.get(code, np.nan)
        if pd.isna(price) or price <= 0:
            price = close_row.get(code, np.nan)
        if pd.notna(price) and price > 0:
            equity_open += int(pos["qty"]) * float(price)

    target_value = equity_open * (1.0 - cash_reserve) / max(len(priced_codes), 1)

    for code in priced_codes:
        price = float(open_row[code])
        current_qty = int(positions.get(code, {}).get("qty", 0))
        current_value = current_qty * price
        diff_value = target_value - current_value
        if diff_value < -price * 100:
            sell_qty = int(abs(diff_value) / price / 100) * 100
            sell_qty = min(sell_qty, current_qty)
            if sell_qty > 0:
                gross = sell_qty * price
                fee = gross * fee_rate
                cash += gross - fee
                positions[code]["qty"] = current_qty - sell_qty
                trades.append({"date": date, "code": code, "action": "TRIM", "qty": sell_qty, "price": price, "fee": fee, "reason": reason})
                if positions[code]["qty"] <= 0:
                    del positions[code]
        elif diff_value > price * 100:
            buy_qty = int(diff_value / (price * (1.0 + fee_rate)) / 100) * 100
            affordable_qty = int(cash / (price * (1.0 + fee_rate)) / 100) * 100
            buy_qty = min(buy_qty, affordable_qty)
            if buy_qty > 0:
                gross = buy_qty * price
                fee = gross * fee_rate
                cash -= gross + fee
                if code in positions:
                    old_qty = int(positions[code]["qty"])
                    old_entry = float(positions[code]["entry_price"])
                    new_qty = old_qty + buy_qty
                    positions[code]["entry_price"] = (old_entry * old_qty + gross) / max(new_qty, 1)
                    positions[code]["qty"] = new_qty
                else:
                    positions[code] = {"qty": buy_qty, "entry_price": price, "entry_date": str(date.date()), "peak_close": price}
                trades.append({"date": date, "code": code, "action": "BUY", "qty": buy_qty, "price": price, "fee": fee, "reason": reason})

    return cash, positions


def run_backtest(
    panel: pd.DataFrame,
    open_wide: pd.DataFrame,
    close_wide: pd.DataFrame,
    cfg: StrategyConfig,
    start: pd.Timestamp,
    end: pd.Timestamp,
    initial_capital: float,
    fee_bps: float,
) -> BacktestResult:
    dates = close_wide.index[(close_wide.index >= start) & (close_wide.index <= end)]
    dates = pd.DatetimeIndex(sorted(dates))
    if len(dates) < 20:
        return BacktestResult(cfg, empty_metrics(initial_capital), pd.DataFrame(), pd.DataFrame())

    fee_rate = fee_bps / 10000.0
    cash = float(initial_capital)
    positions: Dict[str, Dict[str, object]] = {}
    trades: List[Dict[str, object]] = []
    equity_rows: List[Dict[str, object]] = []
    pending_target: Optional[Tuple[List[str], str]] = None
    last_rebalance_i = -cfg.rebalance_days

    for i, date in enumerate(dates):
        if pending_target is not None:
            target_codes, reason = pending_target
            cash, positions = execute_target(
                date, target_codes, positions, cash, open_wide, close_wide, fee_rate, cfg.cash_reserve, trades, reason
            )
            pending_target = None

        close_row = close_wide.loc[date]
        equity = cash
        stopped: List[str] = []
        for code, pos in list(positions.items()):
            close = close_row.get(code, np.nan)
            if pd.isna(close) or close <= 0:
                continue
            close = float(close)
            pos["peak_close"] = max(float(pos.get("peak_close", close)), close)
            qty = int(pos["qty"])
            equity += qty * close
            fixed_stop = close <= float(pos["entry_price"]) * (1.0 - cfg.stop_loss)
            trailing_stop = close <= float(pos["peak_close"]) * (1.0 - cfg.trailing_stop)
            if fixed_stop or trailing_stop:
                stopped.append(code)

        equity_rows.append({"date": date, "equity": equity, "cash": cash, "positions": len(positions)})

        if i >= len(dates) - 1:
            continue

        should_rebalance = (i - last_rebalance_i) >= cfg.rebalance_days
        if should_rebalance:
            target, reason = select_targets(panel, date, cfg)
            pending_target = (target, reason)
            last_rebalance_i = i
        elif stopped:
            target = [code for code in positions if code not in stopped]
            pending_target = (target, "stop_signal")

    equity_df = pd.DataFrame(equity_rows)
    trades_df = pd.DataFrame(trades)
    metrics = compute_metrics(equity_df, trades_df, initial_capital)
    return BacktestResult(cfg, metrics, equity_df, trades_df)


def empty_metrics(initial_capital: float) -> Dict[str, float]:
    return {
        "start_equity": initial_capital,
        "final_equity": initial_capital,
        "total_return": 0.0,
        "cagr": 0.0,
        "max_drawdown": 0.0,
        "sharpe": 0.0,
        "calmar": 0.0,
        "trade_count": 0,
        "monthly_mean": 0.0,
        "monthly_median": 0.0,
        "monthly_min": 0.0,
        "monthly_max": 0.0,
        "positive_month_rate": 0.0,
        "months_ge_30_rate": 0.0,
        "months_ge_100cagr_rate": 0.0,
        "objective_cagr_100": 0.0,
        "objective_all_months_30": 0.0,
    }


def compute_metrics(equity_df: pd.DataFrame, trades_df: pd.DataFrame, initial_capital: float) -> Dict[str, float]:
    if equity_df.empty:
        return empty_metrics(initial_capital)

    df = equity_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    equity = df["equity"].astype(float)
    daily_ret = equity.pct_change().fillna(0.0)
    total_return = float(equity.iloc[-1] / initial_capital - 1.0)
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1.0 / 365.25)
    cagr = float((equity.iloc[-1] / initial_capital) ** (1.0 / years) - 1.0) if equity.iloc[-1] > 0 else -1.0
    drawdown = equity / equity.cummax() - 1.0
    max_dd = float(drawdown.min())
    std = float(daily_ret.std())
    sharpe = float(daily_ret.mean() / std * math.sqrt(252.0)) if std > 0 else 0.0
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else 0.0
    monthly = (1.0 + daily_ret).resample("M").prod() - 1.0
    if len(monthly) > 0:
        monthly_mean = float(monthly.mean())
        monthly_median = float(monthly.median())
        monthly_min = float(monthly.min())
        monthly_max = float(monthly.max())
        positive_month_rate = float((monthly > 0).mean())
        months_ge_30_rate = float((monthly >= 0.30).mean())
        months_ge_100cagr_rate = float((monthly >= MONTHLY_FOR_100_CAGR).mean())
        objective_all_months_30 = float(bool((monthly >= 0.30).all()))
    else:
        monthly_mean = monthly_median = monthly_min = monthly_max = 0.0
        positive_month_rate = months_ge_30_rate = months_ge_100cagr_rate = objective_all_months_30 = 0.0

    return {
        "start_equity": float(initial_capital),
        "final_equity": float(equity.iloc[-1]),
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "calmar": calmar,
        "trade_count": int(len(trades_df)) if trades_df is not None else 0,
        "monthly_mean": monthly_mean,
        "monthly_median": monthly_median,
        "monthly_min": monthly_min,
        "monthly_max": monthly_max,
        "positive_month_rate": positive_month_rate,
        "months_ge_30_rate": months_ge_30_rate,
        "months_ge_100cagr_rate": months_ge_100cagr_rate,
        "objective_cagr_100": float(cagr >= 1.0),
        "objective_all_months_30": objective_all_months_30,
    }


def search_score(metrics: Dict[str, float]) -> float:
    return (
        metrics["cagr"]
        + 0.15 * metrics["sharpe"]
        + 0.40 * metrics["months_ge_100cagr_rate"]
        + 0.25 * metrics["months_ge_30_rate"]
        - 1.25 * abs(metrics["max_drawdown"])
        - 0.75 * max(0.0, -metrics["monthly_min"])
    )


def next_trading_date(dates: pd.DatetimeIndex, value: pd.Timestamp) -> Optional[pd.Timestamp]:
    later = dates[dates >= value]
    return later[0] if len(later) else None


def previous_trading_date(dates: pd.DatetimeIndex, value: pd.Timestamp) -> Optional[pd.Timestamp]:
    earlier = dates[dates <= value]
    return earlier[-1] if len(earlier) else None


def make_folds(
    dates: pd.DatetimeIndex,
    start: pd.Timestamp,
    end: pd.Timestamp,
    train_months: int,
    test_months: int,
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    folds = []
    train_start_raw = start
    while True:
        train_end_raw = train_start_raw + pd.DateOffset(months=train_months) - pd.Timedelta(days=1)
        test_start_raw = train_end_raw + pd.Timedelta(days=1)
        test_end_raw = test_start_raw + pd.DateOffset(months=test_months) - pd.Timedelta(days=1)
        if test_start_raw > end:
            break
        if test_end_raw > end:
            test_end_raw = end
        train_start = next_trading_date(dates, train_start_raw)
        train_end = previous_trading_date(dates, train_end_raw)
        test_start = next_trading_date(dates, test_start_raw)
        test_end = previous_trading_date(dates, test_end_raw)
        if train_start is None or train_end is None or test_start is None or test_end is None:
            break
        if train_start < train_end < test_start <= test_end:
            folds.append((train_start, train_end, test_start, test_end))
        if test_end_raw >= end:
            break
        train_start_raw = train_start_raw + pd.DateOffset(months=test_months)
    return folds


def run_walk_forward(
    panel: pd.DataFrame,
    open_wide: pd.DataFrame,
    close_wide: pd.DataFrame,
    configs: Sequence[StrategyConfig],
    start: pd.Timestamp,
    end: pd.Timestamp,
    train_months: int,
    test_months: int,
    initial_capital: float,
    fee_bps: float,
    bridge: Optional["AISignalBridge"] = None,
    stock_data_map: Optional[Dict[str, pd.DataFrame]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[Dict[str, object]]]:
    dates = close_wide.index[(close_wide.index >= start) & (close_wide.index <= end)]
    folds = make_folds(pd.DatetimeIndex(dates), start, end, train_months, test_months)
    if not folds:
        raise RuntimeError("No walk-forward folds could be created from the requested dates.")

    fold_rows: List[Dict[str, object]] = []
    grid_rows: List[Dict[str, object]] = []
    equity_segments: List[pd.DataFrame] = []
    trade_segments: List[pd.DataFrame] = []
    capital = initial_capital

    use_ai = bridge is not None and stock_data_map is not None
    if use_ai:
        print(f"[AI] walk-forward 启用 AI 信号: {len(configs)} configs (含 ai_weight 网格)")

    for fold_id, (train_start, train_end, test_start, test_end) in enumerate(folds, start=1):
        # === AI 引擎训练 + 预测 (每 fold 一次, 严格遵守时序) ===
        fold_panel = panel
        if use_ai:
            print(f"[AI] fold {fold_id}/{len(folds)}: 训练引擎 (as_of={train_end.date()})")
            bridge.fit(stock_data_map, train_end)
            # 产出该 fold 覆盖期 [train_start, test_end] 的 ai_prob
            fold_dates = [d for d in dates if train_start <= d <= test_end]
            ai_prob_df = bridge.predict_proba(stock_data_map, fold_dates)
            if not ai_prob_df.empty:
                fold_panel = inject_ai_prob(panel, ai_prob_df)
                print(f"[AI] fold {fold_id}: 注入 ai_prob {len(ai_prob_df)} 行 ({ai_prob_df['ai_prob'].min():.3f}~{ai_prob_df['ai_prob'].max():.3f})")
            else:
                print(f"[AI] fold {fold_id}: ai_prob 为空, 该 fold 退化无 AI")

        best_result: Optional[BacktestResult] = None
        best_score = -1e9
        for cfg in configs:
            result = run_backtest(fold_panel, open_wide, close_wide, cfg, train_start, train_end, initial_capital, fee_bps)
            score = search_score(result.metrics)
            grid_rows.append(
                {
                    "fold": fold_id,
                    "phase": "train",
                    "config": cfg.name,
                    "ai_weight": cfg.ai_weight,
                    "search_score": score,
                    **result.metrics,
                }
            )
            if score > best_score:
                best_score = score
                best_result = result
        if best_result is None:
            continue

        test_result = run_backtest(fold_panel, open_wide, close_wide, best_result.config, test_start, test_end, capital, fee_bps)
        capital = float(test_result.metrics["final_equity"])

        fold_rows.append(
            {
                "fold": fold_id,
                "train_start": str(train_start.date()),
                "train_end": str(train_end.date()),
                "test_start": str(test_start.date()),
                "test_end": str(test_end.date()),
                "selected_config": best_result.config.name,
                "train_search_score": best_score,
                **{f"train_{k}": v for k, v in best_result.metrics.items()},
                **{f"test_{k}": v for k, v in test_result.metrics.items()},
            }
        )

        if not test_result.equity.empty:
            seg = test_result.equity.copy()
            seg["fold"] = fold_id
            seg["config"] = best_result.config.name
            equity_segments.append(seg)
        if not test_result.trades.empty:
            trades = test_result.trades.copy()
            trades["fold"] = fold_id
            trades["config"] = best_result.config.name
            trade_segments.append(trades)

    folds_df = pd.DataFrame(fold_rows)
    grid_df = pd.DataFrame(grid_rows)
    equity_df = pd.concat(equity_segments, ignore_index=True) if equity_segments else pd.DataFrame()
    trades_df = pd.concat(trade_segments, ignore_index=True) if trade_segments else pd.DataFrame()
    return folds_df, grid_df, equity_df, trades_df


def summarize_walk_forward(
    folds_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    args,
    universe_rows,
    panel: pd.DataFrame,
) -> Dict[str, object]:
    if equity_df.empty:
        stitched_metrics = empty_metrics(args.initial_capital)
    else:
        stitched = equity_df[["date", "equity", "cash", "positions"]].copy()
        stitched_metrics = compute_metrics(stitched, trades_df, args.initial_capital)

    test_cols = [col for col in folds_df.columns if col.startswith("test_")]
    fold_metric_means = {}
    for col in test_cols:
        if pd.api.types.is_numeric_dtype(folds_df[col]):
            fold_metric_means[col] = float(folds_df[col].mean())

    latest_signal = {}
    if not folds_df.empty:
        latest_signal = {
            "last_fold_config": str(folds_df.iloc[-1]["selected_config"]),
            "last_test_end": str(folds_df.iloc[-1]["test_end"]),
        }
    selected_config_counts = (
        folds_df["selected_config"].value_counts().to_dict() if "selected_config" in folds_df.columns else {}
    )
    profile_counts = {}
    for config_name, count in selected_config_counts.items():
        profile = str(config_name).split("_k", 1)[0]
        profile_counts[profile] = profile_counts.get(profile, 0) + int(count)

    trade_reason_counts = trades_df["reason"].value_counts().to_dict() if "reason" in trades_df.columns else {}
    theme_by_code = {row["code"]: row.get("theme", "unknown") for row in universe_rows}
    traded_theme_counts = {}
    if "code" in trades_df.columns:
        traded_themes = trades_df["code"].map(theme_by_code).fillna("unknown")
        traded_theme_counts = traded_themes.value_counts().head(20).to_dict()

    if "nav_available" in panel.columns:
        panel_indexed = panel.reset_index()
        nav_rows = int(panel_indexed["nav_available"].fillna(False).sum())
        nav_codes = int(panel_indexed.loc[panel_indexed["nav_available"].fillna(False), "code"].nunique())
    else:
        nav_rows = 0
        nav_codes = 0

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "script": str(Path(__file__).relative_to(ROOT)),
        "data_dir": str(DATA_DIR),
        "assumptions": {
            "execution": "signals use close data; trades execute at next trading day's open",
            "lot_size": 100,
            "fee_bps_one_way_model": args.fee_bps,
            "initial_capital": args.initial_capital,
            "survivorship_bias_warning": "universe is built from the current local cache, not a point-in-time ETF master",
            "news_warning": "no live news feed is consumed; this run uses price/volume and ETF name metadata only",
            "theme_signal": "ETF names are mapped to theme clusters; per-date theme heat uses only same-day historical price/volume features",
        },
        "params": vars(args),
        "universe": {
            "selected_count": len(universe_rows),
            "sample": universe_rows[:20],
            "nav_history_rows": nav_rows,
            "nav_history_codes": nav_codes,
        },
        "walk_forward": {
            "fold_count": int(len(folds_df)),
            "stitched_metrics": stitched_metrics,
            "fold_metric_means": fold_metric_means,
            "latest_signal_context": latest_signal,
            "selected_config_counts": selected_config_counts,
            "selected_profile_counts": profile_counts,
            "trade_reason_counts": trade_reason_counts,
            "top_traded_theme_counts": traded_theme_counts,
            "objective": {
                "annual_100pct_pass": bool(stitched_metrics.get("objective_cagr_100", 0.0)),
                "all_months_ge_30pct_pass": bool(stitched_metrics.get("objective_all_months_30", 0.0)),
                "monthly_return_needed_for_100pct_cagr": MONTHLY_FOR_100_CAGR,
            },
        },
    }


def report_paths(prefix: str) -> Dict[str, Path]:
    clean_prefix = str(prefix or "aggressive_walkforward").strip()
    clean_prefix = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in clean_prefix)
    if not clean_prefix:
        clean_prefix = "aggressive_walkforward"
    return {
        "summary": REPORT_DIR / f"{clean_prefix}_summary.json",
        "folds": REPORT_DIR / f"{clean_prefix}_folds.csv",
        "grid": REPORT_DIR / f"{clean_prefix}_grid.csv",
        "equity": REPORT_DIR / f"{clean_prefix}_equity.csv",
        "trades": REPORT_DIR / f"{clean_prefix}_trades.csv",
    }


def save_outputs(
    summary: Dict[str, object],
    folds_df: pd.DataFrame,
    grid_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    prefix: str,
) -> Dict[str, Path]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    paths = report_paths(prefix)
    paths["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    folds_df.to_csv(paths["folds"], index=False, encoding="utf-8-sig")
    grid_df.to_csv(paths["grid"], index=False, encoding="utf-8-sig")
    equity_df.to_csv(paths["equity"], index=False, encoding="utf-8-sig")
    trades_df.to_csv(paths["trades"], index=False, encoding="utf-8-sig")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggressive ETF walk-forward research harness")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default=None, help="default: latest date available in selected universe")
    parser.add_argument("--train-months", type=int, default=24)
    parser.add_argument("--test-months", type=int, default=6)
    parser.add_argument("--initial-capital", type=float, default=100000.0)
    parser.add_argument("--fee-bps", type=float, default=5.0, help="one-way fee + slippage in basis points")
    parser.add_argument("--max-etfs", type=int, default=180)
    parser.add_argument("--min-rows", type=int, default=500)
    parser.add_argument("--min-amount", type=float, default=0.0)
    parser.add_argument("--nav-history", default=str(NAV_HISTORY_PATH), help="dated ETF NAV history CSV; empty string disables NAV features")
    parser.add_argument("--grid", choices=["quick", "default", "full", "event"], default="quick")
    parser.add_argument("--output-prefix", default="aggressive_walkforward", help="report filename prefix under reports/")
    # AI 信号开关 (默认关闭, 开启后 DL/SEQ 引擎接入 walk-forward)
    parser.add_argument("--use-ai", action="store_true", help="enable DL/SEQ AI signal bridge")
    parser.add_argument("--ai-weights", default="0,0.1,0.2,0.3",
                        help="comma-separated ai_weight grid (only with --use-ai)")
    parser.add_argument("--dl-epochs", type=int, default=20)
    parser.add_argument("--seq-epochs", type=int, default=12)
    parser.add_argument("--seq-mode", choices=["lstm", "transformer", "ensemble"], default="ensemble")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    name_map = load_name_map()
    nav_history_path = Path(args.nav_history) if args.nav_history else None
    panel, open_wide, close_wide, universe_rows, stock_data_map = load_feature_panel(
        DATA_DIR,
        name_map,
        max_etfs=args.max_etfs,
        min_rows=args.min_rows,
        min_amount=args.min_amount,
        nav_history_path=nav_history_path,
    )
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end) if args.end else pd.Timestamp(close_wide.index.max())

    # AI 信号桥接: --use-ai 启用时构造 bridge, 否则 None (ai_weight 全 0, 等价于无 AI)
    bridge = None
    ai_weights: Tuple[float, ...] = (0.0,)
    if args.use_ai:
        if AISignalBridge is None:
            raise RuntimeError("--use-ai 需要 backtest.ai_signal_bridge 模块 (检查 models.deep_learning 依赖)")
        bridge = AISignalBridge(
            dl_epochs=args.dl_epochs,
            seq_epochs=args.seq_epochs,
            seq_mode=args.seq_mode,
            verbose=True,
        )
        ai_weights = tuple(float(w) for w in args.ai_weights.split(",") if w.strip() != "")
        if 0.0 not in ai_weights:
            ai_weights = (0.0,) + ai_weights  # 始终含 baseline
        print(f"[AI] DL/SEQ bridge 已启用: ai_weights={ai_weights}, seq_mode={args.seq_mode}")

    configs = build_configs(args.grid, ai_weights)

    print("=" * 72)
    print("Aggressive ETF walk-forward research")
    print("=" * 72)
    nav_codes = int(panel.reset_index().loc[panel.reset_index()["nav_available"].fillna(False), "code"].nunique()) if "nav_available" in panel.columns else 0
    print(f"Universe: {len(universe_rows)} ETFs | configs: {len(configs)} | NAV codes: {nav_codes} | period: {start.date()} -> {end.date()}")
    print(f"Train/test: {args.train_months}m/{args.test_months}m | capital: {args.initial_capital:,.0f} | fee_bps: {args.fee_bps}")
    if args.use_ai:
        print(f"AI: enabled | ai_weights={ai_weights} | dl_epochs={args.dl_epochs} | seq_epochs={args.seq_epochs} | seq_mode={args.seq_mode}")

    folds_df, grid_df, equity_df, trades_df = run_walk_forward(
        panel,
        open_wide,
        close_wide,
        configs,
        start,
        end,
        args.train_months,
        args.test_months,
        args.initial_capital,
        args.fee_bps,
        bridge=bridge,
        stock_data_map=stock_data_map,
    )
    summary = summarize_walk_forward(folds_df, equity_df, trades_df, args, universe_rows, panel)
    output_paths = save_outputs(summary, folds_df, grid_df, equity_df, trades_df, args.output_prefix)

    stitched = summary["walk_forward"]["stitched_metrics"]
    print("-" * 72)
    print(f"Folds: {summary['walk_forward']['fold_count']}")
    print(
        "OOS stitched: "
        f"CAGR={stitched['cagr']*100:.2f}% | "
        f"Total={stitched['total_return']*100:.2f}% | "
        f"MaxDD={stitched['max_drawdown']*100:.2f}% | "
        f"Sharpe={stitched['sharpe']:.2f} | "
        f"Monthly>=30%={stitched['months_ge_30_rate']*100:.1f}%"
    )
    print("Outputs:")
    for path in output_paths.values():
        print(f"  {path}")


if __name__ == "__main__":
    main()
