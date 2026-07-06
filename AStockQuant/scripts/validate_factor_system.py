# -*- coding: utf-8 -*-
"""Validate the 8-layer factor system on cached ETF data.

The script is intentionally offline-first: it validates factor correctness on
`data_cache/*.csv` and disables Layer4 network calls by default so a full batch
audit is deterministic and does not depend on transient AkShare API behavior.

Use `--enable-live-capital` for a small live Layer4 smoke test.
"""

from __future__ import annotations

import argparse
import os
import py_compile
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def compile_core_files() -> None:
    files = [
        ROOT / "layers" / "layer1_macro.py",
        ROOT / "layers" / "layer2_rules.py",
        ROOT / "layers" / "layer3_sector.py",
        ROOT / "layers" / "layer4_capital.py",
        ROOT / "layers" / "layer5_sentiment.py",
        ROOT / "layers" / "layer6_price_vol.py",
        ROOT / "layers" / "layer7_technical.py",
        ROOT / "layers" / "layer8_micro.py",
        ROOT / "core" / "feature_registry.py",
    ]
    for path in files:
        py_compile.compile(str(path), doraise=True)


def load_cached_data(min_rows: int = 120) -> Tuple[Dict[str, pd.DataFrame], Dict[str, int]]:
    data_dir = ROOT / "data_cache"
    all_data: Dict[str, pd.DataFrame] = {}
    invalid: Dict[str, int] = {}
    for path in sorted(data_dir.glob("*.csv")):
        symbol = path.stem
        try:
            df = pd.read_csv(path)
            if "date" not in df.columns:
                invalid[symbol] = len(df)
                continue
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).set_index("date").sort_index()
            required = ["open", "high", "low", "close", "volume"]
            if not all(col in df.columns for col in required):
                invalid[symbol] = len(df)
                continue
            df = df[required].apply(pd.to_numeric, errors="coerce").dropna()
            if len(df) < min_rows:
                invalid[symbol] = len(df)
                continue
            all_data[symbol] = df
        except Exception:
            invalid[symbol] = 0
    return all_data, invalid


def build_registry(disable_live_capital: bool = True):
    from AStockQuant.layers.layer1_macro import MacroLayer
    from AStockQuant.layers.layer2_rules import RulesLayer
    from AStockQuant.layers.layer3_sector import SectorLayer
    import AStockQuant.layers.layer4_capital as layer4_mod
    from AStockQuant.layers.layer4_capital import CapitalLayer
    from AStockQuant.layers.layer5_sentiment import SentimentLayer
    from AStockQuant.layers.layer6_price_vol import PriceVolumeLayer
    from AStockQuant.layers.layer7_technical import TechnicalLayer
    from AStockQuant.layers.layer8_micro import BeliefLayer
    from AStockQuant.core.feature_registry import FeatureRegistry

    if disable_live_capital:
        layer4_mod._AK = False

    registry = FeatureRegistry()
    registry.register("macro", MacroLayer())
    registry.register("rules", RulesLayer())
    registry.register("sector", SectorLayer())
    registry.register("capital", CapitalLayer())
    registry.register("sentiment", SentimentLayer())
    registry.register("price_vol", PriceVolumeLayer())
    registry.register("technical", TechnicalLayer())
    registry.register("belief", BeliefLayer())
    return registry


def assert_range(results: Dict[str, Dict[str, object]], names: Iterable[str], lo: float, hi: float) -> None:
    failures = []
    for symbol, feats in results.items():
        for name in names:
            if name not in feats:
                failures.append((symbol, name, "missing"))
                continue
            value = feats[name]
            if not isinstance(value, (int, float, np.integer, np.floating)) or np.isnan(value):
                failures.append((symbol, name, value))
                continue
            if not lo <= float(value) <= hi:
                failures.append((symbol, name, value))
    if failures:
        raise AssertionError(f"range check failed: {failures[:10]}")


def validate_batch(as_of_date: str, min_rows: int, enable_live_capital: bool) -> None:
    warnings.filterwarnings("error", category=RuntimeWarning)
    compile_core_files()

    all_data, invalid = load_cached_data(min_rows=min_rows)
    registry = build_registry(disable_live_capital=not enable_live_capital)

    results = registry.extract_batch(list(all_data), all_data, as_of_date=as_of_date)

    total_factors = 0
    nan_or_inf = []
    for symbol, feats in results.items():
        total_factors += len(feats)
        for name, value in feats.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                if np.isnan(value) or np.isinf(value):
                    nan_or_inf.append((symbol, name, value))
    if nan_or_inf:
        raise AssertionError(f"NaN/Inf factors detected: {nan_or_inf[:10]}")

    assert_range(results, ("sector_combined_score", "capital_score", "sent_combined_score", "pv_vcp_quality", "tech_pattern_score", "belief_posterior"), 0.0, 1.0)
    assert_range(results, ("pv_rps_combined",), 0.0, 100.0)

    # Time-alignment smoke test: the same symbol at two dates should not be
    # identical for time-dependent factors when enough history is available.
    probe = "510300" if "510300" in all_data else next(iter(all_data))
    early = registry.extract_features(probe, all_data[probe], as_of_date="2025-06-01")
    late = registry.extract_features(probe, all_data[probe], as_of_date=as_of_date)
    if early.get("pv_rps_combined") == late.get("pv_rps_combined") and early.get("belief_posterior") == late.get("belief_posterior"):
        raise AssertionError("time alignment smoke test failed: probe factors are identical")

    sector_counts = Counter(feats.get("sector", "unknown") for feats in results.values())
    factor_counts = [len(feats) for feats in results.values()]
    vcp_count = sum(1 for feats in results.values() if feats.get("pv_vcp_is_pattern"))
    breakout_count = sum(1 for feats in results.values() if feats.get("pv_vcp_breakout"))

    print("VALIDATION_OK")
    print(f"cache_files={len(list((ROOT / 'data_cache').glob('*.csv')))}")
    print(f"valid_etfs={len(all_data)}")
    print(f"invalid_or_short={invalid}")
    print(f"layers={registry.active_layers}")
    print(f"total_factor_values={total_factors}")
    print(f"factor_count_min={min(factor_counts)} max={max(factor_counts)} avg={np.mean(factor_counts):.1f}")
    print(f"sectors={dict(sector_counts)}")
    print(f"vcp_patterns={vcp_count} breakouts={breakout_count}")
    print(f"probe={probe} early_rps={early.get('pv_rps_combined'):.4f} late_rps={late.get('pv_rps_combined'):.4f}")
    print(f"probe_early_belief={early.get('belief_posterior'):.4f} probe_late_belief={late.get('belief_posterior'):.4f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of-date", default="2026-06-18")
    parser.add_argument("--min-rows", type=int, default=120)
    parser.add_argument("--enable-live-capital", action="store_true")
    args = parser.parse_args()
    validate_batch(args.as_of_date, args.min_rows, args.enable_live_capital)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
