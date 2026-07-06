# -*- coding: utf-8 -*-
"""Quick focused parameter search."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))

import pandas as pd
from AStockQuant.scripts.advanced_factor_research import *

CFG = load_config(str(DEFAULT_CONFIG_PATH))
all_data = load_all_data()
cross_dates = get_cross_dates(all_data, CFG['date_range']['start'], CFG['date_range']['end'])
panel = build_factor_panel(all_data, cross_dates, REPORT_DIR / 'factor_panel.csv')
panel = add_forward_returns(panel, all_data)
factors = numeric_factor_columns(panel)
metrics, _, _ = factor_metrics(panel, factors)
selected, _ = select_core_factors(metrics, panel)
panel_clean = panel.dropna(subset=['fwd_ret_20d'])
print(f"Panel ready: {len(panel)} rows, {len(selected)} factors")

results = []
for rps_w in [0.0, 0.3, 0.5, 0.65, 0.8, 1.0]:
    for top_n in [5, 10, 15]:
        cfg = WalkForwardConfig(top_n=top_n, use_regime_gating=True, rps_blend_weight=rps_w)
        folds, equity = run_walk_forward(panel_clean, all_data, selected, cfg)
        stats = perf_stats(equity['equity'], cfg.initial_capital)
        avg_to = float(folds['avg_turnover'].mean())
        results.append((rps_w, top_n, stats['total_return'], stats['max_drawdown'], stats['sharpe'], avg_to))
        print(f"  rps_w={rps_w:.1f} top_n={top_n:2d} → ret={stats['total_return']:7.2%} mdd={stats['max_drawdown']:7.2%} sh={stats['sharpe']:.3f} to={avg_to:.3f}")

print("\n=== TOP 5 ===")
for r in sorted(results, key=lambda x: x[2], reverse=True)[:5]:
    print(f"  rps_w={r[0]:.1f} top_n={r[1]} → ret={r[2]:.2%} mdd={r[3]:.2%} sh={r[4]:.3f} to={r[5]:.3f}")
