# -*- coding: utf-8 -*-
"""
daily_signal.py — v4 实盘信号生成器

用法:
  python daily_signal.py                          # 用最近的数据日 (2026-05-22)
  python daily_signal.py --as-of 2026-05-22       # 指定日期
  python daily_signal.py --budget 50000           # 自定义本金
  python daily_signal.py --current 159628,512480  # 当前持仓（用于对比）

输出:
  - 宏观状态 + 仓位乘数
  - v4 6 层评分筛出的 5 只 ETF
  - 当前持仓 vs 入选对比
  - 退出信号检查 (硬止损/移动止盈/时间止损)
  - 防御信号 (BEAR 时清仓)
  - JSON 信号文件
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

# 复用 v5 的核心模块；当前工作树中 v5 可能位于 _archive。
try:
    from backtest.etf_screener_backtest_v5 import (
        LayeredETFScreener,
        ImprovedMacroLayer,
    )
except ModuleNotFoundError:
    from backtest._archive.etf_screener_backtest_v5 import (
        LayeredETFScreener,
        ImprovedMacroLayer,
    )


# ============================================================
# 风控参数（与 v2.1/v3/v4 完全一致）
# ============================================================

STOP_LOSS_PCT = 0.10
TAKE_PROFIT_FLOOR_PCT = 0.30
BREAKEVEN_TRIGGER_PCT = 0.10
TRAILING_TRIGGER_PCT = 0.15
TRAILING_PCT = 0.08
MAX_HOLD_DAYS = 90
MAX_HOLD_DAYS_HARD = 120
STAGNANT_THRESHOLD = 0.05
MAX_HOLDINGS = 5
MAX_POSITION_PCT = 0.30
EXTERNAL_CACHE_DIR = ROOT / "external_cache"


def load_external_cache(cache_dir: Path = EXTERNAL_CACHE_DIR) -> Dict:
    """读取当前外部信息快照；这些字段只用于当日复核，不回填历史回测。"""
    result = {
        'available': False,
        'metadata': {},
        'etf_by_code': {},
        'theme_by_name': {},
    }
    metadata_path = cache_dir / "external_signals_metadata.json"
    etf_path = cache_dir / "etf_external_snapshot.csv"
    theme_path = cache_dir / "theme_flow_snapshot.csv"

    if metadata_path.exists():
        try:
            result['metadata'] = json.loads(metadata_path.read_text(encoding='utf-8'))
        except Exception:
            result['metadata'] = {}

    if etf_path.exists():
        try:
            etf_df = pd.read_csv(etf_path, dtype={'code': str})
            result['etf_by_code'] = {
                str(row['code']).zfill(6): row.dropna().to_dict()
                for _, row in etf_df.iterrows()
                if str(row.get('code', '')).strip()
            }
        except Exception:
            result['etf_by_code'] = {}

    if theme_path.exists():
        try:
            theme_df = pd.read_csv(theme_path)
            result['theme_by_name'] = {
                str(row['theme']): row.dropna().to_dict()
                for _, row in theme_df.iterrows()
                if str(row.get('theme', '')).strip()
            }
        except Exception:
            result['theme_by_name'] = {}

    result['available'] = bool(result['etf_by_code'] or result['theme_by_name'])
    return result


def external_overlay_for_etfs(etfs: List, external: Dict) -> List[Dict]:
    """为入选 ETF 附加外部快照字段。"""
    overlays = []
    etf_by_code = external.get('etf_by_code', {})
    theme_by_name = external.get('theme_by_name', {})
    for etf in etfs:
        row = etf_by_code.get(etf.code, {})
        theme = str(row.get('theme', ''))
        theme_row = theme_by_name.get(theme, {}) if theme else {}
        overlays.append({
            'code': etf.code,
            'name': row.get('name', ''),
            'theme': theme,
            'market_price': row.get('market_price'),
            'unit_nav': row.get('unit_nav'),
            'discount_rate': row.get('discount_rate'),
            'nav_growth_rate': row.get('nav_growth_rate'),
            'purchase_status': row.get('purchase_status'),
            'redeem_status': row.get('redeem_status'),
            'theme_net_flow_sum': theme_row.get('net_flow_sum'),
            'theme_change_pct_median': theme_row.get('change_pct_median'),
            'theme_best_rank': theme_row.get('best_rank'),
        })
    return overlays


def print_external_overlay(overlays: List[Dict], metadata: Dict) -> None:
    print(f"\n[外部信息快照]")
    if not overlays:
        print("  (未找到 external_cache，跳过 ETF 净值/资金流复核)")
        return
    stamp = metadata.get('timestamp', 'unknown')
    print(f"  快照时间: {stamp}；仅用于当日复核，不回填历史回测")
    print(f"  {'代码':<8} {'主题':<14} {'折价%':>8} {'净值涨%':>8} {'申购':<10} {'主题净流':>10} {'主题涨%':>8}")
    for row in overlays:
        discount = row.get('discount_rate')
        nav_growth = row.get('nav_growth_rate')
        flow = row.get('theme_net_flow_sum')
        theme_change = row.get('theme_change_pct_median')
        print(
            f"  {row['code']:<8} {str(row.get('theme') or '-'):<14} "
            f"{(float(discount) * 100 if discount is not None else 0):>7.2f}% "
            f"{(float(nav_growth) * 100 if nav_growth is not None else 0):>7.2f}% "
            f"{str(row.get('purchase_status') or '-'):<10} "
            f"{(float(flow) if flow is not None else 0):>10.2f} "
            f"{(float(theme_change) * 100 if theme_change is not None else 0):>7.2f}%"
        )


def latest_data_date(data_dir: str) -> str:
    """从本地 ETF 缓存推断最新交易日，优先用 510300。"""
    preferred = Path(data_dir) / "510300.csv"
    candidates = [preferred] if preferred.exists() else []
    if not candidates:
        candidates = list(Path(data_dir).glob("*.csv"))
    latest = None
    for path in candidates:
        try:
            df = pd.read_csv(path, usecols=['date'], parse_dates=['date'])
        except Exception:
            continue
        if df.empty:
            continue
        cur = df['date'].max()
        latest = cur if latest is None or cur > latest else latest
    if latest is None:
        return datetime.now().strftime('%Y-%m-%d')
    return pd.Timestamp(latest).strftime('%Y-%m-%d')


def check_exit_signals(positions: Dict, data_dir: str, as_of: str) -> List[Dict]:
    """对每只持仓 ETF 检查风控信号"""
    signals = []
    for code, pos in positions.items():
        p = Path(data_dir) / f"{code}.csv"
        if not p.exists():
            signals.append({'code': code, 'action': 'NO_DATA', 'reason': '数据缺失'})
            continue

        df = pd.read_csv(p, parse_dates=['date']).sort_values('date')
        df = df[df['date'] <= pd.Timestamp(as_of)]
        if len(df) < 2:
            continue

        current_price = float(df['close'].iloc[-1])
        avg_cost = pos['avg_cost']
        buy_date = pd.Timestamp(pos['buy_date'])
        current_date = pd.Timestamp(as_of)
        days_held = (current_date - buy_date).days
        profit_pct = (current_price - avg_cost) / avg_cost

        # 持仓期间最高价
        df_pos = df[df['date'] >= buy_date]
        if len(df_pos) > 0:
            highest_price = float(df_pos['close'].max())
        else:
            highest_price = current_price
        peak_profit = (highest_price - avg_cost) / avg_cost
        drawdown_from_peak = (current_price - highest_price) / highest_price if highest_price > 0 else 0

        # 退出判断（按 v2.1 顺序）
        action = 'HOLD'
        reason = ''

        if profit_pct <= -STOP_LOSS_PCT:
            action = 'SELL_FULL'
            reason = f'HARD_STOP_LOSS({profit_pct*100:.1f}%)'
        elif peak_profit >= TRAILING_TRIGGER_PCT and drawdown_from_peak <= -TRAILING_PCT:
            action = 'SELL_FULL'
            reason = f'TRAILING_STOP(peak={peak_profit*100:.1f}%,dd={drawdown_from_peak*100:.1f}%)'
        elif peak_profit >= BREAKEVEN_TRIGGER_PCT and profit_pct <= 0:
            action = 'SELL_FULL'
            reason = f'BREAKEVEN_STOP(peak={peak_profit*100:.1f}%)'
        elif profit_pct >= TAKE_PROFIT_FLOOR_PCT:
            action = 'SELL_FULL'
            reason = f'HARD_TAKE_PROFIT({profit_pct*100:.1f}%)'
        elif days_held >= MAX_HOLD_DAYS_HARD:
            action = 'SELL_FULL'
            reason = f'TIME_STOP_HARD({days_held}d,pnl={profit_pct*100:.1f}%)'
        elif days_held >= MAX_HOLD_DAYS and profit_pct < STAGNANT_THRESHOLD:
            action = 'SELL_HALF'
            reason = f'TIME_STOP_HALF({days_held}d,pnl={profit_pct*100:.1f}%)'
        else:
            reason = f'健康持仓 ({days_held}d,pnl={profit_pct*100:.1f}%)'

        signals.append({
            'code': code,
            'avg_cost': avg_cost,
            'current_price': current_price,
            'quantity': pos.get('quantity', 0),
            'days_held': days_held,
            'profit_pct': round(profit_pct * 100, 2),
            'peak_profit_pct': round(peak_profit * 100, 2),
            'drawdown_from_peak_pct': round(drawdown_from_peak * 100, 2),
            'action': action,
            'reason': reason,
            'highest_price': highest_price,
        })
    return signals


def compute_allocation(target_etfs: List, macro_score: float, budget: float) -> List[Dict]:
    """根据宏观得分 + 入选 ETF 计算资金分配"""
    if macro_score > 0.65:
        regime = 'BULL'
        multiplier = 1.0
    elif macro_score < 0.35:
        regime = 'BEAR'
        multiplier = 0.0
    else:
        regime = 'NEUTRAL'
        multiplier = 0.5

    if multiplier == 0.0:
        return [{'note': 'BEAR 市场，建议清仓观望（不建新仓）'}]

    usable = budget * multiplier
    per_etf = usable / min(len(target_etfs), MAX_HOLDINGS)

    allocations = []
    for etf in target_etfs[:MAX_HOLDINGS]:
        p = Path(r"F:\_K-lineChartAnalysis\AStockQuant\data_cache") / f"{etf.code}.csv"
        try:
            df = pd.read_csv(p, parse_dates=['date']).sort_values('date')
            price = float(df['close'].iloc[-1])
        except Exception:
            price = etf.latest_price
        # 整手（100 股）
        qty = int(per_etf / price / 100) * 100
        amount = qty * price
        allocations.append({
            'code': etf.code,
            'latest_price': price,
            'allocation_pct': round(per_etf / budget * 100, 1),
            'quantity': qty,
            'amount': round(amount, 0),
            'score': round(etf.total_score, 3),
        })
    return allocations


def main():
    parser = argparse.ArgumentParser(description='v4 实盘信号生成')
    parser.add_argument('--as-of', type=str, default='', help='信号日期 YYYY-MM-DD；默认自动用本地缓存最新交易日')
    parser.add_argument('--budget', type=float, default=10000.0, help='总资金 (默认 10000)')
    parser.add_argument('--top-n', type=int, default=5, help='选几只 ETF (默认 5)')
    parser.add_argument('--current', type=str, default='', help='当前持仓代码，逗号分隔 (例 159628,512480)')
    parser.add_argument('--current-costs', type=str, default='', help='当前持仓成本，逗号分隔 (与 --current 一一对应)')
    parser.add_argument('--current-dates', type=str, default='', help='当前持仓买入日，逗号分隔 YYYY-MM-DD')
    parser.add_argument('--output', type=str, default='reports/daily_signal.json', help='信号输出文件')
    args = parser.parse_args()

    data_dir = r"F:\_K-lineChartAnalysis\AStockQuant\data_cache"
    as_of = args.as_of or latest_data_date(data_dir)
    budget = args.budget
    top_n = args.top_n

    print("=" * 70)
    print("  AStockQuant v4 实盘信号")
    print(f"  信号日: {as_of}    资金: {budget:,.0f} 元    入选数: {top_n}")
    print("=" * 70)

    # 1) 跑 v4 评分
    print(f"\n[1/3] 跑 v4 6 层评分 (as_of={as_of})...")
    screener = LayeredETFScreener(data_dir)
    top_etfs, ctx = screener.get_top_etfs(min_score=0.3, top_n=top_n, as_of_date=as_of)
    if not top_etfs:
        print("  [失败] 评分失败，请检查数据")
        return

    macro_score = ctx.get('market_score', 0.5)
    macro_regime = ctx.get('market_regime', 'NEUTRAL')
    print(f"  宏观: {macro_regime} (score={macro_score:.3f})")

    print(f"\n  入选 ETF (前 {len(top_etfs)} 只):")
    print(f"  {'代码':<8} {'L1':>5} {'L3m':>5} {'L3p':>5} {'L4':>5} {'L5':>5} {'L6':>5} {'L7':>5} {'综合':>6} {'价格':>7}")
    for etf in top_etfs:
        p = Path(data_dir) / f"{etf.code}.csv"
        try:
            df = pd.read_csv(p, parse_dates=['date'])
            price = float(df['close'].iloc[-1])
        except Exception:
            price = etf.latest_price
        print(f"  {etf.code:<8} {etf.layer1_macro:>5.2f} {etf.layer3_60d_mom:>5.2f} "
              f"{etf.layer3_phase:>5.2f} {etf.layer4_capital:>5.2f} {etf.layer5_sentiment:>5.2f} "
              f"{etf.layer6_pv:>5.2f} {etf.layer7_tech:>5.2f} {etf.total_score:>6.3f} {price:>7.3f}")

    external = load_external_cache()
    external_overlays = external_overlay_for_etfs(top_etfs, external) if external.get('available') else []
    print_external_overlay(external_overlays, external.get('metadata', {}))

    # 2) 资金分配
    print(f"\n[2/3] 资金分配 (按宏观 {macro_regime} → 仓位 {1.0 if macro_regime=='BULL' else 0.5 if macro_regime=='NEUTRAL' else 0.0})...")
    allocations = compute_allocation(top_etfs, macro_score, budget)
    if isinstance(allocations[0], dict) and 'note' in allocations[0]:
        print(f"  [提示] {allocations[0]['note']}")
    else:
        total_amount = sum(a['amount'] for a in allocations)
        usable = budget * (1.0 if macro_regime == 'BULL' else 0.5 if macro_regime == 'NEUTRAL' else 0)
        print(f"  可用资金: {usable:,.0f} 元 (BULL=1.0 / NEUTRAL=0.5 / BEAR=0.0)")
        print(f"  {'代码':<8} {'比例%':>6} {'数量':>8} {'价格':>7} {'金额':>10} {'评分':>6}")
        for a in allocations:
            print(f"  {a['code']:<8} {a['allocation_pct']:>5.1f}% {a['quantity']:>8d} {a['latest_price']:>7.3f} {a['amount']:>9,.0f}元 {a['score']:>6.3f}")
        print(f"  {'合计':<8} {'':>6} {'':>8} {'':>7} {total_amount:>9,.0f}元")

    # 3) 当前持仓检查
    print(f"\n[3/3] 当前持仓检查...")
    current_codes = [c.strip() for c in args.current.split(',') if c.strip()]
    current_costs = [float(c.strip()) for c in args.current_costs.split(',') if c.strip()]
    current_dates = [d.strip() for d in args.current_dates.split(',') if d.strip()]

    target_codes = [e.code for e in top_etfs]
    target_set = set(target_codes)
    exit_signals = []

    if current_codes:
        positions = {}
        for i, code in enumerate(current_codes):
            cost = current_costs[i] if i < len(current_costs) else 1.0
            buy_date = current_dates[i] if i < len(current_dates) else '2024-01-01'
            positions[code] = {'avg_cost': cost, 'buy_date': buy_date, 'quantity': 1000}
        exit_signals = check_exit_signals(positions, data_dir, as_of)
        print(f"\n  当前 {len(current_codes)} 只持仓 vs 入选 {len(target_codes)} 只:")
        print(f"  {'代码':<8} {'成本':>7} {'现价':>7} {'天数':>5} {'盈亏%':>7} {'峰值%':>7} {'回撤%':>7} {'操作':<12} {'原因':<40}")
        for s in exit_signals:
            in_target = s['code'] in target_set
            tag = '[OK] 保留' if in_target else '[X] 调出'
            print(f"  {s['code']:<8} {s.get('avg_cost', 0):>7.3f} {s.get('current_price', 0):>7.3f} "
                  f"{s.get('days_held', 0):>5d} {s.get('profit_pct', 0):>+6.1f}% {s.get('peak_profit_pct', 0):>+6.1f}% "
                  f"{s.get('drawdown_from_peak_pct', 0):>+6.1f}% {s.get('action', '?'):<12} {s.get('reason', '')[:40]:<40} {tag}")

        # 调仓建议
        print(f"\n  调仓建议:")
        to_sell = [c for c in current_codes if c not in target_set]
        to_buy = [c for c in target_codes if c not in set(current_codes)]
        keep = [c for c in current_codes if c in target_set]
        if to_sell:
            print(f"  [卖出] ({len(to_sell)}): {', '.join(to_sell)}")
        if to_buy:
            print(f"  [买入] ({len(to_buy)}): {', '.join(to_buy)}")
        if keep:
            print(f"  [保留] ({len(keep)}): {', '.join(keep)}")
        if not to_sell and not to_buy:
            print(f"  [OK] 持仓与入选一致，无需调仓")
    else:
        print(f"  (未传 --current，跳过持仓检查)")

    # 4) 防御信号
    print(f"\n[防御信号]")
    if macro_regime == 'BEAR':
        print(f"  [BEAR] 建议清仓所有持仓，转为货币基金 (511880 华宝添益)")
    elif macro_score < 0.40:
        print(f"  [警告] 宏观得分偏低 ({macro_score:.3f}) → 谨慎加仓，已有持仓可考虑减半")
    else:
        print(f"  [OK] 宏观状态正常 ({macro_regime}, score={macro_score:.3f})，按上面建议执行")

    # 5) 保存 JSON
    summary = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'as_of': as_of,
        'budget': budget,
        'macro': {
            'regime': macro_regime,
            'score': round(macro_score, 4),
            'multiplier': 1.0 if macro_regime == 'BULL' else (0.5 if macro_regime == 'NEUTRAL' else 0.0),
        },
        'selected_etfs': [
            {'code': e.code, 'total_score': round(e.total_score, 4),
             'L1': round(e.layer1_macro, 3), 'L3_mom': round(e.layer3_60d_mom, 3),
             'L3_phase': round(e.layer3_phase, 3), 'L4': round(e.layer4_capital, 3),
             'L5': round(e.layer5_sentiment, 3), 'L6': round(e.layer6_pv, 3),
             'L7': round(e.layer7_tech, 3)} for e in top_etfs
        ],
        'allocations': allocations,
        'external_snapshot': {
            'metadata': external.get('metadata', {}),
            'selected_etfs': external_overlays,
            'warning': '外部快照是当前数据，只能用于当日复核，不能回填历史回测。',
        },
        'current_positions': exit_signals,
        'recommendation': {
            'to_sell': [c for c in current_codes if c not in set(target_codes)],
            'to_buy': [c for c in target_codes if c not in set(current_codes)],
            'keep': [c for c in current_codes if c in set(target_codes)],
        } if current_codes else {},
    }
    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n信号已保存: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
