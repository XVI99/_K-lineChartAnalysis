# -*- coding: utf-8 -*-
"""
组合构建模块
============
将因子评分转化为实际持仓权重，输出调仓信号并跟踪组合绩效。

管线: 因子Panel → 因子评分聚合 → 标的筛选 → 权重分配 → 调仓信号 → 绩效跟踪

聚合方法:
1. 等权聚合 (baseline)
2. ICIR加权聚合 (用滚动窗口ICIR动态赋权)
3. 滚动IC加权 (用滚动窗口IC均值赋权)

权重分配:
1. 等权 (1/N)
2. 得分加权 (score / sum(scores))
3. 风险平价 (波动率倒数加权)

约束条件:
- 最大持仓数: 10只
- 单只上限: 20%
- 换手率控制: 每期最多换50%

输出: AStockQuant/reports/portfolio/
"""

import os
import sys
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import layers.layer4_capital as _l4_mod
_l4_mod._AK = False

from factor_evaluation import (
    load_all_etf_data,
    get_cross_section_dates,
    build_factor_panel,
    calculate_forward_returns,
    spearmanr,
    N_QUANTILES,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

REPORT_DIR = os.path.join(PROJECT_ROOT, "reports", "portfolio")

# ==================== 配置 ====================
# 组合因子 (复用factor_combination.py的v4最优因子集)
PORTFOLIO_FACTORS = [
    "pv_volume_trend",          # ICIR=0.2549
    "sent_combined_score",      # ICIR=0.1913
    "pv_turnover_change",       # ICIR=0.2539, v4新增独立alpha
    "pv_price_accel",           # ICIR=-0.10, 反向因子(取负号变正向)
    "pv_vol_price_divergence",  # ICIR=0.03, 弱正但独立
    "sector_combined_score",    # ICIR=0.1056
]

# 因子方向: 1=正向(值越大越好), -1=反向(值越小越好)
FACTOR_DIRECTION = {
    "pv_volume_trend": 1,
    "sent_combined_score": 1,
    "pv_turnover_change": 1,
    "pv_price_accel": -1,       # 反向因子, 取负号
    "pv_vol_price_divergence": 1,
    "sector_combined_score": 1,
}

# 组合参数
MAX_POSITIONS = 10       # 最大持仓数
MAX_WEIGHT = 0.20        # 单只上限20%
MAX_TURNOVER = 0.50      # 每期最多换50%
REBALANCE_FREQ = 5       # 每5个交易日调仓(周频)

# 交易成本
SLIPPAGE = 0.001
COMMISSION = 0.0003
TURNOVER_COST = (SLIPPAGE + COMMISSION) * 2  # 0.26%

# 滚动IC窗口
IC_ROLLING_WINDOW = 20


# ==================== 1. 因子评分聚合 ====================
def aggregate_scores(
    panel: pd.DataFrame,
    factors: List[str],
    method: str = "equal",
    ic_window: int = IC_ROLLING_WINDOW,
) -> pd.DataFrame:
    """
    将多个因子聚合为单一综合评分

    Args:
        panel: 因子panel, 含date/symbol/各因子列
        factors: 因子名列表
        method: 聚合方法 (equal / icir_weighted / rolling_ic)
        ic_window: 滚动IC窗口(截面数)

    Returns:
        panel with 'composite_score' column (截面内排名百分位, 0-1)
    """
    result = panel.copy()
    dates = sorted(result["date"].unique())

    # 因子方向调整
    for f in factors:
        if f not in result.columns:
            continue
        direction = FACTOR_DIRECTION.get(f, 1)
        if direction == -1:
            result[f] = -result[f]  # 反向因子取负号

    # 逐截面标准化为排名百分位
    for f in factors:
        if f not in result.columns:
            continue
        norm_col = f"{f}_norm"
        result[norm_col] = result.groupby("date")[f].rank(pct=True)

    norm_cols = [f"{f}_norm" for f in factors if f"{f}_norm" in result.columns]

    if method == "equal":
        # 等权聚合
        result["composite_score"] = result[norm_cols].mean(axis=1)

    elif method == "icir_weighted":
        # ICIR加权聚合 (滚动窗口)
        fwd_col = "fwd_ret_20d"
        if fwd_col not in result.columns:
            print("  警告: 无前瞻收益列, 退化为等权")
            result["composite_score"] = result[norm_cols].mean(axis=1)
            return result

        fwd_days = 20
        ic_lag = max(1, (fwd_days + REBALANCE_FREQ - 1) // REBALANCE_FREQ)

        # 预计算每个因子的已实现IC
        ic_available = {f: [] for f in factors}
        for j_idx, dt in enumerate(dates):
            for f in factors:
                if f not in result.columns:
                    continue
                sub = result[result["date"] == dt][[f, fwd_col]].dropna()
                if len(sub) < 10:
                    continue
                ic, _ = spearmanr(sub[f].values, sub[fwd_col].values)
                if not np.isnan(ic):
                    available_idx = j_idx + ic_lag
                    if available_idx < len(dates):
                        ic_available[f].append((available_idx, ic))

        # 逐截面用滚动ICIR计算权重
        result["composite_score"] = 0.5
        for i, dt in enumerate(dates):
            mask = result["date"] == dt

            # 收集已实现IC
            icirs = {}
            for f in factors:
                past = [ic for (avail_idx, ic) in ic_available[f] if avail_idx <= i]
                if len(past) >= 5:
                    past_arr = np.array(past[-ic_window:])
                    ic_mean = np.mean(past_arr)
                    ic_std = np.std(past_arr, ddof=1)
                    icirs[f] = ic_mean / ic_std if ic_std > 0 else 0
                else:
                    icirs[f] = 0

            positive_icirs = {f: max(0, icir) for f, icir in icirs.items()}
            total = sum(positive_icirs.values())
            if total > 0:
                weights = {f: positive_icirs[f] / total for f in factors}
            else:
                weights = {f: 1.0 / len(factors) for f in factors}

            # 加权聚合
            score = pd.Series(0.0, index=result.loc[mask].index)
            for f in factors:
                norm_col = f"{f}_norm"
                if norm_col in result.columns and weights.get(f, 0) > 0:
                    score += result.loc[mask, norm_col].fillna(0.5) * weights[f]
            result.loc[mask, "composite_score"] = score

    elif method == "rolling_ic":
        # 滚动IC均值加权
        fwd_col = "fwd_ret_20d"
        if fwd_col not in result.columns:
            result["composite_score"] = result[norm_cols].mean(axis=1)
            return result

        fwd_days = 20
        ic_lag = max(1, (fwd_days + REBALANCE_FREQ - 1) // REBALANCE_FREQ)

        ic_available = {f: [] for f in factors}
        for j_idx, dt in enumerate(dates):
            for f in factors:
                if f not in result.columns:
                    continue
                sub = result[result["date"] == dt][[f, fwd_col]].dropna()
                if len(sub) < 10:
                    continue
                ic, _ = spearmanr(sub[f].values, sub[fwd_col].values)
                if not np.isnan(ic):
                    available_idx = j_idx + ic_lag
                    if available_idx < len(dates):
                        ic_available[f].append((available_idx, ic))

        result["composite_score"] = 0.5
        for i, dt in enumerate(dates):
            mask = result["date"] == dt
            ic_means = {}
            for f in factors:
                past = [ic for (avail_idx, ic) in ic_available[f] if avail_idx <= i]
                if len(past) >= 5:
                    ic_means[f] = np.mean(past[-ic_window:])
                else:
                    ic_means[f] = 0

            positive_ics = {f: max(0, ic) for f, ic in ic_means.items()}
            total = sum(positive_ics.values())
            if total > 0:
                weights = {f: positive_ics[f] / total for f in factors}
            else:
                weights = {f: 1.0 / len(factors) for f in factors}

            score = pd.Series(0.0, index=result.loc[mask].index)
            for f in factors:
                norm_col = f"{f}_norm"
                if norm_col in result.columns and weights.get(f, 0) > 0:
                    score += result.loc[mask, norm_col].fillna(0.5) * weights[f]
            result.loc[mask, "composite_score"] = score

    return result


# ==================== 2. 标的筛选 ====================
def select_stocks(
    panel: pd.DataFrame,
    max_positions: int = MAX_POSITIONS,
) -> pd.DataFrame:
    """
    每截面选TOP N标的

    Returns:
        panel with 'selected' column (bool)
    """
    panel = panel.copy()
    panel["rank"] = panel.groupby("date")["composite_score"].rank(ascending=False)
    panel["selected"] = panel["rank"] <= max_positions
    return panel


# ==================== 3. 权重分配 ====================
def allocate_weights(
    panel: pd.DataFrame,
    method: str = "equal",
    max_weight: float = MAX_WEIGHT,
) -> pd.DataFrame:
    """
    为选中标的分配权重

    Args:
        panel: 含selected列的panel
        method: equal / score_weighted / risk_parity
        max_weight: 单只上限

    Returns:
        panel with 'weight' column
    """
    panel = panel.copy()
    panel["weight"] = 0.0

    dates = panel["date"].unique()
    for dt in dates:
        mask = (panel["date"] == dt) & panel["selected"]
        selected = panel[mask]
        n = len(selected)
        if n == 0:
            continue

        if method == "equal":
            w = 1.0 / n
            panel.loc[mask, "weight"] = min(w, max_weight)

        elif method == "score_weighted":
            scores = selected["composite_score"]
            total_score = scores.sum()
            if total_score > 0:
                raw_weights = scores / total_score
                # 截断超限权重
                raw_weights = raw_weights.clip(upper=max_weight)
                # 重新归一化
                raw_weights = raw_weights / raw_weights.sum()
                panel.loc[mask, "weight"] = raw_weights.values

        elif method == "risk_parity":
            # 波动率倒数加权 (需要价格数据)
            # 简化: 用最近20日波动率
            # 如果panel中没有波动率列, 退化为等权
            w = 1.0 / n
            panel.loc[mask, "weight"] = min(w, max_weight)

    return panel


# ==================== 4. 换手率控制 ====================
def apply_turnover_control(
    panel: pd.DataFrame,
    max_turnover: float = MAX_TURNOVER,
) -> pd.DataFrame:
    """限制每期换手率: 新权重 = (1 - max_turnover) * 旧权重 + max_turnover * 目标权重。

    逐期追踪上一期最终权重，对当期目标权重做线性插值，把单期换手
    约束在 max_turnover 以内；每期处理后再对入选(selected=True)标的
    归一化权重，避免组合权重漂移。非入选标的跳过插值。
    """
    if panel.empty or "date" not in panel.columns or "weight" not in panel.columns:
        return panel
    panel = panel.sort_values(["date", "symbol"]).copy()
    prev_weights: dict = {}
    has_selected = "selected" in panel.columns
    for dt in sorted(panel["date"].unique()):
        mask = panel["date"] == dt
        if has_selected:
            sel_mask = mask & panel["selected"].astype(bool)
        else:
            sel_mask = mask
        # 1) 对入选标的做换手插值
        for idx in panel.index[sel_mask]:
            sym = panel.at[idx, "symbol"]
            target = float(panel.at[idx, "weight"])
            old = prev_weights.get(sym, 0.0)
            panel.at[idx, "weight"] = (1.0 - max_turnover) * old + max_turnover * target
        # 2) 入选标的权重归一化，避免漂移
        total = float(panel.loc[sel_mask, "weight"].sum())
        if total > 0:
            panel.loc[sel_mask, "weight"] = panel.loc[sel_mask, "weight"] / total
        # 3) 更新 prev_weights 为归一化后值
        for idx in panel.index[sel_mask]:
            prev_weights[panel.at[idx, "symbol"]] = float(panel.at[idx, "weight"])
    return panel


# ==================== 5. 绩效跟踪 ====================
def track_performance(
    panel: pd.DataFrame,
    all_data: Dict[str, pd.DataFrame],
) -> Dict:
    """
    跟踪组合绩效: 权益曲线、收益、回撤、夏普

    假设: 周频调仓, 持有到下一调仓日
    """
    print("\n[绩效跟踪] 计算组合权益曲线...")

    dates = sorted(panel["date"].unique())
    portfolio_returns = []

    for i, dt in enumerate(dates):
        # 当前截面持仓
        sub = panel[panel["date"] == dt]
        holdings = sub[sub["selected"]]

        if len(holdings) == 0:
            portfolio_returns.append(0.0)
            continue

        # 计算持仓期间收益 (到下一截面日)
        if i + 1 < len(dates):
            next_dt = pd.Timestamp(dates[i + 1])
        else:
            # 最后一期, 无下一截面
            break

        period_return = 0.0
        valid_count = 0
        for _, row in holdings.iterrows():
            sym = row["symbol"]
            weight = row["weight"]
            if sym not in all_data:
                continue

            df = all_data[sym]
            # 调仓日收盘买入
            buy_mask = df.index <= pd.Timestamp(dt)
            if buy_mask.sum() == 0:
                continue
            buy_price = df.loc[buy_mask, "close"].iloc[-1]

            # 下一调仓日收盘卖出
            sell_mask = df.index <= next_dt
            if sell_mask.sum() == 0:
                continue
            sell_price = df.loc[sell_mask, "close"].iloc[-1]

            if buy_price > 0:
                ret = (sell_price / buy_price - 1) - TURNOVER_COST
                period_return += ret * weight
                valid_count += 1

        if valid_count > 0:
            portfolio_returns.append(period_return)
        else:
            portfolio_returns.append(0.0)

    if len(portfolio_returns) < 5:
        return {"error": "收益序列不足"}

    ret_arr = np.array(portfolio_returns)
    cum_ret = np.cumprod(1 + ret_arr) - 1
    total_return = cum_ret[-1] if len(cum_ret) > 0 else 0

    # 年化收益 (52周)
    ann_return = (1 + total_return) ** (52 / len(ret_arr)) - 1 if total_return > -1 else -1

    # 年化波动率
    ann_vol = np.std(ret_arr, ddof=1) * np.sqrt(52)

    # 夏普比率
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    # 最大回撤
    cum_curve = np.cumprod(1 + ret_arr)
    running_max = np.maximum.accumulate(cum_curve)
    drawdowns = (cum_curve - running_max) / running_max
    max_drawdown = np.min(drawdowns)

    # 胜率
    win_rate = np.mean(ret_arr > 0)

    # 周收益统计
    weekly_mean = np.mean(ret_arr)
    weekly_std = np.std(ret_arr, ddof=1)

    return {
        "total_return": round(total_return * 100, 2),
        "ann_return": round(ann_return * 100, 2),
        "ann_vol": round(ann_vol * 100, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown": round(max_drawdown * 100, 2),
        "win_rate": round(win_rate * 100, 2),
        "weekly_mean": round(weekly_mean * 100, 4),
        "weekly_std": round(weekly_std * 100, 4),
        "n_periods": len(ret_arr),
        "cum_returns": cum_ret.tolist(),
    }


# ==================== 6. 信号输出 ====================
def generate_signals(panel: pd.DataFrame) -> pd.DataFrame:
    """
    生成调仓信号表: 日期、标的、权重、操作(买入/卖出/持有)
    """
    signals = []
    dates = sorted(panel["date"].unique())

    prev_holdings = set()
    for dt in dates:
        sub = panel[panel["date"] == dt]
        current_holdings = set(sub[sub["selected"]]["symbol"].values)

        # 新买入
        new_buys = current_holdings - prev_holdings
        # 卖出
        sells = prev_holdings - current_holdings
        # 持有
        holds = current_holdings & prev_holdings

        for _, row in sub[sub["selected"]].iterrows():
            sym = row["symbol"]
            if sym in new_buys:
                action = "BUY"
            elif sym in holds:
                action = "HOLD"
            else:
                action = "BUY"  # 第一期全部是BUY

            signals.append({
                "date": dt,
                "symbol": sym,
                "weight": round(row["weight"], 4),
                "score": round(row["composite_score"], 4),
                "action": action,
            })

        # 卖出信号
        for sym in sells:
            signals.append({
                "date": dt,
                "symbol": sym,
                "weight": 0.0,
                "score": 0.0,
                "action": "SELL",
            })

        prev_holdings = current_holdings

    return pd.DataFrame(signals)


# ==================== 7. 报告生成 ====================
def generate_portfolio_report(
    results: List[Dict],
    signals_df: pd.DataFrame,
):
    """生成组合构建报告"""
    os.makedirs(REPORT_DIR, exist_ok=True)

    print("\n" + "=" * 80)
    print("组合构建报告")
    print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"组合因子: {', '.join(PORTFOLIO_FACTORS)}")
    print(f"最大持仓: {MAX_POSITIONS}只, 单只上限: {MAX_WEIGHT*100}%")
    print(f"交易成本: {TURNOVER_COST*100}%")
    print("=" * 80)

    # 绩效对比
    print("\n一、组合绩效对比")
    print("-" * 100)
    print(f"{'策略':<40} {'总收益%':>10} {'年化%':>10} {'年化波动%':>12} {'夏普':>8} {'最大回撤%':>10} {'胜率%':>8} {'周期数':>8}")
    print("-" * 110)
    for r in results:
        if "error" in r:
            continue
        print(f"{r['name']:<40} {r['total_return']:>10.2f} {r['ann_return']:>10.2f} "
              f"{r['ann_vol']:>12.2f} {r['sharpe']:>8.2f} {r['max_drawdown']:>10.2f} "
              f"{r['win_rate']:>8.2f} {r['n_periods']:>8}")

    # 最佳策略
    valid = [r for r in results if "error" not in r]
    if valid:
        best = max(valid, key=lambda x: x["sharpe"])
        print(f"\n  最佳策略: {best['name']}")
        print(f"  年化收益: {best['ann_return']}%, 夏普: {best['sharpe']}, 最大回撤: {best['max_drawdown']}%")

    # 调仓信号摘要
    print(f"\n二、调仓信号摘要")
    print(f"  总信号数: {len(signals_df)}")
    if len(signals_df) > 0:
        action_counts = signals_df["action"].value_counts()
        for action, count in action_counts.items():
            print(f"  {action}: {count}")

    # 保存
    perf_df = pd.DataFrame(valid)
    perf_df.to_csv(os.path.join(REPORT_DIR, "portfolio_performance.csv"), index=False, encoding="utf-8-sig")
    signals_df.to_csv(os.path.join(REPORT_DIR, "rebalance_signals.csv"), index=False, encoding="utf-8-sig")

    # 摘要
    summary = []
    summary.append("=" * 80)
    summary.append("组合构建报告")
    summary.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary.append(f"组合因子: {', '.join(PORTFOLIO_FACTORS)}")
    summary.append(f"最大持仓: {MAX_POSITIONS}只, 单只上限: {MAX_WEIGHT*100}%")
    summary.append("=" * 80)
    summary.append("")
    summary.append("一、组合绩效对比")
    summary.append(f"{'策略':<40} {'总收益%':>10} {'年化%':>10} {'夏普':>8} {'最大回撤%':>10} {'胜率%':>8}")
    summary.append("-" * 90)
    for r in valid:
        summary.append(f"{r['name']:<40} {r['total_return']:>10.2f} {r['ann_return']:>10.2f} "
                       f"{r['sharpe']:>8.2f} {r['max_drawdown']:>10.2f} {r['win_rate']:>8.2f}")
    if valid:
        best = max(valid, key=lambda x: x["sharpe"])
        summary.append(f"\n最佳策略: {best['name']} 夏普={best['sharpe']} 年化={best['ann_return']}%")

    with open(os.path.join(REPORT_DIR, "portfolio_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary))

    print(f"\n报告已保存到: {REPORT_DIR}")


# ==================== 主流程 ====================
def main():
    print("=" * 80)
    print("组合构建 - 因子评分 → 标的筛选 → 权重分配 → 绩效跟踪")
    print("=" * 80)

    # 1. 加载数据
    all_data = load_all_etf_data()
    if len(all_data) < 20:
        print("错误: 可用ETF数据不足")
        return

    all_dates = set()
    for df in all_data.values():
        all_dates.update(df.index)
    cross_dates = get_cross_section_dates(list(all_dates))
    if len(cross_dates) < 10:
        print("错误: 截面日期不足")
        return

    # 2. 构建因子panel
    panel = build_factor_panel(all_data, cross_dates)
    panel = calculate_forward_returns(all_data, panel)

    available_factors = [f for f in PORTFOLIO_FACTORS if f in panel.columns]
    print(f"\n可用因子: {len(available_factors)}/{len(PORTFOLIO_FACTORS)}")

    # 3. 测试多种策略组合 (共用同一个panel)
    results = []
    all_signals = None

    strategies = [
        # (聚合方法, 选股数, 权重方法)
        ("equal", MAX_POSITIONS, "equal"),
        ("equal", MAX_POSITIONS, "score_weighted"),
        ("equal", 5, "equal"),
        ("icir_weighted", MAX_POSITIONS, "equal"),
    ]

    for agg_method, n_pos, weight_method in strategies:
        strategy_name = f"agg={agg_method}_n={n_pos}_w={weight_method}"

        print(f"\n{'='*60}")
        print(f"[策略] {strategy_name}")
        print(f"{'='*60}")

        # 因子聚合 (在panel副本上操作, 避免污染)
        print(f"  [1] 因子聚合 ({agg_method})...")
        panel_scored = aggregate_scores(panel.copy(), available_factors, method=agg_method)

        # 标的筛选
        print(f"  [2] 标的筛选 (TOP {n_pos})...")
        panel_selected = select_stocks(panel_scored, max_positions=n_pos)

        # 权重分配
        print(f"  [3] 权重分配 ({weight_method})...")
        panel_weighted = allocate_weights(panel_selected, method=weight_method)

        # 换手控制
        panel_final = apply_turnover_control(panel_weighted)

        # 绩效跟踪
        perf = track_performance(panel_final, all_data)
        perf["name"] = strategy_name
        results.append(perf)

        # 保存第一个策略的信号
        if all_signals is None:
            all_signals = generate_signals(panel_final)

    # 4. 报告
    if all_signals is None:
        all_signals = pd.DataFrame()
    generate_portfolio_report(results, all_signals)
    print("\n组合构建完成!")


if __name__ == "__main__":
    main()
