# -*- coding: utf-8 -*-
"""
因子组合优化脚本 (v2 — 修复前视偏差 + 新增walk-forward + 交易成本)
=================================================================
借鉴AlphaEvo最佳实践:
1. 滚动ICIR加权: 修复前视偏差, 只用截面i之前已实现的前瞻收益IC
2. Walk-Forward验证: expanding window分train/test, 报告OOS ICIR
3. 交易成本模型: 滑点+佣金, 扣减多空收益
4. IC加权改为滚动IC加权 (不再用全样本)

输出: AStockQuant/reports/factor_evaluation/combination/
"""

import os
import sys
import warnings
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# 确保能 import AStockQuant 模块
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 禁用akshare网络调用
import layers.layer4_capital as _l4_mod
_l4_mod._AK = False

# 复用 factor_evaluation.py 的函数
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from factor_evaluation import (
    load_all_etf_data,
    get_cross_section_dates,
    build_factor_panel,
    calculate_forward_returns,
    spearmanr,
    EVAL_FACTORS,
    FORWARD_PERIODS,
    N_QUANTILES,
    START_DATE,
    END_DATE,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

REPORT_DIR = os.path.join(PROJECT_ROOT, "reports", "factor_evaluation", "combination")

# 选入组合的因子 (v4: 以pv_volume_trend为核心 + v4新增独立alpha因子)
# 策略: 只保留ICIR>0.1且与pv_volume_trend低相关的因子
COMBINE_FACTORS = [
    "pv_volume_trend",          # ICIR=0.2549, 核心因子
    "sent_combined_score",      # ICIR=0.1913, 情绪综合
    "pv_turnover_change",       # v4新增: 换手率变化率(独立alpha源)
    "pv_price_accel",           # v4新增: 价格加速度(独立alpha源)
    "pv_vol_price_divergence",  # v4新增: 量价背离度(独立alpha源)
    "sector_combined_score",    # ICIR=0.1056, 板块动量
]

# Walk-Forward参数 (v4: 扩大测试集)
WF_TRAIN_PCT = 0.5   # 初始训练集比例 (50%, 留更多测试)
WF_N_FOLDS = 2       # walk-forward折数 (2折, 每折~27截面, 更可靠)

# ==================== 交易成本模型 (借鉴AlphaEvo) ====================
SLIPPAGE = 0.001      # 滑点 0.1% (买入价上浮)
COMMISSION = 0.0003   # 佣金 0.03% (双边)
# 每次调仓的双边成本 = (slippage + commission) * 2
TURNOVER_COST = (SLIPPAGE + COMMISSION) * 2  # 0.26% per round trip

# 滚动ICIR窗口
ROLLING_WINDOW = 20  # 20个截面≈4个月


# ==================== 1. 去共线性检查 ====================
def check_collinearity(panel: pd.DataFrame, factors: List[str]):
    """检查因子间相关性矩阵"""
    print("\n[检查] 因子间Spearman相关性矩阵:")
    sub = panel[factors].dropna()
    if len(sub) < 100:
        print("  警告: 有效数据不足")
        return

    corr_matrix = pd.DataFrame(index=factors, columns=factors, dtype=float)
    for f1 in factors:
        for f2 in factors:
            if f1 == f2:
                corr_matrix.loc[f1, f2] = 1.0
            else:
                rho, _ = spearmanr(sub[f1].values, sub[f2].values)
                corr_matrix.loc[f1, f2] = rho

    high_corr = []
    for i in range(len(factors)):
        for j in range(i + 1, len(factors)):
            rho = corr_matrix.iloc[i, j]
            if abs(rho) > 0.8:
                high_corr.append((factors[i], factors[j], rho))

    if high_corr:
        print("  高相关因子对 (|rho|>0.8):")
        for f1, f2, rho in high_corr:
            print(f"    {f1} <-> {f2}: {rho:.3f}")
    else:
        print("  无高相关因子对 (|rho|<0.8)")

    return corr_matrix


# ==================== 2. 因子标准化 ====================
def normalize_factors(panel: pd.DataFrame, factors: List[str]) -> pd.DataFrame:
    """逐截面将因子标准化到[0,1]区间 (横截面排名百分位)"""
    print("\n[1] 因子标准化 (截面排名百分位)...")
    result = panel.copy()
    for f in factors:
        if f not in result.columns:
            continue
        norm_col = f"{f}_norm"
        result[norm_col] = result.groupby("date")[f].rank(pct=True)
    return result


# ==================== 3. 无前视偏差的IC计算 ====================
def compute_realized_ic_series(
    panel: pd.DataFrame,
    factor_col: str,
    fwd_col: str = "fwd_ret_20d",
    dates: List[str] = None,
) -> List[Tuple[str, float]]:
    """
    计算已实现IC序列: 截面j的IC只在j+fwd_period后才可用
    返回 [(可用日期, IC值)] — IC值在截面j计算, 但标记为j+fwd_period日"可用"
    """
    if dates is None:
        dates = sorted(panel["date"].unique())

    # 前瞻期限(天数) → 截面数偏移
    # fwd_ret_20d = 20个交易日, 截面频率=5天, 所以20/5=4个截面后IC才可用
    fwd_days = int(fwd_col.split("_")[2].rstrip("d"))
    rebalance_freq = 5
    ic_lag = max(1, (fwd_days + rebalance_freq - 1) // rebalance_freq)  # 向上取整

    ic_series = []
    for j_idx, dt in enumerate(dates):
        sub = panel[panel["date"] == dt][[factor_col, fwd_col]].dropna()
        if len(sub) < 10:
            continue
        ic, _ = spearmanr(sub[factor_col].values, sub[fwd_col].values)
        if not np.isnan(ic):
            # IC在截面j计算, 但前瞻收益覆盖[j, j+fwd_days]
            # 所以这个IC只能在截面 j+ic_lag 时才"已知"
            available_idx = j_idx + ic_lag
            if available_idx < len(dates):
                ic_series.append((dates[available_idx], ic, dt))

    return ic_series


# ==================== 4. 组合方法 (全部修复前视偏差) ====================
def combine_equal_weight(panel: pd.DataFrame, factors: List[str]) -> Tuple[pd.Series, Dict]:
    """等权组合: 简单平均 (无前视偏差)"""
    print("  [等权组合]...")
    norm_factors = [f"{f}_norm" for f in factors if f"{f}_norm" in panel.columns]
    return panel[norm_factors].mean(axis=1), {"method": "equal_weight"}


def combine_rolling_icir_fixed(
    panel: pd.DataFrame,
    factors: List[str],
    fwd_col: str = "fwd_ret_20d",
) -> Tuple[pd.Series, Dict]:
    """
    滚动ICIR加权 (v2修复前视偏差):
    - 截面j的IC只在j+ic_lag后才可用(前瞻收益已实现)
    - 计算截面i的权重时, 只用available_date <= i的IC
    """
    print(f"  [滚动ICIR加权-修复前视] 窗口={ROLLING_WINDOW}截面, fwd_lag计算中...")

    dates = sorted(panel["date"].unique())
    fwd_days = int(fwd_col.split("_")[2].rstrip("d"))
    rebalance_freq = 5
    ic_lag = max(1, (fwd_days + rebalance_freq - 1) // rebalance_freq)

    # 预计算每个因子在每个截面j的IC, 以及其"可用日期"索引
    # ic_available[f] = [(available_date_idx, ic_value), ...]
    ic_available = {f: [] for f in factors}
    for j_idx, dt in enumerate(dates):
        for f in factors:
            norm_col = f"{f}_norm"
            if norm_col not in panel.columns:
                continue
            sub = panel[panel["date"] == dt][[norm_col, fwd_col]].dropna()
            if len(sub) < 10:
                continue
            ic, _ = spearmanr(sub[norm_col].values, sub[fwd_col].values)
            if not np.isnan(ic):
                available_idx = j_idx + ic_lag
                if available_idx < len(dates):
                    ic_available[f].append((available_idx, ic))

    # 逐截面计算权重
    combo = pd.Series(0.5, index=panel.index)
    weight_history = []

    for i, dt in enumerate(dates):
        # 只用 available_idx <= i 的IC (即截面j的前瞻收益已实现)
        available_ics = {}
        for f in factors:
            past_ics = [ic for (avail_idx, ic) in ic_available[f] if avail_idx <= i]
            available_ics[f] = past_ics

        # 检查是否有足够的历史IC
        min_ic_count = min(len(v) for v in available_ics.values()) if available_ics else 0

        if min_ic_count < 5:
            # 不够历史数据, 用等权
            weights = {f: 1.0 / len(factors) for f in factors}
        else:
            # 用最近ROLLING_WINDOW个已实现IC计算ICIR
            icirs = {}
            for f in factors:
                past = available_ics[f][-ROLLING_WINDOW:]
                if len(past) < 5:
                    icirs[f] = 0
                else:
                    ic_arr = np.array(past)
                    ic_mean = np.mean(ic_arr)
                    ic_std = np.std(ic_arr, ddof=1)
                    icirs[f] = ic_mean / ic_std if ic_std > 0 else 0

            # 只用正ICIR
            positive_icirs = {f: max(0, icir) for f, icir in icirs.items()}
            total = sum(positive_icirs.values())
            if total > 0:
                weights = {f: positive_icirs[f] / total for f in factors}
            else:
                weights = {f: 1.0 / len(factors) for f in factors}

        weight_history.append({"date": dt, **weights})

        # 应用权重
        mask = panel["date"] == dt
        for f in factors:
            norm_col = f"{f}_norm"
            if norm_col in panel.columns:
                combo.loc[mask] = combo.loc[mask].fillna(0) + panel.loc[mask, norm_col].fillna(0.5) * weights.get(f, 0)

    return combo, {"method": "rolling_icir_fixed", "window": ROLLING_WINDOW, "ic_lag": ic_lag}


def combine_rolling_ic_weighted(
    panel: pd.DataFrame,
    factors: List[str],
    fwd_col: str = "fwd_ret_20d",
) -> Tuple[pd.Series, Dict]:
    """
    滚动IC加权 (v2修复前视偏差):
    替代原combine_ic_weighted(全样本泄漏), 改用滚动窗口的已实现IC均值
    """
    print(f"  [滚动IC加权-修复前视] 窗口={ROLLING_WINDOW}截面...")

    dates = sorted(panel["date"].unique())
    fwd_days = int(fwd_col.split("_")[2].rstrip("d"))
    rebalance_freq = 5
    ic_lag = max(1, (fwd_days + rebalance_freq - 1) // rebalance_freq)

    # 预计算已实现IC
    ic_available = {f: [] for f in factors}
    for j_idx, dt in enumerate(dates):
        for f in factors:
            norm_col = f"{f}_norm"
            if norm_col not in panel.columns:
                continue
            sub = panel[panel["date"] == dt][[norm_col, fwd_col]].dropna()
            if len(sub) < 10:
                continue
            ic, _ = spearmanr(sub[norm_col].values, sub[fwd_col].values)
            if not np.isnan(ic):
                available_idx = j_idx + ic_lag
                if available_idx < len(dates):
                    ic_available[f].append((available_idx, ic))

    # 逐截面计算权重
    combo = pd.Series(0.5, index=panel.index)

    for i, dt in enumerate(dates):
        available_ics = {}
        for f in factors:
            past_ics = [ic for (avail_idx, ic) in ic_available[f] if avail_idx <= i]
            available_ics[f] = past_ics

        min_ic_count = min(len(v) for v in available_ics.values()) if available_ics else 0

        if min_ic_count < 5:
            weights = {f: 1.0 / len(factors) for f in factors}
        else:
            # 用最近ROLLING_WINDOW个IC的均值
            ic_means = {}
            for f in factors:
                past = available_ics[f][-ROLLING_WINDOW:]
                ic_means[f] = np.mean(past) if past else 0

            positive_ics = {f: max(0, ic) for f, ic in ic_means.items()}
            total = sum(positive_ics.values())
            weights = {f: (positive_ics[f] / total if total > 0 else 1.0 / len(factors)) for f in factors}

        mask = panel["date"] == dt
        for f in factors:
            norm_col = f"{f}_norm"
            if norm_col in panel.columns:
                combo.loc[mask] = combo.loc[mask].fillna(0) + panel.loc[mask, norm_col].fillna(0.5) * weights.get(f, 0)

    return combo, {"method": "rolling_ic_weighted", "window": ROLLING_WINDOW}


def combine_rolling_regression(
    panel: pd.DataFrame,
    factors: List[str],
    fwd_col: str = "fwd_ret_20d",
) -> Tuple[pd.Series, Dict]:
    """
    滚动回归法 (v2修复前视偏差):
    替代原combine_regression(双重泄漏), 改用滚动窗口的已实现前瞻收益回归
    """
    print(f"  [滚动回归法-修复前视] 窗口={ROLLING_WINDOW}截面...")

    dates = sorted(panel["date"].unique())
    fwd_days = int(fwd_col.split("_")[2].rstrip("d"))
    rebalance_freq = 5
    ic_lag = max(1, (fwd_days + rebalance_freq - 1) // rebalance_freq)

    # 预计算每个截面的回归系数(用前瞻收益)
    # regression_available = [(available_idx, beta_vector), ...]
    norm_cols = [f"{f}_norm" for f in factors if f"{f}_norm" in panel.columns]
    regression_available = []

    for j_idx, dt in enumerate(dates):
        sub = panel[panel["date"] == dt][norm_cols + [fwd_col]].dropna()
        if len(sub) < 20:
            continue

        X = sub[norm_cols].apply(lambda c: (c - c.mean()) / (c.std() + 1e-8))
        y = sub[fwd_col]

        try:
            XtX = X.T @ X
            Xty = X.T @ y
            beta = np.linalg.solve(XtX, Xty)
            available_idx = j_idx + ic_lag
            if available_idx < len(dates):
                regression_available.append((available_idx, beta))
        except Exception:
            continue

    # 逐截面用滚动窗口的平均回归权重
    combo = pd.Series(0.5, index=panel.index)

    for i, dt in enumerate(dates):
        # 只用 available_idx <= i 的回归系数
        past_betas = [beta for (avail_idx, beta) in regression_available if avail_idx <= i]

        if len(past_betas) < 5:
            weights = {f: 1.0 / len(factors) for f in factors}
        else:
            # 取最近ROLLING_WINDOW个回归系数的平均
            recent_betas = past_betas[-ROLLING_WINDOW:]
            avg_beta = np.mean(recent_betas, axis=0)

            # 正则化: 只保留正权重
            positive_w = {f: max(0, float(avg_beta[j])) for j, f in enumerate(factors) if f"{f}_norm" in norm_cols}
            total = sum(positive_w.values())
            if total > 0:
                weights = {f: positive_w.get(f, 0) / total for f in factors}
            else:
                weights = {f: 1.0 / len(factors) for f in factors}

        mask = panel["date"] == dt
        for f in factors:
            norm_col = f"{f}_norm"
            if norm_col in panel.columns:
                combo.loc[mask] = combo.loc[mask].fillna(0) + panel.loc[mask, norm_col].fillna(0.5) * weights.get(f, 0)

    return combo, {"method": "rolling_regression", "window": ROLLING_WINDOW}


# ==================== 5. 评估组合因子 (含交易成本) ====================
def evaluate_combo_factor(
    panel: pd.DataFrame,
    combo: pd.Series,
    name: str,
    fwd_col: str = "fwd_ret_20d",
) -> Dict:
    """评估组合因子的IC和分层回测 (v2: 含交易成本)"""
    panel = panel.copy()
    panel[name] = combo

    dates = panel["date"].unique()
    ic_list = []
    for dt in dates:
        sub = panel[panel["date"] == dt][[name, fwd_col]].dropna()
        if len(sub) < 10:
            continue
        ic, _ = spearmanr(sub[name].values, sub[fwd_col].values)
        if not np.isnan(ic):
            ic_list.append(ic)

    if len(ic_list) < 5:
        return {"name": name, "ic_mean": np.nan, "icir": np.nan, "ic_tstat": np.nan,
                "long_short_ret": np.nan, "long_short_sharpe": np.nan, "monotonicity": np.nan,
                "long_short_ret_net": np.nan, "long_short_sharpe_net": np.nan,
                "ic_5d": np.nan, "ic_60d": np.nan}

    ic_arr = np.array(ic_list)
    ic_mean = np.mean(ic_arr)
    ic_std = np.std(ic_arr, ddof=1)
    icir = ic_mean / ic_std if ic_std > 0 else 0
    ic_winrate = np.mean(ic_arr > 0)
    ic_tstat = ic_mean / (ic_std / np.sqrt(len(ic_arr))) if ic_std > 0 else 0

    # 分层回测
    quantile_returns = {q: [] for q in range(1, N_QUANTILES + 1)}
    long_short_returns = []

    for dt in dates:
        sub = panel[panel["date"] == dt][[name, fwd_col]].dropna()
        if len(sub) < N_QUANTILES * 2:
            continue
        try:
            sub["quantile"] = pd.qcut(sub[name], N_QUANTILES, labels=False, duplicates="drop")
        except Exception:
            continue
        if sub["quantile"].nunique() < N_QUANTILES:
            continue

        for q in range(N_QUANTILES):
            q_sub = sub[sub["quantile"] == q]
            if len(q_sub) > 0:
                quantile_returns[q + 1].append(q_sub[fwd_col].mean())

        q_high = sub[sub["quantile"] == N_QUANTILES - 1][fwd_col].mean()
        q_low = sub[sub["quantile"] == 0][fwd_col].mean()
        if not np.isnan(q_high) and not np.isnan(q_low):
            long_short_returns.append(q_high - q_low)

    q_means = {q: np.mean(quantile_returns[q]) if quantile_returns[q] else np.nan for q in range(1, N_QUANTILES + 1)}

    if len(long_short_returns) < 5:
        ls_mean, ls_sharpe, ls_cum, mono = np.nan, np.nan, np.nan, np.nan
        ls_mean_net, ls_sharpe_net = np.nan, np.nan
    else:
        ls_arr = np.array(long_short_returns)
        ls_mean = np.mean(ls_arr)
        ls_std = np.std(ls_arr, ddof=1)
        # 年化因子: 252交易日 / 5天截面频率 = 50.4 (修复原sqrt(52))
        ann_factor = np.sqrt(252 / 5)
        ls_sharpe = (ls_mean / ls_std) * ann_factor if ls_std > 0 else 0
        ls_cum = np.sum(ls_arr)

        # v2: 扣减交易成本后的多空收益
        # 每个截面调仓一次, 多头卖出+买入=2次交易, 空头也是2次
        # 简化: 每截面成本 = TURNOVER_COST (假设满仓换手)
        ls_arr_net = ls_arr - TURNOVER_COST
        ls_mean_net = np.mean(ls_arr_net)
        ls_sharpe_net = (ls_mean_net / ls_std) * ann_factor if ls_std > 0 else 0

        q_vals = [q_means[q] for q in range(1, N_QUANTILES + 1) if not np.isnan(q_means[q])]
        if len(q_vals) == N_QUANTILES:
            increases = sum(1 for i in range(1, len(q_vals)) if q_vals[i] > q_vals[i - 1])
            mono = increases / (len(q_vals) - 1)
        else:
            mono = np.nan

    # 多期限IC
    ic_multi = {}
    for p in FORWARD_PERIODS:
        fc = f"fwd_ret_{p}d"
        if fc not in panel.columns:
            continue
        ic_p_list = []
        for dt in dates:
            sub = panel[panel["date"] == dt][[name, fc]].dropna()
            if len(sub) < 10:
                continue
            ic_p, _ = spearmanr(sub[name].values, sub[fc].values)
            if not np.isnan(ic_p):
                ic_p_list.append(ic_p)
        ic_multi[p] = np.mean(ic_p_list) if ic_p_list else np.nan

    return {
        "name": name,
        "ic_mean": ic_mean,
        "icir": icir,
        "ic_winrate": ic_winrate,
        "ic_tstat": ic_tstat,
        "q1": q_means[1], "q2": q_means[2], "q3": q_means[3], "q4": q_means[4], "q5": q_means[5],
        "long_short_ret": ls_mean,
        "long_short_sharpe": ls_sharpe,
        "long_short_ret_net": ls_mean_net,
        "long_short_sharpe_net": ls_sharpe_net,
        "monotonicity": mono,
        "long_short_cum": ls_cum,
        "ic_5d": ic_multi.get(5, np.nan),
        "ic_10d": ic_multi.get(10, np.nan),
        "ic_20d": ic_multi.get(20, np.nan),
        "ic_60d": ic_multi.get(60, np.nan),
    }


# ==================== 6. Walk-Forward验证 (借鉴AlphaEvo) ====================
def walk_forward_evaluation(
    panel: pd.DataFrame,
    factors: List[str],
    fwd_col: str = "fwd_ret_20d",
) -> List[Dict]:
    """
    Walk-Forward验证 (expanding window):
    - 将样本分为多段train/test
    - train段计算ICIR权重, test段评估OOS效果
    - 报告每折的train ICIR vs test ICIR

    借鉴AlphaEvo的compute_walk_forward设计
    """
    print(f"\n[Walk-Forward验证] {WF_N_FOLDS}折, expanding window...")

    dates = sorted(panel["date"].unique())
    n = len(dates)

    # 初始训练集大小
    min_train = max(20, int(n * WF_TRAIN_PCT))
    remaining = n - min_train
    fold_test_size = max(10, remaining // WF_N_FOLDS)

    print(f"  总截面数: {n}, 初始训练: {min_train}, 每折测试: {fold_test_size}")

    folds = []
    train_end = min_train

    for fold_num in range(1, WF_N_FOLDS + 1):
        test_end = min(n, train_end + fold_test_size)
        train_dates = dates[:train_end]
        test_dates = dates[train_end:test_end]

        if len(test_dates) < 5:
            break

        # Train段: 计算每个因子的ICIR
        train_icirs = {}
        for f in factors:
            if f not in panel.columns:
                continue
            ic_list = []
            for dt in train_dates:
                sub = panel[panel["date"] == dt][[f, fwd_col]].dropna()
                if len(sub) < 10:
                    continue
                ic, _ = spearmanr(sub[f].values, sub[fwd_col].values)
                if not np.isnan(ic):
                    ic_list.append(ic)
            if len(ic_list) >= 5:
                ic_arr = np.array(ic_list)
                ic_mean = np.mean(ic_arr)
                ic_std = np.std(ic_arr, ddof=1)
                train_icirs[f] = ic_mean / ic_std if ic_std > 0 else 0
            else:
                train_icirs[f] = 0

        # 用train段ICIR作权重
        positive_icirs = {f: max(0, icir) for f, icir in train_icirs.items()}
        total = sum(positive_icirs.values())
        if total > 0:
            weights = {f: positive_icirs[f] / total for f in factors}
        else:
            weights = {f: 1.0 / len(factors) for f in factors}

        # Test段: 用train权重组合因子, 评估OOS ICIR
        test_ic_list = []
        test_ls_list = []
        for dt in test_dates:
            sub = panel[panel["date"] == dt][[f for f in factors if f in panel.columns] + [fwd_col]].dropna()
            if len(sub) < 10:
                continue

            # 组合因子值
            combo_val = sum(sub[f].rank(pct=True).fillna(0.5) * weights.get(f, 0) for f in factors if f in sub.columns)
            ic, _ = spearmanr(combo_val.values, sub[fwd_col].values)
            if not np.isnan(ic):
                test_ic_list.append(ic)

            # 分层多空
            if len(sub) >= N_QUANTILES * 2:
                try:
                    q_labels = pd.qcut(combo_val, N_QUANTILES, labels=False, duplicates="drop")
                    if q_labels.nunique() == N_QUANTILES:
                        q_high = sub[fwd_col][q_labels == N_QUANTILES - 1].mean()
                        q_low = sub[fwd_col][q_labels == 0].mean()
                        if not np.isnan(q_high) and not np.isnan(q_low):
                            test_ls_list.append(q_high - q_low)
                except Exception:
                    pass

        # Train段ICIR (加权组合)
        train_combo_ic = []
        for dt in train_dates:
            sub = panel[panel["date"] == dt][[f for f in factors if f in panel.columns] + [fwd_col]].dropna()
            if len(sub) < 10:
                continue
            combo_val = sum(sub[f].rank(pct=True).fillna(0.5) * weights.get(f, 0) for f in factors if f in sub.columns)
            ic, _ = spearmanr(combo_val.values, sub[fwd_col].values)
            if not np.isnan(ic):
                train_combo_ic.append(ic)

        train_icir_combo = np.nan
        if len(train_combo_ic) >= 5:
            arr = np.array(train_combo_ic)
            train_icir_combo = np.mean(arr) / np.std(arr, ddof=1) if np.std(arr, ddof=1) > 0 else 0

        test_icir = np.nan
        test_ic_mean = np.nan
        if len(test_ic_list) >= 3:
            arr = np.array(test_ic_list)
            test_ic_mean = np.mean(arr)
            test_std = np.std(arr, ddof=1)
            test_icir = test_ic_mean / test_std if test_std > 0 else 0

        # OOS多空收益(扣成本)
        test_ls_net = np.nan
        if len(test_ls_list) >= 3:
            test_ls_net = np.mean(test_ls_list) - TURNOVER_COST

        gap = abs(train_icir_combo - test_icir) if not (np.isnan(train_icir_combo) or np.isnan(test_icir)) else np.nan

        fold_result = {
            "fold": fold_num,
            "train_dates": f"{train_dates[0][:10]}~{train_dates[-1][:10]}",
            "test_dates": f"{test_dates[0][:10]}~{test_dates[-1][:10]}",
            "train_n": len(train_dates),
            "test_n": len(test_dates),
            "train_icir": round(train_icir_combo, 4) if not np.isnan(train_icir_combo) else "N/A",
            "test_icir": round(test_icir, 4) if not np.isnan(test_icir) else "N/A",
            "test_ic_mean": round(test_ic_mean, 4) if not np.isnan(test_ic_mean) else "N/A",
            "test_ls_net": round(test_ls_net, 4) if not np.isnan(test_ls_net) else "N/A",
            "gap": round(gap, 4) if not np.isnan(gap) else "N/A",
            "overfit": "是" if (not np.isnan(gap) and gap > 0.15) else "否",
        }
        folds.append(fold_result)

        print(f"  Fold {fold_num}: train={fold_result['train_dates']}({len(train_dates)}截面) "
              f"test={fold_result['test_dates']}({len(test_dates)}截面) "
              f"trainICIR={fold_result['train_icir']} testICIR={fold_result['test_icir']} "
              f"gap={fold_result['gap']} 过拟合={fold_result['overfit']}")

        train_end = test_end  # expanding window

    return folds


# ==================== 6b. Regime-Switching评估 (v4新增) ====================
def regime_switching_evaluation(
    panel: pd.DataFrame,
    factors: List[str],
    fwd_col: str = "fwd_ret_20d",
) -> Dict:
    """
    按市场状态分组评估因子ICIR
    - BULL (macro_regime_score >= 0.7): 趋势市, 量价因子应有效
    - NEUTRAL (0.3 < score < 0.7): 震荡市, 因子可能失效
    - BEAR (score <= 0.3): 熊市, 因子表现取决于分化程度

    借鉴AlphaEvo的regime holdout思想
    """
    print(f"\n[Regime-Switching评估] 按市场状态分组ICIR...")

    # 确定市场状态
    if "macro_regime_score" not in panel.columns:
        print("  警告: macro_regime_score不可用, 跳过regime评估")
        return {}

    regime_results = {}
    for regime_name, regime_cond in [
        ("BULL(趋势)", panel["macro_regime_score"] >= 0.7),
        ("NEUTRAL(震荡)", (panel["macro_regime_score"] > 0.3) & (panel["macro_regime_score"] < 0.7)),
        ("BEAR(熊市)", panel["macro_regime_score"] <= 0.3),
    ]:
        regime_panel = panel[regime_cond]
        n_sections = regime_panel["date"].nunique()
        if n_sections < 5:
            print(f"  {regime_name}: 截面数不足 ({n_sections}), 跳过")
            continue

        regime_factors = {}
        for f in factors:
            if f not in regime_panel.columns:
                continue
            ic_list = []
            for dt in regime_panel["date"].unique():
                sub = regime_panel[regime_panel["date"] == dt][[f, fwd_col]].dropna()
                if len(sub) < 10:
                    continue
                ic, _ = spearmanr(sub[f].values, sub[fwd_col].values)
                if not np.isnan(ic):
                    ic_list.append(ic)

            if len(ic_list) >= 5:
                ic_arr = np.array(ic_list)
                ic_mean = np.mean(ic_arr)
                ic_std = np.std(ic_arr, ddof=1)
                icir = ic_mean / ic_std if ic_std > 0 else 0
                regime_factors[f] = {
                    "ic_mean": round(ic_mean, 4),
                    "icir": round(icir, 4),
                    "ic_winrate": round(np.mean(np.array(ic_list) > 0), 4),
                    "n_sections": len(ic_list),
                }

        regime_results[regime_name] = {
            "n_sections": n_sections,
            "n_records": len(regime_panel),
            "factors": regime_factors,
        }

        # 打印该regime下的Top 3因子
        sorted_factors = sorted(regime_factors.items(), key=lambda x: -x[1]["icir"])
        top3 = sorted_factors[:3]
        print(f"  {regime_name} ({n_sections}截面): "
              + " | ".join(f"{f}={info['icir']:.4f}" for f, info in top3))

    return regime_results


# ==================== 7. 报告生成 ====================
def generate_report(
    panel: pd.DataFrame,
    results: List[Dict],
    combos: Dict[str, pd.Series],
    wf_folds: List[Dict],
    regime_results: Dict = None,
):
    """生成组合优化报告"""
    print("\n" + "=" * 80)
    print("因子组合优化报告 (v4 — 新alpha因子 + Regime择时 + 扩大WF)")
    print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"交易成本: 滑点={SLIPPAGE*100}% 佣金={COMMISSION*100}% 双边成本={TURNOVER_COST*100}%")
    print("=" * 80)

    os.makedirs(REPORT_DIR, exist_ok=True)

    # 单因子baseline
    print("\n一、单因子Baseline (Top 5)")
    print("-" * 120)
    print(f"{'因子':<30} {'IC均值':>10} {'ICIR':>10} {'t值':>8} {'多空(毛)':>10} {'多空(净)':>10} {'夏普(毛)':>10} {'夏普(净)':>10} {'单调性':>8}")
    print("-" * 120)
    for r in results[:5]:
        print(f"{r['name']:<30} {r['ic_mean']:>10.4f} {r['icir']:>10.4f} {r['ic_tstat']:>8.2f} "
              f"{r['long_short_ret']:>10.4f} {r['long_short_ret_net']:>10.4f} "
              f"{r['long_short_sharpe']:>10.2f} {r['long_short_sharpe_net']:>10.2f} {r['monotonicity']:>8.2%}")

    # 组合因子
    combo_results = [r for r in results if r["name"].startswith("combo_")]
    print("\n二、组合因子效果 (全部无前视偏差)")
    print("-" * 120)
    print(f"{'组合方法':<30} {'IC均值':>10} {'ICIR':>10} {'t值':>8} {'多空(毛)':>10} {'多空(净)':>10} {'夏普(净)':>10} {'单调性':>8} {'IC衰减':>15}")
    print("-" * 130)
    for r in combo_results:
        decay = f"{r['ic_5d']:.4f}→{r['ic_60d']:.4f}" if not (np.isnan(r['ic_5d']) or np.isnan(r['ic_60d'])) else "N/A"
        print(f"{r['name']:<30} {r['ic_mean']:>10.4f} {r['icir']:>10.4f} {r['ic_tstat']:>8.2f} "
              f"{r['long_short_ret']:>10.4f} {r['long_short_ret_net']:>10.4f} "
              f"{r['long_short_sharpe_net']:>10.2f} {r['monotonicity']:>8.2%} {decay:>15}")

    # 提升对比
    print("\n三、组合 vs 单因子 提升对比")
    print("-" * 80)
    best_single = max(results[:5], key=lambda x: x["icir"] if not np.isnan(x["icir"]) else -999)
    best_combo = max(combo_results, key=lambda x: x["icir"] if not np.isnan(x["icir"]) else -999)

    if not np.isnan(best_single["icir"]) and not np.isnan(best_combo["icir"]):
        icir_lift = (best_combo["icir"] - best_single["icir"]) / best_single["icir"] * 100
        print(f"  最佳单因子: {best_single['name']} (ICIR={best_single['icir']:.4f}, 净夏普={best_single['long_short_sharpe_net']:.2f})")
        print(f"  最佳组合:   {best_combo['name']} (ICIR={best_combo['icir']:.4f}, 净夏普={best_combo['long_short_sharpe_net']:.2f})")
        print(f"  ICIR提升:   {icir_lift:+.1f}%")
        print(f"  组合后t值:  {best_combo['ic_tstat']:.2f} {'(显著!)' if abs(best_combo['ic_tstat']) > 1.96 else '(不显著)'}")

    # Walk-Forward结果
    print("\n四、Walk-Forward验证 (expanding window)")
    print("-" * 100)
    print(f"{'Fold':<6} {'训练期':<25} {'测试期':<25} {'训练ICIR':>10} {'测试ICIR':>10} {'IC均值':>10} {'净多空':>10} {'Gap':>8} {'过拟合':>8}")
    print("-" * 120)
    for f in wf_folds:
        print(f"{f['fold']:<6} {f['train_dates']:<25} {f['test_dates']:<25} "
              f"{f['train_icir']:>10} {f['test_icir']:>10} {f['test_ic_mean']:>10} "
              f"{f['test_ls_net']:>10} {f['gap']:>8} {f['overfit']:>8}")

    # 过拟合判断
    valid_folds = [f for f in wf_folds if f["test_icir"] != "N/A" and f["train_icir"] != "N/A"]
    if valid_folds:
        overfit_count = sum(1 for f in valid_folds if f["overfit"] == "是")
        avg_test_icir = np.mean([f["test_icir"] for f in valid_folds])
        print(f"\n  平均OOS ICIR: {avg_test_icir:.4f}")
        print(f"  过拟合折数: {overfit_count}/{len(valid_folds)}")
        if overfit_count == 0 and avg_test_icir > 0:
            print(f"  结论: 组合因子在样本外仍有效, 无明显过拟合")
        elif avg_test_icir > 0:
            print(f"  结论: 组合因子样本外ICIR为正, 但部分折存在过拟合风险")
        else:
            print(f"  结论: 组合因子样本外ICIR为负, 过拟合严重, 不可用")

    # Regime-Switching结果 (v4新增)
    if regime_results:
        print("\n五、Regime-Switching评估 (按市场状态分组)")
        print("-" * 100)
        for regime_name, info in regime_results.items():
            print(f"\n  [{regime_name}] {info['n_sections']}截面, {info['n_records']}条记录")
            print(f"  {'因子':<30} {'IC均值':>10} {'ICIR':>10} {'IC胜率':>10} {'截面数':>8}")
            print(f"  {'-'*70}")
            sorted_f = sorted(info["factors"].items(), key=lambda x: -x[1]["icir"])
            for f, fi in sorted_f[:5]:
                print(f"  {f:<30} {fi['ic_mean']:>10.4f} {fi['icir']:>10.4f} {fi['ic_winrate']:>10.2%} {fi['n_sections']:>8}")

        # Regime择时建议
        print(f"\n  [Regime择时建议]")
        bull_factors = regime_results.get("BULL(趋势)", {}).get("factors", {})
        neutral_factors = regime_results.get("NEUTRAL(震荡)", {}).get("factors", {})
        bear_factors = regime_results.get("BEAR(熊市)", {}).get("factors", {})

        if bull_factors and neutral_factors:
            bull_best = max(bull_factors.items(), key=lambda x: x[1]["icir"])
            neutral_best = max(neutral_factors.items(), key=lambda x: x[1]["icir"])
            print(f"  趋势市最佳: {bull_best[0]} (ICIR={bull_best[1]['icir']:.4f})")
            print(f"  震荡市最佳: {neutral_best[0]} (ICIR={neutral_best[1]['icir']:.4f})")
            if neutral_best[1]["icir"] < 0.05:
                print(f"  ⚠ 震荡市所有因子接近失效, 建议空仓或降低仓位")
            if bull_best[1]["icir"] > neutral_best[1]["icir"] * 2:
                print(f"  ⚠ 因子在趋势市效果远好于震荡市, 建议只在趋势市使用")

    # CSV输出
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(REPORT_DIR, "combination_results.csv"), index=False, encoding="utf-8-sig")

    wf_df = pd.DataFrame(wf_folds)
    wf_df.to_csv(os.path.join(REPORT_DIR, "walk_forward_results.csv"), index=False, encoding="utf-8-sig")

    panel_with_combos = panel.copy()
    for name, combo in combos.items():
        panel_with_combos[name] = combo
    panel_with_combos.to_csv(os.path.join(REPORT_DIR, "panel_with_combos.csv"), index=False, encoding="utf-8-sig")

    # 摘要
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("因子组合优化报告 (v2 — 修复前视偏差 + Walk-Forward + 交易成本)")
    summary_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append(f"交易成本: 滑点={SLIPPAGE*100}% 佣金={COMMISSION*100}% 双边={TURNOVER_COST*100}%")
    summary_lines.append(f"组合因子数: {len(COMBINE_FACTORS)}")
    summary_lines.append(f"滚动窗口: {ROLLING_WINDOW}截面")
    summary_lines.append("=" * 80)
    summary_lines.append("")
    summary_lines.append("一、单因子Baseline (Top 5)")
    summary_lines.append(f"{'因子':<30} {'ICIR':>10} {'t值':>8} {'多空(净)':>10} {'夏普(净)':>10}")
    summary_lines.append("-" * 80)
    for r in results[:5]:
        summary_lines.append(f"{r['name']:<30} {r['icir']:>10.4f} {r['ic_tstat']:>8.2f} "
                             f"{r['long_short_ret_net']:>10.4f} {r['long_short_sharpe_net']:>10.2f}")
    summary_lines.append("")
    summary_lines.append("二、组合因子效果 (无前视偏差)")
    summary_lines.append(f"{'方法':<30} {'ICIR':>10} {'t值':>8} {'多空(净)':>10} {'夏普(净)':>10}")
    summary_lines.append("-" * 80)
    for r in combo_results:
        summary_lines.append(f"{r['name']:<30} {r['icir']:>10.4f} {r['ic_tstat']:>8.2f} "
                             f"{r['long_short_ret_net']:>10.4f} {r['long_short_sharpe_net']:>10.2f}")
    summary_lines.append("")
    summary_lines.append("三、Walk-Forward验证")
    summary_lines.append(f"{'Fold':<6} {'训练ICIR':>10} {'测试ICIR':>10} {'Gap':>8} {'过拟合':>8}")
    summary_lines.append("-" * 50)
    for f in wf_folds:
        summary_lines.append(f"{f['fold']:<6} {f['train_icir']:>10} {f['test_icir']:>10} {f['gap']:>8} {f['overfit']:>8}")
    if valid_folds:
        avg_test = np.mean([f["test_icir"] for f in valid_folds])
        summary_lines.append(f"\n平均OOS ICIR: {avg_test:.4f}")

    summary_text = "\n".join(summary_lines)
    with open(os.path.join(REPORT_DIR, "combination_report.txt"), "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(f"\n报告已保存到: {REPORT_DIR}")


# ==================== 主流程 ====================
def main():
    print("=" * 80)
    print("因子组合优化 v4 — 新alpha因子 + Regime择时 + 扩大Walk-Forward")
    print("=" * 80)

    # 1. 加载数据并构建panel
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

    panel = build_factor_panel(all_data, cross_dates)
    panel = calculate_forward_returns(all_data, panel)

    # 2. 去共线性检查
    available_factors = [f for f in COMBINE_FACTORS if f in panel.columns]
    print(f"\n可用组合因子: {len(available_factors)}/{len(COMBINE_FACTORS)}")
    check_collinearity(panel, available_factors)

    # 3. 标准化
    panel = normalize_factors(panel, available_factors)

    # 4. 计算各种组合 (全部无前视偏差)
    print("\n[2] 计算因子组合 (全部无前视偏差)...")
    combos = {}

    # 4.1 等权 (无前视)
    combos["combo_equal"], _ = combine_equal_weight(panel, available_factors)

    # 4.2 滚动IC加权 (修复前视, 替代原全样本IC加权)
    combos["combo_rolling_ic"], _ = combine_rolling_ic_weighted(panel, available_factors)

    # 4.3 滚动ICIR加权 (修复前视)
    combos["combo_rolling_icir"], _ = combine_rolling_icir_fixed(panel, available_factors)

    # 4.4 滚动回归法 (修复前视, 替代原全样本回归)
    combos["combo_rolling_reg"], _ = combine_rolling_regression(panel, available_factors)

    # 5. 评估所有因子 (含v4新增因子)
    print("\n[3] 评估组合效果 (含交易成本)...")
    results = []
    # 核心单因子 (含v4新增)
    baseline_factors = [
        "pv_volume_trend", "sent_combined_score",
        "pv_turnover_change", "pv_price_accel", "pv_vol_price_divergence",
        "sector_combined_score",
    ]
    for f in baseline_factors:
        if f in panel.columns:
            r = evaluate_combo_factor(panel, panel[f], f)
            results.append(r)

    for name, combo in combos.items():
        r = evaluate_combo_factor(panel, combo, name)
        results.append(r)

    results.sort(key=lambda x: x["icir"] if not np.isnan(x["icir"]) else -999, reverse=True)

    # 6. Walk-Forward验证 (v4: 扩大测试集)
    wf_folds = walk_forward_evaluation(panel, available_factors)

    # 7. Regime-Switching评估 (v4新增)
    regime_results = regime_switching_evaluation(panel, available_factors)

    # 8. 报告
    generate_report(panel, results, combos, wf_folds, regime_results)
    print("\n优化完成!")


if __name__ == "__main__":
    main()
