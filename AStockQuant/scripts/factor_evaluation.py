# -*- coding: utf-8 -*-
"""
因子对比评估脚本
=================
对8层因子体系中的20个核心因子进行IC分析+分层回测，找出效果最好的因子。

评估方法:
1. IC分析: 截面Rank IC (Spearman), IC均值/ICIR/IC胜率/IC衰减
2. 分层回测: 按因子值分5档, 比较多空收益/单调性/夏普

输出: AStockQuant/reports/factor_evaluation/
"""

import os
import sys
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def spearmanr(x, y):
    """纯pandas实现的Spearman秩相关, 替代scipy.stats.spearmanr"""
    s = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(s) < 3:
        return np.nan, np.nan
    rx = s["x"].rank()
    ry = s["y"].rank()
    n = len(s)
    # Pearson on ranks = Spearman
    mean_rx = rx.mean()
    mean_ry = ry.mean()
    cov = ((rx - mean_rx) * (ry - mean_ry)).sum()
    std_rx = np.sqrt(((rx - mean_rx) ** 2).sum())
    std_ry = np.sqrt(((ry - mean_ry) ** 2).sum())
    if std_rx == 0 or std_ry == 0:
        return np.nan, np.nan
    rho = cov / (std_rx * std_ry)
    return float(rho), 0.0

# 确保能 import AStockQuant 模块
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from layers.layer1_macro import MacroLayer
from layers.layer2_rules import RulesLayer
from layers.layer3_sector import SectorLayer
from layers.layer4_capital import CapitalLayer
from layers.layer5_sentiment import SentimentLayer
from layers.layer6_price_vol import PriceVolumeLayer
from layers.layer7_technical import TechnicalLayer
from layers.layer8_micro import BeliefLayer

warnings.filterwarnings("ignore", category=RuntimeWarning)

# 禁用akshare网络调用, 强制L4资金层使用量价近似模式(避免历史回测中逐标的调用API)
import layers.layer4_capital as _l4_mod
_l4_mod._AK = False

# ==================== 配置 ====================
DATA_CACHE_DIR = os.path.join(PROJECT_ROOT, "data_cache")
REPORT_DIR = os.path.join(PROJECT_ROOT, "reports", "factor_evaluation")
MARKET_PROXY_CODE = "510510"  # 中证500ETF, 2013年起, 作为L1宏观层市场代理
REBALANCE_FREQ = 5  # 每5个交易日一个截面(周频)
LOOKBACK_MIN = 120  # 最少历史数据行数(确保RPS120可算)
FORWARD_PERIODS = [5, 10, 20, 60]  # 前瞻收益期限
N_QUANTILES = 5  # 分层数
START_DATE = "2024-01-01"  # 评估起始日期(确保有足够lookback)
END_DATE = "2026-03-31"  # 评估结束日期(确保60日前瞻收益可算)

# 评估因子清单 (含v3新增/改进因子)
EVAL_FACTORS = [
    # L1宏观 - 原有因子(截面无区分度) + v3新增相对强度因子
    "macro_regime_score", "trend_strength",
    "macro_relative_strength", "macro_relative_score", "macro_relative_position",
    # L3板块 - v3改进sector_combined_score(消除共线性)
    "sector_combined_score", "sector_momentum", "sector_momentum_long", "sector_is_leader",
    # L4资金
    "capital_score",
    # L5情绪 - v3改进sent_rsi_score(正向) + 新增sent_rsi_momentum
    "sent_combined_score", "sent_rsi_score", "sent_rsi_momentum", "sent_volatility_score",
    # L6量价 - v3新增pv_volatility_contraction + v4新增独立alpha因子
    "pv_rps_combined", "pv_rps_50", "pv_vcp_quality", "pv_volatility_contraction",
    "pv_score", "pv_volume_trend", "pv_obv_trend",
    "pv_turnover_change", "pv_price_accel", "pv_vol_price_divergence",
    # L7技术
    "tech_pattern_score", "tech_ma_score", "tech_rsi_score",
    # L8信念
    "belief_posterior",
]


# ==================== 1. 数据加载 ====================
def load_all_etf_data() -> Dict[str, pd.DataFrame]:
    """加载 data_cache/ 下所有ETF的OHLCV数据"""
    all_data = {}
    files = [f for f in os.listdir(DATA_CACHE_DIR) if f.endswith(".csv")]
    print(f"[1/6] 加载数据: 发现 {len(files)} 个CSV文件")

    for f in sorted(files):
        code = f.replace(".csv", "")
        try:
            df = pd.read_csv(os.path.join(DATA_CACHE_DIR, f))
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            # 确保列名统一
            required = ["open", "high", "low", "close", "volume"]
            if all(c in df.columns for c in required):
                df = df[required].dropna()
                if len(df) > 0:
                    all_data[code] = df
        except Exception as e:
            print(f"  警告: 加载 {code} 失败: {e}")

    print(f"  成功加载 {len(all_data)} 只ETF")
    return all_data


def get_cross_section_dates(all_dates: List[pd.Timestamp]) -> List[pd.Timestamp]:
    """生成周频截面日期"""
    dates = pd.DatetimeIndex(sorted(all_dates))
    mask = (dates >= pd.Timestamp(START_DATE)) & (dates <= pd.Timestamp(END_DATE))
    valid_dates = dates[mask]
    # 每5个交易日取一个
    cross_dates = valid_dates[::REBALANCE_FREQ]
    print(f"[2/6] 截面日期: {len(cross_dates)} 个周频截面 ({cross_dates[0].date()} ~ {cross_dates[-1].date()})")
    return list(cross_dates)


# ==================== 2. 因子Panel构建 ====================
def build_factor_panel(
    all_data: Dict[str, pd.DataFrame],
    cross_dates: List[pd.Timestamp],
) -> pd.DataFrame:
    """
    在每个截面日期, 对每只ETF调用8层extract_features, 构建因子panel

    Returns: DataFrame[date, symbol, factor1, factor2, ...]
    """
    print(f"[3/6] 构建因子panel (共 {len(cross_dates)} 个截面)...")

    # 实例化各层
    layers = {
        "macro": MacroLayer(),
        "sector": SectorLayer(),
        "capital": CapitalLayer(),
        "sentiment": SentimentLayer(),
        "price_vol": PriceVolumeLayer(),
        "technical": TechnicalLayer(),
        "belief": BeliefLayer(),
    }

    market_df = all_data.get(MARKET_PROXY_CODE)
    if market_df is None:
        print(f"  错误: 市场代理 {MARKET_PROXY_CODE} 不存在, 改用510300")
        market_df = all_data.get("510300")

    records = []
    for i, as_of in enumerate(cross_dates):
        as_of_str = as_of.strftime("%Y-%m-%d")
        as_of_ts = pd.Timestamp(as_of)

        # 筛选有足够历史数据的ETF
        eligible_symbols = []
        for sym, df in all_data.items():
            df_trunc = df[df.index <= as_of_ts]
            if len(df_trunc) >= LOOKBACK_MIN:
                eligible_symbols.append(sym)

        if len(eligible_symbols) < 10:
            continue

        # 预计算跨标的20日收益率 (供L3板块排名)
        all_returns = {}
        for sym in eligible_symbols:
            df_trunc = all_data[sym][all_data[sym].index <= as_of_ts]
            if len(df_trunc) >= 21:
                all_returns[sym] = float(df_trunc["close"].pct_change(20).iloc[-1])
            else:
                all_returns[sym] = None

        # 截取市场数据
        market_trunc = None
        if market_df is not None:
            market_trunc = market_df[market_df.index <= as_of_ts]

        # 构建ctx
        # 注: 不传all_data避免L5情绪层O(N²)遍历全市场, L5会降级用market_prices_df计算市场情绪
        layer_ctx_base = {
            "name": "",
            "all_sector_returns": all_returns,
            "market_prices_df": market_trunc,
        }

        # 逐ETF计算因子
        for sym in eligible_symbols:
            df_trunc = all_data[sym][all_data[sym].index <= as_of_ts]
            ctx = dict(layer_ctx_base)

            # 按顺序调用各层 (层间证据传递)
            features = {}
            layer_ctx = dict(ctx)
            for layer_name, layer in layers.items():
                try:
                    sig = __import__("inspect").signature(layer.extract_features)
                    if "as_of_date" in sig.parameters:
                        feats = layer.extract_features(sym, df_trunc, layer_ctx, as_of_date=as_of_str)
                    else:
                        feats = layer.extract_features(sym, df_trunc, layer_ctx)
                    if feats:
                        features.update(feats)
                        layer_ctx.update(feats)  # 层间证据传递
                except Exception as e:
                    pass  # 单层失败跳过

            # 收集目标因子
            record = {"date": as_of_str, "symbol": sym}
            for factor in EVAL_FACTORS:
                val = features.get(factor, np.nan)
                # 转换bool为float
                if isinstance(val, bool):
                    val = 1.0 if val else 0.0
                elif isinstance(val, str):
                    val = np.nan
                record[factor] = val
            records.append(record)

        if (i + 1) % 20 == 0 or i == len(cross_dates) - 1:
            print(f"  进度: {i+1}/{len(cross_dates)} 截面, 累计 {len(records)} 条记录")

    panel = pd.DataFrame(records)
    print(f"  panel构建完成: {panel.shape[0]} 行 x {panel.shape[1]} 列")
    return panel


# ==================== 3. 前瞻收益计算 ====================
def calculate_forward_returns(
    all_data: Dict[str, pd.DataFrame],
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """计算各期限前瞻收益并合并到panel"""
    print(f"[4/6] 计算前瞻收益 ({FORWARD_PERIODS}日)...")

    fwd_records = []
    for _, row in panel.iterrows():
        sym = row["symbol"]
        as_of = pd.Timestamp(row["date"])
        df = all_data.get(sym)
        if df is None:
            fwd_records.append({p: np.nan for p in FORWARD_PERIODS})
            continue

        future_df = df[df.index > as_of]
        if len(future_df) == 0:
            fwd_records.append({p: np.nan for p in FORWARD_PERIODS})
            continue

        cur_close = df[df.index <= as_of]["close"].iloc[-1]
        fwd = {}
        for p in FORWARD_PERIODS:
            if len(future_df) >= p:
                fwd[p] = future_df["close"].iloc[p - 1] / cur_close - 1
            else:
                fwd[p] = np.nan
        fwd_records.append(fwd)

    fwd_df = pd.DataFrame(fwd_records)
    fwd_df.columns = [f"fwd_ret_{p}d" for p in FORWARD_PERIODS]
    result = pd.concat([panel.reset_index(drop=True), fwd_df], axis=1)
    return result


# ==================== 4. IC分析 ====================
def compute_ic_analysis(panel: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    计算每个因子的IC指标

    Returns:
        ic_summary: 每因子的IC均值/ICIR/IC胜率/t值 (多期限)
        ic_timeseries: 每因子每截面的IC值 (20日期限)
    """
    print(f"[5/6] IC分析...")

    fwd_cols = [f"fwd_ret_{p}d" for p in FORWARD_PERIODS]
    dates = panel["date"].unique()

    ic_summary_rows = []
    ic_ts_records = []

    for factor in EVAL_FACTORS:
        if factor not in panel.columns:
            continue

        summary = {"factor": factor}
        for fwd_col in fwd_cols:
            # 逐截面计算Spearman IC
            ic_list = []
            for dt in dates:
                sub = panel[panel["date"] == dt][[factor, fwd_col]].dropna()
                if len(sub) < 10:
                    continue
                try:
                    ic, _ = spearmanr(sub[factor].values, sub[fwd_col].values)
                    if not np.isnan(ic):
                        ic_list.append(ic)
                except Exception:
                    continue

            if len(ic_list) < 5:
                summary[f"ic_mean_{fwd_col}"] = np.nan
                summary[f"icir_{fwd_col}"] = np.nan
                summary[f"ic_winrate_{fwd_col}"] = np.nan
                summary[f"ic_tstat_{fwd_col}"] = np.nan
                continue

            ic_arr = np.array(ic_list)
            ic_mean = np.mean(ic_arr)
            ic_std = np.std(ic_arr, ddof=1)
            icir = ic_mean / ic_std if ic_std > 0 else 0
            ic_winrate = np.mean(ic_arr > 0)
            ic_tstat = ic_mean / (ic_std / np.sqrt(len(ic_arr))) if ic_std > 0 else 0

            summary[f"ic_mean_{fwd_col}"] = ic_mean
            summary[f"icir_{fwd_col}"] = icir
            summary[f"ic_winrate_{fwd_col}"] = ic_winrate
            summary[f"ic_tstat_{fwd_col}"] = ic_tstat

            # 20日期限记录时序 (修复日期错位: 用实际产生IC的日期, 而非dates[:len])
            if fwd_col == "fwd_ret_20d":
                valid_dates = []
                for dt in dates:
                    sub = panel[panel["date"] == dt][[factor, fwd_col]].dropna()
                    if len(sub) < 10:
                        continue
                    try:
                        ic, _ = spearmanr(sub[factor].values, sub[fwd_col].values)
                        if not np.isnan(ic):
                            valid_dates.append((dt, ic))
                    except Exception:
                        continue
                for dt, ic in valid_dates:
                    ic_ts_records.append({"date": dt, "factor": factor, "ic": ic})

        ic_summary_rows.append(summary)

    ic_summary = pd.DataFrame(ic_summary_rows)
    ic_timeseries = pd.DataFrame(ic_ts_records)

    # 按ICIR(20日)降序排序
    if "icir_fwd_ret_20d" in ic_summary.columns:
        ic_summary = ic_summary.sort_values("icir_fwd_ret_20d", ascending=False).reset_index(drop=True)

    return ic_summary, ic_timeseries


# ==================== 5. 分层回测 ====================
def compute_quantile_backtest(panel: pd.DataFrame) -> pd.DataFrame:
    """
    对每个因子按值分5档, 计算各组平均收益/多空收益/单调性/夏普
    使用20日前瞻收益作为主期限
    """
    print(f"[5.5] 分层回测 (20日前瞻收益)...")

    fwd_col = "fwd_ret_20d"
    dates = panel["date"].unique()
    results = []

    for factor in EVAL_FACTORS:
        if factor not in panel.columns:
            continue

        # 逐截面分组, 计算各组平均收益
        quantile_returns = {q: [] for q in range(1, N_QUANTILES + 1)}
        long_short_returns = []

        for dt in dates:
            sub = panel[panel["date"] == dt][[factor, fwd_col]].dropna()
            if len(sub) < N_QUANTILES * 2:
                continue

            try:
                sub["quantile"] = pd.qcut(sub[factor], N_QUANTILES, labels=False, duplicates="drop")
            except Exception:
                continue

            if sub["quantile"].nunique() < N_QUANTILES:
                continue

            for q in range(N_QUANTILES):
                q_sub = sub[sub["quantile"] == q]
                if len(q_sub) > 0:
                    quantile_returns[q + 1].append(q_sub[fwd_col].mean())

            # 多空收益: 最高档 - 最低档
            q_high = sub[sub["quantile"] == N_QUANTILES - 1][fwd_col].mean()
            q_low = sub[sub["quantile"] == 0][fwd_col].mean()
            if not np.isnan(q_high) and not np.isnan(q_low):
                long_short_returns.append(q_high - q_low)

        if len(long_short_returns) < 5:
            results.append({
                "factor": factor, "q1_ret": np.nan, "q2_ret": np.nan,
                "q3_ret": np.nan, "q4_ret": np.nan, "q5_ret": np.nan,
                "long_short_ret": np.nan, "long_short_sharpe": np.nan,
                "monotonicity": np.nan, "long_short_cum": np.nan,
            })
            continue

        # 各组平均收益
        q_means = {}
        for q in range(1, N_QUANTILES + 1):
            q_means[q] = np.mean(quantile_returns[q]) if quantile_returns[q] else np.nan

        # 多空收益统计
        ls_arr = np.array(long_short_returns)
        ls_mean = np.mean(ls_arr)
        ls_std = np.std(ls_arr, ddof=1)
        ls_sharpe = (ls_mean / ls_std) * np.sqrt(252 / 5) if ls_std > 0 else 0  # 年化(周频: 252交易日/5天=50.4)
        ls_cum = np.sum(ls_arr)

        # 单调性评分: 5组收益是否单调递增
        q_vals = [q_means[q] for q in range(1, N_QUANTILES + 1) if not np.isnan(q_means[q])]
        if len(q_vals) == N_QUANTILES:
            # 计算相邻组递增的次数
            increases = sum(1 for i in range(1, len(q_vals)) if q_vals[i] > q_vals[i - 1])
            monotonicity = increases / (len(q_vals) - 1)
        else:
            monotonicity = np.nan

        results.append({
            "factor": factor,
            "q1_ret": q_means[1], "q2_ret": q_means[2], "q3_ret": q_means[3],
            "q4_ret": q_means[4], "q5_ret": q_means[5],
            "long_short_ret": ls_mean,
            "long_short_sharpe": ls_sharpe,
            "monotonicity": monotonicity,
            "long_short_cum": ls_cum,
        })

    quantile_df = pd.DataFrame(results)
    # 按多空夏普降序
    quantile_df = quantile_df.sort_values("long_short_sharpe", ascending=False).reset_index(drop=True)
    return quantile_df


# ==================== 6. 报告生成 ====================
def generate_reports(
    ic_summary: pd.DataFrame,
    ic_timeseries: pd.DataFrame,
    quantile_df: pd.DataFrame,
):
    """生成评估报告"""
    print(f"[6/6] 生成报告...")
    os.makedirs(REPORT_DIR, exist_ok=True)

    # CSV输出
    ic_summary.to_csv(os.path.join(REPORT_DIR, "factor_ranking.csv"), index=False, encoding="utf-8-sig")
    ic_timeseries.to_csv(os.path.join(REPORT_DIR, "ic_timeseries.csv"), index=False, encoding="utf-8-sig")
    quantile_df.to_csv(os.path.join(REPORT_DIR, "quantile_returns.csv"), index=False, encoding="utf-8-sig")

    # 摘要报告
    lines = []
    lines.append("=" * 80)
    lines.append("因子对比评估报告")
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"评估窗口: {START_DATE} ~ {END_DATE}")
    lines.append(f"截面频率: 每{REBALANCE_FREQ}个交易日 (周频)")
    lines.append(f"市场代理: {MARKET_PROXY_CODE} (中证500ETF)")
    lines.append(f"L4资金层: 降级为volume_proxy模式 (akshare无法提供历史资金流)")
    lines.append("=" * 80)

    # === IC排名 Top 10 ===
    lines.append("\n" + "=" * 80)
    lines.append("一、IC分析排名 (按20日ICIR降序, Top 10)")
    lines.append("=" * 80)
    lines.append(f"{'因子':<30} {'IC均值':>10} {'ICIR':>10} {'IC胜率':>10} {'IC t值':>10} {'IC衰减(5→60)':>15}")
    lines.append("-" * 95)

    top_ic = ic_summary.head(10)
    for _, row in top_ic.iterrows():
        factor = row["factor"]
        ic_20 = row.get("ic_mean_fwd_ret_20d", np.nan)
        icir_20 = row.get("icir_fwd_ret_20d", np.nan)
        winrate = row.get("ic_winrate_fwd_ret_20d", np.nan)
        tstat = row.get("ic_tstat_fwd_ret_20d", np.nan)
        ic_5 = row.get("ic_mean_fwd_ret_5d", np.nan)
        ic_60 = row.get("ic_mean_fwd_ret_60d", np.nan)
        decay = f"{ic_5:.4f}→{ic_60:.4f}" if not (np.isnan(ic_5) or np.isnan(ic_60)) else "N/A"
        lines.append(f"{factor:<30} {ic_20:>10.4f} {icir_20:>10.4f} {winrate:>10.2%} {tstat:>10.2f} {decay:>15}")

    # === 分层回测排名 Top 10 ===
    lines.append("\n" + "=" * 80)
    lines.append("二、分层回测排名 (按多空夏普降序, Top 10)")
    lines.append("=" * 80)
    lines.append(f"{'因子':<30} {'Q1':>8} {'Q2':>8} {'Q3':>8} {'Q4':>8} {'Q5':>8} {'多空收益':>10} {'夏普':>8} {'单调性':>8}")
    lines.append("-" * 100)

    top_q = quantile_df.head(10)
    for _, row in top_q.iterrows():
        factor = row["factor"]
        q1 = row.get("q1_ret", np.nan)
        q2 = row.get("q2_ret", np.nan)
        q3 = row.get("q3_ret", np.nan)
        q4 = row.get("q4_ret", np.nan)
        q5 = row.get("q5_ret", np.nan)
        ls = row.get("long_short_ret", np.nan)
        shp = row.get("long_short_sharpe", np.nan)
        mono = row.get("monotonicity", np.nan)
        lines.append(
            f"{factor:<30} {q1:>8.4f} {q2:>8.4f} {q3:>8.4f} {q4:>8.4f} {q5:>8.4f} "
            f"{ls:>10.4f} {shp:>8.2f} {mono:>8.2%}"
        )

    # === 综合排名 ===
    lines.append("\n" + "=" * 80)
    lines.append("三、综合排名 (ICIR排名 + 多空夏普排名的平均)")
    lines.append("=" * 80)

    # 合并排名
    ic_ranked = ic_summary.copy()
    ic_ranked["ic_rank"] = ic_ranked["icir_fwd_ret_20d"].rank(ascending=False, method="min")

    q_ranked = quantile_df.copy()
    q_ranked["q_rank"] = q_ranked["long_short_sharpe"].rank(ascending=False, method="min")

    merged = ic_ranked[["factor", "ic_rank", "icir_fwd_ret_20d"]].merge(
        q_ranked[["factor", "q_rank", "long_short_sharpe"]], on="factor", how="outer"
    )
    merged["avg_rank"] = merged[["ic_rank", "q_rank"]].mean(axis=1)
    merged = merged.sort_values("avg_rank")

    lines.append(f"{'排名':<6} {'因子':<30} {'ICIR':>10} {'IC排名':>8} {'夏普':>10} {'夏普排名':>8} {'平均排名':>10}")
    lines.append("-" * 90)
    for i, (_, row) in enumerate(merged.iterrows(), 1):
        factor = row["factor"]
        icir = row.get("icir_fwd_ret_20d", np.nan)
        ic_r = row.get("ic_rank", np.nan)
        shp = row.get("long_short_sharpe", np.nan)
        q_r = row.get("q_rank", np.nan)
        avg_r = row.get("avg_rank", np.nan)
        ic_r_str = str(int(ic_r)) if not pd.isna(ic_r) else "N/A"
        q_r_str = str(int(q_r)) if not pd.isna(q_r) else "N/A"
        icir_str = f"{icir:.4f}" if not pd.isna(icir) else "N/A"
        shp_str = f"{shp:.2f}" if not pd.isna(shp) else "N/A"
        avg_r_str = f"{avg_r:.1f}" if not pd.isna(avg_r) else "N/A"
        lines.append(f"{i:<6} {factor:<30} {icir_str:>10} {ic_r_str:>8} {shp_str:>10} {q_r_str:>8} {avg_r_str:>10}")

    # === 关键发现 ===
    lines.append("\n" + "=" * 80)
    lines.append("四、关键发现")
    lines.append("=" * 80)

    best_factor = merged.iloc[0]["factor"] if len(merged) > 0 else "N/A"
    best_ic = ic_summary.iloc[0]["factor"] if len(ic_summary) > 0 else "N/A"
    best_ls = quantile_df.iloc[0]["factor"] if len(quantile_df) > 0 else "N/A"

    lines.append(f"1. 综合最佳因子: {best_factor}")
    lines.append(f"2. IC分析最佳因子(20日ICIR): {best_ic}")
    lines.append(f"3. 分层回测最佳因子(多空夏普): {best_ls}")

    # 找出IC为正且显著的因子
    if "icir_fwd_ret_20d" in ic_summary.columns:
        positive_ic = ic_summary[
            (ic_summary["icir_fwd_ret_20d"] > 0) &
            (ic_summary["ic_tstat_fwd_ret_20d"].abs() > 1.96)
        ]
        if len(positive_ic) > 0:
            lines.append(f"\n4. IC显著为正的因子(t>1.96, 共{len(positive_ic)}个):")
            for _, row in positive_ic.iterrows():
                lines.append(f"   - {row['factor']}: ICIR={row['icir_fwd_ret_20d']:.4f}, t={row['ic_tstat_fwd_ret_20d']:.2f}")

    # 找出多空收益为正且单调性好的因子
    good_ls = quantile_df[
        (quantile_df["long_short_ret"] > 0) &
        (quantile_df["monotonicity"] >= 0.75)
    ]
    if len(good_ls) > 0:
        lines.append(f"\n5. 多空收益为正且单调性≥75%的因子(共{len(good_ls)}个):")
        for _, row in good_ls.iterrows():
            lines.append(f"   - {row['factor']}: 多空={row['long_short_ret']:.4f}, 单调性={row['monotonicity']:.2%}")

    report_text = "\n".join(lines)
    print("\n" + report_text)

    with open(os.path.join(REPORT_DIR, "summary_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\n报告已保存到: {REPORT_DIR}")


# ==================== 主流程 ====================
def main():
    print("=" * 80)
    print("因子对比评估 - IC分析 + 分层回测")
    print("=" * 80)

    # 1. 加载数据
    all_data = load_all_etf_data()
    if len(all_data) < 20:
        print("错误: 可用ETF数据不足20只, 无法评估")
        return

    # 2. 生成截面日期
    # 收集所有日期的并集
    all_dates = set()
    for df in all_data.values():
        all_dates.update(df.index)
    cross_dates = get_cross_section_dates(list(all_dates))

    if len(cross_dates) < 10:
        print("错误: 截面日期不足10个, 无法评估")
        return

    # 3. 构建因子panel
    panel = build_factor_panel(all_data, cross_dates)
    if panel.shape[0] < 100:
        print(f"错误: panel记录数不足 ({panel.shape[0]}), 无法评估")
        return

    # 4. 计算前瞻收益
    panel = calculate_forward_returns(all_data, panel)

    # 5. IC分析
    ic_summary, ic_timeseries = compute_ic_analysis(panel)

    # 6. 分层回测
    quantile_df = compute_quantile_backtest(panel)

    # 7. 生成报告
    generate_reports(ic_summary, ic_timeseries, quantile_df)

    print("\n评估完成!")


if __name__ == "__main__":
    main()
