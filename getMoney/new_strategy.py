# new_backtest.py
import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

try:
    from k_line_code.kdj_analysis import get_stock_data, calculate_kdj, identify_kdj_signals
    from k_line_code.stochastic_pattern import calculate_stochastic, identify_stochastic_patterns
except ImportError as e:
    logging.error(f"策略模块导入失败: {e}")
    exit(1)



# ============ 1. 生成“沪 A + 深 A”股票列表 ============

def get_cn_a_stock_list() -> pd.DataFrame:
    """
    使用 akshare 获取全 A 股列表，然后筛选：
    沪 A：600 / 601 / 603 / 605 开头
    深 A：000 / 001 / 002 / 003 开头
    """
    try:
        # 获取全 A 股代码和名称
        df = ak.stock_info_a_code_name()
        if df.empty:
            logging.error("获取全 A 股列表失败：数据源返回空数据")
            return pd.DataFrame()

        # 打印列名以确认实际结构
        print("列名:", df.columns.tolist())

        # 确保列名正确映射
        if "代码" in df.columns and "名称" in df.columns:
            df["code"] = df["代码"].astype(str)
        else:
            logging.error("获取全 A 股列表失败：未找到预期的列 '代码' 或 '名称'")
            return pd.DataFrame()

        # 筛选沪 A 和深 A 股票
        prefixes = ("600", "601", "603", "605", "000", "001", "002", "003")
        mask = df["code"].str.startswith(prefixes)
        df_sel = df[mask].copy()
        return df_sel[["code"]]
    except Exception as e:
        logging.error(f"获取全 A 股列表失败: {e}")
        return pd.DataFrame()  # 返回空 DataFrame，避免程序中断

# ============ 2. 组合信号 ============

def build_combined_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    构造 StrongLong / StrongShort 信号
    """
    df = df.copy()
    # 版本 A：最宽，只看 KDJ
    df["StrongLong"] = df["BuySignal"].astype(bool)
    df["StrongShort"] = df["SellSignal"].astype(bool)

    # 版本 B：稍严，叠加随机指标极值
    # df["StrongLong"] = df["BuySignal"].astype(bool) & df["OverSold"].astype(bool)
    # df["StrongShort"] = df["SellSignal"].astype(bool) & df["OverBought"].astype(bool)

    # 当前版本（最严）：KDJ + 随机指标 + K 线形态
    # df["StrongLong"] = df["BuySignal"].astype(bool) & df["BottomSignal"].astype(bool)
    # df["StrongShort"] = df["SellSignal"].astype(bool) & df["TopSignal"].astype(bool)

    return df

# ============ 3. 单个股票的 5 根 K 线回测 ============

def backtest_5bar_for_symbol(symbol: str,
                             days_back: int = 600,
                             max_bars: int = 5) -> dict:
    """
    对单个股票回测：
    - 信号日按收盘价开仓
    - 最多持有 max_bars 根 K 线
    - 到止盈/止损/时间到期就平仓
    """
    df = get_stock_data(symbol, days=days_back)
    if df.empty:
        return {"symbol": symbol, "trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0}

    # 计算 KDJ
    df = calculate_kdj(df, N=9, M1=3, M2=3)
    df = identify_kdj_signals(df)

    # 计算随机指标 + 顶底共振
    df_stoch = calculate_stochastic(df, n_k=14, smooth_k=3, smooth_d=3)
    df_stoch = identify_stochastic_patterns(df_stoch, overbought=80, oversold=20)

    # 对齐合并（修复缺少右括号问题）
    stoch_cols = ["%K", "%D", "OverBought", "OverSold",
                  "BullCross", "BearCross", "TopSignal", "BottomSignal"]
    df_all = df.join(df_stoch[stoch_cols], how="inner")

    if df_all.empty:
        # 没有重叠日期，直接返回
        return {"symbol": symbol, "trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0}

    # 组合强信号（当前版本 A：只看 KDJ）
    df_all = build_combined_signals(df_all)

    # Debug：打印信号数量
    n_long = int(df_all["StrongLong"].sum())
    n_short = int(df_all["StrongShort"].sum())
    print(symbol, "StrongLong:", n_long, "StrongShort:", n_short)

    if n_long + n_short == 0:
        return {"symbol": symbol, "trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0}

    trades = 0
    wins = 0
    losses = 0

    dates = list(df_all.index)

    for i, dt in enumerate(dates):
        row = df_all.loc[dt]

        if not row["StrongLong"] and not row["StrongShort"]:
            continue

        direction = "LONG" if row["StrongLong"] else "SHORT"
        entry_price = row["Close"]

        # 计算最近 20 日波动，用于止损止盈
        hist = df_all.loc[:dt].tail(20)
        if len(hist) < 5:
            continue

        tr = np.maximum(
            hist["High"] - hist["Low"],
            np.maximum(
                (hist["High"] - hist["Close"].shift(1)).abs(),
                (hist["Low"] - hist["Close"].shift(1)).abs(),
            ),
        )
        daily_vol = tr.mean()

        vol_sl = 0.8
        vol_tp = 1.5

        if direction == "LONG":
            stop_loss = entry_price - vol_sl * daily_vol
            take_profit = entry_price + vol_tp * daily_vol
        else:
            stop_loss = entry_price + vol_sl * daily_vol
            take_profit = entry_price - vol_tp * daily_vol

        # 向后最多 max_bars 根 K 线
        exit_index = min(i + max_bars, len(dates) - 1)
        future_dates = dates[i + 1: exit_index + 1]
        if not future_dates:
            continue

        exit_price = df_all.loc[future_dates[-1], "Close"]  # 默认时间到期

        # 在未来每一根 K 线中检查是否触及止损/止盈
        for fdt in future_dates:
            hi = df_all.loc[fdt, "High"]
            lo = df_all.loc[fdt, "Low"]

            if direction == "LONG":
                if lo <= stop_loss:
                    exit_price = stop_loss
                    break
                if hi >= take_profit:
                    exit_price = take_profit
                    break
            else:  # SHORT
                if hi >= stop_loss:
                    exit_price = stop_loss
                    break
                if lo <= take_profit:
                    exit_price = take_profit
                    break

        # 计算盈亏
        if direction == "LONG":
            pnl = exit_price - entry_price
        else:
            pnl = entry_price - exit_price

        trades += 1
        if pnl > 0:
            wins += 1
        else:
            losses += 1

    return {
        "symbol": symbol,
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "win_rate": wins / trades if trades > 0 else 0.0,
    }

# ============ 4. 对沪 A＋深 A 全市场跑回测并统计胜率 ============

def run_universe_backtest(max_symbols: int = 100,
                          days_back: int = 600,
                          max_bars: int = 5):
    stock_list = get_cn_a_stock_list()
    codes = stock_list["code"].tolist()

    # 控制数量，避免一次性跑太多
    codes = codes[:max_symbols]

    results = []
    for code in codes:
        print(f"回测 {code} ...")
        res = backtest_5bar_for_symbol(code, days_back=days_back, max_bars=max_bars)
        if res["trades"] > 0:
            results.append(res)

    if not results:
        print("没有任何有效交易。")
        return

    df_res = pd.DataFrame(results)
    total_trades = df_res["trades"].sum()
    total_wins = df_res["wins"].sum()
    total_losses = df_res["losses"].sum()
    overall_win_rate = total_wins / total_trades if total_trades > 0 else 0.0

    print("\n============== 回测结果汇总（沪 A + 深 A）==============")
    print(df_res.sort_values("win_rate", ascending=False).head(20))
    print("------------------------------------------------------")
    print(f"总交易笔数: {total_trades}")
    print(f"总盈利笔数: {total_wins}")
    print(f"总亏损笔数: {total_losses}")
    print(f"整体胜率: {overall_win_rate:.2%}")

if __name__ == "__main__":
    # 你可以按需调整 max_symbols / days_back / max_bars
    run_universe_backtest(max_symbols=80, days_back=600, max_bars=5)
