import pandas as pd
import numpy as np
import mplfinance as mpf
from datetime import datetime, timedelta
from k_line_code.common.data_fetcher import fetch_stock_data

# ==========================================
# 1. 数据获取函数
# ==========================================
def get_stock_data(symbol: str, days: int = 365) -> pd.DataFrame:
    """
    获取最近 N 天的 A 股复权数据并清洗格式
    """
    end_dt = datetime.today()
    start_dt = end_dt - timedelta(days=days)

    print(f"正在获取 {symbol} 最近 {days} 天的数据...")
    try:
        df_raw = fetch_stock_data(symbol, days=days)
    except Exception as e:
        print(f"数据获取失败: {e}")
        return pd.DataFrame()

    if df_raw.empty:
        print("未获取到数据，请检查代码或网络。")
        return pd.DataFrame()

    # 整理成英文列 + Date 索引
    df = (df_raw
          .rename(columns={'日期': 'Date',
                           '开盘': 'Open',
                           '最高': 'High',
                           '最低': 'Low',
                           '收盘': 'Close',
                           '成交量': 'Volume'})
          .loc[:, ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
          .assign(Date=lambda x: pd.to_datetime(x['Date']))
          .set_index('Date')
          .sort_index())

    return df

# ==========================================
# 2. 平头顶/底识别函数
# ==========================================
def identify_tweezers(df_in: pd.DataFrame,
                      ma_len: int = 20,
                      use_trend_filter: bool = True,
                      tol_pct: float = 0.0015,
                      use_atr_tolerance: bool = True,
                      atr_len: int = 14,
                      atr_mult: float = 0.2,
                      require_long1: bool = True,
                      long_body_pct: float = 0.50, # 注意：原代码逻辑写死 0.5，这里作为参数传入
                      require_small2: bool = True,
                      small_body_pct: float = 0.30, # 注意：原代码逻辑写死 0.3，这里作为参数传入
                      confirm_top: bool = True,
                      confirm_bottom: bool = True) -> pd.DataFrame:
    """
    识别 平头顶 (Tweezers Top) 和 平头底 (Tweezers Bottom)
    """
    df = df_in.copy()

    # 1. 基础计算
    df["Range"] = (df["High"] - df["Low"]).replace(0, 1e-6)
    df["Body"]  = (df["Close"] - df["Open"]).abs()
    df["MA"]    = df["Close"].rolling(ma_len).mean()

    # 2. ATR 计算 (用于容差判断)
    # TR = Max(H-L, Abs(H-Cp), Abs(L-Cp))
    tr = np.maximum.reduce([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift(1)).abs(),
        (df["Low"]  - df["Close"].shift(1)).abs()
    ])
    df["ATR"] = pd.Series(tr, index=df.index).ewm(alpha=1/atr_len, adjust=False).mean()

    # 3. 数据移位 (T-1 和 T-0)
    cols = ["Open","High","Low","Close","Body","Range"]
    o1,h1,l1,c1,b1,r1 = [df[c].shift(1) for c in cols]
    o2,h2,l2,c2,b2,r2 = [df[c] for c in cols]

    # 获取实体的高低点 (用于确认信号判断)
    e1_low, e1_high = np.minimum(o1, c1), np.maximum(o1, c1)
    e2_low, e2_high = np.minimum(o2, c2), np.maximum(o2, c2)

    # 4. 趋势判断
    if use_trend_filter:
        up_trend   = df["Close"].shift(1) > df["MA"].shift(1)
        down_trend = df["Close"].shift(1) < df["MA"].shift(1)
        trend_up_ok   = up_trend
        trend_down_ok = down_trend
    else:
        trend_up_ok   = pd.Series(True, index=df.index)
        trend_down_ok = pd.Series(True, index=df.index)

    # 5. K线形态筛选
    # 第一根需要是长线?
    long1  = (b1 >= long_body_pct * r1) if require_long1  else pd.Series(True, index=df.index)
    # 第二根需要是小线?
    small2 = (b2 <= small_body_pct * r2) if require_small2 else pd.Series(True, index=df.index)

    # 6. 平头判断 (Tolerance)
    # 使用 ATR 倍数 或者 简单的百分比
    tol_top  = df["ATR"] * atr_mult if use_atr_tolerance else h2 * tol_pct
    tol_bot  = df["ATR"] * atr_mult if use_atr_tolerance else l2 * tol_pct

    same_high = (h2 - h1).abs() <= tol_top # 最高价几乎相同
    same_low  = (l2 - l1).abs() <= tol_bot # 最低价几乎相同

    # 7. 初步形态合成
    tweezer_top_basic = same_high & trend_up_ok & long1 & small2
    tweezer_bot_basic = same_low & trend_down_ok & long1 & small2

    # 8. 确认机制 (查看 T+1)
    # 平头顶确认：次日收盘跌破第二根实体低点
    top_confirm_ok = df["Close"].shift(-1) < e2_low
    # 平头底确认：次日收盘突破第二根实体高点
    bot_confirm_ok = df["Close"].shift(-1) > e2_high

    df["Tweezers_Top"]    = tweezer_top_basic & (top_confirm_ok if confirm_top else True)
    df["Tweezers_Bottom"] = tweezer_bot_basic & (bot_confirm_ok if confirm_bottom else True)

    # 9. 计算阻力/支撑位 (用于绘图)
    df["TZ_Res"] = np.where(df["Tweezers_Top"], np.maximum(h1, h2), np.nan)
    df["TZ_Sup"] = np.where(df["Tweezers_Bottom"], np.minimum(l1, l2), np.nan)

    return df

# ==========================================
# 3. 可视化绘图函数
# ==========================================
def plot_tweezers(df: pd.DataFrame, symbol_code: str):
    """
    绘制 K 线及平头顶底标记
    """
    # 结果统计
    n_top = df["Tweezers_Top"].sum()
    n_bot = df["Tweezers_Bottom"].sum()
    print(f'=== {symbol_code} 统计结果 ===')
    print(f'平头顶部 (Tweezers Top)：{n_top} 次')
    print(f'平头底部 (Tweezers Bottom)：{n_bot} 次')

    if n_top + n_bot == 0:
        print('当前参数下无平头形态，可放宽条件 (如 tolerance 或 body_pct) 再试')

    # 准备绘图
    ap = [mpf.make_addplot(df["MA"], color='tab:blue', width=1.2)]
    rng = df["Range"].replace(0, 1e-6)

    # 形态标记
    tt_marks = np.where(df["Tweezers_Top"], df["High"] + rng * 0.18, np.nan)
    tb_marks = np.where(df["Tweezers_Bottom"], df["Low"] - rng * 0.18, np.nan)

    if not np.all(np.isnan(tt_marks)):
        ap.append(mpf.make_addplot(tt_marks, type="scatter", marker="v", markersize=90, color="tab:red", label='Tweezers Top'))
    if not np.all(np.isnan(tb_marks)):
        ap.append(mpf.make_addplot(tb_marks, type="scatter", marker="^", markersize=90, color="tab:green", label='Tweezers Bottom'))

    # 阻力/支撑线 (向前填充)
    # 注意：先计算 ffill 后的 Series，再判断是否全为空
    tz_res_plot = pd.Series(df["TZ_Res"]).ffill()
    tz_sup_plot = pd.Series(df["TZ_Sup"]).ffill()

    if not tz_res_plot.isna().all():
        ap.append(mpf.make_addplot(tz_res_plot, color='tab:red', linestyle='--', width=1))

    if not tz_sup_plot.isna().all():
        ap.append(mpf.make_addplot(tz_sup_plot, color='tab:green', linestyle='--', width=1))

    # 绘图
    mpf.plot(df, type="candle", volume=True, addplot=ap,
             title=f'{symbol_code} Tweezers Pattern (ATR Tolerance + Trend)',
             style="yahoo", figsize=(14, 8))

# ==========================================
# 4. 主程序入口
# ==========================================
if __name__ == "__main__":
    # 配置
    TARGET_SYMBOL = '600519'  # 贵州茅台

    # 1. 获取数据
    df_data = get_stock_data(TARGET_SYMBOL, days=365)

    if not df_data.empty:
        # 2. 计算形态
        df_patterns = identify_tweezers(df_data,
                                        ma_len=20,
                                        use_trend_filter=True,
                                        use_atr_tolerance=True,
                                        atr_mult=0.2,       # ATR 容差倍数
                                        require_long1=True,
                                        require_small2=True,
                                        confirm_top=True,
                                        confirm_bottom=True)

        # 3. 绘图
        plot_tweezers(df_patterns, TARGET_SYMBOL)