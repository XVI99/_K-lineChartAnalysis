from k_line_code.common.data_fetcher import fetch_stock_data
import pandas as pd
import numpy as np
import mplfinance as mpf
from datetime import datetime, timedelta

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
# 2. 形态识别函数 (逻辑保持不变)
# ==========================================
def identify_doji_patterns(df_in: pd.DataFrame,
                           ma_len: int = 20,
                           vol_ma_len: int = 20,
                           doji_thresh: float = 0.10,
                           penetrate_req: float = 0.50,
                           gap_tolerance_pct: float = 0.00,
                           use_trend_filter: bool = True,
                           use_volume_filter: bool = False,
                           use_confirm: bool = False,
                           confirm_shift: int = 1) -> pd.DataFrame:
    """
    计算十字星、弃婴等形态，返回包含信号列的 DataFrame
    """
    df = df_in.copy()

    # 基础指标计算
    df["Range"] = (df["High"] - df["Low"]).replace(0, 1e-6)
    df["Body"]  = (df["Close"] - df["Open"]).abs()
    df["Doji"]  = df["Body"] <= df["Range"] * doji_thresh
    df["MA"]    = df["Close"].rolling(ma_len).mean()
    df["VOL_MA"]= df["Volume"].rolling(vol_ma_len).mean()

    # 辅助内部函数：批量 Shift
    def shift_cols(s):
        return [df[c].shift(s) for c in ["Open","High","Low","Close","Body","Range","Doji","Volume"]]

    # 获取 T-2, T-1, T-0 的数据
    o1,h1,l1,c1,b1,r1,d1,v1 = shift_cols(2)
    o2,h2,l2,c2,b2,r2,d2,v2 = shift_cols(1)
    o3,h3,l3,c3,b3,r3,d3,v3 = shift_cols(0)

    # 实体的高低点（用于判断缺口）
    e1_low, e1_high = np.minimum(o1,c1), np.maximum(o1,c1)
    e2_low, e2_high = np.minimum(o2,c2), np.maximum(o2,c2)
    # e3_low, e3_high = np.minimum(o3,c3), np.maximum(o3,c3) # 未使用，可注释

    # 趋势判断（基于 T-1 时刻）
    uptrend_before   = df["Close"].shift(1) > df["MA"].shift(1)
    downtrend_before = df["Close"].shift(1) < df["MA"].shift(1)
    trend_up_ok   = uptrend_before if use_trend_filter else pd.Series(True, index=df.index)
    trend_down_ok = downtrend_before if use_trend_filter else pd.Series(True, index=df.index)
    vol_ok = (df["Volume"] > df["VOL_MA"]) if use_volume_filter else pd.Series(True, index=df.index)

    # --- 逻辑 A: 十字黄昏星 (Evening Doji Star) ---
    pen_level_evening = o1 + (c1 - o1) * (1 - penetrate_req)
    deep_bear3 = (c3 < o3) & (c3 <= pen_level_evening)
    long_white1 = (c1 > o1) & (b1 > 0.5 * r1)
    doji2 = d2.astype(bool)
    gap12_star = e2_low > e1_high * (1 - gap_tolerance_pct) # 向上跳空
    evening_doji_basic = long_white1 & doji2 & deep_bear3 & trend_up_ok & vol_ok & gap12_star

    confirm_evening = (df["Close"].shift(-confirm_shift) < df["Close"]) if use_confirm else pd.Series(True, index=df.index)
    df["Doji_Evening_Star"] = evening_doji_basic & confirm_evening

    # --- 逻辑 B: 十字启明星 (Morning Doji Star) ---
    pen_level_morning = c1 + (o1 - c1) * penetrate_req
    deep_bull3 = (c3 > o3) & (c3 >= pen_level_morning)
    long_black1 = (c1 < o1) & (b1 > 0.5 * r1)
    gap12_star_dn = h2 < l1 * (1 + gap_tolerance_pct) # 向下跳空
    morning_doji_basic = long_black1 & doji2 & deep_bull3 & trend_down_ok & vol_ok & gap12_star_dn

    confirm_morning = (df["Close"].shift(-confirm_shift) > df["Close"]) if use_confirm else pd.Series(True, index=df.index)
    df["Doji_Morning_Star"] = morning_doji_basic & confirm_morning

    # --- 逻辑 C: 弃婴顶部 & 底部 (Abandoned Baby) ---
    # 弃婴形态要求严格的缺口（上下影线都不接触）
    ab_top = (trend_up_ok & doji2 & (l2 > h1) & (h3 < l2) & (c3 < o3) & vol_ok)
    ab_bot = (trend_down_ok & doji2 & (h2 < l1) & (l3 > h2) & (c3 > o3) & vol_ok)
    df["Abandoned_Baby_Top"] = ab_top
    df["Abandoned_Baby_Bot"] = ab_bot

    # 阻力/支撑计算
    df["Doji_Evening_Res"] = np.where(df["Doji_Evening_Star"], np.maximum.reduce([h1, h2, h3]), np.nan)
    df["Doji_Morning_Sup"] = np.where(df["Doji_Morning_Star"], np.minimum.reduce([l1, l2, l3]), np.nan)
    df["AB_Top_Res"] = np.where(df["Abandoned_Baby_Top"], np.maximum.reduce([h1, h2, h3]), np.nan)
    df["AB_Bot_Sup"] = np.where(df["Abandoned_Baby_Bot"], np.minimum.reduce([l1, l2, l3]), np.nan)

    return df

# ==========================================
# 3. 可视化绘图函数
# ==========================================
def plot_patterns(df: pd.DataFrame, symbol_code: str):
    """
    根据计算结果绘制 K 线及形态标记
    """
    # 结果统计打印
    n_evening = df["Doji_Evening_Star"].sum()
    n_morning = df["Doji_Morning_Star"].sum()
    n_abtop = df["Abandoned_Baby_Top"].sum()
    n_abbot = df["Abandoned_Baby_Bot"].sum()
    print(f'=== {symbol_code} 统计结果 ===')
    print(f'十字黄昏星：{n_evening} 次')
    print(f'十字启明星：{n_morning} 次')
    print(f'弃婴顶部：{n_abtop} 次')
    print(f'弃婴底部：{n_abbot} 次')

    # 准备绘图
    ap = [mpf.make_addplot(df["MA"], color='tab:blue', width=1.2)]

    rng = (df["High"] - df["Low"]).replace(0, 1e-6)

    # 标记坐标
    es_marks = np.where(df["Doji_Evening_Star"], df["High"] + rng * 0.18, np.nan)
    ms_marks = np.where(df["Doji_Morning_Star"], df["Low"] - rng * 0.18, np.nan)
    abtop_mk = np.where(df["Abandoned_Baby_Top"], df["High"] + rng * 0.30, np.nan)
    abbot_mk = np.where(df["Abandoned_Baby_Bot"], df["Low"] - rng * 0.30, np.nan)

    # 添加形态标记（检查是否全 NaN 以避免报错）
    if not np.all(np.isnan(es_marks)):
        ap.append(mpf.make_addplot(es_marks, type="scatter", marker="v", markersize=90, color="tab:red", label='Evening Doji'))
    if not np.all(np.isnan(ms_marks)):
        ap.append(mpf.make_addplot(ms_marks, type="scatter", marker="^", markersize=90, color="tab:green", label='Morning Doji'))
    if not np.all(np.isnan(abtop_mk)):
        ap.append(mpf.make_addplot(abtop_mk, type="scatter", marker="v", markersize=120, color="crimson", label='Aban Baby Top'))
    if not np.all(np.isnan(abbot_mk)):
        ap.append(mpf.make_addplot(abbot_mk, type="scatter", marker="^", markersize=120, color="seagreen", label='Aban Baby Bot'))

    # 阻力/支撑阶梯线 (使用 ffill 向后延伸显示)
    for col, color in zip(["Doji_Evening_Res", "Doji_Morning_Sup", "AB_Top_Res", "AB_Bot_Sup"],
                          ["tab:red", "tab:green", "maroon", "darkgreen"]):
        if not df[col].isna().all():
            plot_col = pd.Series(df[col]).ffill()
            ap.append(mpf.make_addplot(plot_col, color=color, linestyle='--', width=1))

    # 绘图
    mpf.plot(df, type="candle", volume=True, addplot=ap,
             title=f'{symbol_code} Doji Star & Abandoned Baby',
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
        df_patterns = identify_doji_patterns(df_data,
                                             ma_len=20,
                                             doji_thresh=0.10,
                                             penetrate_req=0.50)

        # 3. 绘图
        plot_patterns(df_patterns, TARGET_SYMBOL)