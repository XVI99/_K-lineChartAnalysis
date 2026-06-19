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
# 2. 孕线形态识别函数
# ==========================================
def identify_harami(df_in: pd.DataFrame,
                    ma_len: int = 20,
                    use_trend_filter: bool = True,
                    use_confirm: bool = True) -> pd.DataFrame:
    """
    识别 孕线 (Harami) 和 十字孕线 (Doji Harami)
    """
    df = df_in.copy()

    # 1. 基础指标
    df["Range"] = (df["High"] - df["Low"]).replace(0, 1e-6)
    df["Body"]  = (df["Close"] - df["Open"]).abs()
    df["Bull"]  = (df["Close"] > df["Open"])
    df["Bear"]  = (df["Close"] < df["Open"])
    df["Doji"]  = (df["Body"] <= df["Range"] * 0.10) # 十字星判定
    df["MA"]    = df["Close"].rolling(ma_len).mean()

    # 2. 数据移位（母线 t-1，子线 t）
    # 为了代码整洁，使用列表推导式获取 t-1 和 t 的列
    cols = ["Open","High","Low","Close","Body","Range","Bull","Bear","Doji"]
    o1,h1,l1,c1,b1,r1,bull1,bear1,doji1 = [df[c].shift(1) for c in cols]
    o2,h2,l2,c2,b2,r2,bull2,bear2,doji2 = [df[c] for c in cols]

    # 实体范围计算 (Body Bounds)
    m_low, m_high = np.minimum(o1, c1), np.maximum(o1, c1) # 母线实体
    c_low, c_high = np.minimum(o2, c2), np.maximum(o2, c2) # 子线实体

    # 3. 核心形态逻辑
    mother_long = (b1 >= 0.50 * r1)                      # 母线实体较长
    child_small = (b2 <= 0.30 * r2) & (b2 <= 0.60 * b1)  # 子线实体较小且小于母线实体
    inside = (c_low >= m_low) & (c_high <= m_high)       # 子线实体完全在母线实体内部 (Body inside Body)

    # 4. 趋势过滤
    if use_trend_filter:
        downtrend = df["Close"].shift(1) < df["MA"].shift(1)
        uptrend   = df["Close"].shift(1) > df["MA"].shift(1)
    else:
        downtrend = pd.Series(True, index=df.index)
        uptrend   = pd.Series(True, index=df.index)

    # 基础形态
    bull_harami_basic = mother_long & child_small & inside & downtrend
    bear_harami_basic = mother_long & child_small & inside & uptrend
    doji_bull_basic   = bull_harami_basic & doji2
    doji_bear_basic   = bear_harami_basic & doji2

    # 5. 确认机制 (查看 t+1 日收盘价)
    # 确认逻辑：看涨需次日收盘突破子线实体上沿；看跌需跌破下沿
    entity_high = c_high
    entity_low  = c_low

    # 注意：shift(-1) 意味着我们在当前行使用了未来的数据，这在画图分析历史数据是允许的
    bull_confirm_ok = df["Close"].shift(-1) > entity_high
    bear_confirm_ok = df["Close"].shift(-1) < entity_low

    # 如果不启用确认，则默认确认通过 (True)
    check_bull = bull_confirm_ok if use_confirm else True
    check_bear = bear_confirm_ok if use_confirm else True

    # 6. 最终信号合成
    df["Bull_Harami"]      = bull_harami_basic & check_bull
    df["Bear_Harami"]      = bear_harami_basic & check_bear
    df["Doji_Bull_Harami"] = doji_bull_basic   & check_bull
    df["Doji_Bear_Harami"] = doji_bear_basic   & check_bear

    # 7. 计算用于绘图的支撑/阻力位
    df["Harami_Support"]   = np.where(df["Bull_Harami"] | df["Doji_Bull_Harami"], l1, np.nan)
    df["Harami_Resistance"]= np.where(df["Bear_Harami"] | df["Doji_Bear_Harami"], h1, np.nan)

    return df

# ==========================================
# 3. 可视化绘图函数
# ==========================================
def plot_harami(df: pd.DataFrame, symbol_code: str):
    """
    绘制 K 线及孕线形态标记
    """
    # 统计打印
    n_bull = df["Bull_Harami"].sum()
    n_bear = df["Bear_Harami"].sum()
    n_dbull= df["Doji_Bull_Harami"].sum()
    n_dbear= df["Doji_Bear_Harami"].sum()

    print(f'=== {symbol_code} 统计结果 ===')
    print(f'看涨孕线 (Bullish Harami)：{n_bull} 次')
    print(f'看跌孕线 (Bearish Harami)：{n_bear} 次')
    print(f'十字看涨孕线 (Doji Bull)：{n_dbull} 次')
    print(f'十字看跌孕线 (Doji Bear)：{n_dbear} 次')

    if n_bull + n_bear + n_dbull + n_dbear == 0:
        print('当前参数下无孕线形态，可放宽条件再试')

    # 准备绘图
    ap = [mpf.make_addplot(df["MA"], color='tab:blue', width=1.2)]
    rng = df["Range"].replace(0, 1e-6)

    # 计算标记位置
    bull_marks  = np.where(df["Bull_Harami"],      df["Low"]  - rng * 0.18, np.nan)
    dbull_marks = np.where(df["Doji_Bull_Harami"], df["Low"]  - rng * 0.30, np.nan)
    bear_marks  = np.where(df["Bear_Harami"],      df["High"] + rng * 0.18, np.nan)
    dbear_marks = np.where(df["Doji_Bear_Harami"], df["High"] + rng * 0.30, np.nan)

    # 添加形态标记图层
    # 颜色约定：看涨绿色/深绿，看跌红色/深红
    configs = [
        (bull_marks,  'tab:green', 90,  '^', 'Bull Harami'),
        (dbull_marks, 'seagreen',  110, '^', 'Doji Bull'),
        (bear_marks,  'tab:red',   90,  'v', 'Bear Harami'),
        (dbear_marks, 'crimson',   110, 'v', 'Doji Bear')
    ]

    for marks, color, size, marker, label in configs:
        if not np.all(np.isnan(marks)):
            ap.append(mpf.make_addplot(marks, type="scatter", marker=marker,
                                       markersize=size, color=color, label=label))

    # 支撑/阻力阶梯线 (向前填充 ffill)
    # 将计算好的 Support/Resistance 列进行 forward fill 以形成线条
    if not df["Harami_Support"].isna().all():
        sup_line = df["Harami_Support"].ffill()
        ap.append(mpf.make_addplot(sup_line, color='tab:green', linestyle='--', width=1))

    if not df["Harami_Resistance"].isna().all():
        res_line = df["Harami_Resistance"].ffill()
        ap.append(mpf.make_addplot(res_line, color='tab:red', linestyle='--', width=1))

    # 绘图
    mpf.plot(df, type="candle", volume=True, addplot=ap,
             title=f'{symbol_code} Harami Patterns (Trend+Confirm Filters)',
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
        # use_confirm=True 对应你原代码中的 (... if True else True)
        df_patterns = identify_harami(df_data,
                                      ma_len=20,
                                      use_trend_filter=True,
                                      use_confirm=True)

        # 3. 绘图
        plot_harami(df_patterns, TARGET_SYMBOL)