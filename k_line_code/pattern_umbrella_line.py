# -*- coding: utf-8 -*-
"""
伞形线（锤子线/上吊线）识别与可视化脚本
"""

import pandas as pd
import numpy as np
import mplfinance as mpf
from datetime import datetime, timedelta
from k_line_code.common.data_fetcher import fetch_stock_data

# ==========================================
# 1. 核心功能函数
# ==========================================

def identify_umbrella_lines(
        data: pd.DataFrame,
        body_ratio: float = 0.3,
        lower_shadow_min: float = 2.0,
        upper_shadow_max: float = 0.1,
        trend_lookback: int = 5,
        use_ma50: bool = True,
) -> pd.DataFrame:
    """
    识别伞形线，并区分锤子线（Hammer）与上吊线（Hanging Man）

    要求原始列：Open, High, Low, Close
    可选列：MA50（如果 use_ma50=True 且已有则复用，否则自动计算）

    参数说明：
    - body_ratio:    实体占当日高低区间的最大比例（越小越“纺锤/伞形”）
    - lower_shadow_min: 下影线长度至少是实体的多少倍
    - upper_shadow_max: 上影线长度不超过实体的多少倍
    - trend_lookback:   回看 N 根K线，用收盘价判断是上涨趋势还是下跌趋势
    - use_ma50:     是否使用 MA50 作为“上/下方”辅助过滤
    """

    df = data.copy()

    # -------- 1. 预计算基础量 --------
    o, h, l, c = df['Open'], df['High'], df['Low'], df['Close']

    real_body = (c - o).abs()
    total_range = (h - l).replace(0, np.nan)  # 防止除零
    lower_shadow = np.minimum(o, c) - l
    upper_shadow = h - np.maximum(o, c)

    df['Entity'] = real_body
    df['Lower_Shadow'] = lower_shadow
    df['Upper_Shadow'] = upper_shadow

    # -------- 2. 形态：是否为“伞形线” --------
    # 实体较小 + 下影够长 + 上影很短
    body_small = (real_body / total_range) <= body_ratio
    long_lower = lower_shadow >= lower_shadow_min * real_body
    short_upper = upper_shadow <= upper_shadow_max * real_body

    umbrella = body_small & long_lower & short_upper

    # -------- 3. 趋势过滤：区分锤子线 / 上吊线 --------
    # 3.1 MA50 作为辅助（可选）
    if use_ma50:
        if 'MA50' not in df.columns:
            df['MA50'] = c.rolling(window=50, min_periods=1).mean()
        ma50 = df['MA50']
        close_above_ma = c > ma50
        close_below_ma = c < ma50
    else:
        # 不用 MA50，只用趋势来区分
        close_above_ma = pd.Series(False, index=df.index)
        close_below_ma = pd.Series(False, index=df.index)

    # 3.2 短期趋势：用 N 根前的收盘价判断涨跌
    prev_close = c.shift(trend_lookback)
    down_trend = prev_close.notna() & (c < prev_close)   # 之前在跌
    up_trend   = prev_close.notna() & (c > prev_close)   # 之前在涨

    # -------- 4. 最终信号 --------
    # 锤子线：伞形线 + 之前有下跌趋势 + （可选）当前在 MA50 下方
    hammer = umbrella & down_trend & ( (~use_ma50) | close_below_ma )

    # 上吊线：伞形线 + 之前有上涨趋势 + （可选）当前在 MA50 上方
    hanging_man = umbrella & up_trend & ( (~use_ma50) | close_above_ma )

    df['Hammer'] = hammer
    df['Hanging_Man'] = hanging_man

    # -------- 5. 上吊线验证信号（可选）--------
    # 条件：次日收盘价 < 上吊线实体中点
    # 注意：这使用了未来数据 shift(-1)，最新的 K 线无法计算此验证
    body_mid = (o + c) / 2
    next_close = c.shift(-1)
    hanging_valid = hanging_man & (next_close < body_mid)
    df['Hanging_Man_Valid'] = hanging_valid

    return df


# ==========================================
# 2. 主执行逻辑
# ==========================================

def main():
    print("正在初始化参数...")
    # 1. 计算日期区间
    end_dt   = datetime.today()
    start_dt = end_dt - timedelta(days=365)          # 365 天前
    # start_str = start_dt.strftime('%Y%m%d') # 原始代码未使用的变量
    # end_str   = end_dt.strftime('%Y%m%d')   # 原始代码未使用的变量

    # 2. 股票代码（示例：贵州茅台）
    symbol_code = '600519'          # 贵州茅台

    print(f"正在拉取 {symbol_code} 的数据...")
    # 3. 拉取数据（前复权）
    try:
        df_raw = ak.stock_zh_a_hist(symbol=symbol_code,
                                    period='daily',
                                    start_date=start_dt.strftime('%Y%m%d'),
                                    end_date=end_dt.strftime('%Y%m%d'),
                                    adjust='qfq')
    except Exception as e:
        print(f"数据拉取失败: {e}")
        return

    if df_raw.empty:
        print("未获取到数据，请检查日期或股票代码。")
        return

    print("正在清洗与计算数据...")
    # 4. 整理成英文列名并设 Date 为索引
    df = (df_raw
          .rename(columns={'日期': 'Date',
                           '开盘': 'Open',
                           '收盘': 'Close',
                           '最高': 'High',
                           '最低': 'Low',
                           '成交量': 'Volume'})
          .loc[:, ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
          .assign(Date=lambda x: pd.to_datetime(x['Date']))
          .set_index('Date')
          .sort_index())

    # 5. 计算 50 日均线
    df['MA50'] = df['Close'].rolling(window=50).mean()

    # 6. 识别形态
    df = identify_umbrella_lines(df)

    print("正在准备绘图...")
    # 7. 构造 mplfinance 附加图层
    addplots = [mpf.make_addplot(df['MA50'], color='blue', width=1.2)] # 去掉 title 参数，避免部分版本报错或重叠

    # 信号散点位置
    # 锤子线：在最低价下方一点画绿色上箭头
    hammer_sig = np.where(df['Hammer'], df['Low'] * 0.99, np.nan)

    # 上吊线：在最高价上方一点画红色下箭头
    hanging_sig = np.where(df['Hanging_Man_Valid'], df['High'] * 1.01, np.nan)

    # 只有存在信号时才添加图层
    if not np.all(np.isnan(hammer_sig)):
        addplots.append(
            mpf.make_addplot(
                hammer_sig,
                type='scatter',
                markersize=80,
                marker='^',
                color='green',
                label='Hammer'
            )
        )

    if not np.all(np.isnan(hanging_sig)):
        addplots.append(
            mpf.make_addplot(
                hanging_sig,
                type='scatter',
                markersize=80,
                marker='v',
                color='red',
                label='Hanging Man (valid)'
            )
        )


    # 8. 绘图
    # 注意：mplfinance 默认不支持中文标题显示，如果显示方框，请配置 rcParams 或使用英文标题
    mpf.plot(
        df,
        type='candle',
        style='yahoo',
        title=f'Umbrella Lines Pattern: {symbol_code}', # 改为英文以防乱码
        ylabel='Price',
        volume=True,
        addplot=addplots,
        figsize=(18, 9),
        tight_layout=True,
        block=True # 脚本运行时需要此参数以保持窗口显示
    )
    print("绘图完成。")

if __name__ == "__main__":
    main()