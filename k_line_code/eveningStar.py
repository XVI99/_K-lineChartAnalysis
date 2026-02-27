
# -*- coding: utf-8 -*-
"""
黄昏星 (Evening Star) 形态识别与可视化脚本
形态特征 (顶部反转):
1. 趋势：处于上升趋势。
2. K1：长阳线 (Long White)。
3. K2：小实体星线 (Star/Doji)，向上跳空 (Gap Up)。
4. K3：阴线，向下跳空或低开，且收盘价深入 K1 实体内部 (Penetration)。
"""

import akshare as ak
import pandas as pd
import numpy as np
import mplfinance as mpf
from datetime import datetime, timedelta

# ==========================================
# 1. 核心识别逻辑
# ==========================================

def identify_evening_star(
        df: pd.DataFrame,
        ma_len: int = 20,
        vol_ma_len: int = 20,
        star_small_pct: float = 0.20,    # 星线实体长度/总长度 的最大比例
        gap12_required: bool = True,     # 是否要求 K1 和 K2 之间有缺口
        gap23_required: bool = False,    # 是否要求 K2 和 K3 之间有缺口 (A股通常不严格要求)
        gap_tolerance_pct: float = 0.03, # 缺口容差
        penetrate_req: float = 0.50,     # K3 跌入 K1 实体的深度 (0.5 = 50%)
        use_trend_filter: bool = True,   # 趋势过滤 (Close > MA)
        use_volume_filter: bool = False, # 成交量过滤
        use_confirm: bool = False,       # 确认信号 (次日继续跌)
        confirm_shift: int = 1
) -> pd.DataFrame:
    """
    识别黄昏星及黄昏十字星形态，并计算阻力位
    """
    data = df.copy()

    # --- 1. 基础指标计算 ---
    data["Range"] = (data["High"] - data["Low"]).replace(0, 1e-6)
    data["Body"] = (data["Close"] - data["Open"]).abs()

    # 识别十字星 (Doji) 和 星线 (Star)
    data["Doji"] = data["Body"] <= data["Range"] * 0.1
    data["Star"] = data["Body"] <= data["Range"] * star_small_pct

    data["MA"] = data["Close"].rolling(ma_len).mean()
    data["VOL_MA"] = data["Volume"].rolling(vol_ma_len).mean()

    # --- 2. 准备三根 K 线数据 ---
    # 1=前前日(长阳), 2=前日(星线), 3=当日(阴线)
    o1, h1, l1, c1 = [data[c].shift(2) for c in ["Open", "High", "Low", "Close"]]
    o2, h2, l2, c2 = [data[c].shift(1) for c in ["Open", "High", "Low", "Close"]]
    o3, h3, l3, c3 = data["Open"], data["High"], data["Low"], data["Close"]

    # --- 3. 形态逻辑 ---

    # K1: 长阳线 (实体 > 总长度的一半)
    long_white_1 = (c1 > o1) & ((c1 - o1) > (h1 - l1) * 0.5)

    # K2: 星线 (实体很小)
    # 注意：这里直接使用 shift 后的计算值，逻辑同 data['Star'] 但为了对齐索引
    body2 = (c2 - o2).abs()
    range2 = (h2 - l2).replace(0, 1e-6)
    is_star_2 = body2 <= range2 * star_small_pct

    # K3: 深入 K1 实体
    # 跌入深度计算：Open1 + (Entity1 * (1 - ratio))
    # 因为 K1 是阳线，Bottom=Open1, Top=Close1。我们要求 C3 低于 Top 下方的某个位置
    # 原逻辑：penetration_level = o1 + (c1 - o1) * (1 - penetrate_req)
    penetration_level = o1 + (c1 - o1) * (1 - penetrate_req)
    deep_bear_3 = (c3 < o3) & (c3 <= penetration_level)

    # --- 4. 缺口逻辑 (Gap) ---
    # 定义实体区间 (Entity Range)
    e1_low, e1_high = np.minimum(o1, c1), np.maximum(o1, c1)
    e2_low, e2_high = np.minimum(o2, c2), np.maximum(o2, c2)
    e3_low, e3_high = np.minimum(o3, c3), np.maximum(o3, c3)

    # Gap 1-2: K2 实体低点 > K1 实体高点 (向上跳空)
    if gap12_required:
        gap12 = e2_low > e1_high
    else:
        # 宽松模式
        gap12 = e2_low >= (e1_high * (1 - gap_tolerance_pct))

    # Gap 2-3: K3 实体高点 < K2 实体低点 (向下跳空) - A股较少见，通常设为 False
    if gap23_required:
        gap23 = e3_high < e2_low
    else:
        # 宽松模式
        gap23 = e3_high <= (e2_low * (1 + gap_tolerance_pct))

    # --- 5. 过滤器 ---
    # 趋势过滤：K1 发生时处于均线之上 (shift(2) 对应 K1 时刻，但原代码用 shift(1) 即 K2 时刻判断，也合理)
    if use_trend_filter:
        trend_ok = data["Close"].shift(1) > data["MA"].shift(1)
    else:
        trend_ok = pd.Series(True, index=data.index)

    # 成交量过滤
    if use_volume_filter:
        vol_ok = data["Volume"] > data["VOL_MA"]
    else:
        vol_ok = pd.Series(True, index=data.index)

    # --- 6. 信号合成 ---
    basic = long_white_1 & is_star_2 & deep_bear_3 & gap12 & gap23 & trend_ok & vol_ok

    if use_confirm:
        confirm_ok = data["Close"].shift(-confirm_shift) < data["Close"]
        final_signal = basic & confirm_ok
    else:
        final_signal = basic

    data["Evening_Star"] = final_signal

    # 细分：黄昏十字星 (K2 不仅是星线，还是严格的十字星)
    data["Evening_Doji_Star"] = (final_signal & data["Doji"].shift(1).fillna(False))

    # --- 7. 计算阻力位 ---
    # 取三根 K 线的最高点作为后续压力位
    h1_s, h2_s = data["High"].shift(2), data["High"].shift(1)
    # np.maximum.reduce 可以比较多个数组
    # 注意：计算包含 NaN 时可能会有问题，fill_value=0 确保安全
    res_level = np.maximum.reduce([h1_s.fillna(0), h2_s.fillna(0), data["High"].fillna(0)])

    data["ES_Resist"] = np.where(data["Evening_Star"], res_level, np.nan)
    data["ES_Resist_plot"] = data["ES_Resist"].ffill()

    return data


# ==========================================
# 2. 主执行程序
# ==========================================

def main():
    # --- 参数配置 ---
    symbol_code = '600519'          # 贵州茅台
    days_back = 365

    strategy_params = {
        'ma_len': 20,
        'star_small_pct': 0.20,
        'gap12_required': True,     # 必须有向上跳空
        'gap23_required': False,    # 向下跳空不强制 (A股特性)
        'gap_tolerance_pct': 0.03,
        'penetrate_req': 0.50,      # 跌破一半
        'use_trend_filter': True,
        'use_volume_filter': False
    }

    print(f"正在获取 {symbol_code} 的历史数据...")

    # --- 1. 获取数据 ---
    end_dt = datetime.today()
    start_dt = end_dt - timedelta(days=days_back)

    try:
        df_raw = ak.stock_zh_a_hist(
            symbol=symbol_code,
            period='daily',
            start_date=start_dt.strftime('%Y%m%d'),
            end_date=end_dt.strftime('%Y%m%d'),
            adjust='qfq'
        )
    except Exception as e:
        print(f"数据获取失败: {e}")
        return

    if df_raw.empty:
        print("未获取到数据。")
        return

    # 数据清洗
    df = (
        df_raw
        .rename(columns={'日期': 'Date', '开盘': 'Open', '最高': 'High',
                         '最低': 'Low', '收盘': 'Close', '成交量': 'Volume'})
        .loc[:, ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        .assign(Date=lambda x: pd.to_datetime(x['Date']))
        .set_index('Date')
        .sort_index()
    )

    print("正在计算黄昏星形态...")

    # --- 2. 计算形态 ---
    df = identify_evening_star(df, **strategy_params)

    # 统计
    n_sig = df["Evening_Star"].sum()
    print(f'黄昏星形态出现次数：{n_sig}')

    if n_sig > 0:
        print("最近信号示例：")
        print(df[df["Evening_Star"]][['Open', 'High', 'Low', 'Close']].tail())
    else:
        print('当前参数下无黄昏星形态，可尝试放宽 gap_tolerance_pct 或 star_small_pct。')

    # --- 3. 可视化 ---
    print("正在绘图...")

    apds = [mpf.make_addplot(df["MA"], color='blue', width=1.0)]

    rng = (df["High"] - df["Low"]).replace(0, 1e-6)

    # 信号标记 1: 普通黄昏星 (红色倒三角)
    es_marks = np.where(df["Evening_Star"], df["High"] + rng * 0.15, np.nan)

    # 信号标记 2: 黄昏十字星 (橙色倒三角，更加强烈的反转信号)
    eds_marks = np.where(df["Evening_Doji_Star"], df["High"] + rng * 0.28, np.nan)

    # 阻力线
    res_line = df["ES_Resist_plot"]

    # 只添加非全 NaN 的图层，避免报错
    if not np.all(np.isnan(es_marks)):
        apds.append(
            mpf.make_addplot(es_marks, type="scatter", marker="v", markersize=90, color="tab:red", label='Evening Star')
        )

    if not np.all(np.isnan(eds_marks)):
        apds.append(
            mpf.make_addplot(eds_marks, type="scatter", marker="v", markersize=90, color="tab:orange", label='Evening Doji Star')
        )

    if not np.all(np.isnan(res_line)):
        apds.append(
            mpf.make_addplot(res_line, color="tab:red", linestyle='--', width=1.0, alpha=0.7)
        )

    # 绘图
    # 标题使用英文避免乱码，Yahoo 风格比较经典
    title_str = f'{symbol_code} Evening Star Pattern'

    mpf.plot(
        df,
        type="candle",
        volume=True,
        addplot=apds,
        title=title_str,
        style="yahoo",
        figsize=(14, 8),
        tight_layout=True,
        block=True
    )
    print("完成。")

if __name__ == "__main__":
    main()