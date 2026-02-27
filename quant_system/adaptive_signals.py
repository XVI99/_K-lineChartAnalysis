"""
Adaptive Signal Generator — All-Weather Alpha Engine

核心理念: 顶级交易员不看天吃饭
- 牛市: 趋势跟踪 + 回调买入 (Trend Following + Pullback)
- 震荡市: 均值回归 + 区间交易 (Mean Reversion + Range Trading)
- 熊市: 超跌反弹 + 快进快出 (Oversold Bounce + Scalping)

每种市场状态使用不同的:
1. 信号生成逻辑
2. 止盈止损参数
3. 仓位大小
4. 持仓时间限制
"""
import pandas as pd
import numpy as np


# =========================================================================
# 技术指标计算 (用于信号生成)
# =========================================================================

def compute_rsi(series, period=14):
    """计算RSI"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_bollinger(series, period=20, std_mult=2.0):
    """计算布林带"""
    ma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = ma + std_mult * std
    lower = ma - std_mult * std
    pct_b = (series - lower) / (upper - lower)  # %B indicator
    return ma, upper, lower, pct_b


def compute_atr(df, period=14):
    """计算ATR"""
    high = df['High']
    low = df['Low']
    close = df['Close']
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        abs(high - prev_close),
        abs(low - prev_close)
    ], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def compute_adx(df, period=14):
    """计算ADX (Average Directional Index) — 衡量趋势强度"""
    high = df['High']
    low = df['Low']
    close = df['Close']

    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    atr = compute_atr(df, period)
    atr = atr.replace(0, np.nan)

    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)

    dx_sum = plus_di + minus_di
    dx_sum = dx_sum.replace(0, np.nan)
    dx = 100 * abs(plus_di - minus_di) / dx_sum
    adx = dx.rolling(period).mean()

    return adx, plus_di, minus_di


def detect_support_resistance(df, lookback=60, n_levels=3):
    """
    检测支撑/阻力位 (基于局部极值点)

    Returns:
        tuple: (support_levels, resistance_levels) — 各为价格列表
    """
    close = df['Close'].iloc[-lookback:]
    high = df['High'].iloc[-lookback:]
    low = df['Low'].iloc[-lookback:]

    # 寻找局部最低点作为支撑
    supports = []
    for i in range(2, len(low) - 2):
        if low.iloc[i] <= low.iloc[i-1] and low.iloc[i] <= low.iloc[i-2] \
           and low.iloc[i] <= low.iloc[i+1] and low.iloc[i] <= low.iloc[i+2]:
            supports.append(low.iloc[i])

    # 寻找局部最高点作为阻力
    resistances = []
    for i in range(2, len(high) - 2):
        if high.iloc[i] >= high.iloc[i-1] and high.iloc[i] >= high.iloc[i-2] \
           and high.iloc[i] >= high.iloc[i+1] and high.iloc[i] >= high.iloc[i+2]:
            resistances.append(high.iloc[i])

    # 聚类相近的价位
    supports = _cluster_levels(supports, threshold=0.02)
    resistances = _cluster_levels(resistances, threshold=0.02)

    return sorted(supports)[-n_levels:], sorted(resistances)[:n_levels]


def _cluster_levels(levels, threshold=0.02):
    """将相近的价位聚类为一个"""
    if not levels:
        return []
    levels = sorted(levels)
    clustered = [levels[0]]
    for lvl in levels[1:]:
        if abs(lvl - clustered[-1]) / clustered[-1] < threshold:
            clustered[-1] = (clustered[-1] + lvl) / 2  # 取均值
        else:
            clustered.append(lvl)
    return clustered


def compute_volume_ratio(df, period=20):
    """计算成交量相对均值的比率"""
    if 'Volume' not in df.columns:
        return pd.Series(1.0, index=df.index)
    vol_ma = df['Volume'].rolling(period).mean()
    vol_ma = vol_ma.replace(0, np.nan)
    return df['Volume'] / vol_ma


# =========================================================================
# 市场微观结构分类 (比简单MA交叉更精细)
# =========================================================================

class MarketStructure:
    """
    更精细的市场结构检测

    结合ADX + MA + 波动率判断:
    - STRONG_TREND: ADX > 25 + MA排列一致 → 纯趋势跟踪
    - WEAK_TREND: ADX 20-25 或 MA略微纠缠 → 趋势+回调
    - RANGE: ADX < 20 + 价格在布林带内震荡 → 均值回归
    - HIGH_VOL_DOWN: 急跌 + 放量 → 超跌反弹
    """
    STRONG_TREND_UP = "STRONG_TREND_UP"
    WEAK_TREND_UP = "WEAK_TREND_UP"
    RANGE = "RANGE"
    WEAK_TREND_DOWN = "WEAK_TREND_DOWN"
    STRONG_TREND_DOWN = "STRONG_TREND_DOWN"
    CRASH = "CRASH"  # 恐慌性下跌 → 超跌反弹机会

    @staticmethod
    def classify(df, idx=-1):
        """
        分类当前市场状态

        Args:
            df: OHLCV DataFrame, 至少需要60行数据
            idx: 分析位置 (default -1)

        Returns:
            tuple: (regime, details_dict)
        """
        if len(df) < 60:
            return MarketStructure.RANGE, {'reason': 'insufficient_data'}

        close = df['Close']
        i = idx if idx >= 0 else len(df) + idx

        # 计算指标
        ma20 = close.rolling(20).mean().iloc[i]
        ma60 = close.rolling(60).mean().iloc[i]
        current = close.iloc[i]

        adx_series, plus_di, minus_di = compute_adx(df)
        adx = adx_series.iloc[i] if not pd.isna(adx_series.iloc[i]) else 15

        rsi = compute_rsi(close).iloc[i]
        if pd.isna(rsi):
            rsi = 50

        atr = compute_atr(df).iloc[i]
        atr_pct = (atr / current * 100) if current > 0 else 2.0

        # 近5日涨跌幅
        ret_5d = (current / close.iloc[max(0, i-5)] - 1) * 100 if i >= 5 else 0

        details = {
            'adx': round(adx, 1),
            'rsi': round(rsi, 1),
            'atr_pct': round(atr_pct, 2),
            'ret_5d': round(ret_5d, 2),
            'ma20': round(ma20, 2),
            'ma60': round(ma60, 2),
            'close': round(current, 2),
        }

        # === 分类逻辑 ===

        # 恐慌性下跌: 5日跌幅 > 8% 且 RSI < 25
        if ret_5d < -8 and rsi < 25:
            return MarketStructure.CRASH, details

        # 强趋势上涨: ADX > 25, Close > MA20 > MA60
        if adx > 25 and current > ma20 > ma60:
            return MarketStructure.STRONG_TREND_UP, details

        # 弱趋势上涨: Close > MA60, ADX 15-25
        if current > ma60 and adx >= 15:
            return MarketStructure.WEAK_TREND_UP, details

        # 强趋势下跌: ADX > 25, Close < MA20 < MA60
        if adx > 25 and current < ma20 < ma60:
            return MarketStructure.STRONG_TREND_DOWN, details

        # 弱趋势下跌: Close < MA60
        if current < ma60:
            return MarketStructure.WEAK_TREND_DOWN, details

        # 默认: 区间震荡
        return MarketStructure.RANGE, details


# =========================================================================
# 自适应信号生成器
# =========================================================================

class AdaptiveSignalGenerator:
    """
    全天候自适应信号生成器

    根据市场结构自动选择最佳策略:
    - STRONG_TREND_UP: 追趋势 + 回调买入
    - WEAK_TREND_UP: 回调买入 + 形态确认
    - RANGE: 均值回归 (布林带下轨买, 上轨卖)
    - WEAK_TREND_DOWN: 超跌反弹 + 快速止盈
    - STRONG_TREND_DOWN: 只做超跌反弹, 严格止盈
    - CRASH: 恐慌买入 (极端超跌)
    """

    # 每种市场状态的参数配置
    # 注意: 针对小资金(5000元)优化, position_scale不能太低否则买不到100股
    REGIME_PARAMS = {
        MarketStructure.STRONG_TREND_UP: {
            'position_scale': 1.0,         # 满仓操作
            'risk_per_trade': 0.03,        # 小资金可稍高风险
            'stop_loss_atr_mult': 2.0,     # 宽止损让利润奔跑
            'take_profit_pct': 0.15,       # 高目标
            'trailing_stop_pct': 0.08,     # 紧跟趋势
            'time_stop_days': 30,          # 耐心持有
            'max_position_pct': 0.95,      # 小资金可以重仓
        },
        MarketStructure.WEAK_TREND_UP: {
            'position_scale': 0.8,
            'risk_per_trade': 0.025,
            'stop_loss_atr_mult': 1.5,
            'take_profit_pct': 0.10,
            'trailing_stop_pct': 0.06,
            'time_stop_days': 20,
            'max_position_pct': 0.90,
        },
        MarketStructure.RANGE: {
            'position_scale': 0.7,
            'risk_per_trade': 0.02,
            'stop_loss_atr_mult': 1.2,     # 紧止损
            'take_profit_pct': 0.06,       # 目标小但确定性高
            'trailing_stop_pct': 0.04,     # 快速止盈
            'time_stop_days': 10,          # 短持仓
            'max_position_pct': 0.85,
        },
        MarketStructure.WEAK_TREND_DOWN: {
            'position_scale': 0.5,         # 半仓
            'risk_per_trade': 0.015,
            'stop_loss_atr_mult': 1.0,     # 极紧止损
            'take_profit_pct': 0.05,       # 快速止盈
            'trailing_stop_pct': 0.03,
            'time_stop_days': 7,           # 极短持仓
            'max_position_pct': 0.80,
        },
        MarketStructure.STRONG_TREND_DOWN: {
            'position_scale': 0.5,         # 小资金最低半仓(否则买不够100股)
            'risk_per_trade': 0.015,
            'stop_loss_atr_mult': 0.8,     # 超紧止损
            'take_profit_pct': 0.04,       # 赚一点就走
            'trailing_stop_pct': 0.02,
            'time_stop_days': 5,
            'max_position_pct': 0.80,
        },
        MarketStructure.CRASH: {
            'position_scale': 0.7,         # 恐慌中适度加仓
            'risk_per_trade': 0.025,
            'stop_loss_atr_mult': 1.5,
            'take_profit_pct': 0.08,       # 反弹目标
            'trailing_stop_pct': 0.04,
            'time_stop_days': 10,
            'max_position_pct': 0.90,
        },
    }

    def __init__(self):
        pass

    def generate_signals(self, df, pattern_buy_signals=None, pattern_sell_signals=None,
                         verbose=True):
        """
        根据市场结构生成自适应信号

        Args:
            df: OHLCV DataFrame (需要至少 60 行)
            pattern_buy_signals: Series/array, 原有形态买入信号 (1/0)
            pattern_sell_signals: Series/array, 原有形态卖出信号 (-1/0)
            verbose: 打印详细信息

        Returns:
            tuple: (signals_df, regime_params)
                signals_df: DataFrame 增加 'Adaptive_Signal', 'Adaptive_Reason', 'Regime' 列
                regime_params: dict, 当前市场状态对应的参数
        """
        result = df.copy()
        n = len(result)

        result['Adaptive_Signal'] = 0
        result['Adaptive_Reason'] = ''
        result['Regime'] = ''
        result['Regime_Score'] = 0.0

        if n < 60:
            return result, self.REGIME_PARAMS[MarketStructure.RANGE]

        # 预计算指标
        close = result['Close']
        rsi = compute_rsi(close, 14)
        bb_ma, bb_upper, bb_lower, pct_b = compute_bollinger(close, 20, 2.0)
        atr = compute_atr(result, 14)
        vol_ratio = compute_volume_ratio(result, 20)
        ma5 = close.rolling(5).mean()
        ma10 = close.rolling(10).mean()
        ma20 = close.rolling(20).mean()
        ma60 = close.rolling(60).mean()

        # 转为numpy加速
        close_arr = close.values
        rsi_arr = rsi.values
        pct_b_arr = pct_b.values
        bb_lower_arr = bb_lower.values
        bb_upper_arr = bb_upper.values
        bb_ma_arr = bb_ma.values
        atr_arr = atr.values
        vol_arr = vol_ratio.values
        ma5_arr = ma5.values
        ma10_arr = ma10.values
        ma20_arr = ma20.values
        ma60_arr = ma60.values

        buy_count = 0
        sell_count = 0
        last_regime = None

        for i in range(60, n):
            c = close_arr[i]
            r = rsi_arr[i] if not np.isnan(rsi_arr[i]) else 50
            pb = pct_b_arr[i] if not np.isnan(pct_b_arr[i]) else 0.5
            v = vol_arr[i] if not np.isnan(vol_arr[i]) else 1.0
            m5 = ma5_arr[i] if not np.isnan(ma5_arr[i]) else c
            m10 = ma10_arr[i] if not np.isnan(ma10_arr[i]) else c
            m20 = ma20_arr[i] if not np.isnan(ma20_arr[i]) else c
            m60 = ma60_arr[i] if not np.isnan(ma60_arr[i]) else c
            bl = bb_lower_arr[i] if not np.isnan(bb_lower_arr[i]) else c * 0.95
            bu = bb_upper_arr[i] if not np.isnan(bb_upper_arr[i]) else c * 1.05
            bm = bb_ma_arr[i] if not np.isnan(bb_ma_arr[i]) else c
            at = atr_arr[i] if not np.isnan(atr_arr[i]) else c * 0.02

            # 获取K线形态信号
            has_pattern_buy = False
            has_pattern_sell = False
            if pattern_buy_signals is not None:
                has_pattern_buy = bool(pattern_buy_signals.iloc[i]) if i < len(pattern_buy_signals) else False
            if pattern_sell_signals is not None:
                has_pattern_sell = bool(pattern_sell_signals.iloc[i]) if i < len(pattern_sell_signals) else False

            # 分类市场状态
            regime, details = MarketStructure.classify(result.iloc[:i+1], -1)
            result.iloc[i, result.columns.get_loc('Regime')] = regime

            signal = 0
            reason = ''
            score = 0.0

            # ============================================
            # === 策略1: 强趋势上涨 — 追入 + 回调买入 ===
            # ============================================
            if regime == MarketStructure.STRONG_TREND_UP:
                # 买入: 回调到MA10附近 + RSI从超卖回升
                if c <= m10 * 1.01 and r < 55 and r > 30:
                    signal = 1
                    score = 70 + min(r, 30)
                    reason = f"趋势回调买入(RSI={r:.0f},近MA10)"
                # 买入: 形态信号 + 趋势确认
                elif has_pattern_buy and c > m20:
                    signal = 1
                    score = 65
                    reason = f"趋势形态确认(RSI={r:.0f})"
                # 卖出: RSI极度超买
                elif r > 85:
                    signal = -1
                    reason = f"超买卖出(RSI={r:.0f})"
                elif has_pattern_sell and r > 70:
                    signal = -1
                    reason = f"趋势顶部形态(RSI={r:.0f})"

            # ============================================
            # === 策略2: 弱趋势上涨 — 回调+形态双确认 ===
            # ============================================
            elif regime == MarketStructure.WEAK_TREND_UP:
                # 买入: 回调到MA20 + RSI不超买 + 有形态
                if c <= m20 * 1.02 and r < 50 and r > 25 and has_pattern_buy:
                    signal = 1
                    score = 60 + min(r, 20)
                    reason = f"弱势回调+形态({r:.0f})"
                # 买入: 布林带下轨 + 缩量
                elif pb < 0.15 and r < 35 and v < 0.8:
                    signal = 1
                    score = 55
                    reason = f"布林下轨超卖(B%={pb:.2f})"
                # 卖出
                elif r > 78 or (has_pattern_sell and r > 65):
                    signal = -1
                    reason = f"弱势止盈(RSI={r:.0f})"

            # ============================================
            # === 策略3: 区间震荡 — 均值回归 ===
            # ============================================
            elif regime == MarketStructure.RANGE:
                # 买入: 价格触及布林带下轨 + RSI超卖
                if pb < 0.10 and r < 30:
                    signal = 1
                    score = 75
                    reason = f"均值回归买入(B%={pb:.2f},RSI={r:.0f})"
                # 买入: 接近下轨 + 缩量(主力洗盘)
                elif pb < 0.20 and r < 40 and v < 0.6:
                    signal = 1
                    score = 55
                    reason = f"缩量下轨反弹(B%={pb:.2f},Vol={v:.1f})"
                # 买入: K线形态 + 在区间下半部
                elif has_pattern_buy and pb < 0.40 and r < 45:
                    signal = 1
                    score = 50
                    reason = f"区间形态买入(B%={pb:.2f})"
                # 卖出: 布林带上轨 + RSI超买
                elif pb > 0.90 and r > 70:
                    signal = -1
                    reason = f"均值回归卖出(B%={pb:.2f},RSI={r:.0f})"
                # 卖出: 接近上轨
                elif pb > 0.80 and r > 65:
                    signal = -1
                    reason = f"区间上沿止盈(B%={pb:.2f})"

            # ============================================
            # === 策略4: 弱趋势下跌 — 超跌反弹(快进快出) ===
            # ============================================
            elif regime == MarketStructure.WEAK_TREND_DOWN:
                # 买入: RSI极度超卖 + 放量(恐慌盘结束)
                if r < 25 and v > 1.5:
                    signal = 1
                    score = 60
                    reason = f"超跌放量反弹(RSI={r:.0f},Vol={v:.1f})"
                # 买入: 连续下跌后出现K线反转形态
                elif has_pattern_buy and r < 35 and pb < 0.15:
                    signal = 1
                    score = 50
                    reason = f"弱势反转形态(RSI={r:.0f})"
                # 卖出: 快速止盈 — 反弹到MA20就走
                elif c > m20 * 0.99 and r > 50:
                    signal = -1
                    reason = f"弱势反弹止盈(到MA20)"
                elif r > 60:
                    signal = -1
                    reason = f"弱势RSI止盈({r:.0f})"

            # ============================================
            # === 策略5: 强趋势下跌 — 极端超跌才进场 ===
            # ============================================
            elif regime == MarketStructure.STRONG_TREND_DOWN:
                # 买入: 极端超卖 + 放量(底部放量)
                if r < 20 and v > 2.0:
                    signal = 1
                    score = 55
                    reason = f"极端超跌反弹(RSI={r:.0f},Vol={v:.1f})"
                # 卖出: 稍有反弹就走
                elif r > 50:
                    signal = -1
                    reason = f"熊市快速止盈(RSI={r:.0f})"

            # ============================================
            # === 策略6: 恐慌性暴跌 — "别人恐惧我贪婪" ===
            # ============================================
            elif regime == MarketStructure.CRASH:
                # 恐慌买入: RSI极低 + 大幅偏离均线
                dist_from_ma = (c - m20) / m20
                if r < 20 and dist_from_ma < -0.08:
                    signal = 1
                    score = 80
                    reason = f"恐慌贪婪买入(RSI={r:.0f},偏离MA20={dist_from_ma:.1%})"
                elif r < 30 and has_pattern_buy:
                    signal = 1
                    score = 65
                    reason = f"恐慌反转形态(RSI={r:.0f})"
                # 反弹止盈
                elif r > 45:
                    signal = -1
                    reason = f"恐慌反弹止盈(RSI={r:.0f})"

            # 写入结果
            if signal != 0:
                result.iloc[i, result.columns.get_loc('Adaptive_Signal')] = signal
                result.iloc[i, result.columns.get_loc('Adaptive_Reason')] = reason
                result.iloc[i, result.columns.get_loc('Regime_Score')] = score
                if signal == 1:
                    buy_count += 1
                else:
                    sell_count += 1

            if regime != last_regime and verbose:
                print(f"  [{result.index[i].strftime('%Y-%m-%d')}] 市场状态转换: {last_regime} → {regime} "
                      f"(ADX={details.get('adx', '?')}, RSI={details.get('rsi', '?')})")
                last_regime = regime

        # 确定最后一个bar的市场状态，返回对应参数
        final_regime, _ = MarketStructure.classify(result, -1)
        regime_params = self.REGIME_PARAMS.get(final_regime, self.REGIME_PARAMS[MarketStructure.RANGE])

        if verbose:
            print(f"\n[Adaptive] 当前市场: {final_regime}")
            print(f"[Adaptive] 参数: 仓位={regime_params['position_scale']}, "
                  f"风险={regime_params['risk_per_trade']}, "
                  f"止损={regime_params['stop_loss_atr_mult']}xATR, "
                  f"止盈={regime_params['take_profit_pct']:.0%}")
            print(f"[Adaptive] 信号: {buy_count}买 + {sell_count}卖")

        return result, regime_params

    def get_regime_params(self, df):
        """
        仅获取当前市场状态参数(不生成信号) — 用于BacktestEngine参数设置

        Returns:
            tuple: (regime_name, params_dict)
        """
        regime, details = MarketStructure.classify(df, -1)
        params = self.REGIME_PARAMS.get(regime, self.REGIME_PARAMS[MarketStructure.RANGE])
        return regime, params, details
