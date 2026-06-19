# -*- coding: utf-8 -*-
"""
K线形态识别性能优化基础模块

提供预计算、缓存和向量化操作的基础类，显著提升性能。
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from functools import lru_cache
import warnings

warnings.filterwarnings('ignore')


class PatternBase:
    """
    形态识别基础类 - 提供预计算和缓存机制
    
    性能优化：
    1. 预计算基础指标（只计算一次）
    2. 缓存中间结果
    3. 向量化操作（避免循环）
    4. 使用numpy代替pandas计算
    """
    
    _cache: Dict[str, pd.DataFrame] = {}
    _last_data_hash: Optional[int] = None
    
    def __init__(self, df: pd.DataFrame):
        """初始化并预计算基础指标"""
        self.df = df.copy()
        self._precompute()
    
    def _precompute(self):
        """
        预计算所有基础指标 - 这是性能优化的关键
        
        将原本在每个形态文件中重复计算的 Body, Range, Bull, Bear 等
        统一在这里计算一次，后续所有形态直接使用
        """
        # 基础OHLC
        o = self.df['Open'].astype(float)
        h = self.df['High'].astype(float)
        l = self.df['Low'].astype(float)
        c = self.df['Close'].astype(float)
        v = self.df['Volume'].astype(float) if 'Volume' in self.df.columns else pd.Series(0, index=self.df.index)
        
        # 实体和区间（核心指标）
        self.df['_body'] = (c - o).abs()
        self.df['_range'] = h - l
        self.df['_range_safe'] = self.df['_range'].replace(0, 1e-10)
        
        # 涨跌判断
        self.df['_bull'] = (c > o).astype(int)
        self.df['_bear'] = (c < o).astype(int)
        
        # 影线计算（向量化）
        self.df['_upper_shadow'] = h - np.maximum(o, c)
        self.df['_lower_shadow'] = np.minimum(o, c) - l
        
        # 实体边界
        self.df['_body_low'] = np.minimum(o, c)
        self.df['_body_high'] = np.maximum(o, c)
        
        # 成交量
        self.df['_volume'] = v
        self.df['_vol_ma20'] = v.rolling(20, min_periods=1).mean()
        self.df['_vol_ma50'] = v.rolling(50, min_periods=1).mean()
        
        # 移动平均线
        self.df['_ma20'] = c.rolling(20, min_periods=1).mean()
        self.df['_ma50'] = c.rolling(50, min_periods=1).mean()
        self.df['_ma10'] = c.rolling(10, min_periods=1).mean()
        
        # 趋势判断（向量化）
        self.df['_above_ma20'] = (c > self.df['_ma20']).astype(int)
        self.df['_below_ma20'] = (c < self.df['_ma20']).astype(int)
        self.df['_above_ma50'] = (c > self.df['_ma50']).astype(int)
        self.df['_below_ma50'] = (c < self.df['_ma50']).astype(int)
        
        # ATR (Average True Range)
        tr1 = h - l
        tr2 = (h - c.shift()).abs()
        tr3 = (l - c.shift()).abs()
        self.df['_tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        self.df['_atr14'] = self.df['_tr'].rolling(14, min_periods=1).mean()
        
        # 数据哈希（用于缓存验证）
        self._data_hash = hash(str(c.head(100).tolist()))
    
    def get_cols(self) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """获取OHLC列"""
        return self.df['Open'], self.df['High'], self.df['Low'], self.df['Close']
    
    def get_body(self) -> pd.Series:
        """获取实体"""
        return self.df['_body']
    
    def get_range(self) -> pd.Series:
        """获取区间"""
        return self.df['_range']
    
    def is_bullish(self, shift: int = 0) -> pd.Series:
        """判断阳线"""
        return self.df['_bull'].shift(shift).fillna(0).astype(bool)
    
    def is_bearish(self, shift: int = 0) -> pd.Series:
        """判断阴线"""
        return self.df['_bear'].shift(shift).fillna(0).astype(bool)
    
    def is_doji(self, threshold: float = 0.1) -> pd.Series:
        """判断十字星"""
        return (self.df['_body'] / self.df['_range_safe']) < threshold
    
    def is_small_body(self, threshold: float = 0.3) -> pd.Series:
        """判断小实体"""
        return (self.df['_body'] / self.df['_range_safe']) < threshold
    
    def get_previous(self, shift: int = 1) -> Dict[str, pd.Series]:
        """获取前N根K线数据"""
        return {
            'open': self.df['Open'].shift(shift),
            'high': self.df['High'].shift(shift),
            'low': self.df['Low'].shift(shift),
            'close': self.df['Close'].shift(shift),
            'body': self.df['_body'].shift(shift),
            'range': self.df['_range'].shift(shift),
            'bull': self.df['_bull'].shift(shift),
            'bear': self.df['_bear'].shift(shift),
            'body_low': self.df['_body_low'].shift(shift),
            'body_high': self.df['_body_high'].shift(shift)
        }
    
    def get_trend(self) -> pd.Series:
        """获取趋势（1=上涨，-1=下跌，0=震荡）"""
        c = self.df['Close']
        ma20 = self.df['_ma20']
        return np.where(c > ma20, 1, np.where(c < ma20, -1, 0))
    
    def get_volume_filter(self, threshold: float = 1.0) -> pd.Series:
        """成交量过滤（是否大于均量）"""
        return self.df['_volume'] > self.df['_vol_ma20'] * threshold
    
    def get_range_ratio(self, body: pd.Series = None) -> pd.Series:
        """实体占比（实体/区间）"""
        if body is None:
            body = self.df['_body']
        return body / self.df['_range_safe']
    
    def calculate_engulfing_body(self, prev: Dict, curr: Dict) -> pd.Series:
        """
        计算包容关系（核心优化）
        
        使用向量化操作代替循环，大幅提升性能
        """
        # 当天实体区间
        curr_low = np.minimum(curr['open'], curr['close'])
        curr_high = np.maximum(curr['open'], curr['close'])
        
        # 前一天实体区间
        prev_low = prev['body_low']
        prev_high = prev['body_high']
        
        # 包容判断：当天实体包裹前一天实体
        engulf = (curr_low <= prev_low) & (curr_high >= prev_high)
        
        return engulf
    
    def calculate_gap(self, prev: Dict, curr: Dict) -> Tuple[pd.Series, pd.Series]:
        """
        计算跳空缺口
        
        返回：
        - gap_up: 向上跳空
        - gap_down: 向下跳空
        """
        # K2实体低点 > K1实体高点（向上跳空）
        curr_body_low = np.minimum(curr['open'], curr['close'])
        prev_body_high = prev['body_high']
        gap_up = curr_body_low > prev_body_high
        
        # K2实体高点 < K1实体低点（向下跳空）
        curr_body_high = np.maximum(curr['open'], curr['close'])
        prev_body_low = prev['body_low']
        gap_down = curr_body_high < prev_body_low
        
        return gap_up, gap_down
    
    def calculate_penetration(self, prev: Dict, curr: Dict, ratio: float = 0.5) -> pd.Series:
        """
        计算深入程度（用于黄昏星等形态）
        
        ratio: 深入前一根实体的比例（0.5 = 50%）
        """
        prev_open = prev['open']
        prev_close = prev['close']
        prev_body_high = prev['body_high']
        
        # 深入水平线 = 实体顶部 - 实体长度 * ratio
        penetration_level = prev_open + (prev_close - prev_open) * ratio
        
        # 判断是否深入
        curr_close = curr['close']
        return curr_close <= penetration_level


class OptimizedIndicators:
    """
    优化后的技术指标计算
    
    性能提升：
    1. 使用numpy向量化操作
    2. 避免循环，逐元素计算
    3. 预分配数组
    """
    
    @staticmethod
    def calculate_ema(data: pd.Series, span: int) -> pd.Series:
        """计算EMA（指数移动平均）"""
        return data.ewm(span=span, adjust=False, min_periods=1).mean()
    
    @staticmethod
    def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI（相对强弱指标）- 向量化版本"""
        delta = data.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        # 使用ewm计算移动平均
        avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=1).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=1).mean()
        
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """
        计算MACD - 优化版本
        
        返回：{'macd': ..., 'signal': ..., 'hist': ...}
        """
        # 计算快慢EMA
        ema_fast = data.ewm(span=fast, adjust=False, min_periods=1).mean()
        ema_slow = data.ewm(span=slow, adjust=False, min_periods=1).mean()
        
        # MACD线
        macd_line = ema_fast - ema_slow
        
        # 信号线
        signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=1).mean()
        
        # 柱状图
        hist = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'hist': hist
        }
    
    @staticmethod
    def calculate_kdj(data: pd.Series, high: pd.Series, low: pd.Series, 
                      n: int = 9, k_period: int = 3, d_period: int = 3) -> Dict[str, pd.Series]:
        """
        计算KDJ - 向量化版本（大幅提升性能）
        
        使用pandas的ewm代替循环
        """
        # 计算最低价和最高价的n日移动窗口
        low_min = low.rolling(window=n, min_periods=1).min()
        high_max = high.rolling(window=n, min_periods=1).max()
        
        # 计算RSV
        rsv = (data - low_min) / (high_max - low_min + 1e-10) * 100
        rsv = rsv.fillna(50)
        
        # 使用ewm计算K、D（替代循环）
        k = rsv.ewm(alpha=1/k_period, adjust=False).mean()
        d = k.ewm(alpha=1/d_period, adjust=False).mean()
        j = 3 * k - 2 * d
        
        return {
            'k': k,
            'd': d,
            'j': j
        }
    
    @staticmethod
    def calculate_bollinger_bands(data: pd.Series, window: int = 20, num_std: float = 2) -> Dict[str, pd.Series]:
        """
        计算布林带 - 向量化版本
        """
        middle = data.rolling(window=window, min_periods=1).mean()
        std = data.rolling(window=window, min_periods=1).std()
        
        upper = middle + (std * num_std)
        lower = middle - (std * num_std)
        
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower
        }
    
    @staticmethod
    def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                            k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """
        计算随机指标 - 向量化版本
        """
        low_min = low.rolling(k_period, min_periods=1).min()
        high_max = high.rolling(k_period, min_periods=1).max()
        
        k = (close - low_min) / (high_max - low_min + 1e-10) * 100
        d = k.rolling(d_period, min_periods=1).mean()
        
        return {
            'k': k,
            'd': d
        }


def optimize_pattern_function(func):
    """
    装饰器：优化形态识别函数
    
    使用方法：
    @optimize_pattern_function
    def my_pattern(df, ...):
        ...
    """
    def wrapper(df, *args, **kwargs):
        # 预计算基础指标
        base = PatternBase(df)
        
        # 将基础对象传递给原函数
        if 'base' in func.__code__.co_varnames:
            kwargs['base'] = base
        
        return func(df, *args, **kwargs)
    
    return wrapper


class PatternCache:
    """
    形态识别结果缓存
    
    用于避免重复计算相同的形态
    """
    
    _cache: Dict[str, pd.DataFrame] = {}
    
    @classmethod
    def get(cls, key: str) -> Optional[pd.DataFrame]:
        """获取缓存"""
        return cls._cache.get(key)
    
    @classmethod
    def set(cls, key: str, value: pd.DataFrame):
        """设置缓存"""
        cls._cache[key] = value.copy()
    
    @classmethod
    def clear(cls):
        """清空缓存"""
        cls._cache.clear()
    
    @classmethod
    def get_pattern_key(cls, symbol: str, pattern_name: str, days: int) -> str:
        """生成缓存键"""
        return f"{symbol}_{pattern_name}_{days}"


# 性能测试装饰器
def profile_performance(func):
    """性能分析装饰器"""
    import time
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"函数 {func.__name__} 执行时间: {(end-start)*1000:.2f}ms")
        return result
    
    return wrapper


if __name__ == "__main__":
    # 测试性能
    print("测试优化后的基础模块...")
    
    # 创建测试数据
    dates = pd.date_range('2020-01-01', periods=1000)
    test_df = pd.DataFrame({
        'Open': np.random.uniform(100, 200, 1000),
        'High': np.random.uniform(100, 200, 1000),
        'Low': np.random.uniform(100, 200, 1000),
        'Close': np.random.uniform(100, 200, 1000),
        'Volume': np.random.uniform(1e6, 1e7, 1000)
    }, index=dates)
    
    # 测试PatternBase
    import time
    start = time.time()
    base = PatternBase(test_df)
    print(f"预计算耗时: {(time.time()-start)*1000:.2f}ms")
    
    # 测试指标计算
    start = time.time()
    rsi = OptimizedIndicators.calculate_rsi(test_df['Close'], 14)
    macd = OptimizedIndicators.calculate_macd(test_df['Close'])
    kdj = OptimizedIndicators.calculate_kdj(test_df['Close'], test_df['High'], test_df['Low'])
    print(f"指标计算耗时: {(time.time()-start)*1000:.2f}ms")
    
    print("基础模块测试完成！")