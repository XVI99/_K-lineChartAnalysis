# 三阶段优化实现报告

## 概述

本文档记录了量化交易系统的三阶段优化实现，目标是提高收益率和赢率。

## 第一阶段：信号质量提升

### 已实现功能

#### 1. 多时间框架确认 ([`multi_timeframe.py`](../quant_system/multi_timeframe.py))

```python
class MultiTimeframeConfirm:
    def __init__(self, config=None):
        self.config = config or {
            'require_weekly_trend': True,
            'require_weekly_ma_cross': False,
            'weekly_trend_ma': 20,
            'weekly_rsi_filter': False,
            'weekly_rsi_max': 70,
        }
```

功能：
- 将日线数据重采样为周线
- 检测周线趋势（BULL/NEUTRAL/BEAR）
- 只有当周线趋势为 BULL 或 NEUTRAL 时才允许买入

#### 2. 形态质量评分过滤 ([`pattern_scorer.py`](../quant_system/pattern_scorer.py))

新增功能：
- `filter_signals_by_quality()` - 基于质量分过滤信号
- `get_pattern_quality_score()` - 获取形态质量分
- `PatternQualityFilter` 类 - 可配置的质量过滤器

```python
class PatternQualityFilter:
    def __init__(self, min_quality=70, min_score=5, pattern_weights=None):
        self.min_quality = min_quality  # 最低质量分阈值
        self.min_score = min_score      # 最低综合得分
```

质量评分标准（0-100）：
- 形态质量（30分）：实体与区间比率、影线比例
- 位置评分（25分）：与MA20的距离
- 成交量确认（25分）：成交量与MA20的比率
- 趋势一致性（20分）：与MA20/MA60趋势的一致性

## 第二阶段：盈亏比优化

### 已实现功能 ([`backtest_engine.py`](../quant_system/backtest_engine.py))

#### 1. ATR动态止损

```python
# 初始化参数
atr_stop_loss_multiplier=1.5  # 1.5倍ATR止损
atr_period=14                 # ATR周期

# 计算止损价格
self.dynamic_stop_loss = exec_price - (current_atr * self.atr_stop_loss_multiplier)
```

#### 2. 移动止盈

```python
# 参数
trailing_take_profit_pct=0.05      # 移动止盈百分比
trailing_take_profit_trigger=0.05  # 触发阈值（5%盈利后激活）

# 逻辑
if current_profit_pct >= self.trailing_take_profit_trigger:
    self.trailing_take_profit_active = True
    self.trailing_take_profit_price = close_price * (1 - self.trailing_take_profit_pct)
```

#### 3. 分批止盈

```python
# 参数
partial_take_profit_pct=0.5   # 平仓50%
partial_take_profit_at=2.0    # 2倍风险回报比触发

# 逻辑
target_profit = risk_per_share * self.partial_take_profit_at
if close_price >= target_price:
    self._execute_partial_sell(date, exec_price, exit_reason)
    self.dynamic_stop_loss = self.entry_price  # 移动止损到保本
```

## 第三阶段：市场环境过滤

### 已实现功能

#### 1. 市场状态检测 ([`market_regime.py`](../quant_system/market_regime.py))

```python
class MarketRegime(Enum):
    BULL = "BULL"
    NEUTRAL = "NEUTRAL"
    BEAR = "BEAR"

def check_market_regime(index_df, ma_short=20, ma_long=60):
    # BULL: Close > MA_short > MA_long
    # NEUTRAL: Close > MA_long
    # BEAR: Close < MA_long
```

#### 2. 仓位控制

```python
def get_market_regime_filter(index_code="sh000300", days=120):
    # BULL: 允许开仓，100%仓位
    # NEUTRAL: 允许开仓，50%仓位
    # BEAR: 不允许新开仓
```

## 整合策略

### OptimizedStrategy 类 ([`optimized_strategy.py`](../quant_system/optimized_strategy.py))

```python
class OptimizedStrategy:
    def __init__(self, config=None):
        # 初始化所有组件
        self.pattern_processor = StandardizedPatternProcessor(k_line_dir)
        self.quality_filter = PatternQualityFilter(min_quality=70, min_score=5)
        self.signal_filter = SignalFilter()
        self.mtf_confirm = MultiTimeframeConfirm()
    
    def apply_strategy(self, df, verbose=True):
        # 1. 检测形态
        df = self.pattern_processor.run_all_patterns(df)
        # 2. 多时间框架确认
        df = self.mtf_confirm.add_weekly_trend_column(df)
        # 3. 质量过滤生成信号
        df = self._generate_filtered_signals(df)
        return df
    
    def create_backtest_engine(self):
        return BacktestEngine(
            # Phase 2 参数
            atr_stop_loss_multiplier=1.5,
            trailing_take_profit_pct=0.05,
            partial_take_profit_pct=0.5,
            # ...
        )
```

## 使用方法

### 运行优化策略回测

```bash
# 单只股票回测
python -m quant_system.optimized_strategy --symbol sh600519 --days 800

# 对比原始策略与优化策略
python -m quant_system.optimized_strategy --symbol sh600519 --compare
```

### 代码调用

```python
from quant_system.optimized_strategy import OptimizedStrategy

# 使用默认配置
strategy = OptimizedStrategy()
result = strategy.run_backtest('sh600519', days=800)

# 自定义配置
config = {
    'min_pattern_quality': 70,
    'min_score': 5,
    'atr_stop_loss_multiplier': 1.5,
    'trailing_take_profit_pct': 0.05,
    'use_market_filter': True,
}
strategy = OptimizedStrategy(config)
```

## 文件清单

| 文件 | 功能 |
|------|------|
| `multi_timeframe.py` | 多时间框架确认 |
| `pattern_scorer.py` | 形态质量评分与过滤 |
| `signal_filter.py` | 信号多重过滤 |
| `backtest_engine.py` | 回测引擎（含Phase 2功能） |
| `market_regime.py` | 市场状态检测 |
| `optimized_strategy.py` | 整合优化策略 |

## 预期效果

根据优化设计，预期实现：

1. **赢率提升**：通过质量过滤和多时间框架确认，过滤低质量信号
2. **盈亏比改善**：通过移动止盈和分批止盈，锁定利润
3. **风险控制**：通过ATR动态止损和市场环境过滤，控制回撤

## 注意事项

1. 质量分阈值（`min_quality=70`）可根据实际回测结果调整
2. ATR止损倍数（1.5）可根据市场波动性调整
3. 市场过滤功能需要沪深300指数数据

## 后续优化方向

1. 添加板块轮动过滤
2. 实现自适应参数调整
3. 添加机器学习信号增强
4. 优化仓位管理算法
