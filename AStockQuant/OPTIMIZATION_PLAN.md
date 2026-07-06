# AStockQuant 优化计划

> 基于 8 步 Sequential Thinking 深度分析，覆盖 2026-07-04 代码全量审阅结果。

---

## 一、项目现状诊断

### 1.1 已有成果

| 指标 | 数值 | 说明 |
|------|------|------|
| 筛选型回测总收益 | +43.91% | v1.0 推荐版 |
| 短窗口(2024+) OOS 年化 | +109.92% | quick 网格，12月训练/3月测试 |
| 长窗口 OOS 年化 | +17.24% | 主题热度增强，2021-2026 |
| 最大回撤 | -24.54% → -30.09% | 主题热度增大了回撤 |
| 实盘持仓 | 516160 +28.4%, 512480 +25.9% | 2025-2026 真实交易 |

### 1.2 架构概览

```
DataHub(10数据源降级) → 8层信号 → AI模型(DL/SEQ/RL/PPO) → FusionEngine → MarketScanner(13步) → 回测
```

### 1.3 核心瓶颈

#### 瓶颈 A：数据层不完整（最严重）

| 问题 | 现状 | 影响 |
|------|------|------|
| Layer3/4/5 无历史数据 | 板块/资金/情绪层依赖实时网络数据，回测时无历史面板 | **8层信号只有5层生效** |
| ETF 宇宙不足 | data_cache 仅约100只ETF | 幸存者偏差风险 |
| NAV 覆盖率低 | 仅40只ETF有历史NAV，walk-forward 43只测试宇宙仅27只有NAV | 折溢价因子不可用 |
| 外部信号不可回填 | 资金流/新闻/情绪只有当日快照 | 回测中 Layer3/4/5 空转 |

#### 瓶颈 B：因子设计不适配 ETF

| 问题 | 现状 | 影响 |
|------|------|------|
| VCP 在 ETF 上失效 | 代码注释明确 `ICIR≈0` | Layer6 核心因子无效 |
| 动量周期偏短 | RPS 用 20/50/120日，ETF 波动小需更长周期 | 信号噪声大 |
| 缺少截面动量 | 仅有个股时序 RPS，无跨 ETF 排名 | 丧失相对强度信息 |

#### 瓶颈 C：风险控制缺失

| 问题 | 现状 | 影响 |
|------|------|------|
| 主题热度增大回撤 | -24.54% → -30.09% | 收益提升但风险失控 |
| 无波动率目标 | 仓位不受波动率约束 | 高波期暴露过大 |
| 无回撤约束 | walk-forward 无最大回撤惩罚 | 回撤不可控 |

#### 瓶颈 D：回测统计不严谨

| 问题 | 现状 | 影响 |
|------|------|------|
| 测试期短 | 2024+ 仅覆盖反弹行情 | 缺少完整牛熊验证 |
| 参数网格泛化风险 | full 网格比 quick OOS 更差 | 过拟合迹象 |
| 无多重检验校正 | 未做 Deflated Sharpe Ratio | 可能是数据挖掘偏误 |

#### 瓶颈 E：模型过度设计

| 问题 | 现状 | 影响 |
|------|------|------|
| 四套模型堆叠 | DL + SEQ + RL + PPO | 复杂度高、维护难 |
| RL/PPO 在 ETF 场景价值低 | ETF 波动小于个股 | 仓位分配收益有限 |
| 不确定性量化粗糙 | `abs(dl-seq)` | 无法准确度量风险 |

---

## 二、优化目标重定义

### 2.1 废弃不可达目标

| 旧目标 | 诊断结论 |
|--------|----------|
| 每月收益 ≥ 30% | `diagnose_monthly_return_ceiling.py` 证明：17个月中仅5个月最强ETF达30%，纯ETF轮动无杠杆下不可行 |

### 2.2 新目标体系

| 指标 | 目标值 | 说明 |
|------|--------|------|
| 年化收益 | > 30% | 稳健可达，高于沪深300长期年化 |
| 夏普比率 | > 1.5 | 优秀的风险调整收益 |
| 最大回撤 | < -15% | 可接受的回撤范围 |
| Calmar 比率 | > 1.0 | 年化收益/最大回撤 > 1 |
| 月胜率 | > 60% | 多数月份正收益 |
| 信息比率 | > 0.5 | 相对等权ETF基准的超额收益 |

---

## 三、分阶段优化方案

### Phase 1：数据基础设施补全（1-2周）

**目标**：使 8 层信号在回测中全部生效。

#### 1.1 历史资金流数据库

```
scripts/
├── build_history_capital_flow.py   # 每日定时抓取并持久化
└── load_history_panel.py           # 回测时加载历史面板
```

**数据表结构**（SQLite/Parquet）：

| 表名 | 字段 | 来源 |
|------|------|------|
| `concept_flow_daily` | date, concept, net_inflow, rank | AkShare 同花顺概念资金流 |
| `lhb_daily` | date, symbol, reason, net_buy_amount | AkShare 龙虎榜 |
| `north_flow_daily` | date, net_buy_amount | AkShare 北向资金 |
| `theme_flow_daily` | date, theme, net_inflow, breadth, momentum | 概念资金流按ETF主题映射 |

**实施步骤**：

1. 编写 `scripts/build_history_capital_flow.py`，每日抓取并追加到 `data_cache/history/capital_flow.db`
2. 编写 `core/history_data_loader.py`，提供 `get_capital_flow(date)` / `get_lhb(date)` 等接口
3. 修改 `layers/layer4_capital.py` 的 `extract_features`，回测模式优先从历史库读取

#### 1.2 新闻情绪历史化

```
scripts/
└── build_sentiment_history.py  # 每日跑全市场情绪快照并存储
```

| 表名 | 字段 | 来源 |
|------|------|------|
| `sentiment_daily` | date, symbol, sentiment_score, confidence, summary | LLM NewsSentimentAnalyzer |

**实施步骤**：

1. 编写 `scripts/build_sentiment_history.py`，调用 `llm/News_sentiment.py` 每日生成快照
2. 存为 `data_cache/history/sentiment_history.parquet`（带时间戳）
3. 修改 `layers/layer5_sentiment.py`，回测模式从历史库读取

#### 1.3 板块轮动历史

| 表名 | 字段 | 来源 |
|------|------|------|
| `sector_daily` | date, sector, pct_change, volume, turnover_rank | AkShare 行业板块行情 |
| `sector_rotation` | date, sector, phase(warming/hot/cooling), momentum_score | 计算 |

**实施步骤**：

1. 编写 `scripts/build_sector_history.py`，每日缓存板块涨跌幅排名
2. 修改 `layers/layer3_sector.py`，回测模式从历史库读取

#### 1.4 ETF 宇宙扩展

**当前**：约100只 → **目标**：300+ 只

**实施步骤**：

1. 修改 `scripts/download_all_etf.py`，下载全市场 ETF 列表
2. 按主题/规模/流动性分层：宽基、行业、跨境、债券、商品
3. 过滤条件：日均成交额 > 1000万、上市 > 1年

#### 1.5 NAV 历史全覆盖

**当前**：40只 → **目标**：所有目标 ETF

**实施步骤**：

1. 扩展 `scripts/cache_etf_nav_history.py` 覆盖范围到 300+ ETF
2. 确保 walk-forward 测试宇宙的 NAV 覆盖率达 100%

---

### Phase 2：ETF 适配因子重构（2-3周）

**目标**：替换不适配 ETF 的因子，构建 ETF 专用因子集。

#### 2.1 ETF 专用动量因子

**替换**：RPS 20日 → 多周期长周期 + 截面排名

```python
# layers/layer6_price_vol.py 新增

def calculate_etf_momentum(self, close: pd.Series, all_prices: Dict) -> Dict:
    """ETF专用动量：长周期时序 + 截面排名"""
    # 时序动量：60/120/250日
    mom_60 = close.pct_change(60).iloc[-1]
    mom_120 = close.pct_change(120).iloc[-1]
    mom_250 = close.pct_change(250).iloc[-1]
    # 加权：近期权重更高
    ts_momentum = mom_60 * 0.5 + mom_120 * 0.3 + mom_250 * 0.2

    # 截面动量：跨ETF 20日收益率排名百分位
    # （在 MarketScanner 中批量计算）
    return {
        "pv_etf_ts_momentum": ts_momentum,
        "pv_etf_xs_momentum": None,  # 由 scanner 填充
    }
```

#### 2.2 主题轮动因子强化

```python
# layers/layer3_sector.py 新增

def calculate_theme_rotation(self, date: str, theme: str) -> Dict:
    """主题轮动：动量 + 资金流 + 广度"""
    # 主题动量：过去N日主题内ETF平均涨幅
    theme_momentum = self._get_theme_momentum(date, theme, window=20)
    # 主题资金流：概念资金流汇总
    theme_flow = self._get_theme_flow(date, theme)
    # 主题广度：主题内上涨ETF占比
    theme_breadth = self._get_theme_breadth(date, theme)
    # 综合得分
    score = theme_momentum * 0.4 + theme_flow * 0.35 + theme_breadth * 0.25
    return {
        "sector_theme_momentum": theme_momentum,
        "sector_theme_flow": theme_flow,
        "sector_theme_breadth": theme_breadth,
        "sector_theme_score": score,
    }
```

#### 2.3 波动率 Regime 自适应

```python
# layers/layer6_price_vol.py 新增

def detect_volatility_regime(self, close: pd.Series) -> str:
    """检测波动率regime，动态选择参数"""
    vol_20 = close.pct_change().rolling(20).std().iloc[-1]
    vol_60 = close.pct_change().rolling(60).std().iloc[-1]
    ratio = vol_20 / vol_60 if vol_60 > 0 else 1.0

    if ratio < 0.7:
        return "low_vol"   # 低波期：用长周期(120日)
    elif ratio > 1.3:
        return "high_vol"  # 高波期：用短周期(20日)
    else:
        return "normal"    # 正常：用中周期(60日)
```

#### 2.4 VCP 替代方案

**替换**：离散 VCP 检测 → 连续波动率收缩度（已有 `pv_volatility_contraction`）

```python
# 已有实现，需在walk-forward中启用
# layers/layer6_price_vol.py 第280-292行
# pv_volatility_contraction: 20日波动率/60日波动率，sigmoid映射
# 该因子在ETF上ICIR优于离散VCP，需加入walk-forward训练窗
```

#### 2.5 跨市场宏观因子

```python
# layers/layer1_macro.py 新增

def calculate_cross_market_factors(self, date: str) -> Dict:
    """跨市场宏观因子"""
    return {
        "macro_ah_premium": self._get_ah_premium(date),       # AH溢价率
        "macro_equity_bond": self._get_equity_bond_ratio(date), # 股债性价比
        "macro_commodity_rotation": self._get_commodity_signal(date),  # 商品-股票轮动
    }
```

#### 2.6 流动性因子

```python
# layers/layer6_price_vol.py 新增

def calculate_liquidity_factors(self, df: pd.DataFrame, nav_data: Dict) -> Dict:
    """流动性因子"""
    return {
        "pv_discount_premium": self._calc_discount_premium(df, nav_data),  # 折溢价率
        "pv_scale_change": nav_data.get("scale_change", 0),                # 规模变化
        "pv_turnover_rank": self._calc_turnover_rank(df),                  # 成交额排名
    }
```

#### 2.7 因子评估验证

```python
# scripts/factor_evaluation.py 扩展

def evaluate_etf_factors(self, factor_names: List[str], period: str = "2019-2026"):
    """评估ETF因子的IC/IR/分层收益"""
    for factor in factor_names:
        ic_series = self._calc_ic(factor)
        ir = ic_series.mean() / ic_series.std()
        # 分层回测：按因子值分5档，对比各档收益
        quintile_returns = self._calc_quintile_returns(factor)
```

---

### Phase 3：风险控制体系（1-2周）

**目标**：将最大回撤控制在 -15% 以内，收益提升与风险控制并重。

#### 3.1 波动率目标控制

```python
# core/risk_manager.py（新建）

class VolatilityTargeting:
    """波动率目标控制：动态调仓位使实际波动率跟踪目标"""

    def __init__(self, target_vol: float = 0.20, lookback: int = 60):
        self.target_vol = target_vol  # 目标年化波动率20%
        self.lookback = lookback

    def calculate_position_scale(self, returns: pd.Series) -> float:
        """计算仓位缩放因子"""
        recent_vol = returns.rolling(self.lookback).std().iloc[-1] * np.sqrt(252)
        if recent_vol > 0:
            scale = self.target_vol / recent_vol
            return max(0.0, min(1.5, scale))  # 允许最高1.5倍杠杆
        return 1.0
```

#### 3.2 最大回撤约束

```python
# core/risk_manager.py

class DrawdownControl:
    """最大回撤约束：超过阈值时降仓或转防御"""

    def __init__(self, max_drawdown: float = 0.15, defense_etf: str = "511260"):
        self.max_drawdown = max_drawdown
        self.defense_etf = defense_etf  # 国债ETF

    def check_drawdown(self, equity_curve: pd.Series) -> dict:
        """检查回撤并返回调仓建议"""
        peak = equity_curve.cummax()
        drawdown = (equity_curve - peak) / peak
        current_dd = drawdown.iloc[-1]

        if current_dd < -self.max_drawdown:
            return {"action": "switch_to_defense", "defense_ratio": 0.7}
        elif current_dd < -self.max_drawdown * 0.7:
            return {"action": "reduce_position", "scale": 0.5}
        return {"action": "normal", "scale": 1.0}
```

#### 3.3 市场状态分层风控

```python
# scanners/market_scanner.py 修改 _detect_regime

def _detect_regime(self, index_df: pd.DataFrame) -> Tuple[MarketRegime, float]:
    """市场状态检测 + 建议仓位"""
    # 原有 MA20/MA60 判断
    regime = self._ma_regime(index_df)

    # 新增：波动率状态
    vol = index_df["close"].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252)
    vol_regime = "high" if vol > 0.25 else "normal"

    # 仓位建议
    position_suggestion = {
        MarketRegime.BULL: 1.0,
        MarketRegime.NEUTRAL: 0.5,
        MarketRegime.BEAR: 0.0,  # 清仓转债券ETF
    }[regime]

    if vol_regime == "high":
        position_suggestion *= 0.7  # 高波动降仓

    return regime, position_suggestion
```

#### 3.4 止损系统优化

```python
# core/risk_manager.py

class StopLossSystem:
    """ETF级 + 组合级止损"""

    def check_etf_stop(self, symbol: str, df: pd.DataFrame, entry_price: float) -> bool:
        """ETF级止损：跌破20日均线-3%确认"""
        ma20 = df["close"].rolling(20).mean().iloc[-1]
        current = df["close"].iloc[-1]
        return current < ma20 * 0.97  # 跌破MA20的3%

    def check_portfolio_stop(self, daily_return: float, threshold: float = -0.03) -> bool:
        """组合级止损：单日回撤>3%降仓"""
        return daily_return < threshold
```

#### 3.5 凯利公式仓位

```python
# core/risk_manager.py

class KellyPositionSizer:
    """凯利公式仓位：用历史胜率和盈亏比计算最优仓位"""

    def calculate_kelly_weight(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """凯利公式：f* = (p*b - q) / b"""
        if avg_loss == 0 or avg_win == 0:
            return 0.0
        b = avg_win / avg_loss  # 盈亏比
        p = win_rate             # 胜率
        q = 1 - p
        kelly = (p * b - q) / b
        # 使用半凯利（更保守）
        return max(0.0, min(0.25, kelly * 0.5))  # 单只最高25%
```

#### 3.6 相关性过滤

```python
# core/risk_manager.py

class CorrelationFilter:
    """相关性过滤：去除高相关ETF，保留动量最强"""

    def filter_by_correlation(self, candidates: pd.DataFrame,
                               returns_matrix: pd.DataFrame,
                               threshold: float = 0.8) -> pd.DataFrame:
        """相关性>0.8时只保留动量最强的一只"""
        corr = returns_matrix.corr()
        to_drop = set()
        for i in range(len(corr)):
            for j in range(i + 1, len(corr)):
                if abs(corr.iloc[i, j]) > threshold:
                    # 保留动量得分更高的
                    sym_i = corr.index[i]
                    sym_j = corr.columns[j]
                    score_i = candidates.loc[candidates["symbol"] == sym_i, "ai_rank_score"].values[0]
                    score_j = candidates.loc[candidates["symbol"] == sym_j, "ai_rank_score"].values[0]
                    if score_i < score_j:
                        to_drop.add(sym_i)
                    else:
                        to_drop.add(sym_j)
        return candidates[~candidates["symbol"].isin(to_drop)]
```

---

### Phase 4：回测引擎严谨性（1周）

**目标**：确保回测结果具有统计显著性和泛化能力。

#### 4.1 全周期 Walk-Forward 验证

```python
# backtest/aggressive_etf_walkforward.py 修改

# 当前：2024+ 或 2021+
# 目标：2019-2026 覆盖完整牛熊周期
TEST_PERIODS = [
    ("2019-01-01", "2019-12-31"),  # 牛市
    ("2020-01-01", "2020-12-31"),  # 疫情
    ("2021-01-01", "2021-12-31"),  # 核心资产
    ("2022-01-01", "2022-12-31"),  # 回调
    ("2023-01-01", "2023-12-31"),  # 震荡
    ("2024-01-01", "2026-05-29"),  # 反弹
]
```

**验证指标**：
- 每个子周期的年化收益、夏普、最大回撤
- 各周期收益的离散度（标准差）
- 最差周期的表现是否仍可接受

#### 4.2 参数敏感性分析

```python
# scripts/param_sensitivity.py（新建）

def parameter_perturbation_test(self, base_params: dict, perturbation: float = 0.1):
    """参数微变测试：关键参数±10%时收益波动"""
    results = []
    for key, value in base_params.items():
        for delta in [-perturbation, 0, +perturbation]:
            test_params = base_params.copy()
            test_params[key] = value * (1 + delta)
            result = self._run_backtest(test_params)
            results.append({
                "param": key, "delta": delta,
                "annual_return": result["annual_return"],
                "sharpe": result["sharpe"],
                "max_drawdown": result["max_drawdown"],
            })
    # 判断标准：参数±10%时，年化收益变化<20%
    return results
```

#### 4.3 交易成本真实化

```python
# backtest/aggressive_etf_walkforward.py 修改

class RealisticCostModel:
    """真实交易成本模型"""

    def __init__(self):
        self.commission_rate = 0.0005    # 佣金万五（单边）
        self.slippage_base = 0.0005      # 基础滑点5bp
        self.impact_coeff = 0.1           # 冲击成本系数

    def calculate_total_cost(self, trade_amount: float, avg_daily_volume: float) -> float:
        """计算总交易成本"""
        # 佣金
        commission = trade_amount * self.commission_rate
        # 基础滑点
        slippage = trade_amount * self.slippage_base
        # 冲击成本（与交易量占比成正比）
        volume_ratio = trade_amount / (avg_daily_volume + 1)
        impact = trade_amount * self.impact_coeff * volume_ratio
        return commission + slippage + impact
```

#### 4.4 基准对比

| 基准 | 说明 |
|------|------|
| 等权 ETF 组合 | 每月等权持有全部候选ETF |
| 沪深300指数 | 宽基基准 |
| 中证500指数 | 中盘基准 |
| 60/40股债组合 | 传统配置基准 |

**计算指标**：超额收益、信息比率、跟踪误差、胜率

#### 4.5 显著性检验

```python
# scripts/significance_test.py（新建）

def deflated_sharpe_ratio(self, sharpe: float, n_trials: int,
                           sample_length: int, skew: float, kurtosis: float) -> float:
    """Deflated Sharpe Ratio：纠正多重检验偏差"""
    # 参考 Bailey & Lopez de Prado (2014)
    # 考虑试错次数、样本长度、非正态性
    from scipy.stats import norm
    # ...
    return dsr

def whites_reality_check(self, strategy_returns: pd.Series,
                          benchmark_returns: pd.Series) -> dict:
    """White's Reality Check：检验策略相对基准的显著性"""
    # Bootstrap 重采样检验
    # ...
    return {"p_value": p_value, "significant": p_value < 0.05}
```

---

### Phase 5：AI 模型简化与优化（2-3周）

**目标**：简化模型栈，适配 ETF 场景，提升可维护性。

#### 5.1 简化模型栈

```
当前（4套模型）：
  DL(DeepLearningSignalEngine) + SEQ(TemporalEnsembleSignalEngine)
  + RL(RiskAwareReinforcementAllocator) + PPO(PPOAllocationEngine)

优化后（2套模型 + 风险预算）：
  DL(DeepLearningSignalEngine)    # 主力信号源
  + SEQ(TemporalEnsembleSignalEngine)  # 辅助验证
  + RiskBudgetAllocator             # 风险预算分配（替代RL/PPO）
```

```python
# core/risk_budget.py（新建，替代RL/PPO）

class RiskBudgetAllocator:
    """风险预算分配：基于波动率倒数加权"""

    def allocate(self, candidates: pd.DataFrame) -> Dict[str, float]:
        """根据各ETF的波动率倒数分配权重"""
        weights = {}
        for _, row in candidates.iterrows():
            vol = row.get("volatility", 0.2)
            weight = 1.0 / vol if vol > 0 else 0
            weights[row["symbol"]] = weight
        # 归一化 + 仓位上限
        total = sum(weights.values())
        weights = {k: min(0.3, v / total) for k, v in weights.items()}
        return weights
```

#### 5.2 ETF 专用特征集

```python
# models/deep_learning.py 修改 build_feature_label_dataset

def build_etf_feature_dataset(self, stock_data_map: Dict, history_data: Dict) -> Tuple:
    """构建ETF专用特征集"""
    features = []
    labels = []
    for symbol, df in stock_data_map.items():
        # 基础价量特征
        feat = extract_latest_features(df)
        # ETF专用特征
        feat["theme_momentum"] = history_data["theme"].get(symbol, {}).get("momentum", 0)
        feat["nav_discount"] = history_data["nav"].get(symbol, {}).get("discount", 0)
        feat["capital_flow"] = history_data["flow"].get(symbol, {}).get("net_inflow", 0)
        feat["scale_change"] = history_data["nav"].get(symbol, {}).get("scale_change", 0)
        # 标签：未来20日收益率是否跑赢等权基准
        future_return = df["close"].shift(-20).pct_change(20).iloc[-1]
        benchmark_return = ...  # 等权基准收益
        labels.append(1 if future_return > benchmark_return else 0)
        features.append(feat)
    return np.array(features), np.array(labels)
```

#### 5.3 动态集成权重

```python
# scanners/market_scanner.py 修改

def _calculate_dynamic_weights(self, recent_ic: Dict[str, float]) -> Tuple[float, float]:
    """根据近期IC值动态分配DL/SEQ权重"""
    ic_dl = recent_ic.get("dl", 0.02)
    ic_seq = recent_ic.get("seq", 0.02)
    total_ic = abs(ic_dl) + abs(ic_seq)
    if total_ic > 0:
        w_dl = abs(ic_dl) / total_ic
        w_seq = abs(ic_seq) / total_ic
    else:
        w_dl, w_seq = 0.5, 0.5
    return w_dl, w_seq
```

#### 5.4 不确定性量化

```python
# models/deep_learning.py 新增

class MCDropoutUncertainty:
    """MC Dropout：多次前向传播计算预测方差"""

    def __init__(self, model, n_forward: int = 50):
        self.model = model
        self.n_forward = n_forward

    def predict_with_uncertainty(self, x) -> Tuple[float, float]:
        """返回 (均值, 不确定性)"""
        predictions = []
        for _ in range(self.n_forward):
            pred = self.model.predict_proba(x, dropout=True)
            predictions.append(pred)
        mean_pred = np.mean(predictions)
        uncertainty = np.std(predictions)
        return mean_pred, uncertainty
```

#### 5.5 模型重训练频率

| 当前 | 优化后 |
|------|--------|
| 月度重训练 | 季度重训练 + 滚动窗口扩展 |
| 全量重训练 | 增量训练（新数据追加） |

#### 5.6 LLM 定位调整

| 模块 | 当前定位 | 优化后定位 |
|------|----------|------------|
| `FactorExplainer` | 因子解释 | 保留，向用户解释选股逻辑 |
| `NewsSentimentAnalyzer` | 新闻情绪（策略核心） | 降级为辅助：每日情绪快照存储（供回测） |
| `ReportAnalyzer` | 研报分析 | 降级为辅助：每日研报摘要 |
| `StrategyReporter` | 策略报告 | 保留，自动生成周报/月报 |

**核心原则**：LLM 不直接参与信号生成，避免幻觉影响交易决策。

---

## 四、实施路线图与时间表

```
Week 1-2:  Phase 1 数据基建
           ├── 历史资金流数据库构建
           ├── 新闻情绪历史化
           ├── 板块轮动历史
           ├── ETF宇宙扩展到300+
           └── NAV历史全覆盖

Week 3-5:  Phase 2 因子重构
           ├── ETF专用动量因子
           ├── 主题轮动因子强化
           ├── 波动率Regime自适应
           ├── VCP替代方案启用
           ├── 跨市场宏观因子
           ├── 流动性因子
           └── 因子IC/IR评估

Week 6-7:  Phase 3 风控体系
           ├── 波动率目标控制
           ├── 最大回撤约束
           ├── 市场状态分层风控
           ├── 止损系统优化
           ├── 凯利公式仓位
           └── 相关性过滤

Week 8:    Phase 4 回测验证
           ├── 2019-2026全周期walk-forward
           ├── 参数敏感性分析
           ├── 交易成本真实化
           ├── 基准对比
           └── 显著性检验

Week 9-11: Phase 5 模型优化
           ├── 简化模型栈
           ├── ETF专用特征集
           ├── 动态集成权重
           ├── MC Dropout不确定性
           ├── 重训练频率调整
           └── LLM定位调整
```

---

## 五、验证方法与成功标准

### 5.1 各 Phase 验收标准

| Phase | 验收指标 | 成功标准 |
|-------|----------|----------|
| Phase 1 | 8层信号回测覆盖率 | Layer3/4/5 在回测中有历史数据，非空转 |
| Phase 1 | ETF宇宙规模 | ≥ 300只，NAV覆盖率100% |
| Phase 2 | 新因子IC/IR | ETF动量因子 IR > 0.3，主题轮动 IR > 0.4 |
| Phase 2 | VCP替代效果 | `pv_volatility_contraction` IR > 0（原VCP ICIR≈0） |
| Phase 3 | 最大回撤控制 | 全周期回测最大回撤 < -15% |
| Phase 3 | 波动率跟踪误差 | 实际年化波动率在目标20%±5%以内 |
| Phase 4 | 全周期夏普 | 2019-2026 OOS 夏普 > 1.0 |
| Phase 4 | 最差周期表现 | 任一子周期年化 > 0（无亏损年） |
| Phase 4 | 参数敏感性 | 参数±10%时收益变化 < 20% |
| Phase 4 | DSR显著性 | Deflated Sharpe Ratio > 0（非数据挖掘偏误） |
| Phase 5 | 模型简化效果 | 简化后OOS夏普不低于原4模型栈 |
| Phase 5 | 不确定性校准 | MC Dropout预测方差与实际误差相关性 > 0.5 |

### 5.2 最终成功标准

| 指标 | 目标 | 当前 |
|------|------|------|
| 年化收益 | > 30% | 长窗口17.24% / 短窗口109.92% |
| 夏普比率 | > 1.5 | 0.45-2.20（不稳定） |
| 最大回撤 | < -15% | -24.54% ~ -30.09% |
| Calmar 比率 | > 1.0 | 未达标 |
| 月胜率 | > 60% | 未统计 |
| 信息比率 | > 0.5 | 未统计 |
| 全周期稳定性 | 各子周期年化 > 0 | 未验证 |

---

## 六、风险与依赖项

### 6.1 技术风险

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| AkShare 接口限流 | 历史数据构建缓慢 | 增加请求间隔、批量下载、多源降级 |
| LLM API 成本 | 每日情绪快照成本高 | 仅对候选ETF（非全市场）调用LLM |
| 历史数据缺失 | 部分数据源无早期历史 | 标注缺失日期，回测时自动降级 |
| 参数过拟合 | 新因子可能过拟合 | 严格样本外验证 + 参数敏感性测试 |

### 6.2 数据依赖

| 数据 | 来源 | 可获取性 |
|------|------|----------|
| 概念资金流 | AkShare | ✅ 免费，有历史 |
| 龙虎榜 | AkShare | ✅ 免费，有历史 |
| 北向资金 | AkShare | ✅ 免费，有历史 |
| 行业板块行情 | AkShare | ✅ 免费，有历史 |
| ETF NAV | AkShare | ✅ 免费，有历史 |
| 新闻情绪 | LLM 生成 | ⚠️ 需每日累积，无历史回填 |

### 6.3 关键假设

> **核心假设**：补全数据层 + 适配ETF因子 + 严格风控，比堆叠复杂模型更能提升稳健收益。

**验证方式**：
1. 对比 Phase 1 完成前后（8层全部生效 vs 5层生效）的 OOS 夏普
2. 对比 Phase 3 完成前后（有风控 vs 无风控）的最大回撤
3. 对比 Phase 5 完成前后（简化模型 vs 4模型栈）的 OOS 夏普和可维护性

---

## 七、附录

### A. 新建文件清单

| 文件 | Phase | 说明 |
|------|-------|------|
| `scripts/build_history_capital_flow.py` | 1 | 历史资金流数据库构建 |
| `scripts/build_sentiment_history.py` | 1 | 新闻情绪历史化 |
| `scripts/build_sector_history.py` | 1 | 板块轮动历史 |
| `core/history_data_loader.py` | 1 | 历史数据加载接口 |
| `core/risk_manager.py` | 3 | 风险管理器（波动率目标/回撤约束/止损/凯利/相关性） |
| `core/risk_budget.py` | 5 | 风险预算分配器（替代RL/PPO） |
| `scripts/param_sensitivity.py` | 4 | 参数敏感性分析 |
| `scripts/significance_test.py` | 4 | 显著性检验 |

### B. 修改文件清单

| 文件 | Phase | 修改内容 |
|------|-------|----------|
| `layers/layer6_price_vol.py` | 2 | 新增ETF专用动量/波动率regime/流动性因子 |
| `layers/layer3_sector.py` | 2 | 新增主题轮动因子（动量+资金流+广度） |
| `layers/layer1_macro.py` | 2 | 新增跨市场宏观因子 |
| `layers/layer4_capital.py` | 1 | 回测模式从历史库读取资金流 |
| `layers/layer5_sentiment.py` | 1 | 回测模式从历史库读取情绪 |
| `scanners/market_scanner.py` | 3,5 | 市场状态分层风控+动态集成权重 |
| `backtest/aggressive_etf_walkforward.py` | 4 | 全周期验证+真实成本模型 |
| `models/deep_learning.py` | 5 | ETF特征集+MC Dropout不确定性 |
| `scripts/factor_evaluation.py` | 2 | 扩展ETF因子IC/IR评估 |
| `scripts/cache_etf_nav_history.py` | 1 | 扩展覆盖到300+ETF |
| `scripts/download_all_etf.py` | 1 | 下载全市场ETF列表 |
| `config.yaml` | 3 | 新增风控参数配置 |

### C. 优化前后对比

| 维度 | 当前 | 优化后 |
|------|------|--------|
| 信号层回测覆盖率 | 5/8层生效 | 8/8层全部生效 |
| ETF宇宙 | ~100只 | 300+只 |
| NAV覆盖率 | 40只(27/43) | 100% |
| 核心因子 | VCP(ICIR≈0) | 连续波动率收缩度+ETF专用动量 |
| 风控体系 | 无 | 波动率目标+回撤约束+止损+凯利+相关性 |
| 回测周期 | 2024+(短窗口) | 2019-2026全周期 |
| 模型栈 | 4套(DL+SEQ+RL+PPO) | 2套(DL+SEQ)+风险预算 |
| 不确定性 | abs(dl-seq) | MC Dropout |
| 目标 | 每月30%(不可行) | 年化>30%+Sharpe>1.5+回撤<-15% |
| LLM定位 | 策略核心 | 辅助分析 |

---

## 八、总结

本优化计划的核心思路是 **"先补基础再提性能"**：

1. **Phase 1-2（数据+因子）**：补全数据基建，让8层信号全部在回测中生效，这是当前最大的短板——再复杂的模型也无法弥补数据缺失
2. **Phase 3（风控）**：在收益提升的同时控制风险，将回撤从-30%降到-15%
3. **Phase 4（回测）**：用全周期验证确保策略稳健，避免短窗口过拟合
4. **Phase 5（模型）**：简化模型栈，降低维护复杂度，提升可解释性

**预期最终成果**：年化收益>30%、夏普>1.5、最大回撤<-15%、Calmar>1.0，在2019-2026全周期验证下稳健达标。

---

*文档生成时间：2026-07-04*
*作者：AStockQuant 优化分析（Sequential Thinking 8步深度分析）*
