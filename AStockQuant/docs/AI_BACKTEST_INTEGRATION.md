# AI 引擎接入 walkforward 回测

## 概述

本模块把 `MarketScanner` 里的 `DeepLearningSignalEngine`（MLP）和 `TemporalEnsembleSignalEngine`（LSTM+Transformer）接入 `aggressive_etf_walkforward.py`，让 AI 选股模型**首次在历史回测中被验证**。

AI 概率 `ai_prob` 作为打分特征注入 `score_candidates`，`ai_weight` 进入 grid search 网格自动寻优，与无 AI 的 baseline 直接对比。

## 架构

```
walkforward fold k
├─ bridge.fit(stock_data_map, train_end)     # 只看 ≤train_end 训练 DL+SEQ
├─ bridge.predict_proba(map, [train_start..test_end])  # 产出 ai_prob
├─ inject_ai_prob(panel, ai_prob_df)         # ai_prob 列注入 fold panel 副本
├─ grid search (configs × ai_weight)         # 各 config 跑 train backtest
│     └─ score_candidates 读 df["ai_prob"]   # ai_weight>0 时参与打分
└─ best config 跑 test backtest               # OOS 验证
```

### 文件清单

| 文件 | 角色 |
|------|------|
| `backtest/ai_signal_bridge.py` | **新建**：`AISignalBridge` 类（fit/predict_proba）+ `inject_ai_prob` 辅助函数 |
| `backtest/aggressive_etf_walkforward.py` | 改造：`StrategyConfig.ai_weight`、`build_configs(ai_weights)`、`score_candidates` 注入行、`run_walk_forward` bridge 循环、`--use-ai` CLI |
| `models/deep_learning.py` | 不动：复用现有引擎类 + 特征构造函数 |

## 时序对齐（防 look-ahead）

**核心原则**：AI 信号严格遵守 walk-forward 时序，测试期数据绝不进入训练。

1. **训练**：`bridge.fit(stock_data_map, train_end)` → 每个 code 的 df 先 `df[df['date'] <= train_end]` 截断，再调 `build_feature_label_dataset` / `build_sequence_dataset`。这两个函数内部 `df.iloc[:t+1]` 已保证 t 时刻只看历史。
2. **预测**：`bridge.predict_proba(..., dates)` → 对 date d，取 `df[df['date'] <= d]`（含当日 close），符合"T 日 close 出信号，T+1 open 执行"。
3. **fold 隔离**：fold k 引擎只见 ≤train_end_k 数据；fold k+1 的 train_end > train_end_k，引擎重训，不会用到 fold k 测试期数据训练（rolling forward）。
4. **fold 级 ai_prob 复用**：同一 fold 内所有 config 共享同一份 ai_prob（引擎训练与 config 无关），grid search 只反复跑 backtest，不反复训练引擎。

## CLI 用法

### 基线（无 AI，与改造前 100% 一致）
```bash
python backtest/aggressive_etf_walkforward.py --grid quick --train-months 24 --test-months 6
```

### 启用 AI
```bash
python backtest/aggressive_etf_walkforward.py \
    --use-ai \
    --grid quick \
    --train-months 24 --test-months 6 \
    --ai-weights 0,0.1,0.2,0.3 \
    --dl-epochs 20 --seq-epochs 12 --seq-mode ensemble
```

### 参数说明

| 参数 | 默认 | 说明 |
|------|------|------|
| `--use-ai` | off | 启用 DL/SEQ AI 信号桥接 |
| `--ai-weights` | `0,0.1,0.2,0.3` | ai_weight 网格（逗号分隔），自动含 0.0 baseline |
| `--dl-epochs` | 20 | DL 引擎训练轮数 |
| `--seq-epochs` | 12 | SEQ 引擎训练轮数 |
| `--seq-mode` | ensemble | `lstm` / `transformer` / `ensemble` |

## grid search 语义

`--use-ai --ai-weights 0,0.1,0.2,0.3` 时，每个 (profile, top_k, rebalance, stop, risk_off) 组合生成 4 个 config（ai_weight=0/0.1/0.2/0.3）。`search_score` 自动比较：
- `ai_weight=0` 的 config = 无 AI baseline
- `ai_weight>0` 的 config = 有 AI

**判读**：若 AI 有效，best config 的 ai_weight 应 >0；若无效，best config 的 ai_weight 应=0。

输出 `grid.csv` 含每个 config 的 `ai_weight` 和 `search_score` 列，可对比同 profile 下不同 ai_weight 的表现。

## 注入机制

`score_candidates` 在公共修正项之后加一行（零侵入，签名不变）：
```python
if cfg.ai_weight > 0 and "ai_prob" in df.columns:
    score = score + cfg.ai_weight * (df["ai_prob"].fillna(0.5) - 0.5)
```
减 0.5 保持中性（ai_prob=0.5 时贡献为 0），避免整体抬高所有分数。

## 与 MarketScanner 的关系

| | MarketScanner.scan() | walkforward (本模块) |
|---|---|---|
| 引擎 | DL + SEQ + RL + PPO | 仅 DL + SEQ |
| 训练时机 | 每次扫描从零重训 | 每 fold 重训（walk-forward） |
| 预测用途 | 实时选股 + RL 组合权重 | 历史 ai_prob 注入打分 |
| 持久化 | 无（内存） | 无（每 fold 重训） |
| 验证方式 | 无历史验证 | **walk-forward OOS** |

两者共享 `models/deep_learning.py` 的引擎类，但调用方式独立。本任务不重构 `MarketScanner.scan()`。

## 性能预估

- panel ~180 ETF × ~1500 日 ≈ 270k 行。fold 内复制一份注入 ai_prob ≈ 2-4MB。
- 引擎训练：DL ~30s/fold，SEQ ~3-5min/fold（LSTM）。6 folds → 总训练 ~30min。
- 预测：6 folds × ~150 交易日 × 180 ETF × (DL+SEQ) ≈ 162k 次预测 → ~30min。
- 缓存：fold 内 (code, date) 特征向量缓存，避免重复 extract。

**若太慢**：减小 `--max-etfs`、降低 `--dl-epochs`/`--seq-epochs`、用 `--seq-mode lstm`（比 ensemble 快）。

## 限制与边界

- **不接 PPO/RL 引擎**：PPO 是组合权重分配，不是选股打分，与 `score_candidates` 语义不匹配。仅接 DL/SEQ（概率型，天然适配打分）。
- **不给引擎加 save/load**：walkforward 每 fold 重训，不需要持久化。
- **不改其他回测脚本**（etf_screener_backtest_v4 / final_strategy_v8 / turtle）：只接 walkforward 这一个最严谨的入口。
- **训练阶段 ai_prob 是 in-sample**：grid search 的 train 期 ai_prob 由见过 ≤train_end 数据的引擎产出（in-sample 偏差），但 test 期是 OOS。这是标准 walk-forward 特性，test 期会暴露真实效果。

## 验证

1. **baseline 回归**：不带 `--use-ai` 跑，确认结果与改造前一致（ai_weight=0，注入行不触发）。
2. **集成测试**：带 `--use-ai` 跑小范围，确认 `[AI] fold k: 训练引擎` / `注入 ai_prob N 行` 日志出现、`grid.csv` 含 `ai_weight` 列、不报错。
3. **对比**：相同参数下，比较 `--use-ai` 与 baseline 的 `stitched_metrics`，看 AI 是否提升 CAGR/Sharpe。
