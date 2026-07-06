# -*- coding: utf-8 -*-
"""
strategy_reporter.py — 策略报告生成模块

使用 LLM 根据回测结果生成 Markdown 格式的策略分析报告。
用于回测后的自动化报告生成。

输入: 回测结果（folds, equity, benchmarks, regime_perf, selected_factors）
输出: Markdown 格式的策略分析报告
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from AStockQuant.core.llm_client import llm


SYSTEM_PROMPT = """你是一位资深的量化投资策略分析师，擅长从回测结果中提取有价值的投资洞见。
你的报告需要：
1. 客观分析策略表现，不回避问题
2. 对比策略与基准的差异，找出alpha来源
3. 分析不同市场状态下的表现差异
4. 提出具体的改进建议
5. 用专业但易懂的中文撰写
请直接输出Markdown格式文本。"""


class StrategyReporter:
    """策略报告生成器"""

    def __init__(self):
        self._enabled = llm.enabled and llm.config.get("tasks", {}).get("strategy_report", True)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def generate_report(
        self,
        strategy_stats: Dict,
        benchmark_stats: List[Dict],
        regime_performance: Optional[pd.DataFrame] = None,
        selected_factors: Optional[pd.DataFrame] = None,
        folds: Optional[pd.DataFrame] = None,
        config: Optional[Dict] = None,
    ) -> str:
        """
        生成完整的策略分析报告

        Args:
            strategy_stats: 策略统计 {"total_return": 0.02, "max_drawdown": -0.19, ...}
            benchmark_stats: 基准对比 [{"benchmark": "buy_hold_510300", "total_return": 0.004, ...}]
            regime_performance: 分regime表现 DataFrame
            selected_factors: 选中因子 DataFrame
            folds: walk-forward 分折结果
            config: 策略配置

        Returns:
            Markdown 格式报告文本
        """
        if not self._enabled:
            return self._generate_fallback_report(
                strategy_stats, benchmark_stats, regime_performance, selected_factors, folds
            )

        data_summary = self._prepare_data_summary(
            strategy_stats, benchmark_stats, regime_performance, selected_factors, folds, config
        )

        prompt = f"""请根据以下量化ETF轮动策略的回测结果，生成一份详细的策略分析报告。

回测数据:
{data_summary}

报告需包含以下部分：
1. ## 策略概览 — 总体表现概述
2. ## 基准对比 — 与各基准的差异分析，找出alpha来源
3. ## Regime分析 — 不同市场状态下的表现差异
4. ## 因子分析 — 核心因子的贡献和问题
5. ## Walk-Forward稳定性 — 各折的稳定性和退化趋势
6. ## 风险分析 — 回撤特征和风险点
7. ## 改进建议 — 具体可执行的优化方向

要求：
- 数据要具体，引用实际数字
- 不回避策略的问题（如收益低、回撤大等）
- 改进建议要具体可执行
- 全文用中文撰写，Markdown格式"""

        report = llm.chat(
            prompt,
            system=SYSTEM_PROMPT,
            temperature=0.5,
            max_tokens=8192,
            timeout=180,
        )
        return report if report else self._generate_fallback_report(
            strategy_stats, benchmark_stats, regime_performance, selected_factors, folds
        )

    def _prepare_data_summary(
        self,
        strategy_stats: Dict,
        benchmark_stats: List[Dict],
        regime_performance: Optional[pd.DataFrame],
        selected_factors: Optional[pd.DataFrame],
        folds: Optional[pd.DataFrame],
        config: Optional[Dict],
    ) -> str:
        """准备给LLM的数据摘要"""
        parts = []

        parts.append("### 策略统计")
        for k, v in strategy_stats.items():
            if isinstance(v, float):
                parts.append(f"- {k}: {v:.4f}")
            else:
                parts.append(f"- {k}: {v}")

        parts.append("\n### 基准对比")
        for b in benchmark_stats:
            parts.append(f"- {b.get('benchmark', '?')}: "
                         f"收益={b.get('total_return', 0):.4f}, "
                         f"回撤={b.get('max_drawdown', 0):.4f}, "
                         f"夏普={b.get('sharpe', 0):.4f}")

        if regime_performance is not None and not regime_performance.empty:
            parts.append("\n### 分Regime表现")
            for _, row in regime_performance.iterrows():
                parts.append(f"- {row['regime_label']}: n={int(row.get('n', 0))}, "
                           f"平均收益={row.get('mean_period_return', 0):.4f}, "
                           f"胜率={row.get('win_rate', 0):.2f}")

        if selected_factors is not None and not selected_factors.empty:
            parts.append("\n### 核心因子")
            for _, row in selected_factors.iterrows():
                parts.append(f"- {row['factor']}: IC={row.get('ic_mean', 0):.4f}, "
                           f"ICIR={row.get('icir', 0):.3f}, "
                           f"LS={row.get('long_short_ret', 0):.4f}")

        if folds is not None and not folds.empty:
            parts.append("\n### Walk-Forward分折结果")
            for _, row in folds.iterrows():
                parts.append(f"- Fold {row.get('fold', '?')}: "
                           f"收益={row.get('fold_return_after_cost', 0):.4f}, "
                           f"换手={row.get('avg_turnover', 0):.2f}, "
                           f"成本={row.get('cost_paid', 0):.0f}")

        if config:
            parts.append("\n### 策略配置")
            for k, v in config.items():
                parts.append(f"- {k}: {v}")

        return "\n".join(parts)

    def _generate_fallback_report(
        self,
        strategy_stats: Dict,
        benchmark_stats: List[Dict],
        regime_performance: Optional[pd.DataFrame],
        selected_factors: Optional[pd.DataFrame],
        folds: Optional[pd.DataFrame],
    ) -> str:
        """LLM不可用时的降级报告（纯模板）"""
        lines = ["# 策略回测报告（自动生成）", ""]

        lines.append("## 策略概览")
        for k, v in strategy_stats.items():
            if isinstance(v, float):
                lines.append(f"- **{k}**: {v:.4f}")
            else:
                lines.append(f"- **{k}**: {v}")
        lines.append("")

        lines.append("## 基准对比")
        lines.append("| 基准 | 总收益 | 最大回撤 | 夏普 |")
        lines.append("|------|--------|----------|------|")
        for b in benchmark_stats:
            lines.append(f"| {b.get('benchmark', '?')} | "
                        f"{b.get('total_return', 0):.4f} | "
                        f"{b.get('max_drawdown', 0):.4f} | "
                        f"{b.get('sharpe', 0):.4f} |")
        lines.append("")

        if regime_performance is not None and not regime_performance.empty:
            lines.append("## 分Regime表现")
            lines.append("| Regime | 样本数 | 平均收益 | 胜率 |")
            lines.append("|--------|--------|----------|------|")
            for _, row in regime_performance.iterrows():
                lines.append(f"| {row['regime_label']} | {int(row.get('n', 0))} | "
                            f"{row.get('mean_period_return', 0):.4f} | "
                            f"{row.get('win_rate', 0):.2f} |")
            lines.append("")

        if selected_factors is not None and not selected_factors.empty:
            lines.append("## 核心因子")
            lines.append("| 因子 | IC | ICIR | 多空收益 |")
            lines.append("|------|-----|------|---------|")
            for _, row in selected_factors.iterrows():
                lines.append(f"| {row['factor']} | {row.get('ic_mean', 0):.4f} | "
                            f"{row.get('icir', 0):.3f} | "
                            f"{row.get('long_short_ret', 0):.4f} |")
            lines.append("")

        lines.append("> 注: LLM不可用，以上为纯数据模板报告。设置 ARK_API_KEY 环境变量可启用AI分析。")
        return "\n".join(lines)

    def save_report(self, report: str, path: str) -> None:
        """保存报告到文件"""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(report, encoding="utf-8")
        print(f"[StrategyReporter] 报告已保存: {p}")
