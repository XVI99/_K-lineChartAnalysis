# -*- coding: utf-8 -*-
"""
auto_evaluator.py — AStockQuant 自动化回测评估器 [因果诊断版]

功能：
- 读取 reports/ 目录下的最新 JSON 回测结果
- 对比新旧回测的夏普比率，判断 Alpha 是否提升
- 提取亏损最大的 N 笔交易，支持因果诊断
- 输出评估结论（pass / fail / neutral）
- 提供 is_alpha_improved() 接口供外部调用

【因果诊断协议】
当评估结果为 fail 时：
1. 调用 get_worst_trades() 提取失败案例
2. 分析亏损共性（板块趋势、换手率、形态出现位置）
3. 提出量化假设并编码到对应 Layer

使用方式：
    from core.auto_evaluator import AutoEvaluator
    evaluator = AutoEvaluator()
    result = evaluator.evaluate()
    worst = evaluator.get_worst_trades(n=10)
    print(result)
    print(worst)
"""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple, List


# ==================== 数据结构 ====================

@dataclass
class EvalResult:
    """评估结果 dataclass"""
    status: str          # "pass" | "fail" | "neutral"
    new_sharpe: Optional[float]
    old_sharpe: Optional[float]
    improvement: Optional[float]   # 相对提升比例
    verdict: str         # 结论描述
    report_path: Optional[str]      # 本次评估所基于的报告路径


@dataclass
class TradeRecord:
    """单笔交易记录"""
    date: Optional[str] = None
    code: Optional[str] = None
    symbol: Optional[str] = None    # 股票代码/名称
    pnl: float = 0.0               # 盈亏金额
    return_pct: float = 0.0        # 收益率（%）
    volume: Optional[float] = None  # 成交量
    turnover_rate: Optional[float] = None  # 换手率
    sector: Optional[str] = None   # 板块
    pattern: Optional[str] = None  # 触发形态
    holding_days: int = 0          # 持仓天数
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    # 扩展字段（兼容不同报告格式）
    open_price: Optional[float] = None
    close_price: Optional[float] = None
    metadata: dict = field(default_factory=dict)

    @property
    def loss(self) -> float:
        """亏损金额（正数表示亏损）"""
        return -self.pnl if self.pnl < 0 else 0.0


@dataclass
class DiagnosticReport:
    """因果诊断报告"""
    worst_trades: List[TradeRecord]
    total_loss: float
    avg_loss: float
    loss_by_pattern: dict  # 各形态的亏损总额
    loss_by_sector: dict   # 各板块的亏损总额
    common_patterns: List[str]  # 共同出现的形态
    sector_correlation: str    # 板块共性描述
    volume_profile: str         # 成交量特征描述
    hypothesis: str             # 量化假设
    suggested_fix: str          # 建议修复方案


# ==================== 评估器主体 ====================

class AutoEvaluator:
    """自动化回测评估器 [因果诊断版]"""

    REPORTS_DIR = str(Path(__file__).resolve().parent.parent / "reports")
    MIN_IMPROVEMENT_THRESHOLD = 0.05  # 5%

    def __init__(self, reports_dir: str = None):
        self.reports_dir = reports_dir or self.REPORTS_DIR

    # ---- 报告读取 ----

    def get_latest_reports(self, n: int = 2) -> list[dict]:
        """获取最近 n 次回测报告，按修改时间倒序"""
        reports_path = Path(self.reports_dir)
        if not reports_path.exists():
            return []

        json_files = list(reports_path.glob("backtest_*.json"))
        if not json_files:
            return []

        json_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        results = []
        for f in json_files[:n]:
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    results.append(json.load(fp))
            except (json.JSONDecodeError, IOError):
                continue
        return results

    def _get_latest_report_path(self) -> Optional[str]:
        """获取最新报告的文件路径"""
        reports_path = Path(self.reports_dir)
        json_files = list(reports_path.glob("backtest_*.json"))
        if not json_files:
            return None
        json_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        return str(json_files[0])

    # ---- 字段提取 ----

    def extract_sharpe(self, report: dict) -> Optional[float]:
        """从报告字典中提取夏普比率（兼容多种字段名）"""
        for key in ["sharpe_ratio", "sharpe", "sharpeRatio", "annual_sharpe"]:
            if key in report and isinstance(report[key], (int, float)):
                return float(report[key])
        if "stats" in report and isinstance(report["stats"], dict):
            for key in ["sharpe_ratio", "sharpe"]:
                if key in report["stats"] and isinstance(report["stats"][key], (int, float)):
                    return float(report["stats"][key])
        return None

    def extract_trades(self, report: dict) -> List[TradeRecord]:
        """从报告中提取交易列表（兼容多种字段名）"""
        trades = []

        # 兼容：trades / results / positions / history
        raw = None
        for key in ["trades", "results", "positions", "history", "deals"]:
            if key in report and isinstance(report[key], list):
                raw = report[key]
                break

        if not raw:
            return trades

        for t in raw:
            if not isinstance(t, dict):
                continue

            # 提取公共字段（兼容命名）
            date = t.get("date") or t.get("time") or t.get("datetime") or t.get("trade_date")
            code = t.get("code") or t.get("symbol") or t.get("stock_code") or t.get("instrument")
            pnl = float(t.get("pnl") or t.get("profit") or t.get("return") or t.get("gain") or 0)
            ret = float(t.get("return_pct") or t.get("return") or t.get("ret") or t.get("profit_pct") or 0)

            volume = None
            for vk in ["volume", "vol", "turnover_volume"]:
                if vk in t:
                    try:
                        volume = float(t[vk])
                        break
                    except (ValueError, TypeError):
                        pass

            turnover = None
            for tk in ["turnover_rate", "turnover", "换手率"]:
                if tk in t:
                    try:
                        turnover = float(t[tk])
                        break
                    except (ValueError, TypeError):
                        pass

            sector = t.get("sector") or t.get("industry") or t.get("板块")
            pattern = t.get("pattern") or t.get("signal") or t.get("trigger")
            holding = int(t.get("holding_days") or t.get("days") or t.get("holding") or 0)
            entry = t.get("entry_price") or t.get("open_price") or t.get("open") or t.get("buy_price")
            exit = t.get("exit_price") or t.get("close_price") or t.get("close") or t.get("sell_price")

            if entry is not None:
                try:
                    entry = float(entry)
                except (ValueError, TypeError):
                    entry = None
            if exit is not None:
                try:
                    exit = float(exit)
                except (ValueError, TypeError):
                    exit = None

            trades.append(TradeRecord(
                date=date,
                code=code,
                symbol=code,
                pnl=pnl,
                return_pct=ret,
                volume=volume,
                turnover_rate=turnover,
                sector=sector,
                pattern=pattern,
                holding_days=holding,
                entry_price=entry,
                exit_price=exit,
                metadata=t
            ))

        return trades

    # ---- 核心评估函数 ----

    def is_alpha_improved(self, new_report: dict, old_report: dict) -> Tuple[bool, Optional[float]]:
        """对比新旧回测，判断 Alpha（夏普比率）是否提升"""
        new_sharpe = self.extract_sharpe(new_report)
        old_sharpe = self.extract_sharpe(old_report)

        if new_sharpe is None or old_sharpe is None:
            return False, None

        if old_sharpe == 0:
            return new_sharpe > 0, None if new_sharpe <= 0 else (new_sharpe - old_sharpe)

        improvement = (new_sharpe - old_sharpe) / abs(old_sharpe)
        return improvement >= self.MIN_IMPROVEMENT_THRESHOLD, improvement

    def evaluate(self) -> EvalResult:
        """执行完整评估流程，返回 EvalResult"""
        reports = self.get_latest_reports(n=2)

        if len(reports) < 2:
            if len(reports) == 1:
                sharpe = self.extract_sharpe(reports[0])
                return EvalResult(
                    status="neutral",
                    new_sharpe=sharpe,
                    old_sharpe=None,
                    improvement=None,
                    verdict=f"仅找到1份报告（sharpe={sharpe}），无法对比，标记为 neutral",
                    report_path=self._get_latest_report_path()
                )
            return EvalResult(
                status="neutral",
                new_sharpe=None,
                old_sharpe=None,
                improvement=None,
                verdict="未找到任何回测报告，请先运行回测生成报告",
                report_path=None
            )

        new_report, old_report = reports[0], reports[1]
        improved, improvement = self.is_alpha_improved(new_report, old_report)

        new_sharpe = self.extract_sharpe(new_report)
        old_sharpe = self.extract_sharpe(old_report)

        if improvement is not None:
            verdict = (
                f"Alpha {'提升 ✓' if improved else '未提升 ✗'} "
                f"（旧 Sharpe={old_sharpe:.4f} → 新 Sharpe={new_sharpe:.4f}，"
                f"变化率={improvement*100:+.2f}%）"
            )
        else:
            verdict = (
                f"无法计算提升幅度 "
                f"（旧 Sharpe={old_sharpe}，新 Sharpe={new_sharpe}）"
            )

        return EvalResult(
            status="pass" if improved else "fail",
            new_sharpe=new_sharpe,
            old_sharpe=old_sharpe,
            improvement=improvement,
            verdict=verdict,
            report_path=self._get_latest_report_path()
        )

    # ---- 因果诊断 ----

    def get_worst_trades(self, report: dict = None, n: int = 10) -> List[TradeRecord]:
        """获取亏损最大的 N 笔交易

        Args:
            report: 若为 None，自动取最新报告
            n: 返回数量（默认10）
        Returns:
            按亏损金额降序排列的 TradeRecord 列表
        """
        if report is None:
            reports = self.get_latest_reports(n=1)
            if not reports:
                return []
            report = reports[0]

        trades = self.extract_trades(report)
        # 按亏损排序（pnl 升序，亏损最大的在最前）
        losing_trades = [t for t in trades if t.pnl < 0]
        losing_trades.sort(key=lambda t: t.pnl)
        return losing_trades[:n]

    def diagnose(self, report: dict = None) -> DiagnosticReport:
        """执行因果诊断，分析亏损交易的共性

        Args:
            report: 若为 None，自动取最新报告
        Returns:
            DiagnosticReport — 包含诊断结论和假设
        """
        if report is None:
            reports = self.get_latest_reports(n=1)
            if not reports:
                raise ValueError("未找到回测报告，无法诊断")
            report = reports[0]

        worst = self.get_worst_trades(report, n=10)

        if not worst:
            return DiagnosticReport(
                worst_trades=[],
                total_loss=0.0,
                avg_loss=0.0,
                loss_by_pattern={},
                loss_by_sector={},
                common_patterns=[],
                sector_correlation="无亏损交易",
                volume_profile="无数据",
                hypothesis="无亏损案例，无需假设",
                suggested_fix="保持当前策略"
            )

        total_loss = sum(t.pnl for t in worst)
        avg_loss = total_loss / len(worst)

        # 按形态分组亏损
        loss_by_pattern: dict = {}
        for t in worst:
            p = t.pattern or "unknown"
            loss_by_pattern[p] = loss_by_pattern.get(p, 0.0) + t.pnl

        # 按板块分组亏损
        loss_by_sector: dict = {}
        for t in worst:
            s = t.sector or "unknown"
            loss_by_sector[s] = loss_by_sector.get(s, 0.0) + t.pnl

        # 共性形态
        common_patterns = sorted(loss_by_pattern, key=lambda x: loss_by_pattern[x])[:3]

        # 板块共性
        if len(loss_by_sector) <= 3:
            sector_correlation = f"亏损集中在 {', '.join(loss_by_sector.keys())} 板块"
        else:
            top_sectors = sorted(loss_by_sector, key=lambda x: loss_by_sector[x], reverse=True)[:3]
            sector_correlation = f"亏损主要集中在 {', '.join(top_sectors)} 等板块"

        # 成交量特征
        volumes = [t.volume for t in worst if t.volume is not None]
        if volumes:
            avg_vol = sum(volumes) / len(volumes)
            turnover_rates = [t.turnover_rate for t in worst if t.turnover_rate is not None]
            if turnover_rates:
                avg_turnover = sum(turnover_rates) / len(turnover_rates)
                volume_profile = f"平均成交量 {avg_vol:.0f}，平均换手率 {avg_turnover:.2%}"
            else:
                volume_profile = f"平均成交量 {avg_vol:.0f}（换手率数据缺失）"
        else:
            volume_profile = "成交量数据不足"

        # 生成量化假设
        hypothesis_parts = []
        if common_patterns:
            hypothesis_parts.append(f"形态共性：{', '.join(common_patterns)} 信号在弱势环境下易失效")

        if len(loss_by_sector) <= 3 and len(loss_by_sector) > 0:
            sectors = list(loss_by_sector.keys())
            hypothesis_parts.append(f"板块陷阱：{', '.join(sectors)} 板块在下跌趋势中，看涨信号成功率低")

        if volumes and avg_vol < 1000000:  # 假设小于100万股为缩量
            hypothesis_parts.append("量价背离：亏损案例普遍成交量偏低（缩量上涨后反转向下）")

        hypothesis = "；".join(hypothesis_parts) if hypothesis_parts else "未发现明显共性"

        # 建议修复方案
        if common_patterns:
            suggested_fix = (
                f"在 {', '.join(common_patterns[:2])} 信号触发前，"
                f"增加板块/大盘趋势过滤（例如：要求 MA5 > MA20 才允许看涨信号通过）"
            )
        else:
            suggested_fix = "建议增加大盘趋势过滤和成交量确认机制"

        return DiagnosticReport(
            worst_trades=worst,
            total_loss=total_loss,
            avg_loss=avg_loss,
            loss_by_pattern=loss_by_pattern,
            loss_by_sector=loss_by_sector,
            common_patterns=common_patterns,
            sector_correlation=sector_correlation,
            volume_profile=volume_profile,
            hypothesis=hypothesis,
            suggested_fix=suggested_fix
        )


# ==================== 独立运行入口 ====================

if __name__ == "__main__":
    evaluator = AutoEvaluator()
    result = evaluator.evaluate()
    print(f"【评估结论】 {result.status.upper()}")
    print(f"【详细说明】 {result.verdict}")
    print(f"【报告路径】 {result.report_path}")
    print()

    # 因果诊断
    try:
        diag = evaluator.diagnose()
        print(f"【因果诊断】")
        print(f"  总亏损：{diag.total_loss:.2f} 元")
        print(f"  平均亏损：{diag.avg_loss:.2f} 元")
        print(f"  板块共性：{diag.sector_correlation}")
        print(f"  成交量特征：{diag.volume_profile}")
        print(f"  共性形态：{', '.join(diag.common_patterns) if diag.common_patterns else '无'}")
        print(f"  量化假设：{diag.hypothesis}")
        print(f"  建议修复：{diag.suggested_fix}")
        print()
        print("【亏损最大10笔】")
        for i, t in enumerate(diag.worst_trades, 1):
            print(f"  {i}. {t.date} {t.code} 亏损:{t.pnl:.2f} 收益率:{t.return_pct:.2%} 形态:{t.pattern} 板块:{t.sector}")
    except Exception as e:
        print(f"【诊断失败】 {e}")
