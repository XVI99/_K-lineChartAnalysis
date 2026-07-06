# -*- coding: utf-8 -*-
"""
test_llm.py — LLM 接入测试脚本

测试内容:
1. LLM 客户端初始化和 API Key 检测
2. 基本对话调用
3. JSON 模式调用
4. 新闻情绪分析
5. 研报/板块解读
6. 因子解释
7. 策略报告生成（用模拟数据）

用法:
    # 先设置 API Key
    set ARK_API_KEY=your_api_key_here

    # 运行测试
    python test_llm.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_connection():
    """测试 1: LLM 客户端初始化"""
    print("\n" + "=" * 60)
    print("测试 1: LLM 客户端初始化")
    print("=" * 60)

    from AStockQuant.core.llm_client import llm

    print(f"  enabled: {llm.enabled}")
    print(f"  config: base_url={llm.config.get('base_url', '?')}")
    print(f"  config: model={llm.config.get('model', '?')}")

    api_key_env = llm.config.get("api_key_env", "ARK_API_KEY")
    api_key = os.environ.get(api_key_env, "")
    print(f"  API Key (env {api_key_env}): {'已设置 (' + api_key[:8] + '...)' if api_key else '未设置'}")

    if not llm.enabled:
        print("\n  [SKIP] LLM 未启用，请设置环境变量:")
        print(f"    set {api_key_env}=your_api_key_here")
        print("  或在 config.yaml llm.api_key 中填入")
        return False

    print("  [OK] LLM 已启用")
    return True


def test_basic_chat():
    """测试 2: 基本对话"""
    print("\n" + "=" * 60)
    print("测试 2: 基本对话")
    print("=" * 60)

    from AStockQuant.core.llm_client import llm

    resp = llm.chat("用一句话描述A股ETF轮动策略的核心逻辑", temperature=0.3, max_tokens=200)
    if resp:
        print(f"  [OK] 回复: {resp[:200]}")
    else:
        print("  [FAIL] 返回空")
    return bool(resp)


def test_json_mode():
    """测试 3: JSON 模式"""
    print("\n" + "=" * 60)
    print("测试 3: JSON 模式")
    print("=" * 60)

    from AStockQuant.core.llm_client import llm

    result = llm.chat_json(
        '分析"半导体板块今日大涨3%"的市场情绪，返回JSON：{"sentiment_score": 0.0, "label": "", "reason": ""}',
        system="你是金融分析师，返回合法JSON。",
        temperature=0.1,
        max_tokens=256,
    )
    if result:
        print(f"  [OK] JSON: {result}")
    else:
        print("  [FAIL] 返回 None")
    return result is not None


def test_news_sentiment():
    """测试 4: 新闻情绪分析"""
    print("\n" + "=" * 60)
    print("测试 4: 新闻情绪分析")
    print("=" * 60)

    from AStockQuant.llm.news_sentiment import NewsSentimentAnalyzer

    analyzer = NewsSentimentAnalyzer()
    if not analyzer.enabled:
        print("  [SKIP] 新闻情绪分析未启用")
        return False

    result = analyzer.analyze(
        title="央行宣布降准0.5个百分点，释放长期资金约1万亿元",
        content="中国人民银行决定于2024年X月X日下调金融机构存款准备金率0.5个百分点。此次降准将释放长期资金约1万亿元，降低金融机构资金成本每年约56亿元。",
        source="新华社",
    )
    if result:
        print(f"  [OK] 情绪分: {result.get('sentiment_score', '?')}")
        print(f"  标签: {result.get('sentiment_label', '?')}")
        print(f"  关键词: {result.get('keywords', [])}")
        print(f"  相关板块: {result.get('related_sectors', [])}")
        print(f"  影响程度: {result.get('impact_level', '?')}")
        print(f"  摘要: {result.get('summary', '?')}")
    else:
        print("  [FAIL] 返回 None")
    return result is not None


def test_report_analysis():
    """测试 5: 研报/板块解读"""
    print("\n" + "=" * 60)
    print("测试 5: 研报/板块解读")
    print("=" * 60)

    from AStockQuant.llm.report_analysis import ReportAnalyzer

    analyzer = ReportAnalyzer()
    if not analyzer.enabled:
        print("  [SKIP] 研报分析未启用")
        return False

    result = analyzer.analyze_sector(
        sector_name="半导体",
        sector_data={
            "20日涨幅": 0.085,
            "成交额_亿": 1250,
            "领涨股": ["中芯国际", "北方华创", "韦尔股份"],
            "PE_TTM": 65.2,
            "周环比": 0.032,
        },
        market_context="trend_bull_mid_vol_theme_market",
    )
    if result:
        print(f"  [OK] 评级: {result.get('rating', '?')}")
        print(f"  评分: {result.get('score', '?')}")
        print(f"  驱动因素: {result.get('drivers', [])}")
        print(f"  风险点: {result.get('risks', [])}")
        print(f"  相关ETF: {result.get('related_etfs', [])}")
        print(f"  总结: {result.get('summary', '?')}")
    else:
        print("  [FAIL] 返回 None")
    return result is not None


def test_factor_explainer():
    """测试 6: 因子解释"""
    print("\n" + "=" * 60)
    print("测试 6: 因子解释")
    print("=" * 60)

    from AStockQuant.llm.factor_explainer import FactorExplainer

    explainer = FactorExplainer()
    if not explainer.enabled:
        print("  [SKIP] 因子解释未启用")
        return False

    result = explainer.explain_factor(
        factor_name="pv_rps_20",
        factor_value=0.85,
        rank_pct=0.95,
        regime={"trend_regime": "trend_bull", "vol_regime": "mid_vol", "leadership_regime": "theme_market"},
        ic_mean=0.039,
    )
    if result:
        print(f"  [OK] 解释:\n  {result[:500]}")
    else:
        print("  [FAIL] 返回空")
    return bool(result)


def test_strategy_reporter():
    """测试 7: 策略报告生成"""
    print("\n" + "=" * 60)
    print("测试 7: 策略报告生成")
    print("=" * 60)

    from AStockQuant.llm.strategy_reporter import StrategyReporter

    reporter = StrategyReporter()

    strategy_stats = {"total_return": 0.0203, "max_drawdown": -0.1969, "sharpe": 0.11, "ann_return": 0.005}
    benchmark_stats = [
        {"benchmark": "strategy", "total_return": 0.0203, "max_drawdown": -0.1969, "sharpe": 0.11},
        {"benchmark": "buy_hold_510300", "total_return": 0.0042, "max_drawdown": -0.3314, "sharpe": 0.10},
        {"benchmark": "simple_rps_momentum", "total_return": 0.6948, "max_drawdown": -0.1898, "sharpe": 0.66},
    ]

    report = reporter.generate_report(
        strategy_stats=strategy_stats,
        benchmark_stats=benchmark_stats,
    )
    if report:
        print(f"  [OK] 报告长度: {len(report)} 字符")
        print(f"  前500字:\n{report[:500]}")

        save_path = ROOT / "reports" / "llm_test_report.md"
        reporter.save_report(report, str(save_path))
    else:
        print("  [FAIL] 返回空")
    return bool(report)


def main():
    print("=" * 60)
    print("LLM 接入测试 (火山方舟 Coding Plan)")
    print("=" * 60)

    if not test_connection():
        print("\n所有测试跳过（LLM 未启用）")
        return 1

    results = []
    results.append(("基本对话", test_basic_chat()))
    results.append(("JSON模式", test_json_mode()))
    results.append(("新闻情绪", test_news_sentiment()))
    results.append(("研报解读", test_report_analysis()))
    results.append(("因子解释", test_factor_explainer()))
    results.append(("策略报告", test_strategy_reporter()))

    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    for name, ok in results:
        status = "[PASS]" if ok else "[FAIL]"
        print(f"  {status} {name}")

    passed = sum(1 for _, ok in results if ok)
    print(f"\n  {passed}/{len(results)} 通过")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
