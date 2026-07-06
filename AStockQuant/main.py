# -*- coding: utf-8 -*-
"""
main.py — AStockQuant 统一命令行入口

使用方法:
    python main.py scan [--top N] [--conf FLOAT]     # 全市场扫描
    python main.py backtest [--start DATE]            # 历史回测
    python main.py download [--days N]                # 下载数据
    python main.py quick-test                         # 快速验证

示例:
    python main.py scan --top 30 --conf 0.55
    python main.py backtest --start 2024-01-01
    python main.py download --days 500
"""

import sys
import os

# 确保项目根目录在 sys.path 中
proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parent_root = os.path.dirname(proj_root)
if parent_root not in sys.path:
    sys.path.insert(0, parent_root)
os.environ["PYTHONPATH"] = parent_root
os.chdir(proj_root)

import argparse
from datetime import datetime


def cmd_scan(args):
    """执行全市场扫描"""
    from AStockQuant.scanners.market_scanner import MarketScanner
    
    top_n = args.top or 30
    min_conf = args.conf or 0.55
    
    print("=" * 60)
    print("AStockQuant 全市场扫描")
    print("=" * 60)
    print(f"扫描数量: {top_n}")
    print(f"最低置信度: {min_conf}")
    print()
    
    scanner = MarketScanner(budget=5000, max_price=45.0)
    results = scanner.scan(top_n=top_n, min_confidence=min_conf)
    
    if results is not None and not results.empty:
        buy_signals = results[results['signal'].isin(['STRONG_BUY', 'BUY'])]
        print(f"\n--- Scan complete. Buy signals: {len(buy_signals)} ---")
    else:
        print("\n--- Scan: no results ---")


def cmd_backtest(args):
    """执行历史回测"""
    from AStockQuant.backtest.historical_runner import run_historical_backtest
    
    start_date = args.start or "2024-01-01"
    rebalance = args.rebalance or 5
    capital = args.capital or 5000.0
    top_n = args.positions or 3
    
    print("=" * 60)
    print("AStockQuant 历史回测")
    print("=" * 60)
    print(f"开始日期: {start_date}")
    print(f"调仓周期: {rebalance} 个交易日")
    print(f"初始资金: {capital} 元")
    print(f"持仓数量: {top_n} 只")
    print()
    
    account, stats = run_historical_backtest(
        start_date=start_date,
        rebalance_days=rebalance,
        initial_capital=capital,
        top_n=top_n,
        enable_limit_up_down=True,
        enable_dynamic_slippage=True
    )
    
    if stats:
        print(f"\n--- Backtest complete. Total return: {stats.get('total_return', 0):.2f}% ---")


def cmd_download(args):
    """下载市场数据"""
    from AStockQuant.backtest.data_downloader import main as download_main
    
    print("=" * 60)
    print("AStockQuant 数据下载")
    print("=" * 60)
    print()
    
    download_main()


def cmd_quick_test(args):
    """快速验证系统可用性"""
    import importlib
    importlib.import_module("AStockQuant")  # 确保路径生效

    print("=" * 60)
    print("AStockQuant 快速验证")
    print("=" * 60)
    
    checks = []
    
    # 1. 检查配置
    try:
        from AStockQuant.core.config_loader import ConfigLoader
        config = ConfigLoader.get_instance()
        checks.append(("配置加载", True, f"配置文件: {config._config_path}"))
    except Exception as e:
        checks.append(("配置加载", False, str(e)))

    # 2. 检查数据层
    try:
        from AStockQuant.core.data_hub import ETFDataHub as DataHub
        dh = DataHub()
        checks.append(("数据中枢", True, "OK"))
    except Exception as e:
        checks.append(("数据中枢", False, str(e)))

    # 3. 检查特征注册
    try:
        from AStockQuant.core.feature_registry import FeatureRegistry
        fr = FeatureRegistry()
        checks.append(("特征注册", True, "OK"))
    except Exception as e:
        checks.append(("特征注册", False, str(e)))

    # 4. 检查扫描器
    try:
        from AStockQuant.scanners.market_scanner import MarketScanner
        checks.append(("市场扫描器", True, "OK"))
    except Exception as e:
        checks.append(("市场扫描器", False, str(e)))

    # 5. 检查回测引擎
    try:
        from AStockQuant.backtest.engine import VirtualAccount
        va = VirtualAccount()
        checks.append(("回测引擎", True, "OK"))
    except Exception as e:
        checks.append(("回测引擎", False, str(e)))

    # 6. 检查贝叶斯信念层（L8）
    try:
        from AStockQuant.layers.layer8_micro import BeliefLayer
        bl = BeliefLayer()
        checks.append(("信念引擎L8", True, "OK"))
    except Exception as e:
        checks.append(("信念引擎L8", False, str(e)))
    
    # 输出结果
    print()
    all_pass = True
    for name, passed, msg in checks:
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status} {name}: {msg}")
        if not passed:
            all_pass = False
    
    print()
    if all_pass:
        print("PASS: All modules verified!")
    else:
        print("WARN: Some modules failed.")


def cmd_belief(args):
    """贝叶斯信念诊断"""
    from AStockQuant.layers.layer8_micro import BeliefLayer

    print("=" * 60)
    print("贝叶斯信念诊断 (L8)")
    print("=" * 60)

    layer = BeliefLayer()
    summary = layer.get_market_summary()

    if not summary:
        print("\n【市场信念汇总】无数据（BeliefLayer 为空，需先运行 scan 累积信念后才能汇总）")
        return

    print(f"\n【市场信念汇总】")
    print(f"  标的数: {summary['count']}")
    print(f"  均值后验: {summary['mean_posterior']:.3f}")
    print(f"  中位数后验: {summary['median_posterior']:.3f}")
    print(f"  后验标准差: {summary['std_posterior']:.3f}")
    print(f"  看涨: {summary['bullish_count']}  看跌: {summary['bearish_count']}")
    print(f"  强烈看涨: {summary['strongly_bullish_count']}  强烈看跌: {summary['strongly_bearish_count']}")
    print(f"  均值KL: {summary['mean_kl']:.4f}  最大KL: {summary['max_kl']:.4f}")
    print(f"  漂移告警数: {summary['drift_alert_count']}")

    print(f"\n【信念最强 Top-{args.top}】")
    top_df = layer.get_top_beliefs(n=args.top)
    if not top_df.empty:
        print(top_df.to_string(index=False))
    else:
        print("无数据（需要先运行 scan）")

    if args.drift_kl:
        alerts = layer.get_drift_alerts(min_kl=args.drift_kl)
        if alerts:
            print(f"\n【漂移告警 (KL>{args.drift_kl})】{len(alerts)} 只")
            for a in alerts[:10]:
                print(f"  {a['symbol']:8s}  KL={a['kl']:.4f}  "
                      f"P={a['posterior']:.3f}  base={a['base_prior']:.3f}  "
                      f"drift={a['drift_pct']:+.1f}%  [{a['level']}]")
        else:
            print(f"\n【漂移告警】无 (KL<{args.drift_kl})")


def main():
    parser = argparse.ArgumentParser(
        description="AStockQuant — A股量化分析框架",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py scan --top 30 --conf 0.55
  python main.py backtest --start 2024-01-01 --capital 5000
  python main.py download --days 500
  python main.py quick-test
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # scan 命令
    scan_parser = subparsers.add_parser("scan", help="全市场扫描")
    scan_parser.add_argument("--top", type=int, help="扫描股票数量 (默认: 30)")
    scan_parser.add_argument("--conf", type=float, help="最低置信度 (默认: 0.55)")
    
    # backtest 命令
    bt_parser = subparsers.add_parser("backtest", help="历史回测")
    bt_parser.add_argument("--start", type=str, help="开始日期 YYYY-MM-DD (默认: 2024-01-01)")
    bt_parser.add_argument("--rebalance", type=int, help="调仓周期天数 (默认: 5)")
    bt_parser.add_argument("--capital", type=float, help="初始资金 (默认: 5000)")
    bt_parser.add_argument("--positions", type=int, help="持仓数量 (默认: 3)")
    
    # download 命令
    dl_parser = subparsers.add_parser("download", help="下载市场数据")
    dl_parser.add_argument("--days", type=int, help="下载天数 (默认: 500)")
    
    # quick-test 命令
    subparsers.add_parser("quick-test", help="快速验证系统可用性")

    # belief 命令：贝叶斯信念诊断
    belief_parser = subparsers.add_parser("belief", help="贝叶斯信念诊断")
    belief_parser.add_argument("--symbol", type=str, help="标的代码 (默认: 全部)")
    belief_parser.add_argument("--top", type=int, default=10, help="显示 top N (默认: 10)")
    belief_parser.add_argument("--drift-kl", type=float, default=0.05, help="漂移 KL 阈值 (默认: 0.05)")

    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        print("\n📋 可用命令:")
        print("  scan       — 全市场扫描")
        print("  backtest   — 历史回测")
        print("  download   — 下载数据")
        print("  quick-test — 快速验证")
        return
    
    if args.command == "scan":
        cmd_scan(args)
    elif args.command == "backtest":
        cmd_backtest(args)
    elif args.command == "download":
        cmd_download(args)
    elif args.command == "quick-test":
        cmd_quick_test(args)
    elif args.command == "belief":
        cmd_belief(args)


if __name__ == "__main__":
    main()