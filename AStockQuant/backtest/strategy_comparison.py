"""
策略综合对比分析 v8.0
============================================================

汇总所有策略版本的表现，并创建可视化对比
"""

import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 读取所有回测结果
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output')

# 策略结果汇总
strategy_results = [
    # 版本1: 简单ETF回测
    {'name': 'v1.0 简单回测', 'total': 0.0945, 'annual': 0.0709, 'drawdown': -0.1460, 'sharpe': 0.65, 'trades': 45},
    
    # 版本2: D2策略
    {'name': 'v2.0 D2策略', 'total': -0.3806, 'annual': -0.0850, 'drawdown': -0.4000, 'sharpe': -0.30, 'trades': 30},
    
    # 版本3: 优化回测
    {'name': 'v3.0 优化回测', 'total': 0.0679, 'annual': 0.0478, 'drawdown': -0.0821, 'sharpe': 0.45, 'trades': 40},
    
    # 版本4: 海龟策略
    {'name': 'v4.0 海龟策略', 'total': 0.0001, 'annual': 0.0000, 'drawdown': -0.0003, 'sharpe': 0.08, 'trades': 86},
    
    # 版本5: 动量趋势 (激进)
    {'name': 'v5.0 动量激进', 'total': 1.8023, 'annual': 0.2947, 'drawdown': -0.9561, 'sharpe': 0.58, 'trades': 414},
    
    # 版本5.1: 动量+风控
    {'name': 'v5.1 动量+风控', 'total': 1.2572, 'annual': 0.2264, 'drawdown': -0.9599, 'sharpe': 0.89, 'trades': 280},
    
    # 版本6: 稳健版
    {'name': 'v6.0 稳健版', 'total': 0.3070, 'annual': 0.0694, 'drawdown': -0.7789, 'sharpe': 0.47, 'trades': 200},
    
    # 版本7: 严格选股
    {'name': 'v7.0 严格选股', 'total': 0.3251, 'annual': 0.0731, 'drawdown': -0.5543, 'sharpe': 0.41, 'trades': 120},
]

def create_comparison_chart():
    """创建策略对比图表"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'sans-serif']
    
    names = [r['name'] for r in strategy_results]
    anns = [r['annual'] * 100 for r in strategy_results]
    dds = [r['drawdown'] * 100 for r in strategy_results]
    sharpes = [r['sharpe'] for r in strategy_results]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 年化收益
    colors = ['green' if x > 0 else 'red' for x in anns]
    axes[0].barh(names, anns, color=colors, alpha=0.7)
    axes[0].set_xlabel('Annualized Return (%)')
    axes[0].set_title('Strategy Annualized Returns')
    axes[0].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    for i, v in enumerate(anns):
        axes[0].text(v + 1, i, f'{v:.1f}%', va='center', fontsize=9)
    
    # 最大回撤
    colors = ['orange' if x > -50 else 'red' for x in dds]
    axes[1].barh(names, dds, color=colors, alpha=0.7)
    axes[1].set_xlabel('Max Drawdown (%)')
    axes[1].set_title('Strategy Max Drawdown')
    axes[1].axvline(x=-50, color='red', linestyle='--', linewidth=1, label='-50% threshold')
    for i, v in enumerate(dds):
        axes[1].text(v - 3, i, f'{v:.1f}%', va='center', fontsize=9)
    
    # 夏普比率
    colors = ['blue' if x > 0.5 else 'gray' for x in sharpes]
    axes[2].barh(names, sharpes, color=colors, alpha=0.7)
    axes[2].set_xlabel('Sharpe Ratio')
    axes[2].set_title('Strategy Sharpe Ratio')
    axes[2].axvline(x=1.0, color='green', linestyle='--', linewidth=1, label='Sharpe=1.0')
    for i, v in enumerate(sharpes):
        axes[2].text(v + 0.05, i, f'{v:.2f}', va='center', fontsize=9)
    
    plt.tight_layout()
    chart_path = os.path.join(output_dir, 'strategy_comparison.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    print(f"\n图表已保存: {chart_path}")
    
    return chart_path

def create_equity_curves():
    """生成各策略权益曲线"""
    pass

def print_summary():
    """打印汇总表"""
    print("\n" + "="*80)
    print("  量化策略回测结果汇总 (2019-01 ~ 2024-12)")
    print("="*80)
    print(f"{'策略名称':<20} {'总收益':<12} {'年化':<12} {'最大回撤':<12} {'夏普':<8} {'交易次数':<10}")
    print("-"*80)
    
    for r in strategy_results:
        print(f"{r['name']:<20} {r['total']*100:>+10.2f}% {r['annual']*100:>+10.2f}% {r['drawdown']*100:>-10.2f}% {r['sharpe']:>8.2f} {r['trades']:>8}")
    
    print("-"*80)
    
    # 计算性价比 (年化/回撤)
    for r in strategy_results:
        if r['drawdown'] != 0:
            r['return_per_drawdown'] = r['annual'] / abs(r['drawdown'])
        else:
            r['return_per_drawdown'] = 0
    
    # 按风险收益比排序
    sorted_results = sorted(strategy_results, key=lambda x: x['return_per_drawdown'], reverse=True)
    
    print("\n风险收益比排名 (年化收益/最大回撤):")
    print("-"*60)
    for i, r in enumerate(sorted_results[:5], 1):
        print(f"{i}. {r['name']:<18}: {r['return_per_drawdown']:.3f} (年化{r['annual']*100:.1f}% / 回撤{r['drawdown']*100:.1f}%)")
    
    print("\n" + "="*80)
    print("  关键发现")
    print("="*80)
    
    # 最高收益
    best_return = max(strategy_results, key=lambda x: x['annual'])
    print(f"• 最高年化: {best_return['name']} ({best_return['annual']*100:.2f}%)")
    
    # 最低回撤
    lowest_dd = min(strategy_results, key=lambda x: x['drawdown'])
    print(f"• 最低回撤: {lowest_dd['name']} ({lowest_dd['drawdown']*100:.2f}%)")
    
    # 最高夏普
    best_sharpe = max(strategy_results, key=lambda x: x['sharpe'])
    print(f"• 最高夏普: {best_sharpe['name']} ({best_sharpe['sharpe']:.2f})")
    
    # 最佳平衡
    best_balance = max(strategy_results, key=lambda x: x['return_per_drawdown'])
    print(f"• 最佳平衡: {best_balance['name']} (风险收益比:{best_balance['return_per_drawdown']:.3f})")

def create_final_recommendation():
    """生成最终推荐"""
    print("\n" + "="*80)
    print("  策略推荐")
    print("="*80)
    
    print("""
根据2019-2024年回测结果，建议如下:

【激进方案】v5.1 动量+风控
  年化: 22.64% | 回撤: -95.99%
  适合: 高风险承受能力, 长期持有不动
  注意: 需承受接近100%回撤风险

【平衡方案】v7.0 严格选股 (配置D)
  年化: 7.31% | 回撤: -55.43%
  适合: 中等风险承受能力
  特点: 严格选股条件减少亏损交易

【保守方案】v3.0 优化回测
  年化: 4.78% | 回撤: -8.21%
  适合: 低风险承受能力
  特点: 回撤极低, 收益稳定

【建议】
实际使用时建议:
1. 使用【平衡方案】作为主要策略
2. 结合市场趋势判断 (大盘MA200向上时使用)
3. 配合8-10%止损保护
4. 保持30%现金作为安全垫
    """)

if __name__ == '__main__':
    print_summary()
    create_comparison_chart()
    create_final_recommendation()