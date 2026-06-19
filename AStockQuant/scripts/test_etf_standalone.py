"""ETF双动量低频轮动系统 - 独立测试版本"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json


class ETFPosition:
    """ETF持仓"""
    def __init__(self, code: str, name: str, shares: int, avg_cost: float):
        self.code = code
        self.name = name
        self.shares = shares
        self.avg_cost = avg_cost


class ETFDualMomentumEngine:
    """ETF双动量低频轮动引擎"""

    def __init__(self, initial_capital: float = 5000, min_trade_cost_pct: float = 0.002):
        self.initial_capital = initial_capital
        self.min_trade_cost_pct = min_trade_cost_pct
        self.cash = initial_capital
        self.positions: Dict[str, ETFPosition] = {}
        self.portfolio_nav: List[Dict] = []
        self.paper_trades: List[Dict] = []
        self.daily_signals: List[Dict] = []
        self.weekly_reviews: List[Dict] = []

        # 评分参数
        self.weight_60d = 0.5
        self.weight_20d = 0.3
        self.weight_120d = 0.2
        self.weight_vol = 0.3

        # 风险切换参数
        self.risk_etf_codes = ['510300', '510500', '159605']
        self.risk_ma_period = 120
        self.risk_threshold = 3

        # 交易规则
        self.max_weekly_trades = 1
        self.min_holding_weeks = 4
        self.cash_reserve_pct = 0.05

    def calculate_momentum_score(self, prices: pd.DataFrame, code: str) -> Optional[float]:
        """计算动量评分: score = 0.5x60日 + 0.3x20日 + 0.2x120日 - 0.3x波动率"""
        if code not in prices.columns:
            return None

        price_series = prices[code].dropna()
        if len(price_series) < 121:
            return None

        current_price = price_series.iloc[-1]
        ret_60d = (current_price / price_series.iloc[-61] - 1) * 100 if len(price_series) >= 61 else 0
        ret_20d = (current_price / price_series.iloc[-21] - 1) * 100 if len(price_series) >= 21 else 0
        ret_120d = (current_price / price_series.iloc[-121] - 1) * 100 if len(price_series) >= 121 else 0

        if len(price_series) >= 61:
            daily_returns = price_series.pct_change().dropna()[-60:]
            vol_60d = daily_returns.std() * np.sqrt(252) * 100
        else:
            vol_60d = 0

        score = (self.weight_60d * ret_60d +
                 self.weight_20d * ret_20d +
                 self.weight_120d * ret_120d -
                 self.weight_vol * vol_60d)

        return score

    def check_risk_switch(self, prices: pd.DataFrame) -> Tuple[bool, int]:
        """检查风险切换"""
        count_above_ma = 0
        for code in self.risk_etf_codes:
            if code in prices.columns:
                price_series = prices[code].dropna()
                if len(price_series) >= 121:
                    ma120 = price_series.rolling(120).mean().iloc[-1]
                    if price_series.iloc[-1] > ma120:
                        count_above_ma += 1
        defensive = count_above_ma < self.risk_threshold
        return defensive, count_above_ma

    def execute_trade(self, date: str, code: str, name: str, price: float, action: str, shares: int) -> float:
        """执行交易"""
        if action == 'BUY':
            cost = shares * price
            commission = max(cost * self.min_trade_cost_pct, 5)
            total_cost = cost + commission

            if total_cost > self.cash:
                max_shares = int((self.cash * (1 - self.cash_reserve_pct)) / price / 100) * 100
                if max_shares < 100:
                    return 0
                shares = max_shares
                cost = shares * price
                commission = max(cost * self.min_trade_cost_pct, 5)
                total_cost = cost + commission

            self.cash -= total_cost
            self.positions[code] = ETFPosition(code, name, shares, price)

            self.paper_trades.append({
                'date': date,
                'code': code,
                'name': name,
                'action': 'BUY',
                'price': price,
                'shares': shares,
                'amount': cost,
                'commission': commission
            })
            return total_cost

        elif action == 'SELL':
            if code not in self.positions:
                return 0
            position = self.positions[code]
            amount = position.shares * price
            commission = max(amount * self.min_trade_cost_pct, 5)
            net_proceeds = amount - commission

            self.cash += net_proceeds
            del self.positions[code]

            self.paper_trades.append({
                'date': date,
                'code': code,
                'name': name,
                'action': 'SELL',
                'price': price,
                'shares': position.shares,
                'amount': amount,
                'commission': commission
            })
            return net_proceeds
        return 0

    def update_nav(self, date: str, close_prices: Dict[str, float]):
        """更新每日净值"""
        total_value = self.cash
        for code, pos in self.positions.items():
            if code in close_prices:
                total_value += pos.shares * close_prices[code]
            else:
                total_value += pos.shares * pos.avg_cost

        self.portfolio_nav.append({
            'date': date,
            'cash': self.cash,
            'position_value': total_value - self.cash,
            'total_nav': total_value,
            'nav_pct': (total_value / self.initial_capital - 1) * 100
        })

    def add_signal(self, date: str, code: str, name: str, score: float, action: str, reason: str):
        """添加每日信号"""
        self.daily_signals.append({
            'date': date,
            'code': code,
            'name': name,
            'score': round(score, 2) if score else 0,
            'action': action,
            'reason': reason,
            'cash': round(self.cash, 2)
        })

    def run_backtest(self, prices: pd.DataFrame, etf_info: Dict[str, Dict],
                     start_date: str = None, end_date: str = None,
                     weekly_interval: int = 5) -> Dict:
        """运行回测"""
        print("=" * 60)
        print("ETF双动量低频轮动系统回测")
        print("核心公式: score = 0.5x60日 + 0.3x20日 + 0.2x120日 - 0.3x波动率")
        print("=" * 60)

        if start_date:
            prices = prices[prices.index >= start_date]
        if end_date:
            prices = prices[prices.index <= end_date]

        candidates = [c for c in prices.columns if c in etf_info]
        holding_weeks = {code: 0 for code in self.positions.keys()}
        weekly_trades = 0
        last_week_idx = -1
        trade_dates = prices.index.tolist()

        for i, date in enumerate(trade_dates):
            is_evaluation_day = (i - last_week_idx) >= weekly_interval

            if not is_evaluation_day:
                close_prices = {code: prices.loc[date, code] for code in candidates if code in prices.columns}
                self.update_nav(str(date)[:10], close_prices)
                continue

            last_week_idx = i

            # 计算所有ETF评分
            scores_data = []
            for code in candidates:
                score = self.calculate_momentum_score(prices, code)
                if score is not None:
                    scores_data.append({
                        'code': code,
                        'name': etf_info.get(code, {}).get('name', code),
                        'score': score
                    })

            if not scores_data:
                continue

            scores_data.sort(key=lambda x: x['score'], reverse=True)
            top_etfs = scores_data[:5]

            # 检查风险切换
            defensive, risk_count = self.check_risk_switch(prices.loc[:date])
            risk_status = "DEFENSIVE" if defensive else "NORMAL"

            # 选择最优ETF
            selected_code = scores_data[0]['code'] if scores_data else None
            current_code = list(self.positions.keys())[0] if self.positions else None

            # 更新持有周数
            if current_code:
                holding_weeks[current_code] = holding_weeks.get(current_code, 0) + 1

            # 决策交易
            action = "HOLD"
            reason = ""

            date_str = str(date)[:10]
            
            if selected_code and selected_code != current_code:
                current_price = prices.loc[date, selected_code]
                current_name = etf_info.get(selected_code, {}).get('name', selected_code)

                if current_code:
                    sell_price = prices.loc[date, current_code]
                    sell_name = etf_info.get(current_code, {}).get('name', current_code)
                    self.execute_trade(date_str, current_code, sell_name,
                                      sell_price, 'SELL', self.positions[current_code].shares)
                    action = "SELL " + current_code
                    reason = f"切换到{selected_code}"

                shares = int((self.cash * (1 - self.cash_reserve_pct)) / current_price / 100) * 100
                if shares >= 100:
                    self.execute_trade(date_str, selected_code, current_name,
                                      current_price, 'BUY', shares)
                    if action == "HOLD":
                        action = "BUY " + selected_code
                    reason = f"评分最高:{scores_data[0]['score']:.1f}"
                weekly_trades = 1

            elif not current_code and selected_code:
                current_price = prices.loc[date, selected_code]
                current_name = etf_info.get(selected_code, {}).get('name', selected_code)
                shares = int((self.cash * (1 - self.cash_reserve_pct)) / current_price / 100) * 100
                if shares >= 100:
                    self.execute_trade(date_str, selected_code, current_name,
                                      current_price, 'BUY', shares)
                    action = "BUY " + selected_code
                    reason = "空仓建仓"
                weekly_trades = 1
            else:
                action = "HOLD"
                reason = f"继续持有{current_code}"

            selected_score = scores_data[0]['score'] if scores_data else 0
            self.add_signal(date_str, selected_code or '', '', selected_score, action, reason)

            close_prices = {code: prices.loc[date, code] for code in candidates if code in prices.columns}
            self.update_nav(date_str, close_prices)

            current_nav = self.portfolio_nav[-1]['nav_pct'] if self.portfolio_nav else 0
            print(f"{date_str} | {action:15} | {risk_status:9} | 净值:{current_nav:+.1f}%")

            weekly_trades = 0

        return self.get_statistics()

    def get_statistics(self) -> Dict:
        """计算回测统计"""
        if not self.portfolio_nav:
            return {}

        nav_df = pd.DataFrame(self.portfolio_nav)
        total_return = (nav_df['total_nav'].iloc[-1] / self.initial_capital - 1) * 100

        if len(nav_df) > 1:
            years = len(nav_df) / 252
            annualized = ((nav_df['total_nav'].iloc[-1] / self.initial_capital) ** (1/years) - 1) * 100 if years > 0 else 0
        else:
            annualized = 0

        nav_df['peak'] = nav_df['total_nav'].cummax()
        nav_df['drawdown'] = (nav_df['total_nav'] - nav_df['peak']) / nav_df['peak'] * 100
        max_drawdown = nav_df['drawdown'].min()

        if len(nav_df) > 1:
            daily_returns = nav_df['total_nav'].pct_change().dropna()
            sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
        else:
            sharpe = 0

        stats = {
            'initial_capital': self.initial_capital,
            'final_nav': nav_df['total_nav'].iloc[-1],
            'total_return': total_return,
            'annualized_return': annualized,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'total_trades': len(self.paper_trades),
        }

        print("\n" + "=" * 60)
        print("回测结果统计")
        print("=" * 60)
        print(f"初始资金:    ¥{stats['initial_capital']:,.2f}")
        print(f"最终净值:    ¥{stats['final_nav']:,.2f}")
        print(f"总收益率:    {stats['total_return']:+.2f}%")
        print(f"年化收益率:  {stats['annualized_return']:+.2f}%")
        print(f"最大回撤:    {stats['max_drawdown']:.2f}%")
        print(f"夏普比率:    {stats['sharpe_ratio']:.2f}")
        print(f"总交易次数:  {stats['total_trades']}")
        print("=" * 60)

        return stats

    def save_outputs(self, output_dir: str = 'output'):
        """保存四个标准输出文件"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        if self.daily_signals:
            signal_df = pd.DataFrame(self.daily_signals)
            signal_df.to_csv(f'{output_dir}/etf_daily_signal.csv', index=False, encoding='utf-8-sig')
            print(f"已保存: {output_dir}/etf_daily_signal.csv")

        if self.paper_trades:
            trades_df = pd.DataFrame(self.paper_trades)
            trades_df.to_csv(f'{output_dir}/etf_paper_trades.csv', index=False, encoding='utf-8-sig')
            print(f"已保存: {output_dir}/etf_paper_trades.csv")

        if self.portfolio_nav:
            nav_df = pd.DataFrame(self.portfolio_nav)
            nav_df.to_csv(f'{output_dir}/etf_portfolio_nav.csv', index=False, encoding='utf-8-sig')
            print(f"已保存: {output_dir}/etf_portfolio_nav.csv")

        if self.weekly_reviews:
            review_df = pd.DataFrame(self.weekly_reviews)
            review_df['top_etfs'] = review_df['top_etfs'].apply(lambda x: json.dumps(x, ensure_ascii=False))
            review_df.to_csv(f'{output_dir}/etf_weekly_review.csv', index=False, encoding='utf-8-sig')
            print(f"已保存: {output_dir}/etf_weekly_review.csv")


# ========== 主程序 ==========
if __name__ == '__main__':
    print("=" * 60)
    print("ETF双动量低频轮动系统")
    print("核心理念: 双动量 + 风险切换")
    print("=" * 60)

    # 生成模拟数据
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', '2026-05-31', freq='B')
    n_days = len(dates)

    etf_list = {
        '510300': {'name': '沪深300ETF'},
        '510500': {'name': '中证500ETF'},
        '159919': {'name': '沪深300ETF深'},
        '510050': {'name': '上证50ETF'},
        '159915': {'name': '创业板ETF'},
        '512880': {'name': '证券ETF'},
        '512760': {'name': '芯片ETF'},
    }

    prices_data = {}
    for code in etf_list.keys():
        initial_price = 5.0
        daily_ret = np.random.normal(0.0003, 0.015, n_days)
        cumulative_ret = np.cumprod(1 + daily_ret)
        trend = np.linspace(0, 0.2, n_days)
        prices = initial_price * cumulative_ret * (1 + trend)
        prices_data[code] = prices

    prices_df = pd.DataFrame(prices_data, index=dates)

    # 运行回测
    engine = ETFDualMomentumEngine(initial_capital=5000, min_trade_cost_pct=0.002)
    stats = engine.run_backtest(prices_df, etf_list, start_date='2024-01-01')
    engine.save_outputs('output')

    print()
    print("=" * 60)
    print("买卖信号示例:")
    print("=" * 60)
    for s in engine.daily_signals[:15]:
        print(f"{s['date']} | {s['action']:15} | 评分:{s['score']:6.1f} | {s['reason']}")

    print()
    print("=" * 60)
    print("交易记录:")
    print("=" * 60)
    for t in engine.paper_trades:
        print(f"{t['date']} | {t['action']:4} | {t['code']} | {t['name']} | {t['shares']}股 @ ¥{t['price']:.3f}")
