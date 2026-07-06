# -*- coding: utf-8 -*-
"""
Aggressive Leverage ETF Strategy
Target: 100%+ annual return, 30%+ monthly return
Features:
- Daily rebalancing, only hold top 1-3 strongest ETFs
- 2x margin leverage
- 5% take profit, 2% stop loss
- 100% cash position in down market
"""
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime, timedelta

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data_cache"
REPORT_DIR = ROOT / "reports"

LEVERAGE = 2.0
TAKE_PROFIT_PCT = 0.05
STOP_LOSS_PCT = 0.02
MAX_HOLD_DAYS = 3
TOP_N_ETF = 2
MIN_RANK_THRESHOLD = 0.8

def load_etf_data():
    """Load all ETF daily data"""
    all_data = {}
    for f in DATA_DIR.glob("*.csv"):
        code = f.stem
        df = pd.read_csv(f, parse_dates=["date"], index_col="date")
        if len(df) < 100: continue
        all_data[code] = df
    return all_data

def calculate_momentum_score(df, window=5):
    """Calculate short-term momentum score"""
    if len(df) < window + 1: return 0.0
    recent_return = (df["close"].iloc[-1] - df["close"].iloc[-window -1]) / df["close"].iloc[-window -1]
    volatility = df["close"].pct_change().rolling(window).std().iloc[-1]
    if volatility == 0: return 0.0
    return recent_return / volatility

def get_market_regime(date, data):
    """判断市场趋势，下跌时空仓"""
    hs300 = data.get("510300")
    if hs300 is None or date not in hs300.index: return "unknown"
    ma20 = hs300["close"].rolling(20).mean().loc[:date].iloc[-1]
    ma60 = hs300["close"].rolling(60).mean().loc[:date].iloc[-1]
    current_price = hs300["close"].loc[date]
    if current_price < ma20 and current_price < ma60:
        return "down"
    if current_price > ma20 and current_price > ma60:
        return "bull"
    return "range"

def backtest(start_date="2024-01-01", end_date="2026-03-31", initial_capital=1000000):
    etf_data = load_etf_data()
    all_dates = pd.date_range(start=start_date, end=end_date, freq="B")
    valid_dates = [d for d in all_dates if d in etf_data["510300"].index]
    
    equity = initial_capital
    cash = initial_capital * LEVERAGE
    positions = {}
    trade_history = []
    equity_curve = []
    
    for date in valid_dates:
        # 先处理持仓的止盈止损
        to_sell = []
        for code, pos in positions.items():
            if date not in etf_data[code].index: continue
            current_price = etf_data[code]["open"].loc[date]
            return_pct = (current_price - pos["entry_price"]) / pos["entry_price"]
            hold_days = (date - pos["entry_date"]).days
            
            if return_pct >= TAKE_PROFIT_PCT or return_pct <= -STOP_LOSS_PCT or hold_days >= MAX_HOLD_DAYS:
                sell_amount = pos["quantity"] * current_price
                cash += sell_amount
                trade_history.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "code": code,
                    "direction": "sell",
                    "price": current_price,
                    "quantity": pos["quantity"],
                    "profit": (current_price - pos["entry_price"]) * pos["quantity"]
                })
                to_sell.append(code)
        for code in to_sell:
            del positions[code]
        
        # 市场下跌时空仓，不买入
        regime = get_market_regime(date, etf_data)
        if regime == "down" and len(positions) == 0:
            position_value = sum([etf_data[code]["close"].loc[date] * pos["quantity"] for code, pos in positions.items() if date in etf_data[code].index])
            net_equity = (cash + position_value) - (initial_capital * (LEVERAGE - 1))
            equity_curve.append({
                "date": date.strftime("%Y-%m-%d"),
                "equity": net_equity,
                "cash": cash,
                "position_count": len(positions),
                "regime": regime
            })
            continue
        
        # 计算所有ETF的动量得分，选最强的
        momentum_scores = []
        for code, df in etf_data.items():
            if date not in df.index: continue
            score = calculate_momentum_score(df.loc[:date])
            if score > MIN_RANK_THRESHOLD:
                momentum_scores.append((code, score))
        momentum_scores.sort(key=lambda x: x[1], reverse=True)
        top_codes = [x[0] for x in momentum_scores[:TOP_N_ETF]]
        
        # 分配资金买入
        if len(top_codes) and len(positions) < TOP_N_ETF:
            available_cash = cash / (TOP_N_ETF - len(positions))
            for code in top_codes:
                if code in positions: continue
                if date not in etf_data[code].index: continue
                current_price = etf_data[code]["open"].loc[date]
                quantity = int(available_cash / current_price / 100) * 100
                if quantity >= 100:
                    positions[code] = {
                        "entry_price": current_price,
                        "entry_date": date,
                        "quantity": quantity
                    }
                    cash -= quantity * current_price
                    trade_history.append({
                        "date": date.strftime("%Y-%m-%d"),
                        "code": code,
                        "direction": "buy",
                        "price": current_price,
                        "quantity": quantity,
                        "profit": 0
                    })
        
        # 计算当前净值
        position_value = 0
        for code, pos in positions.items():
            if date in etf_data[code].index:
                position_value += etf_data[code]["close"].loc[date] * pos["quantity"]
        net_equity = (cash + position_value) - (initial_capital * (LEVERAGE - 1))
        equity = max(net_equity, 0.0)
        equity_curve.append({
            "date": date.strftime("%Y-%m-%d"),
            "equity": equity,
            "cash": cash,
            "position_value": position_value,
            "position_count": len(positions),
            "regime": regime
        })
    
    # 计算收益指标
    df_equity = pd.DataFrame(equity_curve)
    df_equity["date"] = pd.to_datetime(df_equity["date"])
    df_equity.set_index("date", inplace=True)
    
    total_return = (df_equity["equity"].iloc[-1] / initial_capital) - 1
    years = (df_equity.index[-1] - df_equity.index[0]).days / 365
    annualized_return = (1 + total_return) ** (1 / years) - 1
    
    # 计算最大回撤
    peak = df_equity["equity"].cummax()
    drawdown = (df_equity["equity"] - peak) / peak
    max_drawdown = drawdown.min()
    
    # 计算月度收益
    monthly_returns = df_equity["equity"].resample("M").last().pct_change().dropna()
    months_above_30pct = (monthly_returns >= 0.3).sum()
    month_30pct_ratio = months_above_30pct / len(monthly_returns)
    sell_trades = [t for t in trade_history if t["direction"] == "sell"]
    win_count = sum([t["profit"] > 0 for t in sell_trades])
    win_rate = win_count / max(1, len(sell_trades))
    
    results = {
        "metrics": {
            "initial_capital": initial_capital,
            "final_equity": df_equity["equity"].iloc[-1],
            "total_return": total_return,
            "annualized_return": annualized_return,
            "max_drawdown": abs(max_drawdown),
            "sharpe_ratio": (annualized_return - 0.03) / (df_equity["equity"].pct_change().std() * np.sqrt(252)) if annualized_return > 0 else 0,
            "monthly_30_percent_ratio": month_30pct_ratio,
            "total_trades": len(trade_history),
            "win_rate": win_rate
        },
        "equity_curve": [{"date": d.strftime("%Y-%m-%d"), "value": e} for d, e in df_equity["equity"].items()],
        "drawdown_curve": [{"date": d.strftime("%Y-%m-%d"), "value": abs(dd)} for d, dd in drawdown.items()],
        "monthly_returns": [{"month": idx.strftime("%Y-%m"), "value": r} for idx, r in monthly_returns.items()],
        "trades": trade_history
    }
    
    output_path = REPORT_DIR / "aggressive_leverage_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("=== Aggressive Leverage ETF Strategy Results ===")
    print(f"回测区间: {df_equity.index[0].strftime('%Y-%m-%d')} ~ {df_equity.index[-1].strftime('%Y-%m-%d')}")
    print(f"初始资金: {initial_capital:,.2f}")
    print(f"最终资金: {df_equity['equity'].iloc[-1]:,.2f}")
    print(f"累计收益: {total_return*100:.2f}%")
    print(f"年化收益: {annualized_return*100:.2f}%")
    print(f"最大回撤: {abs(max_drawdown)*100:.2f}%")
    print(f"月收益≥30%占比: {month_30pct_ratio*100:.2f}% ({months_above_30pct}/{len(monthly_returns)})")
    print(f"胜率: {win_rate*100:.2f}%")
    print(f"总交易次数: {len(trade_history)}")
    print(f"结果已保存到: {output_path}")
    
    return results

if __name__ == "__main__":
    backtest()
