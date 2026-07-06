# -*- coding: utf-8 -*-
"""
web_app.py — AStockQuant 量化交易网页端

基于 Flask 的 Web 应用，提供：
- 市场扫描结果展示
- 回测结果可视化
- 每日信号推送
- 持仓管理
- 8层因子雷达图

用法:
    cd AStockQuant
    python web_app.py --port 5000
    # 浏览器访问 http://localhost:5000
"""

from __future__ import annotations

import os
import sys
import json
import glob
import argparse
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
from flask import Flask, jsonify, render_template, send_from_directory, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)

app = Flask(__name__, template_folder="templates", static_folder="static")

HOLDINGS_FILE = os.path.join(BASE_DIR, "data_cache", "holdings", "holdings.json")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")


# ==================== 持仓数据管理 ====================

def load_holdings():
    if os.path.exists(HOLDINGS_FILE):
        with open(HOLDINGS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"current_positions": [], "trade_history": [], "daily_snapshots": [], "config": {"initial_capital": 100000}}

def save_holdings(data):
    os.makedirs(os.path.dirname(HOLDINGS_FILE), exist_ok=True)
    with open(HOLDINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ==================== 技术指标计算 ====================

def calc_indicators(df):
    close = df["close"]
    r = {}
    r["ma5"] = close.rolling(5).mean().tolist()
    r["ma10"] = close.rolling(10).mean().tolist()
    r["ma20"] = close.rolling(20).mean().tolist()
    r["ma60"] = close.rolling(60).mean().tolist()
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, 0.001)
    r["rsi"] = (100 - (100 / (1 + rs))).tolist()
    e12 = close.ewm(span=12).mean()
    e26 = close.ewm(span=26).mean()
    dif = e12 - e26
    dea = dif.ewm(span=9).mean()
    r["macd_dif"] = dif.tolist()
    r["macd_dea"] = dea.tolist()
    r["macd_hist"] = ((dif - dea) * 2).tolist()
    return r


def sell_advice(pos):
    """根据技术指标和持仓信息给出卖出建议"""
    etf = get_etf_data(pos["symbol"])
    prices = etf.get("prices", [])
    if not prices:
        return {"action": "HOLD", "reason": "无价格数据", "indicators": {}}
    close = pd.Series([float(x) if x else 0 for x in prices])
    if len(close) < 20:
        return {"action": "HOLD", "reason": "数据不足", "indicators": {}}
    ind = calc_indicators(pd.DataFrame({"close": close}))
    rsi = ind["rsi"][-1] if ind["rsi"] and not np.isnan(ind["rsi"][-1]) else 50
    ma5 = ind["ma5"][-1] if ind["ma5"] and not np.isnan(ind["ma5"][-1]) else 0
    ma20 = ind["ma20"][-1] if ind["ma20"] and not np.isnan(ind["ma20"][-1]) else 0
    cur = float(close.iloc[-1])
    buy = pos.get("buy_price", 0)
    sl = pos.get("stop_loss")
    tp = pos.get("take_profit")
    pct = (cur / buy - 1) if buy > 0 else 0
    reasons = []
    action = "HOLD"
    if sl and cur <= sl:
        action = "SELL"
        reasons.append(f"触发止损线({sl})")
    if tp and cur >= tp:
        action = "SELL"
        reasons.append(f"触发止盈线({tp})")
    if rsi > 70:
        if action != "SELL":
            action = "REDUCE"
        reasons.append(f"RSI超买({rsi:.1f})")
    if ma5 < ma20 and ma5 > 0:
        if action == "HOLD":
            action = "REDUCE"
        reasons.append("MA5下穿MA20")
    dif = ind["macd_dif"][-1] if ind["macd_dif"] else 0
    dea = ind["macd_dea"][-1] if ind["macd_dea"] else 0
    if dif < dea and dif > 0:
        if action == "HOLD":
            action = "REDUCE"
        reasons.append("MACD死叉")
    if pct > 0.2 and rsi > 65:
        if action == "HOLD":
            action = "REDUCE"
        reasons.append(f"盈利{pct*100:.1f}%,建议止盈一半")
    if not reasons:
        reasons.append("各项指标正常,继续持有")
    return {
        "action": action,
        "reason": "; ".join(reasons),
        "indicators": {
            "rsi": round(rsi, 1),
            "ma5": round(ma5, 3),
            "ma20": round(ma20, 3),
            "macd_dif": round(dif, 4),
            "macd_dea": round(dea, 4),
            "profit_pct": round(pct * 100, 2),
            "current_price": cur,
        }
    }


def get_scan_results() -> List[Dict]:
    scan_files = sorted(glob.glob(os.path.join(BASE_DIR, "scan_results_*.csv")), reverse=True)
    if not scan_files:
        return []
    df = pd.read_csv(scan_files[0], encoding="utf-8-sig")
    return df.to_dict("records")


def get_backtest_results() -> Dict:
    result_file = os.path.join(BASE_DIR, "reports", "screener_backtest_results.json")
    if os.path.exists(result_file):
        with open(result_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def get_daily_signals() -> List[Dict]:
    signal_dir = os.path.join(BASE_DIR, "signals")
    files = sorted(glob.glob(os.path.join(signal_dir, "*_alerts.json")), reverse=True)
    if not files:
        return []
    with open(files[0], "r", encoding="utf-8") as f:
        return json.load(f)


def get_etf_data(symbol: str) -> Dict:
    csv_path = os.path.join(BASE_DIR, "data_cache", f"{symbol}.csv")
    if not os.path.exists(csv_path):
        return {"symbol": symbol, "dates": [], "prices": [], "volumes": []}

    df = pd.read_csv(csv_path)
    cols_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl in ("date", "日期", "trade_date"):
            cols_map[c] = "date"
        elif cl in ("close", "收盘价", "close_price"):
            cols_map[c] = "close"
        elif cl in ("open", "开盘价"):
            cols_map[c] = "open"
        elif cl in ("high", "最高价"):
            cols_map[c] = "high"
        elif cl in ("low", "最低价"):
            cols_map[c] = "low"
        elif cl in ("volume", "成交量"):
            cols_map[c] = "volume"
    df = df.rename(columns=cols_map)

    if "date" not in df.columns:
        df["date"] = range(len(df))
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.tail(250)

    return {
        "symbol": symbol,
        "dates": df["date"].dt.strftime("%Y-%m-%d").tolist() if "date" in df else [],
        "prices": df["close"].tolist() if "close" in df else [],
        "volumes": df["volume"].tolist() if "volume" in df else [],
        "opens": df["open"].tolist() if "open" in df else [],
        "highs": df["high"].tolist() if "high" in df else [],
        "lows": df["low"].tolist() if "low" in df else [],
    }


def get_etf_list() -> List[str]:
    cache_dir = os.path.join(BASE_DIR, "data_cache")
    symbols = sorted([
        f.replace(".csv", "") for f in os.listdir(cache_dir)
        if f.endswith(".csv") and f.replace(".csv", "").isdigit()
    ])
    return symbols


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/scan")
def api_scan():
    results = get_scan_results()
    return jsonify({"data": results, "count": len(results)})


@app.route("/api/backtest")
def api_backtest():
    results = get_backtest_results()
    return jsonify(results)


@app.route("/api/signals")
def api_signals():
    signals = get_daily_signals()
    return jsonify({"data": signals, "count": len(signals)})


@app.route("/api/etf/<symbol>")
def api_etf_data(symbol):
    data = get_etf_data(symbol)
    return jsonify(data)


@app.route("/api/etf_list")
def api_etf_list():
    symbols = get_etf_list()
    return jsonify({"symbols": symbols, "count": len(symbols)})


@app.route("/api/profitability")
def api_profitability():
    report_file = os.path.join(BASE_DIR, "PROFITABILITY_REPORT.md")
    if os.path.exists(report_file):
        with open(report_file, "r", encoding="utf-8") as f:
            return jsonify({"content": f.read()})
    return jsonify({"content": ""})


@app.route("/api/health")
def api_health():
    return jsonify({
        "status": "ok",
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "version": "1.0.0"
    })


@app.route("/api/scan_live")
def api_scan_live():
    results = get_scan_results()
    if not results:
        return jsonify({"buy_signals": [], "total_scanned": 0, "scan_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
    buys = [r for r in results if str(r.get("signal", "")).startswith("BUY") or str(r.get("signal", "")) == "STRONG_BUY"]
    return jsonify({
        "buy_signals": buys[:10],
        "total_scanned": len(results),
        "scan_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })



# ==================== 回测扩展 API ====================

@app.route("/api/backtest/equity")
def api_bt_equity():
    p = os.path.join(REPORTS_DIR, "reports_quick_test__equity.csv")
    if os.path.exists(p):
        df = pd.read_csv(p)
        return jsonify({"dates": df.iloc[:, 0].tolist(), "values": df.iloc[:, 1].tolist()})
    return jsonify({"dates": [], "values": []})

@app.route("/api/backtest/trades")
def api_bt_trades():
    p = os.path.join(REPORTS_DIR, "reports_quick_test__trades.csv")
    if os.path.exists(p):
        df = pd.read_csv(p)
        return jsonify({"data": df.to_dict("records")})
    return jsonify({"data": []})

@app.route("/api/backtest/folds")
def api_bt_folds():
    p = os.path.join(REPORTS_DIR, "reports_quick_test__folds.csv")
    if os.path.exists(p):
        df = pd.read_csv(p)
        return jsonify({"data": df.to_dict("records")})
    return jsonify({"data": []})

# ==================== 组合与因子 API ====================

@app.route("/api/portfolio/signals")
def api_portfolio_signals():
    p = os.path.join(REPORTS_DIR, "portfolio", "rebalance_signals.csv")
    if os.path.exists(p):
        df = pd.read_csv(p)
        ld = df["date"].iloc[-1] if "date" in df.columns else ""
        latest = df[df["date"] == ld] if ld else df.tail(20)
        return jsonify({"data": latest.to_dict("records"), "total": len(df), "latest_date": str(ld)})
    return jsonify({"data": [], "total": 0, "latest_date": ""})

@app.route("/api/factors/ranking")
def api_factors_ranking():
    p = os.path.join(REPORTS_DIR, "factor_evaluation", "factor_ranking.csv")
    if os.path.exists(p):
        df = pd.read_csv(p)
        return jsonify({"data": df.to_dict("records")})
    return jsonify({"data": []})

@app.route("/api/etf/<symbol>/indicators")
def api_etf_indicators(symbol):
    d = get_etf_data(symbol)
    if not d["prices"]:
        return jsonify({})
    close = pd.Series([float(x) if x else 0 for x in d["prices"]])
    ind = calc_indicators(pd.DataFrame({"close": close}))
    ind["dates"] = d["dates"]
    return jsonify(ind)

# ==================== 持仓管理 API ====================

@app.route("/api/holdings/current")
def api_holdings_current():
    data = load_holdings()
    positions = data.get("current_positions", [])
    result = []
    for pos in positions:
        etf = get_etf_data(pos["symbol"])
        cur_price = float(etf["prices"][-1]) if etf["prices"] else pos.get("buy_price", 0)
        buy_price = pos.get("buy_price", 0)
        qty = pos.get("quantity", 0)
        cost = buy_price * qty
        value = cur_price * qty
        profit = value - cost
        detail = dict(pos)
        detail["current_price"] = round(cur_price, 3)
        detail["cost"] = round(cost, 2)
        detail["value"] = round(value, 2)
        detail["profit"] = round(profit, 2)
        detail["profit_pct"] = round(profit / cost * 100, 2) if cost > 0 else 0
        if pos.get("stop_loss"):
            detail["stop_loss_pct"] = round((cur_price / pos["stop_loss"] - 1) * 100, 2)
        result.append(detail)
    return jsonify({"data": result, "count": len(result)})

@app.route("/api/holdings/trades")
def api_holdings_trades():
    data = load_holdings()
    return jsonify({"data": data.get("trade_history", []), "count": len(data.get("trade_history", []))})

@app.route("/api/holdings/summary")
def api_holdings_summary():
    data = load_holdings()
    positions = data.get("current_positions", [])
    total_cost = 0
    total_value = 0
    for pos in positions:
        etf = get_etf_data(pos["symbol"])
        cur_price = float(etf["prices"][-1]) if etf["prices"] else pos.get("buy_price", 0)
        total_cost += pos.get("buy_price", 0) * pos.get("quantity", 0)
        total_value += cur_price * pos.get("quantity", 0)
    init_cap = data.get("config", {}).get("initial_capital", 100000)
    return jsonify({
        "count": len(positions),
        "total_cost": round(total_cost, 2),
        "total_value": round(total_value, 2),
        "total_profit": round(total_value - total_cost, 2),
        "profit_pct": round((total_value / total_cost - 1) * 100, 2) if total_cost > 0 else 0,
        "cash": round(init_cap - total_cost, 2),
        "total_account": round(init_cap + total_value - total_cost, 2),
    })

@app.route("/api/holdings/trade", methods=["POST"])
def api_holdings_trade():
    data = load_holdings()
    req = request.json
    symbol = req.get("symbol", "")
    action = req.get("action", "buy")
    price = float(req.get("price", 0))
    quantity = int(req.get("quantity", 0))
    date = req.get("date", datetime.now().strftime("%Y-%m-%d"))
    stop_loss = float(req.get("stop_loss", 0)) if req.get("stop_loss") else None
    take_profit = float(req.get("take_profit", 0)) if req.get("take_profit") else None
    reason = req.get("reason", "")
    if not symbol or not price or not quantity:
        return jsonify({"error": "Missing required fields"}), 400
    trade = {"date": date, "symbol": symbol, "action": action, "price": price,
             "quantity": quantity, "amount": round(price * quantity, 2), "reason": reason}
    data["trade_history"].append(trade)
    positions = data["current_positions"]
    if action == "buy":
        existing = [p for p in positions if p["symbol"] == symbol]
        if existing:
            p = existing[0]
            total_qty = p["quantity"] + quantity
            p["buy_price"] = round((p["buy_price"] * p["quantity"] + price * quantity) / total_qty, 3)
            p["quantity"] = total_qty
            if stop_loss:
                p["stop_loss"] = stop_loss
            if take_profit:
                p["take_profit"] = take_profit
        else:
            positions.append({"symbol": symbol, "buy_date": date, "buy_price": price,
                              "quantity": quantity, "stop_loss": stop_loss,
                              "take_profit": take_profit, "notes": reason})
    elif action == "sell":
        existing = [p for p in positions if p["symbol"] == symbol]
        if existing:
            p = existing[0]
            if quantity >= p["quantity"]:
                positions.remove(p)
            else:
                p["quantity"] -= quantity
    save_holdings(data)
    return jsonify({"status": "ok", "trade": trade, "message": "交易记录成功"})

@app.route("/api/holdings/detail/<symbol>")
def api_holdings_detail(symbol):
    data = load_holdings()
    positions = [p for p in data.get("current_positions", []) if p["symbol"] == symbol]
    if not positions:
        return jsonify({"error": "持仓不存在"}), 404
    pos = positions[0]
    etf_data = get_etf_data(symbol)
    advice = sell_advice(pos)
    trades = [t for t in data.get("trade_history", []) if t["symbol"] == symbol]
    return jsonify({"position": pos, "etf_data": etf_data, "advice": advice, "trades": trades})


def main():
    parser = argparse.ArgumentParser(description="AStockQuant Web App")
    parser.add_argument("--port", type=int, default=5000, help="端口号")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="绑定地址")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"AStockQuant 量化交易网页端")
    print(f"{'='*60}")
    print(f"访问地址: http://localhost:{args.port}")
    print(f"ETF数量: {len(get_etf_list())}")
    print(f"{'='*60}\n")

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()