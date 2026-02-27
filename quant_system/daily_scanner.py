import argparse
import pandas as pd
import akshare as ak
import datetime
import os
import sys

# Ensure custom modules can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from optimized_strategy import OptimizedStrategy, validate_stock_code, INITIAL_CAPITAL
from custom_data import get_price

def get_all_valid_stocks(max_price=45.0):
    """
    Fetch all A-share stocks and filter for valid prefixes and price.
    """
    print("[1] 获取全市场A股实时数据...")
    try:
        df_spot = ak.stock_zh_a_spot_em()
    except Exception as e:
        print(f"获取实时数据失败: {e}")
        return pd.DataFrame()

    print(f"    获取到 {len(df_spot)} 只股票。")
    print("[2] 过滤股票池 (仅保留沪深主板 & 价格合适)...")
    
    valid_stocks = []
    
    # 假设列名: "代码", "名称", "最新价"
    for _, row in df_spot.iterrows():
        code = str(row['代码']).zfill(6)
        name = row['名称']
        price = row['最新价']
        
        # 忽略停牌或无价格数据
        if pd.isna(price) or price <= 0:
            continue
            
        # 资金限制：如果买一手(100股)超过 INITIAL_CAPITAL，或者超过设定的最高单价
        if price * 100 > INITIAL_CAPITAL * 0.95 or price > max_price:
            continue
            
        # 添加前缀
        if code.startswith(('600', '601', '603', '605')):
            symbol = f"sh{code}"
        elif code.startswith(('000', '001', '002', '003')):
            symbol = f"sz{code}"
        else:
            continue  # 忽略创业板、科创板等
            
        # 再次确认通过验证
        valid, _ = validate_stock_code(symbol)
        if valid:
            valid_stocks.append({
                'symbol': symbol,
                'name': name,
                'price': price
            })
            
    df_valid = pd.DataFrame(valid_stocks)
    print(f"    符合条件 (主板 + 现价<={max_price}元): {len(df_valid)} 只")
    return df_valid

import concurrent.futures
import warnings
from contextlib import redirect_stdout
from tqdm import tqdm

def process_single_stock(row_tuple, days):
    """
    处理单只股票的独立函数 (用于多进程)
    """
    i, row = row_tuple
    symbol = row['symbol']
    name = row['name']
    price = row['price']
    
    try:
        # 抑制所有警告和标准输出，避免刷屏
        with warnings.catch_warnings(), open(os.devnull, 'w') as f, redirect_stdout(f):
            warnings.simplefilter("ignore")
            strategy = OptimizedStrategy()
            
            df = get_price(symbol, count=days, frequency='1d')
            if df is None or df.empty or len(df) < 5:
                return None
                
            df = df.rename(columns={
                'open': 'Open', 'high': 'High', 'low': 'Low',
                'close': 'Close', 'volume': 'Volume'
            })
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            df_signals = strategy.apply_strategy(df, verbose=False)
            
            if len(df_signals) > 0:
                last_row = df_signals.iloc[-1]
                last_date = df_signals.index[-1].strftime('%Y-%m-%d')
                
                # 当前没有信号的话再看前一天，有时信号在昨天触发今天适合买入
                if last_row['Signal'] == 1:
                    regime_str = last_row.get('Regime', 'Unknown')
                    reason = last_row.get('Signal_Reason', '')
                    
                    return {
                        'Date': last_date,
                        'Symbol': symbol,
                        'Name': name,
                        'Price': price,
                        'Regime': regime_str,
                        'Reason': reason,
                        'Score': last_row.get('Aggregate_Score', 0)
                    }
    except Exception:
        pass
        
    return None

def scan_market(max_price=45.0, days=40):
    """
    扫描全市场，寻找最新的 BUY 信号 (多进程加速版)。
    """
    stocks_df = get_all_valid_stocks(max_price)
    if stocks_df.empty:
        print("没有找到符合条件的股票。")
        return
        
    total = len(stocks_df)
    print(f"\n[3] 启动多进程扫描 {total} 只股票 (获取最近{days}天数据)...")
    
    buy_recommendations = []
    
    # 使用进程池加速扫描
    max_workers = min(16, os.cpu_count() or 4)
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = {executor.submit(process_single_stock, row_tuple, days): row_tuple[1]['symbol'] 
                   for row_tuple in stocks_df.iterrows()}
        
        # 使用 tqdm 显示进度条
        for future in tqdm(concurrent.futures.as_completed(futures), total=total, desc="扫描进度"):
            result = future.result()
            if result:
                buy_recommendations.append(result)
                # 实时打印发现的信号 (带上擦除当前行避免破坏进度条)
                tqdm.write(f"    !!! 发现买入信号: {result['Name']} ({result['Symbol']}) - {result['Price']}元 - {result['Regime']}")
            
    print("\n" + "="*80)
    print("🎯 今日买入推荐汇总 (适合 5000元 微型账户)")
    print("="*80)
    
    if not buy_recommendations:
        print("目前没有任何符合我们全天候策略的买入信号。顶级交易员的耐心是等出来的，空仓也是一种策略！")
    else:
        # 按分数(Score)排序
        buy_recommendations.sort(key=lambda x: x['Score'], reverse=True)
        
        for i, rec in enumerate(buy_recommendations, 1):
            print(f"{i}. {rec['Name']} ({rec['Symbol']})")
            print(f"   现价: {rec['Price']:.2f}元 (买一手需 ~{rec['Price']*100:.0f}元)")
            print(f"   信号日期: {rec['Date']}")
            print(f"   市场环境: {rec['Regime']}")
            print(f"   买入理由: {rec['Reason']}")
            print("-" * 60)
            
        print("\n💡 操作建议:")
        print("1. 优选前 1-2 只买入，每次只买 1 手 (100股)。")
        print("2. 纪律: 买入后必须根据当前ATR或者市场环境设置止损单！")
        
if __name__ == "__main__":
    # 为了解决多进程在 Windows 下的冻结问题
    import multiprocessing
    multiprocessing.freeze_support()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_price', type=float, default=45.0, help='允许的最高股价 (默认: 45元)')
    parser.add_argument('--days', type=int, default=40, help='获取的历史数据天数 (默认: 40)')
    args = parser.parse_args()
    
    # 确保终端不会因为 print 大量数据报错
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    scan_market(max_price=args.max_price, days=args.days)
