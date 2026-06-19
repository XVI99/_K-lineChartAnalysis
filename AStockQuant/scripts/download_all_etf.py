# 下载所有ETF历史数据
import requests
import json
import time
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

DATA_DIR = r"F:\_K-lineChartAnalysis\AStockQuant\data_cache"

def _get_no_proxy_session():
    session = requests.Session()
    session.trust_env = False
    session.proxies = {"http": None, "https": None}
    return session

def fetch_etf_data(symbol, days=500):
    """从腾讯API获取ETF历史数据"""
    session = _get_no_proxy_session()
    
    # 转换代码格式
    if symbol.startswith("5"):
        tx_code = f"sh{symbol}"
    else:
        tx_code = f"sz{symbol}"
    
    url = f"http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={tx_code},day,,,{days},qfq"
    
    try:
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        
        if "data" not in data or tx_code not in data["data"]:
            return None
        
        stk = data["data"][tx_code]
        buf = stk.get("qfqday") or stk.get("day", [])
        
        if not buf or len(buf) < 10:
            return None
        
        # 解析数据
        rows = []
        for row in buf:
            if len(row) >= 6:
                try:
                    rows.append({
                        "date": row[0],
                        "open": float(row[1]),
                        "close": float(row[2]),
                        "high": float(row[3]),
                        "low": float(row[4]),
                        "volume": float(row[5]) * 100
                    })
                except:
                    continue
        
        if len(rows) < 10:
            return None
        
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        df = df[["open", "high", "low", "close", "volume"]]
        
        return df
        
    except Exception as e:
        return None

def download_etf(symbol):
    """下载单个ETF数据"""
    df = fetch_etf_data(symbol, days=500)
    
    if df is not None and not df.empty:
        filepath = os.path.join(DATA_DIR, f"{symbol}.csv")
        df.index = df.index.strftime("%Y-%m-%d")
        df.to_csv(filepath, index=True, index_label="date", encoding="utf-8")
        return symbol, len(df)
    
    return symbol, None

def main():
    # 加载ETF列表
    with open(r"F:\_K-lineChartAnalysis\AStockQuant\all_etf_list.json", "r", encoding="utf-8") as f:
        etf_list = json.load(f)
    
    print("=" * 60)
    print("全量ETF数据下载器")
    print("=" * 60)
    print(f"数据目录: {DATA_DIR}")
    print(f"ETF数量: {len(etf_list)}")
    print()
    
    # 确保目录存在
    os.makedirs(DATA_DIR, exist_ok=True)
    
    success_count = 0
    failed_symbols = []
    results = {}
    
    print("开始下载...")
    start_time = time.time()
    
    # 使用线程池加速
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(download_etf, etf["code"]): etf["code"] for etf in etf_list}
        
        completed = 0
        for future in as_completed(futures):
            try:
                symbol, data_len = future.result()
                if data_len is not None:
                    success_count += 1
                    results[symbol] = data_len
                else:
                    failed_symbols.append(symbol)
                
                completed += 1
                
                if completed % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    remaining = len(etf_list) - completed
                    eta = remaining / rate if rate > 0 else 0
                    print(f"  进度: {completed}/{len(etf_list)} ({rate:.1f}/s) - 成功: {success_count} - 预计剩余: {eta:.0f}s")
                    
            except Exception as e:
                failed_symbols.append(futures[future])
                completed += 1
    
    elapsed = time.time() - start_time
    
    print()
    print("=" * 60)
    print(f"下载完成!")
    print(f"耗时: {elapsed:.1f} 秒")
    print(f"成功: {success_count} 只")
    print(f"失败: {len(failed_symbols)} 只")
    print("=" * 60)
    
    # 打印成功的前20个
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    print("\n数据量最多的20个ETF:")
    for symbol, count in sorted_results[:20]:
        print(f"  {symbol}: {count} 条")
    
    # 保存下载结果
    result = {
        "download_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total": len(etf_list),
        "success": success_count,
        "failed_count": len(failed_symbols),
        "failed": failed_symbols[:100],  # 只保存前100个失败记录
        "results": results
    }
    
    result_file = os.path.join(DATA_DIR, "all_download_result.json")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到: {result_file}")

if __name__ == "__main__":
    main()