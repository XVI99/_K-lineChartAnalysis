# 使用新浪HTTP API批量探测所有ETF
import requests
import json
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

def _get_no_proxy_session():
    session = requests.Session()
    session.trust_env = False
    session.proxies = {"http": None, "https": None}
    return session

def test_batch(batch_codes):
    """测试一批代码，返回有效的ETF"""
    session = _get_no_proxy_session()
    
    # 转换代码格式
    tx_codes = []
    for code in batch_codes:
        if code.startswith("5"):
            tx_codes.append(f"sh{code}")
        else:
            tx_codes.append(f"sz{code}")
    
    url = f"http://hq.sinajs.cn/list={','.join(tx_codes)}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://finance.sina.com.cn"
    }
    
    valid_etfs = []
    
    try:
        resp = session.get(url, headers=headers, timeout=15)
        if resp.status_code == 200:
            # 解析新浪返回的数据: var hq_str_sh510050="名称,当前价,..."
            lines = resp.text.strip().split('\n')
            for line in lines:
                # 格式: var hq_str_sh510050="上证50ETF华夏,2.988,..."
                match = re.search(r'hq_str_(sh\d{6}|sz\d{6})="([^"]*)"', line)
                if match:
                    code_full = match.group(1)
                    data_str = match.group(2)
                    code = code_full[2:]  # 去掉sh/sz前缀
                    
                    if data_str and len(data_str) > 5:
                        parts = data_str.split(',')
                        if parts and parts[0]:
                            name = parts[0].strip()
                            valid_etfs.append({"code": code, "name": name})
    except Exception as e:
        pass
    
    return valid_etfs

def main():
    print("=" * 60)
    print("全量ETF扫描器 (新浪HTTP API)")
    print("=" * 60)
    
    # 生成所有可能的ETF代码
    print("\n生成候选ETF代码...")
    
    # 上海ETF: 510000-519999
    sh_codes = [f"5{str(i).zfill(4)}" for i in range(10000, 20000)]
    
    # 深圳ETF: 159000-159999
    sz_codes = [f"159{str(i).zfill(3)}" for i in range(1000)]
    
    all_codes = sh_codes + sz_codes
    
    print(f"上海候选: {len(sh_codes)} 个")
    print(f"深圳候选: {len(sz_codes)} 个")
    print(f"总计候选: {len(all_codes)} 个")
    print()
    
    # 分批处理
    batch_size = 50
    all_etfs = []
    processed = 0
    
    print("开始扫描...")
    start_time = time.time()
    
    # 使用线程池加速
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        
        for i in range(0, len(all_codes), batch_size):
            batch = all_codes[i:i+batch_size]
            futures.append(executor.submit(test_batch, batch))
        
        for future in as_completed(futures):
            try:
                result = future.result()
                all_etfs.extend(result)
                processed += batch_size
                
                if processed % 1000 == 0:
                    elapsed = time.time() - start_time
                    rate = processed / elapsed if elapsed > 0 else 0
                    print(f"  已处理: {processed}/{len(all_codes)} ({rate:.1f}/s) - 发现: {len(all_etfs)} 只")
            except Exception as e:
                pass
    
    # 去重
    seen = set()
    unique_etfs = []
    for etf in all_etfs:
        if etf['code'] not in seen:
            seen.add(etf['code'])
            unique_etfs.append(etf)
    
    elapsed = time.time() - start_time
    
    print()
    print("=" * 60)
    print(f"扫描完成!")
    print(f"耗时: {elapsed:.1f} 秒")
    print(f"发现有效ETF: {len(unique_etfs)} 只")
    print("=" * 60)
    
    # 按代码排序
    unique_etfs.sort(key=lambda x: x['code'])
    
    # 打印前50个
    print("\n前50个ETF:")
    for i, etf in enumerate(unique_etfs[:50]):
        print(f"  {i+1:3d}. {etf['code']} - {etf['name']}")
    
    # 统计代码分布
    print("\nETF代码分布:")
    prefix_count = {}
    for etf in unique_etfs:
        code = etf['code']
        if code.startswith('510'):
            prefix = '510'
        elif code.startswith('511'):
            prefix = '511'
        elif code.startswith('512'):
            prefix = '512'
        elif code.startswith('513'):
            prefix = '513'
        elif code.startswith('515'):
            prefix = '515'
        elif code.startswith('159'):
            prefix = '159'
        else:
            prefix = code[:3]
        prefix_count[prefix] = prefix_count.get(prefix, 0) + 1
    
    for prefix in sorted(prefix_count.keys()):
        print(f"  {prefix}**: {prefix_count[prefix]} 只")
    
    # 保存结果
    result_file = r"F:\_K-lineChartAnalysis\AStockQuant\all_etf_list.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(unique_etfs, f, ensure_ascii=False, indent=2)
    
    print(f"\nETF列表已保存到: {result_file}")

if __name__ == "__main__":
    main()