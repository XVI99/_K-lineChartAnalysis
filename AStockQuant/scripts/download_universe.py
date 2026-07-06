# -*- coding: utf-8 -*-
"""
download_universe.py — 批量下载宽基+行业ETF池的2年历史数据

用法:
    python scripts/download_universe.py                  # 默认2年(730天)
    python scripts/download_universe.py --days 500       # 自定义天数
    python scripts/download_universe.py --broad-only     # 仅宽基
    python scripts/download_universe.py --industry-only  # 仅行业
    python scripts/download_universe.py --concurrent 5   # 并发数(默认3)

产出: data_cache/{symbol}.csv (自动增量更新，与本地数据合并)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# 确保项目根目录在 sys.path
PROJ_ROOT = Path(__file__).resolve().parent.parent
PARENT_ROOT = PROJ_ROOT.parent
if str(PARENT_ROOT) not in sys.path:
    sys.path.insert(0, str(PARENT_ROOT))
os.chdir(PROJ_ROOT)

from AStockQuant.core.config_loader import load_config
from AStockQuant.core.data_hub import DataHub

# ============================================================
# ETF 标的池（宽基 ~50 + 行业 ~50）
# ============================================================

BROAD_ETFS = {
    # === 沪深300系列 ===
    "510300": "沪深300ETF华泰柏瑞",
    "510310": "沪深300ETF易方达",
    "510330": "沪深300ETF华夏",
    "159919": "沪深300ETF嘉实",
    # === 中证500系列 ===
    "510500": "中证500ETF南方",
    "510510": "中证500ETF华泰柏瑞",
    "159922": "中证500ETF嘉实",
    "512500": "中证500ETF华夏",
    # === 中证1000系列 ===
    "512100": "中证1000ETF南方",
    "560010": "中证1000ETF摩根",
    "159845": "中证1000ETF汇添富",
    # === 中证A500系列 ===
    "159338": "中证A500ETF景顺",
    "159352": "中证A500ETF南方",
    "563360": "中证A500ETF华泰柏瑞",
    "159788": "中证A500ETF中欧",
    "159692": "中证A500ETF嘉实",
    # === 创业板系列 ===
    "159915": "创业板ETF易方达",
    "159952": "创业板ETF广发",
    "159949": "创业板50ETF华安",
    "159977": "创业板ETF天弘",
    # === 科创板系列 ===
    "588000": "科创50ETF华夏",
    "588050": "科创50ETF华泰柏瑞",
    "588090": "科创50ETF易方达",
    "588160": "科创100ETF华夏",
    "588200": "科创芯片ETF华安",
    # === 上证系列 ===
    "510050": "上证50ETF华夏",
    # === 深证系列 ===
    "159628": "深证100ETF银华",
    "159901": "深证100ETF易方达",
    "159975": "深证成指ETF工银",
    # === 红利系列 ===
    "510880": "红利ETF华泰柏瑞",
    "515080": "中证红利ETF招商",
    "159339": "红利低波ETF景顺",
    "512890": "红利低波100ETF华泰柏瑞",
    # === 跨境ETF ===
    "513100": "纳指ETF国泰",
    "513500": "标普500ETF博时",
    "159941": "纳指ETF广发",
    "513050": "中概互联ETF易方达",
    "164906": "中国互联ETF交银",
    "159920": "恒生ETF华夏",
    "510900": "H股ETF易方达",
    "513060": "恒生医疗ETF华夏",
    "159940": "恒生科技ETF华泰柏瑞",
    "513130": "恒生科技ETF华安",
    "159509": "纳指科技ETF华夏",
    "513520": "日经ETF华夏",
    "159866": "日经ETF工银",
    "513030": "德国ETF华安",
    "159612": "法国ETF华安",
    # === 主题宽基 ===
    "562500": "中证2000ETF华泰柏瑞",
    "159531": "中证2000ETF华安",
}

INDUSTRY_ETFS = {
    # === 科技/半导体 ===
    "512760": "芯片ETF国泰",
    "512480": "半导体ETF国联安",
    "159801": "芯片ETF广发",
    "159995": "芯片ETF华夏",
    "515050": "5G通信ETF华夏",
    "515070": "人工智能ETF平安",
    "515980": "人工智能ETF华富",
    "515000": "科技ETF华宝",
    "159732": "消费电子ETF国泰",
    "159998": "计算机ETF华夏",
    # === 消费 ===
    "512690": "酒ETF鹏华",
    "159928": "消费ETF汇添富",
    "510150": "消费ETF招商",
    "512980": "传媒ETF鹏华",
    "159869": "游戏ETF华夏",
    "516160": "零售ETF华夏",
    "159825": "农业ETF华夏",
    # === 金融 ===
    "512000": "券商ETF华宝",
    "512880": "证券ETF华泰柏瑞",
    "512070": "保险ETF易方达",
    "512800": "银行ETF华宝",
    "512730": "银行ETF鹏华",
    # === 新能源/碳中和 ===
    "515030": "新能源车ETF华夏",
    "516160": "新能源ETF华夏",
    "159875": "新能源ETF鹏华",
    "562890": "光伏ETF华泰柏瑞",
    "515790": "光伏ETF天弘",
    "159611": "电力ETF鹏华",
    "159811": "电池ETF华夏",
    "159615": "新材料ETF华泰柏瑞",
    # === 医药/医疗 ===
    "512170": "医疗ETF华宝",
    "512010": "医药ETF易方达",
    "159938": "医药ETF广发",
    "513060": "恒生医疗ETF华夏",
    "159828": "生物医药ETF华宝",
    # === 军工/国防 ===
    "512660": "军工ETF国泰",
    "512680": "军工ETF华宝",
    "512670": "国防ETF鹏华",
    # === 资源/周期 ===
    "515220": "煤炭ETF国泰",
    "512400": "有色金属ETF南方",
    "515210": "钢铁ETF华宝",
    "562900": "稀土ETF华泰柏瑞",
    "159871": "有色金属ETF嘉实",
    # === 房地产/基建 ===
    "512200": "房地产ETF南方",
    "159767": "建材ETF华夏",
    # === 公用事业 ===
    "159611": "电力ETF鹏华",
    "561560": "水利ETF华泰柏瑞",
    # === 通信/传媒 ===
    "512220": "通信ETF国泰",
    "515000": "科技ETF华宝",
    # === 大宗商品(黄金) ===
    "518880": "黄金ETF华安",
    "159934": "黄金ETF易方达",
    "159937": "黄金ETF博时",
    # === 债券 ===
    "511010": "国债ETF国泰",
    "511260": "十年国债ETF海富通",
    "511220": "城投债ETF海富通",
    "511990": "华宝现金添益",
    # === 货币 ===
    "511920": "广发货币ETF",
}


def download_one(hub: DataHub, symbol: str, name: str, days: int) -> dict:
    """下载单个ETF数据（利用 DataHub 自动落盘）"""
    t0 = time.time()
    try:
        df = hub.get_stock_data_ex(symbol, days=days)
        elapsed = time.time() - t0
        if df.empty:
            return {"symbol": symbol, "name": name, "rows": 0, "elapsed": elapsed, "status": "EMPTY"}
        return {
            "symbol": symbol,
            "name": name,
            "rows": len(df),
            "start": str(df.index[0].date()),
            "end": str(df.index[-1].date()),
            "elapsed": elapsed,
            "status": "OK",
        }
    except Exception as e:
        return {
            "symbol": symbol,
            "name": name,
            "rows": 0,
            "elapsed": time.time() - t0,
            "status": f"ERROR: {type(e).__name__}: {e}",
        }


def main():
    parser = argparse.ArgumentParser(description="批量下载宽基+行业ETF历史数据")
    parser.add_argument("--days", type=int, default=730, help="历史天数（默认730=2年）")
    parser.add_argument("--broad-only", action="store_true", help="仅下载宽基ETF")
    parser.add_argument("--industry-only", action="store_true", help="仅下载行业ETF")
    parser.add_argument("--concurrent", type=int, default=3, help="并发数（默认3，避免API限流）")
    args = parser.parse_args()

    # 构建标的池
    universe = {}
    if not args.industry_only:
        universe.update(BROAD_ETFS)
    if not args.broad_only:
        # 去重：行业池中与宽基池重复的跳过
        for code, name in INDUSTRY_ETFS.items():
            if code not in universe:
                universe[code] = name

    print("=" * 70)
    print(f"AStockQuant ETF Universe Batch Download")
    print("=" * 70)
    print(f"  标的数量: {len(universe)} 只")
    print(f"  历史天数: {args.days} 天 ({args.days/365:.1f} 年)")
    print(f"  并发数:   {args.concurrent}")
    print(f"  落盘目录: {PROJ_ROOT / 'data_cache'}")
    print(f"  开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 初始化 DataHub
    cfg = load_config()
    dsc = cfg.get_data_source_config()

    # 并发下载时每个线程需要独立的 DataHub 实例（避免 socket 竞争）
    # 但 Redis 缓存是共享的，本地 CSV 写入需要加锁
    import threading
    csv_lock = threading.Lock()

    class LockedDataHub(DataHub):
        def _save_local_csv(self, symbol: str, df) -> None:
            with csv_lock:
                super()._save_local_csv(symbol, df)

    # 分批处理
    symbols = list(universe.items())
    results = []
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=args.concurrent) as executor:
        futures = {}
        for code, name in symbols:
            hub = LockedDataHub(data_source_config=dsc)
            future = executor.submit(download_one, hub, code, name, args.days)
            futures[future] = (code, name)

        completed = 0
        success = 0
        for future in as_completed(futures):
            completed += 1
            code, name = futures[future]
            result = future.result()
            results.append(result)
            if result["status"] == "OK":
                success += 1
            # 进度显示
            pct = completed / len(symbols) * 100
            elapsed = time.time() - t_start
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (len(symbols) - completed) / rate if rate > 0 else 0
            status_icon = "OK" if result["status"] == "OK" else "FAIL"
            rows = result["rows"]
            print(
                f"  [{completed:>3}/{len(symbols)}] {code} {name[:12]:<12} "
                f"{status_icon:>4} {rows:>4}行  "
                f"({pct:5.1f}% | {rate:.1f}/s | ETA {eta:.0f}s)"
            )

    total_elapsed = time.time() - t_start

    # 汇总
    print("\n" + "=" * 70)
    print("下载完成汇总")
    print("=" * 70)
    print(f"  总数: {len(results)}")
    print(f"  成功: {success}")
    print(f"  失败: {len(results) - success}")
    print(f"  耗时: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    print(f"  平均: {total_elapsed/len(results):.1f}s/只")

    # 数据覆盖统计
    ok_results = [r for r in results if r["status"] == "OK"]
    if ok_results:
        total_rows = sum(r["rows"] for r in ok_results)
        avg_rows = total_rows / len(ok_results)
        print(f"\n  数据统计:")
        print(f"    总行数: {total_rows:,}")
        print(f"    平均行数: {avg_rows:.0f}")
        print(f"    最小行数: {min(r['rows'] for r in ok_results)}")
        print(f"    最大行数: {max(r['rows'] for r in ok_results)}")

    # 失败列表
    failed = [r for r in results if r["status"] != "OK"]
    if failed:
        print(f"\n  失败列表 ({len(failed)} 只):")
        for r in failed:
            print(f"    {r['symbol']} {r['name']}: {r['status']}")

    # 检查 data_cache 目录
    cache_dir = PROJ_ROOT / "data_cache"
    csv_files = list(cache_dir.glob("*.csv"))
    total_size = sum(f.stat().st_size for f in csv_files)
    print(f"\n  data_cache/ 目录:")
    print(f"    CSV 文件数: {len(csv_files)}")
    print(f"    总大小: {total_size / 1024 / 1024:.1f} MB")

    print(f"\n  完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
