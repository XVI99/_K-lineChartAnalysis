"""
Layer3 - 板块层
=====================

功能: 行业板块轮动分析，热点判断

分析:
- 标的所属板块（覆盖103只ETF全量映射）
- 板块动量（5/20/60日）
- 板块热度与轮动阶段
- 是否为龙头
- 板块内相对强度排名

v2 改进:
- 板块映射从10只扩展到103只ETF全覆盖
- 新增名称自动分类兜底（未知代码按名称关键词归类）
- 板块轮动阶段用连续评分替代离散阈值
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class SectorLayer:
    """
    板块层 - 行业板块轮动分析
    热点板块中的标的更值得关注
    """

    # ==================== 完整板块映射（103只ETF） ====================
    # 格式: "代码": ("板块", "子类")
    SECTOR_MAP: Dict[str, tuple] = {
        # === 宽基指数 (大盘) ===
        "510050": ("宽基", "上证50"),
        "510300": ("宽基", "沪深300"),
        "510310": ("宽基", "沪深300"),
        "510330": ("宽基", "沪深300"),
        "159919": ("宽基", "沪深300"),
        "510500": ("宽基", "中证500"),
        "510510": ("宽基", "中证500"),
        "512500": ("宽基", "中证500"),
        "159922": ("宽基", "中证500"),
        "512100": ("宽基", "中证1000"),
        "560010": ("宽基", "中证1000"),
        "159845": ("宽基", "中证1000"),
        "159338": ("宽基", "中证A500"),
        "159352": ("宽基", "中证A500"),
        "563360": ("宽基", "中证A500"),
        "159788": ("宽基", "中证A500"),
        "159692": ("宽基", "中证A500"),
        "159915": ("宽基", "创业板"),
        "159952": ("宽基", "创业板"),
        "159949": ("宽基", "创业板"),
        "159977": ("宽基", "创业板"),
        "588000": ("宽基", "科创50"),
        "588050": ("宽基", "科创50"),
        "588090": ("宽基", "科创50"),
        "588160": ("宽基", "科创100"),
        "588200": ("宽基", "科创芯片"),
        "159628": ("宽基", "深证100"),
        "159901": ("宽基", "深证100"),
        "159975": ("宽基", "深证成指"),
        "562500": ("宽基", "中证2000"),
        "159531": ("宽基", "中证2000"),

        # === 红利策略 ===
        "510880": ("红利", "红利"),
        "515080": ("红利", "中证红利"),
        "159339": ("红利", "红利低波"),
        "512890": ("红利", "红利低波100"),

        # === 跨境ETF ===
        "513100": ("跨境", "纳指"),
        "159941": ("跨境", "纳指"),
        "159509": ("跨境", "纳指科技"),
        "513500": ("跨境", "标普500"),
        "513050": ("跨境", "中概互联"),
        "164906": ("跨境", "中国互联"),
        "159920": ("跨境", "恒生"),
        "510900": ("跨境", "H股"),
        "513060": ("跨境", "恒生医疗"),
        "159940": ("跨境", "恒生科技"),
        "513130": ("跨境", "恒生科技"),
        "513520": ("跨境", "日经"),
        "159866": ("跨境", "日经"),
        "513030": ("跨境", "德国"),
        "159612": ("跨境", "法国"),

        # === 科技/半导体 ===
        "512760": ("科技", "芯片"),
        "512480": ("科技", "半导体"),
        "159801": ("科技", "芯片"),
        "159995": ("科技", "芯片"),
        "515050": ("科技", "5G通信"),
        "515070": ("科技", "人工智能"),
        "515980": ("科技", "人工智能"),
        "515000": ("科技", "科技"),
        "159732": ("科技", "消费电子"),
        "159998": ("科技", "计算机"),
        "512220": ("科技", "通信"),

        # === 消费 ===
        "512690": ("消费", "酒"),
        "159928": ("消费", "消费"),
        "510150": ("消费", "消费"),
        "512980": ("消费", "传媒"),
        "159869": ("消费", "游戏"),
        "516160": ("消费", "零售"),
        "159825": ("消费", "农业"),

        # === 金融 ===
        "512000": ("金融", "券商"),
        "512880": ("金融", "证券"),
        "512070": ("金融", "保险"),
        "512800": ("金融", "银行"),
        "512730": ("金融", "银行"),

        # === 新能源/碳中和 ===
        "515030": ("新能源", "新能源车"),
        "516160": ("新能源", "新能源"),
        "159875": ("新能源", "新能源"),
        "562890": ("新能源", "光伏"),
        "515790": ("新能源", "光伏"),
        "159611": ("新能源", "电力"),
        "159811": ("新能源", "电池"),
        "159615": ("新能源", "新材料"),

        # === 医药/医疗 ===
        "512170": ("医药", "医疗"),
        "512010": ("医药", "医药"),
        "159938": ("医药", "医药"),
        "159828": ("医药", "生物医药"),

        # === 军工/国防 ===
        "512660": ("军工", "军工"),
        "512680": ("军工", "军工"),
        "512670": ("军工", "国防"),

        # === 资源/周期 ===
        "515220": ("资源", "煤炭"),
        "512400": ("资源", "有色金属"),
        "515210": ("资源", "钢铁"),
        "562900": ("资源", "稀土"),
        "159871": ("资源", "有色金属"),

        # === 房地产/基建 ===
        "512200": ("房地产", "房地产"),
        "159767": ("房地产", "建材"),

        # === 公用事业 ===
        "561560": ("公用事业", "水利"),

        # === 大宗商品(黄金) ===
        "518880": ("商品", "黄金"),
        "159934": ("商品", "黄金"),
        "159937": ("商品", "黄金"),

        # === 债券 ===
        "511010": ("债券", "国债"),
        "511260": ("债券", "十年国债"),
        "511220": ("债券", "城投债"),

        # === 货币 ===
        "511990": ("货币", "货币"),
        "511920": ("货币", "货币"),
    }

    # 名称关键词兜底分类（当代码不在映射表时按名称归类）
    NAME_KEYWORD_MAP: List[tuple] = [
        (("沪深300", "300ETF"), "宽基", "沪深300"),
        (("中证500", "500ETF"), "宽基", "中证500"),
        (("中证1000", "1000ETF"), "宽基", "中证1000"),
        (("A500",), "宽基", "中证A500"),
        (("创业板",), "宽基", "创业板"),
        (("科创",), "宽基", "科创"),
        (("上证50", "上证ETF"), "宽基", "上证50"),
        (("深证", "深成"), "宽基", "深证"),
        (("红利",), "红利", "红利"),
        (("纳指", "纳斯达克"), "跨境", "纳指"),
        (("标普",), "跨境", "标普500"),
        (("恒生",), "跨境", "恒生"),
        (("日经",), "跨境", "日经"),
        (("德国",), "跨境", "德国"),
        (("中概", "互联"), "跨境", "中概互联"),
        (("芯片", "半导体"), "科技", "芯片"),
        (("5G", "通信"), "科技", "通信"),
        (("人工智能", "AI"), "科技", "人工智能"),
        (("科技",), "科技", "科技"),
        (("计算机",), "科技", "计算机"),
        (("消费电子",), "科技", "消费电子"),
        (("酒",), "消费", "酒"),
        (("消费",), "消费", "消费"),
        (("传媒", "游戏"), "消费", "传媒"),
        (("农业",), "消费", "农业"),
        (("券商", "证券"), "金融", "证券"),
        (("保险",), "金融", "保险"),
        (("银行",), "金融", "银行"),
        (("新能源车",), "新能源", "新能源车"),
        (("新能源",), "新能源", "新能源"),
        (("光伏",), "新能源", "光伏"),
        (("电池",), "新能源", "电池"),
        (("电力",), "新能源", "电力"),
        (("新材料",), "新能源", "新材料"),
        (("医疗", "医药", "生物"), "医药", "医疗"),
        (("军工", "国防"), "军工", "军工"),
        (("煤炭",), "资源", "煤炭"),
        (("有色",), "资源", "有色金属"),
        (("钢铁",), "资源", "钢铁"),
        (("稀土",), "资源", "稀土"),
        (("房地产",), "房地产", "房地产"),
        (("基建",), "房地产", "基建"),
        (("建材",), "房地产", "建材"),
        (("黄金", "Gold"), "商品", "黄金"),
        (("国债", "债券", "城投"), "债券", "债券"),
        (("货币", "现金", "Money"), "货币", "货币"),
    ]

    @classmethod
    def classify_by_name(cls, name: str) -> tuple:
        """根据ETF名称关键词自动分类"""
        name_upper = (name or "").upper()
        for keywords, sector, sub_sector in cls.NAME_KEYWORD_MAP:
            if any(kw.upper() in name_upper for kw in keywords):
                return (sector, sub_sector)
        return ("其他", "未知")

    @classmethod
    def get_sector(cls, symbol: str, name: str = "") -> tuple:
        """获取标的的板块归属，返回 (sector, sub_sector)"""
        if symbol in cls.SECTOR_MAP:
            return cls.SECTOR_MAP[symbol]
        if name:
            return cls.classify_by_name(name)
        return ("其他", "未知")

    def extract_features(
        self,
        symbol: str,
        df: pd.DataFrame,
        ctx: Dict,
        as_of_date: Optional[str] = None,
    ) -> Dict:
        """提取板块层特征

        Args:
            symbol: ETF代码
            df: OHLCV数据
            ctx: 上下文（可含 all_sector_returns: Dict[str, float] 全板块动量）
            as_of_date: 截止日期（防未来函数，只用到此日期前的数据）
        """
        features = {}

        # 时序对齐：截取到 as_of_date
        if as_of_date and not df.empty:
            df = df[df.index <= pd.Timestamp(as_of_date)]

        # 确定板块
        name = ctx.get("name", "")
        sector, sub_sector = self.get_sector(symbol, name)
        features["sector"] = sector
        features["sector_sub"] = sub_sector

        # v3: 回测模式从历史数据库读取板块数据
        if as_of_date:
            try:
                from core.history_data_loader import HistoryDataLoader
                loader = HistoryDataLoader.get_instance()
                if loader.has_history_data():
                    sector_data = loader.get_sector_data(sector, as_of_date=as_of_date)
                    if sector_data["momentum"] != 0.0:
                        features["sector_history_momentum"] = sector_data["momentum"]
                        features["sector_history_breadth"] = sector_data["breadth"]
                        features["sector_history_turnover_rank"] = sector_data["turnover_rank"]
                        features["sector_history_phase"] = sector_data["phase"]
                    theme_data = loader.get_theme_flow(sector, as_of_date=as_of_date)
                    if theme_data["net_inflow"] != 0.0:
                        features["sector_theme_flow"] = theme_data["net_inflow"]
                        features["sector_theme_breadth"] = theme_data["breadth"]
                        features["sector_theme_momentum"] = theme_data["momentum"]
            except Exception:
                pass

        if df.empty or len(df) < 20:
            return features

        close = df["close"]
        ret_5d = close.pct_change(5).iloc[-1]
        ret_20d = close.pct_change(20).iloc[-1]
        ret_60d = close.pct_change(60).iloc[-1] if len(df) >= 60 else ret_20d

        features["sector_momentum"] = ret_20d * 100
        features["sector_momentum_short"] = ret_5d * 100
        features["sector_momentum_long"] = ret_60d * 100

        # 板块是否热门 (短期动量 > 5%)
        features["sector_is_hot"] = bool(ret_5d > 0.05)

        # 板块综合得分 (连续映射，0-1)
        # v3改进: 原版仅用sigmoid(ret_20d*10), 与sector_momentum完全共线(ICIR相同)
        # 改为融合多维度: 动量40% + 板块内排名30% + 板块宽度20% + 短期动量10%
        momentum_score = 1.0 / (1.0 + np.exp(-np.clip(ret_20d * 10, -50.0, 50.0)))

        # 预计算板块内排名和宽度 (后续也会用到)
        intra_rank = 50.0
        breadth = 0.5
        all_sector_returns = ctx.get("all_sector_returns", {})
        if all_sector_returns:
            same_sector_returns = [
                r for sym, r in all_sector_returns.items()
                if self.get_sector(sym)[0] == sector and r is not None
            ]
            if same_sector_returns:
                rank_val = sum(1 for r in same_sector_returns if r <= ret_20d) / len(same_sector_returns)
                intra_rank = float(rank_val * 100)
                breadth = float(sum(1 for r in same_sector_returns if r > 0) / len(same_sector_returns))

        # 短期动量sigmoid
        short_momentum_score = 1.0 / (1.0 + np.exp(-np.clip(ret_5d * 10, -50.0, 50.0)))

        # 多维度融合
        features["sector_combined_score"] = float(
            momentum_score * 0.40
            + (intra_rank / 100.0) * 0.30
            + breadth * 0.20
            + short_momentum_score * 0.10
        )

        # 板块轮动阶段 (基于短期+长期动量的连续判断)
        if ret_5d > 0.03 and ret_20d > 0.10:
            features["sector_phase"] = "hot"
        elif ret_5d > 0.01 and ret_20d > 0.05:
            features["sector_phase"] = "warming"
        elif ret_5d < -0.03:
            features["sector_phase"] = "cooling"
        else:
            features["sector_phase"] = "neutral"

        # 是否为龙头 (动量最强的前20%)
        features["sector_is_leader"] = bool(ret_20d > 0.15)

        # v2 新增: 板块内相对强度（如果 ctx 提供了同板块其他ETF的收益）
        all_sector_returns = ctx.get("all_sector_returns", {})
        if all_sector_returns:
            same_sector_returns = [
                r for sym, r in all_sector_returns.items()
                if self.get_sector(sym)[0] == sector and r is not None
            ]
            if same_sector_returns:
                rank = sum(1 for r in same_sector_returns if r <= ret_20d) / len(same_sector_returns)
                features["sector_intra_rank"] = float(rank * 100)
            else:
                features["sector_intra_rank"] = 50.0
        else:
            features["sector_intra_rank"] = 50.0

        # v2 新增: 板块宽度（板块内有多少比例的ETF上涨）
        if all_sector_returns:
            same_sector_returns_valid = [
                r for sym, r in all_sector_returns.items()
                if self.get_sector(sym)[0] == sector and r is not None
            ]
            if same_sector_returns_valid:
                breadth = sum(1 for r in same_sector_returns_valid if r > 0) / len(same_sector_returns_valid)
                features["sector_breadth"] = float(breadth)
            else:
                features["sector_breadth"] = 0.5
        else:
            features["sector_breadth"] = 0.5

        return features

    def get_hot_sectors(
        self, all_data: Dict[str, pd.DataFrame], as_of_date: Optional[str] = None
    ) -> List[str]:
        """获取当前热门板块"""
        hot_sectors = []

        for symbol, df in all_data.items():
            if len(df) < 20:
                continue
            # 时序对齐
            if as_of_date:
                df = df[df.index <= pd.Timestamp(as_of_date)]
            if df.empty or len(df) < 20:
                continue
            ret_5d = df["close"].pct_change(5).iloc[-1]
            if ret_5d > 0.05:
                sector = self.get_sector(symbol)[0]
                if sector not in hot_sectors:
                    hot_sectors.append(sector)

        return hot_sectors
