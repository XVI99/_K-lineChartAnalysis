# -*- coding: utf-8 -*-
"""
feature_registry.py — 特征注册中心

负责收集各层 (Layer) 产生的因子, 将它们拼接成统一的特征张量供 AI 模型消费。
每一层只需要实现 extract_features 方法, 注册到此处即可自动参与融合。

v2 改进:
- 层间时序对齐：支持 as_of_date 参数，所有层统一截取到同一时间点
- 层间证据传递：前序层输出作为后序层的上下文（Layer8 获取 Layer4/6 的因子）
- 定义标准执行顺序：Layer1→2→3→4→5→6→7→8
- 向后兼容：支持旧接口 extract() 和无 as_of_date 参数的层
"""

from __future__ import annotations

from typing import Dict, List, Optional
import inspect

import numpy as np
import pandas as pd


class FeatureRegistry:
    """因子注册中心: 管理所有层级的因子, 按需拼接。"""

    # 标准层执行顺序（序号小的先执行，输出作为后序层的上下文）
    LAYER_ORDER: List[str] = [
        "macro",       # Layer1: 宏观环境
        "rules",       # Layer2: 规则过滤
        "sector",      # Layer3: 板块轮动
        "capital",     # Layer4: 资金流向
        "sentiment",   # Layer5: 市场情绪
        "price_vol",   # Layer6: 量价/RPS/VCP
        "technical",   # Layer7: 技术指标
        "belief",      # Layer8: 贝叶斯信念（需要前序层的证据）
    ]

    def __init__(self):
        self._layers: Dict[str, object] = {}
        self._enabled: Dict[str, bool] = {}

    def register(self, name: str, layer, enabled: bool = True):
        """注册一个 Layer 实例"""
        self._layers[name] = layer
        self._enabled[name] = enabled

    def enable(self, name: str):
        self._enabled[name] = True

    def disable(self, name: str):
        self._enabled[name] = False

    @property
    def active_layers(self) -> List[str]:
        """返回按执行顺序排序的已启用层"""
        ordered = []
        for name in self.LAYER_ORDER:
            if name in self._layers and self._enabled.get(name, True):
                ordered.append(name)
        # 加上不在标准顺序中的层
        for name in self._layers:
            if name not in self.LAYER_ORDER and self._enabled.get(name, True):
                ordered.append(name)
        return ordered

    def get_layer(self, name: str):
        """获取指定层"""
        return self._layers.get(name)

    def _call_layer_extract(
        self,
        layer,
        symbol: str,
        stock_df: pd.DataFrame,
        ctx: dict,
        as_of_date: Optional[str] = None,
    ) -> Dict:
        """调用层的 extract_features，自动适配新旧接口"""
        try:
            sig = inspect.signature(layer.extract_features)
            params = sig.parameters

            if "as_of_date" in params:
                return layer.extract_features(symbol, stock_df, ctx, as_of_date=as_of_date)
            else:
                # 旧接口：手动截取数据
                df = stock_df
                if as_of_date and not df.empty:
                    df = df[df.index <= pd.Timestamp(as_of_date)]
                return layer.extract_features(symbol, df, ctx)
        except Exception as e:
            # 尝试旧接口 extract()
            try:
                return layer.extract(symbol, stock_df, ctx)
            except Exception:
                print(f"[FeatureRegistry] layer 提取失败: {e}")
                return {}

    def extract_features(
        self,
        symbol: str,
        stock_df: pd.DataFrame,
        context: Optional[dict] = None,
        as_of_date: Optional[str] = None,
    ) -> Dict[str, object]:
        """
        从所有已启用层中提取因子, 返回 {因子名: 值} 字典

        Args:
            symbol: 标的代码
            stock_df: OHLCV数据
            context: 上下文（可含 market_prices_df, all_data, all_sector_returns 等）
            as_of_date: 截止日期（防未来函数，所有层统一截取到此日期）

        v2: 层间证据传递——前序层的输出会合并到 ctx 中供后序层使用
        """
        features: Dict[str, object] = {}
        ctx = dict(context or {})  # 复制一份，避免修改原始context

        for name in self.active_layers:
            layer = self._layers[name]
            layer_feats = self._call_layer_extract(
                layer, symbol, stock_df, ctx, as_of_date
            )
            features.update(layer_feats)

            # v2: 层间证据传递——将当前层的输出注入 ctx 供后序层使用
            # 特别是 Layer8(belief) 需要 Layer4(capital) 和 Layer6(price_vol) 的因子
            ctx.update(layer_feats)

        return features

    def extract_vector(
        self,
        symbol: str,
        stock_df: pd.DataFrame,
        context: Optional[dict] = None,
        as_of_date: Optional[str] = None,
    ) -> np.ndarray:
        """返回扁平化的 float32 特征向量 (适合 ML 输入)"""
        feats = self.extract_features(symbol, stock_df, context, as_of_date)
        vals = []
        for v in feats.values():
            if isinstance(v, (int, float, np.floating, np.integer)):
                vals.append(float(v))
            elif isinstance(v, bool):
                vals.append(1.0 if v else 0.0)
            elif isinstance(v, str):
                vals.append(hash(v) % 100 / 100)
        arr = np.array(vals, dtype=np.float32)
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    def extract_batch(
        self,
        symbols: List[str],
        all_data: Dict[str, pd.DataFrame],
        context: Optional[dict] = None,
        as_of_date: Optional[str] = None,
    ) -> Dict[str, Dict[str, object]]:
        """
        批量提取多个标的的因子

        Args:
            symbols: 标的代码列表
            all_data: {symbol: OHLCV DataFrame}
            context: 上下文
            as_of_date: 截止日期

        Returns:
            {symbol: {因子名: 值}}
        """
        ctx = dict(context or {})

        # 预计算跨标的因子（如板块动量排名、市场广度等）
        if "all_data" not in ctx:
            ctx["all_data"] = all_data

        # 预计算所有标的的20日收益（供 Layer3 板块排名用）
        all_returns = {}
        for sym in symbols:
            df = all_data.get(sym)
            if df is not None and len(df) >= 21:
                if as_of_date:
                    df = df[df.index <= pd.Timestamp(as_of_date)]
                if len(df) >= 21:
                    all_returns[sym] = float(df["close"].pct_change(20).iloc[-1])
                else:
                    all_returns[sym] = None
            else:
                all_returns[sym] = None
        ctx["all_sector_returns"] = all_returns

        # 逐标的提取
        results = {}
        for sym in symbols:
            df = all_data.get(sym)
            if df is None or df.empty:
                results[sym] = {}
                continue
            results[sym] = self.extract_features(sym, df, ctx, as_of_date)

        return results

    def get_all_features_description(self) -> Dict[str, List[str]]:
        """获取所有已注册层的特征名"""
        result = {}
        for name in self._layers.keys():
            layer = self._layers[name]
            try:
                if hasattr(layer, "get_feature_names"):
                    result[name] = layer.get_feature_names()
                else:
                    result[name] = [f"{name}_*"]
            except Exception:
                result[name] = [f"{name}_*"]
        return result
