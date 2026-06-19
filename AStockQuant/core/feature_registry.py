# -*- coding: utf-8 -*-
"""
feature_registry.py — 特征注册中心

负责收集各层 (Layer) 产生的因子, 将它们拼接成统一的特征张量供 AI 模型消费。
每一层只需要实现 extract_features 方法, 注册到此处即可自动参与融合。
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class FeatureRegistry:
    """因子注册中心: 管理所有层级的因子, 按需拼接。"""

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
        return [n for n, e in self._enabled.items() if e]
    
    def get_layer(self, name: str):
        """获取指定层"""
        return self._layers.get(name)

    def extract_features(
        self,
        symbol: str,
        stock_df: pd.DataFrame,
        context: Optional[dict] = None,
    ) -> Dict[str, object]:
        """从所有已启用层中提取因子, 返回 {因子名: 值} 字典"""
        features: Dict[str, object] = {}
        ctx = context or {}
        
        for name in self.active_layers:
            layer = self._layers[name]
            try:
                # 调用layer的extract_features方法
                layer_feats = layer.extract_features(symbol, stock_df, ctx)
                features.update(layer_feats)
            except Exception as e:
                # 可能层使用的是旧接口extract
                try:
                    layer_feats = layer.extract(symbol, stock_df, ctx)
                    features.update(layer_feats)
                except Exception as e2:
                    print(f"[FeatureRegistry] '{name}' 提取失败: {e2}")
        
        return features

    def extract_vector(
        self,
        symbol: str,
        stock_df: pd.DataFrame,
        context: Optional[dict] = None,
    ) -> np.ndarray:
        """返回扁平化的 float32 特征向量 (适合 ML 输入)"""
        feats = self.extract_features(symbol, stock_df, context)
        vals = []
        for v in feats.values():
            if isinstance(v, (int, float, np.floating, np.integer)):
                vals.append(float(v))
            elif isinstance(v, bool):
                vals.append(1.0 if v else 0.0)
            elif isinstance(v, str):
                # 字符串转为编码
                vals.append(hash(v) % 100 / 100)
        arr = np.array(vals, dtype=np.float32)
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    
    def get_all_features_description(self) -> Dict[str, List[str]]:
        """获取所有已注册层的特征名"""
        result = {}
        for name in self._layers.keys():
            layer = self._layers[name]
            try:
                # 尝试获取层的特征列表
                if hasattr(layer, 'get_feature_names'):
                    result[name] = layer.get_feature_names()
                else:
                    result[name] = [f"{name}_*"]
            except:
                result[name] = [f"{name}_*"]
        return result