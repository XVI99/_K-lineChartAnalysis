# -*- coding: utf-8 -*-
"""
config_loader.py — 全局配置加载器

从 config.yaml 读取系统配置，提供统一的配置访问接口。
支持热重载和默认值兜底。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class ConfigLoader:
    """配置加载器 - 单例模式"""

    _instance: Optional["ConfigLoader"] = None
    _config: Dict[str, Any] = {}
    _config_path: str = ""

    def __new__(cls, config_path: Optional[str] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            if config_path:
                cls._instance._load_config(config_path)
        return cls._instance

    def _load_config(self, config_path: str):
        """加载 YAML 配置文件"""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        with open(path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f) or {}
            self._config_path = str(path)

    @classmethod
    def get_instance(cls, config_path: Optional[str] = None) -> "ConfigLoader":
        """获取单例实例"""
        if cls._instance is None:
            # 默认配置文件路径
            if config_path is None:
                # 尝试从 AStockQuant 根目录查找
                proj_root = Path(__file__).parent.parent
                default_path = proj_root / "config.yaml"
                if default_path.exists():
                    config_path = str(default_path)
                else:
                    raise FileNotFoundError("未找到 config.yaml，请指定配置文件路径")
            cls._instance = cls(config_path)
        return cls._instance

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        通过点分路径获取配置值
        
        Args:
            key_path: 配置键路径，如 "layers.macro" 或 "trading.budget"
            default: 默认值
            
        Returns:
            配置值或默认值
        """
        keys = key_path.split(".")
        value = self._config
        
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key, default)
            else:
                return default
                
        return value if value is not None else default

    def get_layer_config(self) -> Dict[str, bool]:
        """获取各层启用配置"""
        layers = self.get("layers", {})
        return {
            "macro": bool(layers.get("macro", False)),
            "rules": bool(layers.get("rules", True)),
            "sector": bool(layers.get("sector", False)),
            "capital": bool(layers.get("capital", False)),
            "sentiment": bool(layers.get("sentiment", False)),
            "price_vol": bool(layers.get("price_vol", True)),
            "technical": bool(layers.get("technical", True)),
            "micro": bool(layers.get("micro", False)),
            "news": bool(layers.get("news", False)),
        }

    def get_trading_config(self) -> Dict[str, Any]:
        """获取交易配置"""
        trading = self.get("trading", {})
        return {
            "budget": float(trading.get("budget", 5000)),
            "max_price": float(trading.get("max_price", 45.0)),
            "instrument_type": str(trading.get("instrument_type", "etf")),
            "etf_prefixes": tuple(
                trading.get(
                    "etf_prefixes",
                    ["510", "511", "512", "513", "515", "516", "517", "518", "560", "561", "562", "563", "588", "159"],
                )
            ),
            "main_board_prefixes": tuple(
                trading.get("main_board_prefixes", [])
            ),
        }

    def get_model_config(self) -> Dict[str, Any]:
        """获取模型配置"""
        models = self.get("models", {})
        return {
            "deep_learning": {
                "epochs": int(models.get("deep_learning", {}).get("epochs", 24)),
                "lr": float(models.get("deep_learning", {}).get("lr", 0.001)),
            },
            "temporal": {
                "lookback": int(models.get("temporal", {}).get("lookback", 30)),
                "epochs": int(models.get("temporal", {}).get("epochs", 12)),
                "lr": float(models.get("temporal", {}).get("lr", 0.0008)),
                "mode": str(models.get("temporal", {}).get("mode", "ensemble")),
            },
            "rl_allocator": {
                "alpha": float(models.get("rl_allocator", {}).get("alpha", 0.35)),
                "beta": float(models.get("rl_allocator", {}).get("beta", 0.20)),
                "gamma_risk": float(models.get("rl_allocator", {}).get("gamma_risk", 0.35)),
                "max_weight": float(models.get("rl_allocator", {}).get("max_weight", 0.25)),
                "turnover_penalty": float(models.get("rl_allocator", {}).get("turnover_penalty", 0.30)),
            },
            "ppo": {
                "max_weight": float(models.get("ppo", {}).get("max_weight", 0.25)),
                "commission": float(models.get("ppo", {}).get("commission", 0.001)),
                "risk_aversion": float(models.get("ppo", {}).get("risk_aversion", 0.25)),
            },
        }

    def get_scanner_config(self) -> Dict[str, Any]:
        """获取扫描器配置"""
        scanner = self.get("scanner", {})
        return {
            "top_n": int(scanner.get("top_n", 30)),
            "min_confidence": float(scanner.get("min_confidence", 0.55)),
        }

    def get_fusion_weights(self) -> Dict[str, float]:
        """获取信号融合权重"""
        fusion = self.get("fusion", {})
        return {
            "ml_weight": float(fusion.get("ml_weight", 0.35)),
            "rps_weight": float(fusion.get("rps_weight", 0.30)),
            "vcp_weight": float(fusion.get("vcp_weight", 0.20)),
            "pattern_weight": float(fusion.get("pattern_weight", 0.15)),
        }

    def reload(self):
        """重新加载配置文件"""
        if self._config_path:
            self._load_config(self._config_path)

    def __repr__(self):
        return f"ConfigLoader(config_path='{self._config_path}')"


# 便捷函数
def load_config(config_path: Optional[str] = None) -> ConfigLoader:
    """快速加载配置的便捷函数"""
    return ConfigLoader.get_instance(config_path)
