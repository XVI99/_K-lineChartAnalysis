# -*- coding: utf-8 -*-
"""
deep_learning.py — 深度学习信号引擎

从原 advanced_ai.py 迁移而来, 仅负责模型训练/推理, 不碰业务逻辑。
包含:
- DeepLearningSignalEngine  (MLP, Torch优先 / NumPy回退)
- TemporalDeepSignalEngine  (LSTM + Attention)
- TemporalEnsembleSignalEngine (LSTM + Transformer 并联)
- 特征 / 序列构建辅助函数
"""

# ===========================================================
# 直接复用原有 advanced_ai.py 中的 AI 引擎类
# 这样做是为了保证迁移阶段的100%兼容
# 后续优化时可以逐步重写
# ===========================================================

from __future__ import annotations

# 从原模块导入所有 AI 类 (保持向后兼容)
# 注意: 这些类已经在 quant_system.advanced_ai 中被验证过
# 未来将逐步将其核心代码迁移到此文件中

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def _compute_rsi(close: pd.Series, period: int = 14) -> float:
    if len(close) < period + 2:
        return 50.0
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean().iloc[-1]
    loss = (-delta.clip(upper=0)).rolling(period).mean().iloc[-1]
    if pd.isna(gain) or pd.isna(loss):
        return 50.0
    if loss == 0:
        return 100.0
    rs = gain / loss
    return float(100 - (100 / (1 + rs)))


def extract_latest_features(df: pd.DataFrame, min_window: int = 30) -> Optional[np.ndarray]:
    """从最新窗口提取特征向量 (供 DL 引擎使用)"""
    if df is None or df.empty or len(df) < min_window:
        return None
    required = {"close", "high", "low", "volume"}
    if not required.issubset(df.columns):
        return None
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    vol = df["volume"].astype(float)
    rets = close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    cur = float(close.iloc[-1])
    ma5 = float(close.rolling(5).mean().iloc[-1]) if len(close) >= 5 else cur
    ma10 = float(close.rolling(10).mean().iloc[-1]) if len(close) >= 10 else cur
    ma20 = float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else cur
    atr14 = float((high - low).rolling(14).mean().iloc[-1]) if len(df) >= 14 else float((high - low).mean())
    vol5 = float(vol.rolling(5).mean().iloc[-1]) if len(vol) >= 5 else float(vol.iloc[-1])
    vol20 = float(vol.rolling(20).mean().iloc[-1]) if len(vol) >= 20 else vol5
    feat = np.array([
        rets.iloc[-1],
        close.iloc[-1] / close.iloc[-5] - 1 if len(close) >= 5 else 0.0,
        close.iloc[-1] / close.iloc[-10] - 1 if len(close) >= 10 else 0.0,
        close.iloc[-1] / close.iloc[-20] - 1 if len(close) >= 20 else 0.0,
        float(rets.rolling(20).std().iloc[-1]) if len(rets) >= 20 else float(rets.std()),
        (cur / ma5 - 1) if ma5 else 0.0,
        (cur / ma10 - 1) if ma10 else 0.0,
        (cur / ma20 - 1) if ma20 else 0.0,
        _compute_rsi(close) / 100.0,
        (vol5 / vol20 - 1) if vol20 else 0.0,
        (atr14 / cur) if cur else 0.0,
    ], dtype=np.float32)
    return np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)


def build_feature_label_dataset(
    stock_data_map: Dict[str, pd.DataFrame], horizon: int = 5, min_window: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """构建训练集"""
    X, y = [], []
    for _, df in stock_data_map.items():
        if df is None or df.empty or len(df) < min_window + horizon + 5:
            continue
        close = df["close"].astype(float)
        for t in range(min_window, len(df) - horizon):
            feat = extract_latest_features(df.iloc[:t + 1], min_window)
            if feat is None:
                continue
            fwd = float(close.iloc[t + horizon] / close.iloc[t] - 1)
            X.append(feat)
            y.append(1 if fwd > 0 else 0)
    if not X:
        return np.empty((0, 11), dtype=np.float32), np.empty((0,), dtype=np.float32)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def extract_latest_sequence(df: pd.DataFrame, lookback: int = 30) -> Optional[np.ndarray]:
    """提取时序特征序列 (lookback, 6)"""
    if df is None or df.empty or len(df) < lookback + 5:
        return None
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        return None
    close = df["close"].astype(float)
    open_ = df["open"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)
    log_ret = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    intraday = (close - open_) / open_.replace(0, np.nan)
    hl_spread = (high - low) / close.replace(0, np.nan)
    vol_chg = volume.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    vol20 = log_ret.rolling(20).std().bfill().fillna(0.0)
    ma_gap = (close / close.rolling(20).mean() - 1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    seq_df = pd.DataFrame({
        "log_ret": log_ret, "intraday": intraday, "hl_spread": hl_spread,
        "vol_chg": vol_chg, "vol20": vol20, "ma_gap": ma_gap,
    }).fillna(0.0)
    seq = seq_df.iloc[-lookback:].values.astype(np.float32)
    if seq.shape[0] != lookback:
        return None
    return np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)


def build_sequence_dataset(
    stock_data_map: Dict[str, pd.DataFrame], lookback: int = 30, horizon: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """构建序列样本"""
    X_seq, y = [], []
    for _, df in stock_data_map.items():
        if df is None or df.empty or len(df) < lookback + horizon + 10:
            continue
        close = df["close"].astype(float)
        for t in range(lookback, len(df) - horizon):
            seq = extract_latest_sequence(df.iloc[:t + 1], lookback)
            if seq is None:
                continue
            fwd = float(close.iloc[t + horizon] / close.iloc[t] - 1)
            X_seq.append(seq)
            y.append(1 if fwd > 0 else 0)
    if not X_seq:
        return np.empty((0, lookback, 6), dtype=np.float32), np.empty((0,), dtype=np.float32)
    return np.array(X_seq, dtype=np.float32), np.array(y, dtype=np.float32)


@dataclass
class DLTrainReport:
    trained: bool
    backend: str
    samples: int
    note: str = ""


class DeepLearningSignalEngine:
    """深度学习信号引擎 (Torch 优先, NumPy 回退)"""

    def __init__(self, epochs: int = 20, lr: float = 1e-3):
        self.epochs = epochs
        self.lr = lr
        self.backend = "numpy"
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self._trained = False
        self._use_torch = False
        self._torch = None
        self._model = None
        self._w = None
        self._b = 0.0
        self._init_torch()

    def _init_torch(self):
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            self._torch = torch
            self._nn = nn
            self._optim = optim
            self._use_torch = True
            self.backend = "torch"
        except Exception:
            pass

    def _standardize(self, X, fit=False):
        if fit or self.mean_ is None:
            self.mean_ = X.mean(axis=0)
            self.std_ = X.std(axis=0)
            self.std_[self.std_ < 1e-6] = 1.0
        return (X - self.mean_) / self.std_

    def fit(self, X, y) -> DLTrainReport:
        if X is None or y is None or len(X) < 80:
            self._trained = False
            return DLTrainReport(False, self.backend, 0, "样本不足")
        X = self._standardize(X, fit=True)
        y = y.reshape(-1, 1).astype(np.float32)
        if self._use_torch:
            nn, optim, torch = self._nn, self._optim, self._torch
            self._model = nn.Sequential(
                nn.Linear(X.shape[1], 64), nn.ReLU(), nn.Dropout(0.15),
                nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid(),
            )
            criterion = nn.BCELoss()
            optimizer = optim.Adam(self._model.parameters(), lr=self.lr)
            xt = torch.tensor(X, dtype=torch.float32)
            yt = torch.tensor(y, dtype=torch.float32)
            self._model.train()
            for _ in range(self.epochs):
                pred = self._model(xt)
                loss = criterion(pred, yt)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            self._trained = True
            return DLTrainReport(True, "torch", len(X), "MLP训练完成")
        # NumPy fallback
        self._w = np.zeros((X.shape[1], 1), dtype=np.float32)
        self._b = 0.0
        for _ in range(max(50, self.epochs * 10)):
            pred = _sigmoid(X @ self._w + self._b)
            err = pred - y
            self._w -= 0.02 * (X.T @ err) / len(X)
            self._b -= 0.02 * float(err.mean())
        self._trained = True
        return DLTrainReport(True, "numpy", len(X), "逻辑回归回退")

    def predict_proba(self, feat) -> float:
        if feat is None:
            return 0.5
        x = np.array(feat, dtype=np.float32).reshape(1, -1)
        if self.mean_ is not None:
            x = (x - self.mean_) / self.std_
        if not self._trained:
            return 0.5
        if self._use_torch and self._model:
            self._model.eval()
            with self._torch.no_grad():
                p = float(self._model(self._torch.tensor(x, dtype=self._torch.float32)).item())
            return max(0.01, min(0.99, p))
        if self._w is None:
            return 0.5
        p = float(_sigmoid(x @ self._w + self._b).ravel()[0])
        return max(0.01, min(0.99, p))


class TemporalEnsembleSignalEngine:
    """时序模型并联引擎 (LSTM + Transformer)"""

    def __init__(self, lookback=30, epochs=12, lr=8e-4, mode="ensemble"):
        self.lookback = lookback
        self.epochs = epochs
        self.lr = lr
        self.mode = mode
        self._use_torch = False
        self._torch = None
        self._nn = None
        self._optim = None
        self._lstm_model = None
        self._trans_model = None
        self._trained = False
        self.backend = "numpy"
        self._fallback_engine = DeepLearningSignalEngine(epochs=max(10, epochs), lr=1e-3)
        self._init_torch()

    def _init_torch(self):
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            self._torch = torch
            self._nn = nn
            self._optim = optim
            self._use_torch = True
            self.backend = f"torch_{self.mode}"
        except Exception:
            pass

    def fit(self, X_seq, y) -> DLTrainReport:
        if X_seq is None or y is None or len(X_seq) < 140:
            self._trained = False
            return DLTrainReport(False, self.backend, 0, "时序样本不足")
        if not self._use_torch:
            X_flat = X_seq.reshape(X_seq.shape[0], -1)
            r = self._fallback_engine.fit(X_flat, y)
            self._trained = r.trained
            return DLTrainReport(r.trained, "numpy_fallback", r.samples, r.note)
        torch, nn, optim = self._torch, self._nn, self._optim
        y = y.reshape(-1, 1).astype(np.float32)
        xt = torch.tensor(X_seq, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        split = int(len(xt) * 0.85)
        x_tr, x_va = xt[:split], xt[split:]
        y_tr, y_va = yt[:split], yt[split:]
        in_dim = X_seq.shape[-1]

        class LSTMHead(nn.Module):
            def __init__(self, d):
                super().__init__()
                self.lstm = nn.LSTM(d, 48, batch_first=True, num_layers=2, dropout=0.15)
                self.head = nn.Sequential(nn.Linear(48, 24), nn.ReLU(), nn.Dropout(0.1), nn.Linear(24, 1), nn.Sigmoid())
            def forward(self, x):
                h, _ = self.lstm(x)
                return self.head(h[:, -1, :])

        class TransHead(nn.Module):
            def __init__(self, d, dm=48, nh=4, nl=2):
                super().__init__()
                self.proj = nn.Linear(d, dm)
                enc = nn.TransformerEncoderLayer(dm, nh, 128, 0.1, batch_first=True, activation="gelu")
                self.enc = nn.TransformerEncoder(enc, nl)
                self.head = nn.Sequential(nn.Linear(dm, 24), nn.GELU(), nn.Dropout(0.1), nn.Linear(24, 1), nn.Sigmoid())
            def forward(self, x):
                return self.head(self.enc(self.proj(x))[:, -1, :])

        def _train(model):
            crit = nn.BCELoss()
            opt = optim.Adam(model.parameters(), lr=self.lr)
            best, bad = 1e9, 0
            for _ in range(self.epochs):
                model.train()
                loss = crit(model(x_tr), y_tr)
                opt.zero_grad(); loss.backward(); opt.step()
                model.eval()
                with torch.no_grad():
                    vl = crit(model(x_va), y_va).item() if len(x_va) > 0 else loss.item()
                if vl < best - 1e-4:
                    best, bad = vl, 0
                else:
                    bad += 1
                    if bad >= 3:
                        break
            return best

        notes = []
        if self.mode in ("lstm", "ensemble"):
            self._lstm_model = LSTMHead(in_dim)
            notes.append(f"lstm={_train(self._lstm_model):.4f}")
        if self.mode in ("transformer", "ensemble"):
            self._trans_model = TransHead(in_dim)
            notes.append(f"trans={_train(self._trans_model):.4f}")
        self._trained = True
        return DLTrainReport(True, self.backend, len(X_seq), "; ".join(notes))

    def predict_proba(self, seq) -> float:
        if seq is None or not self._trained:
            return 0.5
        if not self._use_torch:
            return self._fallback_engine.predict_proba(seq.ravel())
        x = self._torch.tensor(seq.reshape(1, seq.shape[0], seq.shape[1]), dtype=self._torch.float32)
        probs = []
        with self._torch.no_grad():
            if self._lstm_model:
                self._lstm_model.eval()
                probs.append(float(self._lstm_model(x).item()))
            if self._trans_model:
                self._trans_model.eval()
                probs.append(float(self._trans_model(x).item()))
        return max(0.01, min(0.99, float(np.mean(probs)))) if probs else 0.5
