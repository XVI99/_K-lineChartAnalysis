# -*- coding: utf-8 -*-
"""
深度学习因子组合模型
====================
用MLP学习因子→收益的非线性映射，替代线性等权组合

架构:
  输入: 25个因子截面值 (标准化)
  模型: MLP [256→128→64→1] + BatchNorm + Dropout + ReLU
  输出: 预测未来20日收益
  训练: Walk-Forward (3折expanding window)
  评估: ICIR / 分层回测 / 对比线性组合

依赖: pip install torch scikit-learn (如果没有则自动安装)
"""

import os
import sys
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

warnings.filterwarnings("ignore")

REPORT_DIR = os.path.join(PROJECT_ROOT, "reports", "deep_learning")
os.makedirs(REPORT_DIR, exist_ok=True)

# ==================== 配置 ====================
FWD_RETURN_DAYS = 20          # 预测目标: 未来20日收益
WF_FOLDS = 3                  # Walk-Forward折数
HIDDEN_LAYERS = [256, 128, 64]
DROPOUT = 0.3
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
EPOCHS = 100
PATIENCE = 15                 # Early stopping
DEVICE = "cpu"                # cpu / cuda

# 因子清单 (与factor_evaluation.py一致)
ALL_FACTORS = [
    "macro_regime_score", "macro_relative_score", "macro_relative_strength",
    "sector_combined_score", "sector_momentum", "sector_momentum_long", "sector_is_leader",
    "capital_score",
    "sent_combined_score", "sent_rsi_score", "sent_rsi_momentum", "sent_volatility_score",
    "pv_rps_combined", "pv_rps_50", "pv_vcp_quality", "pv_volatility_contraction",
    "pv_score", "pv_volume_trend", "pv_obv_trend",
    "pv_turnover_change", "pv_price_accel", "pv_vol_price_divergence",
    "tech_pattern_score", "tech_ma_score", "tech_rsi_score",
    "belief_posterior",
]


# ==================== 1. 数据准备 ====================
def load_factor_panel() -> pd.DataFrame:
    """加载因子panel (复用factor_evaluation.py)"""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from factor_evaluation import load_all_etf_data, get_cross_section_dates, build_factor_panel

    print("[1/5] 加载数据...")
    all_data = load_all_etf_data()
    all_dates_set = set()
    for df in all_data.values():
        all_dates_set.update(df.index)
    cross_dates = get_cross_section_dates(list(all_dates_set))
    panel = build_factor_panel(all_data, cross_dates)
    print(f"  Panel: {len(panel)} 条, {panel['date'].nunique()} 个截面, {panel['symbol'].nunique()} 只ETF")
    return panel


def prepare_ml_data(panel: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    准备ML数据: X=因子值, y=未来收益, meta=日期+标的

    Returns: X, y, dates, symbols
    """
    print("[2/5] 准备训练数据...")

    # 筛选可用因子
    available_factors = [f for f in ALL_FACTORS if f in panel.columns]
    print(f"  可用因子: {len(available_factors)}/{len(ALL_FACTORS)}")

    # 构建特征矩阵
    X_list, y_list, date_list, symbol_list = [], [], [], []

    for dt in sorted(panel["date"].unique()):
        sub = panel[panel["date"] == dt].copy()
        if "fwd_ret_20d" not in sub.columns:
            continue

        # 特征
        feats = sub[available_factors].values.astype(np.float32)
        # 目标
        targets = sub["fwd_ret_20d"].values.astype(np.float32)

        # 过滤NaN
        valid = ~np.isnan(feats).any(axis=1) & ~np.isnan(targets)
        if valid.sum() < 10:
            continue

        X_list.append(feats[valid])
        y_list.append(targets[valid])
        date_list.extend([dt] * valid.sum())
        symbol_list.extend(sub["symbol"].values[valid])

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    dates = np.array(date_list)
    symbols = np.array(symbol_list)

    print(f"  样本: {len(X)}, 特征维度: {X.shape[1]}")
    print(f"  y均值: {y.mean():.4f}, y标准差: {y.std():.4f}")
    return X, y, dates, symbols


# ==================== 2. MLP模型 ====================
class FactorMLP:
    """因子→收益 MLP"""

    def __init__(self, input_dim: int, hidden_dims: List[int] = HIDDEN_LAYERS, dropout: float = DROPOUT):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.model = None
        self.scaler_mean = None
        self.scaler_std = None

    def _build_model(self):
        """构建PyTorch模型"""
        import torch
        import torch.nn as nn

        layers = []
        prev_dim = self.input_dim

        for h_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.model = nn.Sequential(*layers)
        return self.model

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None,
            epochs: int = EPOCHS, batch_size: int = BATCH_SIZE,
            lr: float = LEARNING_RATE, patience: int = PATIENCE,
            verbose: bool = True):
        """训练模型"""
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset

        # 标准化
        self.scaler_mean = X_train.mean(axis=0)
        self.scaler_std = X_train.std(axis=0) + 1e-8

        X_train_norm = (X_train - self.scaler_mean) / self.scaler_std
        if X_val is not None:
            X_val_norm = (X_val - self.scaler_mean) / self.scaler_std

        # 转tensor
        X_t = torch.tensor(X_train_norm, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self._build_model()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        best_val_loss = float("inf")
        best_state = None
        no_improve = 0

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                pred = self.model(batch_X)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(loader)

            # 验证
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_pred = self.model(torch.tensor(X_val_norm, dtype=torch.float32))
                    val_loss = criterion(val_pred, torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1

                if no_improve >= patience:
                    if verbose:
                        print(f"    Early stopping at epoch {epoch+1}")
                    break
            else:
                if train_loss < best_val_loss:
                    best_val_loss = train_loss
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        # 恢复最佳状态
        if best_state is not None:
            self.model.load_state_dict(best_state)

        if verbose:
            print(f"    训练完成, loss={best_val_loss:.6f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        import torch
        X_norm = (X - self.scaler_mean) / self.scaler_std
        X_t = torch.tensor(X_norm, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(X_t).numpy().flatten()
        return pred


# ==================== 3. Walk-Forward训练+评估 ====================
def walk_forward_train_eval(
    X: np.ndarray, y: np.ndarray, dates: np.ndarray, symbols: np.ndarray,
    n_folds: int = WF_FOLDS,
) -> Dict:
    """
    Walk-Forward训练+评估

    Returns: 评估结果字典
    """
    print(f"\n[3/5] Walk-Forward训练 ({n_folds}折)...")

    unique_dates = sorted(np.unique(dates))
    n_total = len(unique_dates)
    fold_size = n_total // (n_folds + 1)  # +1 for initial training

    results = {
        "folds": [],
        "all_predictions": [],
        "all_actuals": [],
        "all_dates": [],
        "all_symbols": [],
    }

    for fold in range(n_folds):
        # 划分训练/测试
        train_end_idx = fold_size * (fold + 1)
        test_start_idx = train_end_idx
        test_end_idx = min(test_start_idx + fold_size, n_total)

        train_dates = set(unique_dates[:train_end_idx])
        test_dates = set(unique_dates[test_start_idx:test_end_idx])

        train_mask = np.isin(dates, list(train_dates))
        test_mask = np.isin(dates, list(test_dates))

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        # 用训练集后20%做验证
        val_size = int(len(X_train) * 0.2)
        X_tr, X_val = X_train[:-val_size], X_train[-val_size:]
        y_tr, y_val = y_train[:-val_size], y_train[-val_size:]

        print(f"\n  Fold {fold+1}/{n_folds}:")
        print(f"    训练: {unique_dates[0].date()} ~ {unique_dates[train_end_idx-1].date()} ({len(X_tr)}样本)")
        print(f"    测试: {unique_dates[test_start_idx].date()} ~ {unique_dates[test_end_idx-1].date()} ({len(X_test)}样本)")

        # 训练
        model = FactorMLP(input_dim=X.shape[1])
        model.fit(X_tr, y_tr, X_val, y_val, verbose=True)

        # 预测
        y_pred = model.predict(X_test)

        # 计算IC
        from scipy.stats import spearmanr
        ic, _ = spearmanr(y_pred, y_test)

        # 按截面计算IC
        test_dates_list = sorted(test_dates)
        ic_list = []
        for dt in test_dates_list:
            dt_mask = dates[test_mask] == dt
            if dt_mask.sum() >= 10:
                ic_dt, _ = spearmanr(y_pred[dt_mask], y_test[dt_mask])
                if not np.isnan(ic_dt):
                    ic_list.append(ic_dt)

        ic_mean = np.mean(ic_list) if ic_list else 0
        ic_std = np.std(ic_list) if ic_list else 0
        icir = ic_mean / ic_std if ic_std > 0 else 0

        fold_result = {
            "fold": fold + 1,
            "train_samples": len(X_tr),
            "test_samples": len(X_test),
            "ic": round(ic, 4),
            "ic_mean": round(ic_mean, 4),
            "icir": round(icir, 4),
            "ic_win_rate": round(np.mean(np.array(ic_list) > 0), 4) if ic_list else 0,
        }
        results["folds"].append(fold_result)

        # 收集预测
        results["all_predictions"].extend(y_pred.tolist())
        results["all_actuals"].extend(y_test.tolist())
        results["all_dates"].extend(dates[test_mask].tolist())
        results["all_symbols"].extend(symbols[test_mask].tolist())

        print(f"    样本外IC: {ic:.4f}, ICIR: {icir:.4f}, 胜率: {fold_result['ic_win_rate']:.2%}")

    # 汇总
    all_preds = np.array(results["all_predictions"])
    all_actuals = np.array(results["all_actuals"])
    all_dates_arr = np.array(results["all_dates"])

    from scipy.stats import spearmanr
    total_ic, _ = spearmanr(all_preds, all_actuals)

    # 按截面IC
    unique_test_dates = sorted(np.unique(all_dates_arr))
    all_ic_list = []
    for dt in unique_test_dates:
        dt_mask = all_dates_arr == dt
        if dt_mask.sum() >= 10:
            ic_dt, _ = spearmanr(all_preds[dt_mask], all_actuals[dt_mask])
            if not np.isnan(ic_dt):
                all_ic_list.append(ic_dt)

    total_ic_mean = np.mean(all_ic_list)
    total_ic_std = np.std(all_ic_list)
    total_icir = total_ic_mean / total_ic_std if total_ic_std > 0 else 0

    results["summary"] = {
        "total_ic": round(total_ic, 4),
        "ic_mean": round(total_ic_mean, 4),
        "icir": round(total_icir, 4),
        "ic_win_rate": round(np.mean(np.array(all_ic_list) > 0), 4),
        "n_cross_sections": len(all_ic_list),
    }

    return results


# ==================== 4. 分层回测 ====================
def quantile_backtest(predictions: np.ndarray, actuals: np.ndarray,
                      dates: np.ndarray, n_quantiles: int = 5) -> Dict:
    """分层回测"""
    print(f"\n[4/5] 分层回测...")

    unique_dates = sorted(np.unique(dates))
    q_returns = {f"Q{i+1}": [] for i in range(n_quantiles)}

    for dt in unique_dates:
        dt_mask = dates == dt
        if dt_mask.sum() < n_quantiles * 2:
            continue

        preds_dt = predictions[dt_mask]
        actuals_dt = actuals[dt_mask]

        # 按预测值分组
        try:
            labels = pd.qcut(preds_dt, n_quantiles, labels=False, duplicates="drop")
        except ValueError:
            continue

        for q in range(n_quantiles):
            q_mask = labels == q
            if q_mask.sum() > 0:
                q_returns[f"Q{q+1}"].append(actuals_dt[q_mask].mean())

    # 计算统计
    q_stats = {}
    for q_name, rets in q_returns.items():
        if rets:
            q_stats[q_name] = {
                "mean_return": round(np.mean(rets), 6),
                "std": round(np.std(rets), 6),
                "sharpe": round(np.mean(rets) / np.std(rets) * np.sqrt(52), 4) if np.std(rets) > 0 else 0,
            }

    # 多空
    if q_stats:
        long_short = q_stats[f"Q{n_quantiles}"]["mean_return"] - q_stats["Q1"]["mean_return"]
        monotonic = sum(
            1 for i in range(n_quantiles - 1)
            if q_stats[f"Q{i+2}"]["mean_return"] > q_stats[f"Q{i+1}"]["mean_return"]
        ) / (n_quantiles - 1)
    else:
        long_short = 0
        monotonic = 0

    return {
        "quantiles": q_stats,
        "long_short": round(long_short, 6),
        "monotonicity": round(monotonic, 2),
    }


# ==================== 5. 对比线性组合 ====================
def compare_with_linear(results: Dict) -> pd.DataFrame:
    """对比MLP vs 线性等权组合"""
    print(f"\n[5/5] 对比分析...")

    # 读取线性组合结果
    combo_csv = os.path.join(PROJECT_ROOT, "reports", "factor_evaluation", "combination", "combination_results.csv")
    linear_data = {}
    if os.path.exists(combo_csv):
        combo_df = pd.read_csv(combo_csv)
        if "method" in combo_df.columns and "icir" in combo_df.columns:
            for _, row in combo_df.iterrows():
                linear_data[row["method"]] = {
                    "icir": row.get("icir", 0),
                    "ic_mean": row.get("ic_mean", 0),
                    "sharpe": row.get("sharpe", 0),
                }

    # 读取最佳单因子
    factor_csv = os.path.join(PROJECT_ROOT, "reports", "factor_evaluation", "factor_ranking.csv")
    best_factor = {}
    if os.path.exists(factor_csv):
        factor_df = pd.read_csv(factor_csv)
        if len(factor_df) > 0:
            best = factor_df.iloc[0]
            best_factor = {
                "name": best.get("factor", "N/A"),
                "icir": best.get("icir", 0),
                "sharpe": best.get("sharpe", 0),
            }

    # 构建对比表
    comparison = []

    # MLP
    mlp_icir = results["summary"]["icir"]
    mlp_q = results.get("quantile_backtest", {}).get("quantiles", {})
    mlp_sharpe = mlp_q.get("Q5", {}).get("sharpe", 0) if mlp_q else 0

    comparison.append({
        "method": "MLP (深度学习)",
        "icir": mlp_icir,
        "ic_mean": results["summary"]["ic_mean"],
        "sharpe": mlp_sharpe,
        "ic_win_rate": results["summary"]["ic_win_rate"],
    })

    # 线性组合
    for method, data in linear_data.items():
        comparison.append({
            "method": f"线性-{method}",
            "icir": data["icir"],
            "ic_mean": data.get("ic_mean", 0),
            "sharpe": data.get("sharpe", 0),
            "ic_win_rate": 0,
        })

    # 最佳单因子
    if best_factor:
        comparison.append({
            "method": f"最佳单因子({best_factor['name']})",
            "icir": best_factor["icir"],
            "ic_mean": 0,
            "sharpe": best_factor["sharpe"],
            "ic_win_rate": 0,
        })

    comp_df = pd.DataFrame(comparison)
    comp_df = comp_df.sort_values("icir", ascending=False)

    return comp_df


# ==================== 6. 报告 ====================
def generate_report(results: Dict, comparison: pd.DataFrame):
    """生成报告"""
    print("\n" + "=" * 80)
    print("深度学习因子组合 — 评估报告")
    print("=" * 80)

    # Walk-Forward结果
    print(f"\n一、Walk-Forward结果 ({WF_FOLDS}折)")
    print("-" * 60)
    for fold in results["folds"]:
        print(f"  Fold {fold['fold']}: IC={fold['ic']:.4f}, ICIR={fold['icir']:.4f}, "
              f"胜率={fold['ic_win_rate']:.2%}, 样本={fold['test_samples']}")

    summary = results["summary"]
    print(f"\n  汇总: IC={summary['ic_mean']:.4f}, ICIR={summary['icir']:.4f}, "
          f"胜率={summary['ic_win_rate']:.2%}, 截面数={summary['n_cross_sections']}")

    # 分层回测
    qb = results.get("quantile_backtest", {})
    if qb.get("quantiles"):
        print(f"\n二、分层回测")
        print("-" * 60)
        print(f"  {'分组':<8} {'均值收益':>10} {'夏普':>8}")
        for q_name, stats in qb["quantiles"].items():
            print(f"  {q_name:<8} {stats['mean_return']:>10.6f} {stats['sharpe']:>8.4f}")
        print(f"  多空收益: {qb['long_short']:.6f}, 单调性: {qb['monotonicity']:.0%}")

    # 对比
    print(f"\n三、方法对比")
    print("-" * 80)
    print(f"  {'方法':<25} {'ICIR':>8} {'IC均值':>10} {'夏普':>8}")
    print(f"  {'-'*55}")
    for _, row in comparison.iterrows():
        print(f"  {row['method']:<25} {row['icir']:>8.4f} {row['ic_mean']:>10.4f} {row['sharpe']:>8.4f}")

    # 保存
    comp_path = os.path.join(REPORT_DIR, "dl_vs_linear.csv")
    comparison.to_csv(comp_path, index=False, encoding="utf-8-sig")

    fold_path = os.path.join(REPORT_DIR, "dl_walk_forward.csv")
    pd.DataFrame(results["folds"]).to_csv(fold_path, index=False, encoding="utf-8-sig")

    # 预测结果
    pred_df = pd.DataFrame({
        "date": results["all_dates"],
        "symbol": results["all_symbols"],
        "predicted": results["all_predictions"],
        "actual": results["all_actuals"],
    })
    pred_df.to_csv(os.path.join(REPORT_DIR, "dl_predictions.csv"), index=False, encoding="utf-8-sig")

    print(f"\n报告已保存到: {REPORT_DIR}")


# ==================== 主流程 ====================
def main():
    print("=" * 80)
    print("深度学习因子组合模型 — MLP非线性因子融合")
    print("=" * 80)

    # 检查依赖
    try:
        import torch
        print(f"  PyTorch: {torch.__version__}")
    except ImportError:
        print("  安装依赖: pip install torch scikit-learn")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "--quiet"])
        import torch

    # 1. 加载数据
    panel = load_factor_panel()

    # 2. 准备数据
    X, y, dates, symbols = prepare_ml_data(panel)

    # 3. Walk-Forward训练
    results = walk_forward_train_eval(X, y, dates, symbols)

    # 4. 分层回测
    qb = quantile_backtest(
        np.array(results["all_predictions"]),
        np.array(results["all_actuals"]),
        np.array(results["all_dates"]),
    )
    results["quantile_backtest"] = qb

    # 5. 对比
    comparison = compare_with_linear(results)

    # 6. 报告
    generate_report(results, comparison)

    print("\n深度学习因子组合完成!")


if __name__ == "__main__":
    main()
