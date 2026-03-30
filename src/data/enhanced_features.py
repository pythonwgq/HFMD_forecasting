"""
精简版特征工程 - 只保留最有效的特征
"""
import numpy as np
import pandas as pd
from typing import Tuple


def build_features(X: np.ndarray, y: np.ndarray = None,
                   window_size: int = 6) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    精简版特征工程 - 约15-18个特征

    保留最有效的特征类型：
    - 统计特征
    - 趋势特征
    - 近期特征
    - 波动特征
    - 峰值特征
    """
    n_samples = X.shape[0]
    features = []

    for i in range(n_samples):
        row = X[i]
        feat = {}

        # ========== 1. 统计特征（4个） ==========
        feat['mean'] = np.mean(row)
        feat['std'] = np.std(row)
        feat['max'] = np.max(row)
        feat['min'] = np.min(row)

        # ========== 2. 趋势特征（2个） ==========
        feat['slope'] = (row[-1] - row[0]) / window_size
        feat['trend_ratio'] = row[-1] / (row[0] + 1e-6)

        # ========== 3. 近期特征（4个） ==========
        feat['last_1'] = row[-1]
        feat['last_2'] = row[-2] if window_size >= 2 else row[-1]
        feat['last_3'] = row[-3] if window_size >= 3 else row[-1]
        feat['ma_3'] = np.mean(row[-3:]) if window_size >= 3 else row[-1]

        # ========== 4. 波动特征（2个） ==========
        feat['volatility'] = np.std(row)
        feat['volatility_recent'] = np.std(row[-3:]) if window_size >= 3 else np.std(row)

        # ========== 5. 变化率特征（2个） ==========
        feat['change_1_2'] = (row[-1] - row[-2]) / (row[-2] + 1e-6) if window_size >= 2 else 0
        feat['mean_change'] = np.mean(np.diff(row[-3:])) if window_size >= 3 else 0

        # ========== 6. 峰值特征（2个） ==========
        feat['is_peak'] = 1 if row[-1] == np.max(row) else 0
        feat['peak_ratio'] = row[-1] / (np.max(row) + 1e-6)

        # ========== 7. 滞后值（3个） ==========
        # 只保留最后3个滞后值，减少冗余
        feat['lag_1'] = row[-1]
        feat['lag_2'] = row[-2] if window_size >= 2 else row[-1]
        feat['lag_3'] = row[-3] if window_size >= 3 else row[-1]

        features.append(feat)

    X_feat = pd.DataFrame(features)

    # 去重：移除重复的列（lag_1 和 last_1 重复）
    X_feat = X_feat.drop(columns=['lag_1'], errors='ignore')

    # 处理缺失值
    X_feat = X_feat.fillna(0)
    X_feat = X_feat.replace([np.inf, -np.inf], 0)

    if y is not None:
        return X_feat, y
    return X_feat, None


def get_feature_names() -> list:
    """返回特征名称列表"""
    return [
        'mean', 'std', 'max', 'min',  # 统计特征
        'slope', 'trend_ratio',  # 趋势特征
        'last_1', 'last_2', 'last_3', 'ma_3',  # 近期特征
        'volatility', 'volatility_recent',  # 波动特征
        'change_1_2', 'mean_change',  # 变化率
        'is_peak', 'peak_ratio',  # 峰值特征
        'lag_2', 'lag_3'  # 滞后值
    ]