"""
数据处理器 - 数据预处理、滑动窗口生成
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler


class DataProcessor:
    """数据处理器"""

    def __init__(self, config: dict):
        self.config = config
        self.window_size = config.get('window_size', 6)
        self.test_ratio = config.get('test_ratio', 0.1)
        self.scaler = StandardScaler()
        self._is_fitted = False

    def split_train_test(self, data: pd.Series) -> Tuple[np.ndarray, np.ndarray, pd.Index, pd.Index]:
        """
        按时间顺序划分训练集和测试集

        Returns:
            train_data, test_data, train_indices, test_indices
        """
        data_values = data.values
        data_indices = data.index

        test_size = int(len(data_values) * self.test_ratio)

        train_data = data_values[:-test_size]
        test_data = data_values[-test_size:]
        train_indices = data_indices[:-test_size]
        test_indices = data_indices[-test_size:]

        print(f"[DataProcessor] 数据划分完成")
        print(f"  训练集: {len(train_data)} 条 ({train_indices[0]} ~ {train_indices[-1]})")
        print(f"  测试集: {len(test_data)} 条 ({test_indices[0]} ~ {test_indices[-1]})")

        return train_data, test_data, train_indices, test_indices

    def create_sequences(self, data: np.ndarray, window_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成滑动窗口序列

        Args:
            data: 一维时间序列
            window_size: 窗口大小，默认使用配置中的值

        Returns:
            X: (n_samples, window_size)
            y: (n_samples,)
        """
        if window_size is None:
            window_size = self.window_size

        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:i + window_size])
            y.append(data[i + window_size])

        X = np.array(X)
        y = np.array(y)

        return X, y

    def normalize(self, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        标准化数据（仅对训练集拟合）
        """
        # 重塑为2D
        X_train_2d = X_train.reshape(-1, 1)

        self.scaler.fit(X_train_2d)
        self._is_fitted = True

        X_train_norm = self.scaler.transform(X_train_2d).reshape(X_train.shape)
        X_test_norm = self.scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)

        return X_train_norm, X_test_norm

    def inverse_normalize(self, data: np.ndarray) -> np.ndarray:
        """反标准化"""
        if not self._is_fitted:
            raise ValueError("请先调用 normalize 方法")
        return self.scaler.inverse_transform(data.reshape(-1, 1)).flatten()

    def get_train_test_sequences(self, data: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        获取完整的训练和测试序列（滑动窗口形式）

        Returns:
            X_train, y_train, X_test, y_test
        """
        train_raw, test_raw, _, _ = self.split_train_test(data)

        X_train, y_train = self.create_sequences(train_raw)
        X_test, y_test = self.create_sequences(test_raw)

        return X_train, y_train, X_test, y_test