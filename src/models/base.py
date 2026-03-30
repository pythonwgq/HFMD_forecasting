"""
模型基类 - 定义统一的模型接口
"""
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Any
import joblib
import os


class BaseModel(ABC):
    """
    所有预测模型的基类

    统一接口:
        - fit(X_train, y_train): 训练模型
        - predict(X_test): 预测
        - evaluate(X_test, y_test): 评估
        - save(path): 保存模型
        - load(path): 加载模型
    """

    def __init__(self, name: str, config: dict = None):
        self.name = name
        self.config = config or {}
        self.model = None
        self.is_fitted = False

    @abstractmethod
    def _build_model(self):
        """构建模型结构（子类实现）"""
        pass

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> 'BaseModel':
        """训练模型"""
        self._build_model()
        self._fit_impl(X_train, y_train)
        self.is_fitted = True
        return self

    @abstractmethod
    def _fit_impl(self, X_train: np.ndarray, y_train: np.ndarray):
        """具体训练逻辑（子类实现）"""
        pass

    @abstractmethod
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """预测"""
        pass

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        评估模型

        Returns:
            dict: {'rmse': float, 'mae': float}
        """
        from src.evaluation.metrics import calculate_metrics

        y_pred = self.predict(X_test)
        metrics = calculate_metrics(y_test, y_pred)
        return metrics

    def save(self, path: str):
        """
        保存模型

        Args:
            path: 保存路径，建议使用 .keras 或 .h5 扩展名
        """
        import os
        import joblib

        os.makedirs(os.path.dirname(path), exist_ok=True)

        # 对于深度学习模型（有 self.model），使用 Keras 保存
        if hasattr(self, 'model') and self.model is not None:
            # 确保扩展名正确
            if not (path.endswith('.keras') or path.endswith('.h5')):
                path = path.replace('.pkl', '.keras')
                if not (path.endswith('.keras') or path.endswith('.h5')):
                    path = path + '.keras'

            self.model.save(path)
            print(f"[{self.name}] 模型已保存: {path}")
        else:
            # 对于传统模型，使用 joblib
            joblib.dump(self.model, path)
            print(f"[{self.name}] 模型已保存: {path}")

    def load(self, path: str):
        """加载模型"""
        self.model = joblib.load(path)
        self.is_fitted = True
        print(f"[{self.name}] 模型已加载: {path}")