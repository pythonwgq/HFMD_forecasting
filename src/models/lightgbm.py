"""
LightGBM模型
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

from .base import BaseModel


class LightGBMModel(BaseModel):
    """
    LightGBM梯度提升树模型

    适用于小样本数据，通过特征工程提取时间序列特征
    """

    def __init__(self, config: dict):
        super().__init__('LightGBM', config)

        # 模型参数
        self.n_estimators = config.get('n_estimators', 500)
        self.max_depth = config.get('max_depth', 6)
        self.learning_rate = config.get('learning_rate', 0.05)
        self.num_leaves = config.get('num_leaves', 31)
        self.feature_fraction = config.get('feature_fraction', 0.8)
        self.bagging_fraction = config.get('bagging_fraction', 0.8)
        self.bagging_freq = config.get('bagging_freq', 5)
        self.reg_alpha = config.get('reg_alpha', 0.1)
        self.reg_lambda = config.get('reg_lambda', 0.1)
        self.window_size = config.get('window_size',3)

        self.verbose = config.get('verbose', -1)

    def _build_model(self):
        """构建LightGBM模型"""
        self.model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            feature_fraction=self.feature_fraction,
            bagging_fraction=self.bagging_fraction,
            bagging_freq=self.bagging_freq,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=42,
            verbose=self.verbose
        )

    def _extract_features(self, X: np.ndarray) -> pd.DataFrame:
        """
            从滑动窗口数据中提取增强特征
            """
        from src.data.enhanced_features import build_features
        X_feat, _ = build_features(X, window_size=self.window_size)
        return X_feat

    def _fit_impl(self, X_train: np.ndarray, y_train: np.ndarray):
        """训练LightGBM模型"""
        # 提取特征
        X_train_feat = self._extract_features(X_train)

        # 划分验证集（时间顺序）
        val_size = int(len(X_train_feat) * 0.1)
        X_val_feat = X_train_feat[-val_size:]
        y_val = y_train[-val_size:]
        X_train_feat = X_train_feat[:-val_size]
        y_train = y_train[:-val_size]

        # 构建模型
        self._build_model()

        # 训练
        self.model.fit(
            X_train_feat, y_train,
            eval_set=[(X_val_feat, y_val)],
            eval_metric='rmse',
            callbacks=[
                lgb.early_stopping(50),
                lgb.log_evaluation(0)  # 不输出训练过程
            ]
        )

        # 保存特征提取器（用于预测时保持一致）
        self.feature_columns = X_train_feat.columns.tolist()



    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """预测"""
        X_test_feat = self._extract_features(X_test)

        # 确保列顺序一致
        X_test_feat = X_test_feat[self.feature_columns]

        y_pred = self.model.predict(X_test_feat)
        return y_pred.flatten()

    def save(self, path: str):
        """保存 LightGBM 模型"""
        import joblib
        import os

        if self.model is None:
            raise ValueError("模型未训练，请先调用 fit 方法")

        os.makedirs(os.path.dirname(path), exist_ok=True)

        # 确保使用 .pkl 扩展名
        if not path.endswith('.pkl'):
            path = path.replace('.keras', '.pkl')
            if not path.endswith('.pkl'):
                path = path + '.pkl'

        joblib.dump(self.model, path)
        print(f"[{self.name}] 模型已保存: {path}")

    def load(self, path: str):
        """加载 LightGBM 模型"""
        import joblib

        self.model = joblib.load(path)
        self.is_fitted = True
        print(f"[{self.name}] 模型已加载: {path}")