"""
LSTM模型
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings

warnings.filterwarnings('ignore')

from .base import BaseModel


class LSTMModel(BaseModel):
    """LSTM模型"""

    def __init__(self, config: dict):
        super().__init__('LSTM', config)
        self.units = config.get('units', 128)
        self.dropout = config.get('dropout', 0.2)
        self.epochs = config.get('epochs', 200)
        self.batch_size = config.get('batch_size', 32)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.verbose = config.get('verbose', 1)
        self.random_seed = config.get('random_seed', 42)

        # 设置 TensorFlow 随机种子
        tf.random.set_seed(self.random_seed)

    def _build_model(self):
        self.model = Sequential([
            LSTM(self.units, return_sequences=False, input_shape=(None, 1)),
            Dropout(self.dropout),
            Dense(1)
        ])

    def _fit_impl(self, X_train: np.ndarray, y_train: np.ndarray):
        # 重塑为3D输入 (samples, timesteps, features)
        X_train_3d = X_train.reshape(-1, X_train.shape[1], 1)

        # 划分验证集
        val_size = int(len(X_train_3d) * 0.1)
        X_val = X_train_3d[-val_size:]
        y_val = y_train[-val_size:]
        X_train_3d = X_train_3d[:-val_size]
        y_train = y_train[:-val_size]

        self._build_model()

        self.model.compile(
            optimizer=Adam(self.learning_rate),
            loss='mse'
        )

        callbacks = [
            EarlyStopping(patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10)
        ]

        self.model.fit(
            X_train_3d, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=self.verbose
        )

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        X_test_3d = X_test.reshape(-1, X_test.shape[1], 1)
        y_pred = self.model.predict(X_test_3d, verbose=0)
        return y_pred.flatten()

    def save(self, path: str):
        """保存 LSTM 模型"""
        import os

        if self.model is None:
            raise ValueError("模型未训练")

        os.makedirs(os.path.dirname(path), exist_ok=True)

        # 确保使用正确的扩展名
        if not (path.endswith('.keras') or path.endswith('.h5')):
            path = path.replace('.pkl', '.keras')
            if not (path.endswith('.keras') or path.endswith('.h5')):
                path = path + '.keras'

        self.model.save(path)
        print(f"[{self.name}] 模型已保存: {path}")