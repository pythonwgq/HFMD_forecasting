"""
CNN-BiLSTM模型 - 简化版（适合小样本数据）
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, Bidirectional, LSTM, Dense, Dropout,
    BatchNormalization, GlobalAveragePooling1D, Flatten
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os

from .base import BaseModel


class CNNBiLSTMModel(BaseModel):
    """
    CNN-BiLSTM混合模型 - 简化版

    特点:
        - 单尺度CNN（减少参数量）
        - 简化的BiLSTM层
        - 无解码器，直接输出
    """

    def __init__(self, config: dict):
        super().__init__('CNN-BiLSTM', config)
        self.random_seed = config.get('random_seed', 42)

        # 设置 TensorFlow 随机种子
        tf.random.set_seed(self.random_seed)

        # CNN参数（简化）
        self.cnn_filters = config.get('cnn_filters', 32)  # 64 → 32
        self.cnn_kernel_size = config.get('cnn_kernel_size', 3)  # 单尺度
        self.cnn_dilation_rate = config.get('cnn_dilation_rate', 1)  # 无膨胀

        # LSTM参数（简化）
        self.lstm_units = config.get('lstm_units', 64)  # 128 → 64
        self.bidirectional = config.get('bidirectional', True)

        # 训练参数
        self.epochs = config.get('epochs', 150)  # 200 → 150
        self.batch_size = config.get('batch_size', 16)  # 32 → 16
        self.learning_rate = config.get('learning_rate', 0.001)
        self.dropout = config.get('dropout', 0.3)  # 0.2 → 0.3

        self.verbose = config.get('verbose', 1)
        self.window_size = None

    def _build_model(self):
        """构建简化的CNN-BiLSTM模型"""

        model = Sequential([
            # 输入层
            tf.keras.layers.Input(shape=(self.window_size, 1)),

            # CNN层
            Conv1D(
                self.cnn_filters,
                kernel_size=self.cnn_kernel_size,
                activation='relu',
                padding='same'
            ),
            BatchNormalization(),
            Dropout(self.dropout),

            # 第二层CNN（可选，增加一点深度）
            Conv1D(
                self.cnn_filters // 2,
                kernel_size=self.cnn_kernel_size,
                activation='relu',
                padding='same'
            ),
            BatchNormalization(),
            Dropout(self.dropout),

            # BiLSTM层
            Bidirectional(
                LSTM(self.lstm_units, return_sequences=False),
                merge_mode='concat'
            ),
            Dropout(self.dropout),

            # 输出层
            Dense(1)
        ])

        self.model = model

    def _fit_impl(self, X_train: np.ndarray, y_train: np.ndarray):
        """训练模型"""
        self.window_size = X_train.shape[1]

        # 重塑输入为3D (samples, timesteps, features)
        X_train_3d = X_train.reshape(-1, self.window_size, 1)

        # 划分验证集
        val_size = int(len(X_train_3d) * 0.1)
        X_val = X_train_3d[-val_size:]
        y_val = y_train[-val_size:]
        X_train_3d = X_train_3d[:-val_size]
        y_train = y_train[:-val_size]

        # 构建模型
        self._build_model()

        # 打印模型结构
        if self.verbose:
            self.model.summary()

        # 编译模型
        self.model.compile(
            optimizer=Adam(self.learning_rate),
            loss='mse'
        )

        # 回调函数
        callbacks = [
            EarlyStopping(
                patience=15,  # 20 → 15
                restore_best_weights=True,
                min_delta=0.001
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-6
            )
        ]

        # 训练
        history = self.model.fit(
            X_train_3d, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=self.verbose
        )

        # 保存训练历史
        self.history = history

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """预测"""
        X_test_3d = X_test.reshape(-1, self.window_size, 1)
        y_pred = self.model.predict(X_test_3d, verbose=0)
        return y_pred.flatten()

    def save(self, path: str):
        """保存模型"""
        if self.model is None:
            raise ValueError("模型未训练，请先调用 fit 方法")

        os.makedirs(os.path.dirname(path), exist_ok=True)

        # 确保使用正确的扩展名
        if not (path.endswith('.keras') or path.endswith('.h5')):
            path = path.replace('.pkl', '.keras')
            if not (path.endswith('.keras') or path.endswith('.h5')):
                path = path + '.keras'

        self.model.save(path)
        print(f"[{self.name}] 模型已保存: {path}")

    def load(self, path: str):
        """加载模型"""
        from tensorflow.keras.models import load_model
        self.model = load_model(path)
        self.is_fitted = True
        print(f"[{self.name}] 模型已加载: {path}")