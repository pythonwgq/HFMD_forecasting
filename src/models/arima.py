"""
ARIMA模型
"""
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from statsmodels.tsa.arima.model import ARIMA as StatsmodelsARIMA
from statsmodels.tsa.stattools import adfuller
import warnings

warnings.filterwarnings('ignore')

from .base import BaseModel


class ARIMAModel(BaseModel):
    """ARIMA时间序列模型"""

    def __init__(self, config: dict):
        super().__init__('ARIMA', config)
        self.order_search = config.get('order_search', True)
        self.p = config.get('p', 2)
        self.d = config.get('d', 1)
        self.q = config.get('q', 2)
        self.seasonal = config.get('seasonal', False)
        self.seasonal_order = config.get('seasonal_order', (0, 1, 0, 12))
        self.train_series = None

    def _build_model(self):
        pass

    def _check_stationarity(self, data: np.ndarray) -> int:
        data_series = pd.Series(data)
        d = 0
        current_data = data_series.copy()

        while d < 3:
            result = adfuller(current_data, autolag='AIC')
            if result[1] < 0.05:
                break
            current_data = current_data.diff().dropna()
            d += 1
        return d

    def _find_best_order(self, data: np.ndarray, max_p: int = 5, max_q: int = 5) -> Tuple[int, int, int]:
        from statsmodels.tsa.arima.model import ARIMA

        best_aic = np.inf
        best_order = (2, 1, 2)
        d = self._check_stationarity(data)

        for p in range(0, max_p + 1):
            for q in range(0, max_q + 1):
                try:
                    model = ARIMA(data, order=(p, d, q))
                    model_fit = model.fit()
                    if model_fit.aic < best_aic:
                        best_aic = model_fit.aic
                        best_order = (p, d, q)
                except:
                    continue
        return best_order

    def _fit_impl(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        ARIMA 使用原始时间序列训练
        如果传入的是滑动窗口，尝试重建原始序列
        """
        # 判断是否为原始序列（一维）
        if len(X_train.shape) == 1:
            self.train_series = X_train.flatten()
        elif len(X_train.shape) == 2 and X_train.shape[1] == 1:
            self.train_series = X_train.flatten()
        else:
            # 从滑动窗口重建原始序列
            first_window = X_train[0]
            self.train_series = np.concatenate([first_window, y_train])

        print(f"[ARIMA] 训练序列长度: {len(self.train_series)}")

        if self.order_search:
            self.p, self.d, self.q = self._find_best_order(self.train_series)
            print(f"[ARIMA] 最优阶数: p={self.p}, d={self.d}, q={self.q}")

        self.model = StatsmodelsARIMA(self.train_series, order=(self.p, self.d, self.q))
        if self.seasonal:
            self.model = StatsmodelsARIMA(
                self.train_series, order=(self.p, self.d, self.q),
                seasonal_order=self.seasonal_order
            )
        self.model_fit = self.model.fit()

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        n_steps = len(X_test)
        forecast = self.model_fit.forecast(steps=n_steps)
        return np.array(forecast).flatten()