"""
Prophet模型
"""
import numpy as np
import pandas as pd
from prophet import Prophet
import warnings

warnings.filterwarnings('ignore')

from .base import BaseModel


class ProphetModel(BaseModel):
    """Prophet时间序列模型"""

    def __init__(self, config: dict):
        super().__init__('Prophet', config)
        self.yearly_seasonality = config.get('yearly_seasonality', True)
        self.weekly_seasonality = config.get('weekly_seasonality', False)
        self.daily_seasonality = config.get('daily_seasonality', False)
        self.seasonality_mode = config.get('seasonality_mode', 'additive')
        self.train_series = None
        self.last_date = None

    def _build_model(self):
        self.model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            seasonality_mode=self.seasonality_mode,
            # 添加以下参数避免 Windows 优化问题
            interval_width=0.95,
            mcmc_samples=0,  # 不使用 MCMC
            stan_backend='CMDSTANPY'  # 指定后端
        )

    def _fit_impl(self, X_train: np.ndarray, y_train: np.ndarray):
        # 判断是否为原始序列
        if len(X_train.shape) == 1:
            self.train_series = X_train.flatten()
        elif len(X_train.shape) == 2 and X_train.shape[1] == 1:
            self.train_series = X_train.flatten()
        else:
            first_window = X_train[0]
            self.train_series = np.concatenate([first_window, y_train])

        # 生成日期索引
        n = len(self.train_series)
        dates = pd.date_range(start='2006-01-01', periods=n, freq='MS')
        self.last_date = dates[-1]

        # 创建DataFrame
        df = pd.DataFrame({'ds': dates, 'y': self.train_series})

        self._build_model()

        try:

            self.model.fit(df)
        except Exception as e:
            print(f"[Prophet] 优化失败，使用简化参数重试...")
            # 简化模型：只使用年度季节性
            self.model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=5.0
            )
            self.model.fit(df)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        n_steps = len(X_test)

        future_dates = pd.date_range(
            start=self.last_date + pd.DateOffset(months=1),
            periods=n_steps,
            freq='MS'
        )
        future_df = pd.DataFrame({'ds': future_dates})

        forecast = self.model.predict(future_df)
        return forecast['yhat'].values.flatten()