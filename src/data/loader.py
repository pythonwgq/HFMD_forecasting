"""
数据加载器 - 统一数据加载接口
"""
import pandas as pd
import numpy as np
from typing import Optional, Tuple


class DataLoader:
    """数据加载器"""

    def __init__(self, config: dict):
        self.config = config
        self.exclude_years = config.get('data', {}).get('exclude_years', [2020, 2021, 2022])

    def _load_csv(self, path: str, province_name: str = '') -> pd.Series:
        """加载CSV文件并返回发病率序列"""
        df = pd.read_csv(path)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        # 剔除疫情年份
        df = df[~df.index.year.isin(self.exclude_years)]

        # 取发病率列
        if 'rate' in df.columns:
            data = df['rate']
        else:
            data = df.iloc[:, 0]

        print(f"[DataLoader] {province_name} 数据加载成功")
        print(f"  时间范围: {data.index[0]} ~ {data.index[-1]}")
        print(f"  数据长度: {len(data)}")

        return data

    def load_yunnan(self) -> pd.Series:
        """加载云南数据"""
        path = self.config['data']['yunnan']
        return self._load_csv(path, '云南')

    def load_guangdong(self) -> pd.Series:
        """加载广东数据"""
        path = self.config['data']['guangdong']
        return self._load_csv(path, '广东')

    def load_shandong(self) -> pd.Series:
        """加载山东数据"""
        path = self.config['data']['shandong']
        return self._load_csv(path, '山东')

    def load_beijing(self) -> pd.Series:
        """加载北京数据"""
        path = self.config['data']['beijing']
        return self._load_csv(path, '北京')

    def load_all(self) -> dict:
        """加载所有省份数据"""
        return {
            'yunnan': self.load_yunnan(),
            'guangdong': self.load_guangdong(),
            'shandong': self.load_shandong(),
            'beijing': self.load_beijing()
        }