"""
数据增强器 - 基于TSAUG的时序数据增强
"""
import numpy as np
import pandas as pd
import pickle
import os
from typing import Optional, List, Tuple
from tsaug import TimeWarp, Quantize, Drift, AddNoise
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import STL
import warnings

warnings.filterwarnings('ignore')


class DataAugmenter:
    """数据增强器（带缓存功能）"""

    def __init__(self, config: dict):
        self.config = config
        aug_config = config.get('augmentation', {})

        self.enabled = aug_config.get('enabled', False)
        # 兼容两种配置：n_sequences（轮数）和 m_sequences（目标段数）
        self.max_attempts = aug_config.get('n_sequences', 5000)  # 最大尝试次数
        self.n_sequences = aug_config.get('m_sequences', 50)  # 目标增强段数
        self.similarity_threshold = aug_config.get('similarity_threshold', 0.85)

        # 原始数据重复倍数
        self.original_repeat = aug_config.get('original_repeat', 2)

        # 缓存路径（从配置文件获取）
        self.cache_path = aug_config.get('cache_path', None)

        # 配置增强器
        augmenters = aug_config.get('augmenters', {})

        self.augmenter = None
        if self.enabled:
            self._build_augmenter(augmenters)

        # 缓存数据
        self._cached_augmented_sequences = None
        self._cached_X_train = None
        self._cached_y_train = None

    def _build_augmenter(self, augmenters: dict):
        """构建增强器组合"""
        augmenter_list = []

        # 时间扭曲
        tw = augmenters.get('time_warp', {})
        if tw:
            augmenter_list.append(TimeWarp(
                n_speed_change=tw.get('n_speed_change', 3),
                max_speed_ratio=tw.get('max_speed_ratio', 1.03)
            ))

        # 量化
        q = augmenters.get('quantize', {})
        if q:
            augmenter_list.append(Quantize(n_levels=q.get('n_levels', 30)))

        # 趋势漂移
        d = augmenters.get('drift', {})
        if d:
            augmenter_list.append(Drift(
                max_drift=d.get('max_drift', 0.03),
                n_drift_points=d.get('n_drift_points', 2)
            ))

        # 噪声
        n = augmenters.get('noise', {})
        if n:
            augmenter_list.append(AddNoise(scale=n.get('scale', 0.002)))

        # 组合增强器
        if augmenter_list:
            self.augmenter = sum(augmenter_list[1:], augmenter_list[0])

    def _check_seasonality(self, original: np.ndarray, augmented: np.ndarray,
                           period: int = 12) -> Tuple[float, float]:
        """检查增强序列是否保持季节性和趋势"""
        min_len = min(len(original), len(augmented))

        try:
            orig = STL(original[:min_len], period=period).fit()
            aug = STL(augmented[:min_len], period=period).fit()

            similarity = np.corrcoef(orig.seasonal, aug.seasonal)[0, 1]
            trend = np.corrcoef(orig.trend, aug.trend)[0, 1]
        except Exception as e:
            print(f"  STL分解失败: {e}")
            similarity = 0.5
            trend = 0.5

        return similarity, trend

    def _generate_augmented_sequences(self, train_data: np.ndarray) -> List[np.ndarray]:
        """生成增强序列"""
        print(f"[DataAugmenter] 开始生成增强序列...")
        print(f"  目标增强段数: {self.n_sequences}")
        print(f"  最大尝试次数: {self.max_attempts}")

        # 标准化
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(train_data.reshape(-1, 1))
        ts_data = scaled_data.reshape(1, -1, 1)

        # 生成增强序列
        augmented_sequences = []

        for i in range(self.max_attempts):
            if len(augmented_sequences) >= self.n_sequences:
                break

            try:
                aug = self.augmenter.augment(ts_data)
                aug_flat = aug.reshape(-1, 1)
                aug_inv = scaler.inverse_transform(aug_flat).flatten()

                # 保真性检查
                similarity, trend = self._check_seasonality(train_data, aug_inv)

                if similarity >= self.similarity_threshold and trend >= self.similarity_threshold:
                    augmented_sequences.append(aug_inv)
                    if len(augmented_sequences) % 10 == 0:
                        print(f"  已生成: {len(augmented_sequences)}/{self.n_sequences}")
            except Exception as e:
                continue

        print(f"[DataAugmenter] 共生成 {len(augmented_sequences)} 个有效增强序列")
        return augmented_sequences

    def _save_to_cache(self, augmented_sequences: List[np.ndarray]):
        """保存增强序列到缓存"""
        if self.cache_path is None:
            print("[DataAugmenter] 未配置缓存路径，跳过保存")
            return

        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)

        cache_data = {
            'augmented_sequences': augmented_sequences,
            'n_sequences': self.n_sequences,
            'similarity_threshold': self.similarity_threshold,
            'original_repeat': self.original_repeat
        }

        with open(self.cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"[DataAugmenter] 增强序列已保存至缓存: {self.cache_path}")

    def _load_from_cache(self) -> Optional[List[np.ndarray]]:
        """从缓存加载增强序列"""
        if self.cache_path is None:
            return None

        if not os.path.exists(self.cache_path):
            return None

        try:
            with open(self.cache_path, 'rb') as f:
                cache_data = pickle.load(f)

            # 验证缓存参数是否匹配
            if (cache_data.get('n_sequences') == self.n_sequences and
                    cache_data.get('similarity_threshold') == self.similarity_threshold):
                print(f"[DataAugmenter] 从缓存加载增强序列: {self.cache_path}")
                print(f"  缓存包含 {len(cache_data['augmented_sequences'])} 个增强序列")
                return cache_data['augmented_sequences']
            else:
                print(f"[DataAugmenter] 缓存参数不匹配，将重新生成")
                return None
        except Exception as e:
            print(f"[DataAugmenter] 加载缓存失败: {e}")
            return None

    def augment(self, train_data: np.ndarray,
                X_train: np.ndarray, y_train: np.ndarray,
                force_regen: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        对训练数据进行增强

        Args:
            train_data: 原始训练序列（用于保真性检查）
            X_train: 滑动窗口输入
            y_train: 滑动窗口目标
            force_regen: 是否强制重新生成

        Returns:
            增强后的 X_train, y_train
        """
        if not self.enabled or self.augmenter is None:
            print("[DataAugmenter] 数据增强未启用")
            return X_train, y_train

        # 尝试从缓存加载
        augmented_sequences = None
        if not force_regen:
            augmented_sequences = self._load_from_cache()

        # 如果没有缓存或强制重新生成，则生成新的
        if augmented_sequences is None:
            augmented_sequences = self._generate_augmented_sequences(train_data)
            if augmented_sequences:
                self._save_to_cache(augmented_sequences)

        if not augmented_sequences:
            print("[DataAugmenter] 警告：未生成有效增强序列，使用原始数据")
            return X_train, y_train

        # 从增强序列生成滑动窗口样本
        X_list = []
        y_list = []

        # 1. 加入原始训练样本（重复多次）
        if self.original_repeat > 0:
            X_orig_repeated = np.repeat(X_train, self.original_repeat, axis=0)
            y_orig_repeated = np.repeat(y_train, self.original_repeat, axis=0)
            X_list.append(X_orig_repeated)
            y_list.append(y_orig_repeated)
            print(f"[DataAugmenter] 加入原始样本（重复{self.original_repeat}倍），共 {len(X_orig_repeated)} 条")

        # 2. 加入增强序列样本
        window_size = X_train.shape[1]
        for seq in augmented_sequences:
            X_seq, y_seq = self._create_sequences(seq, window_size)
            if len(X_seq) > 0:
                X_list.append(X_seq)
                y_list.append(y_seq)

        X_all = np.concatenate(X_list, axis=0)
        y_all = np.concatenate(y_list, axis=0)

        # 打乱
        indices = np.random.permutation(len(X_all))
        X_all = X_all[indices]
        y_all = y_all[indices]

        print(f"[DataAugmenter] 增强后总样本数: {len(X_all)}")
        print(f"  - 原始样本占比: {len(X_orig_repeated) / len(X_all) * 100:.1f}%")
        print(f"  - 增强样本占比: {(len(X_all) - len(X_orig_repeated)) / len(X_all) * 100:.1f}%")

        return X_all, y_all

    def _create_sequences(self, data: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """生成滑动窗口序列"""
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:i + window_size])
            y.append(data[i + window_size])
        return np.array(X), np.array(y)