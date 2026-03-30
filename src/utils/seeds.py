"""
随机种子管理 - 保证实验可复现
"""
import random
import numpy as np
import torch
import os
from typing import Optional


def set_global_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    固定所有随机种子

    Args:
        seed: 随机种子值
        deterministic: 是否启用确定性算法（可能降低性能）
    """
    # 环境变量 - 必须在导入 TensorFlow 之前设置
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    # Python随机种子
    random.seed(seed)

    # NumPy随机种子
    np.random.seed(seed)

    # 尝试导入并设置 TensorFlow
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        if deterministic:
            tf.config.experimental.enable_op_determinism()
    except ImportError:
        pass

    # PyTorch随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"[Seeds] 全局随机种子已固定: {seed}")


def get_random_state(seed: Optional[int] = None) -> np.random.RandomState:
    """获取随机状态对象"""
    if seed is None:
        seed = 42
    return np.random.RandomState(seed)


def reset_seeds(seed: int = 42) -> None:
    """重置所有随机种子"""
    set_global_seed(seed, deterministic=True)


class SeedManager:
    """种子管理器 - 用于上下文管理"""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._original_state = None

    def __enter__(self):
        """进入上下文，保存当前状态并设置新种子"""
        self._original_state = {
            'random_state': random.getstate(),
            'numpy_state': np.random.get_state()
        }
        set_global_seed(self.seed)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文，恢复原始状态"""
        if self._original_state:
            random.setstate(self._original_state['random_state'])
            np.random.set_state(self._original_state['numpy_state'])
        self._original_state = None