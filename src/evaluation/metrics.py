"""
评估指标计算
"""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import norm
from typing import Dict, Tuple


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    计算评估指标

    Args:
        y_true: 真实值
        y_pred: 预测值

    Returns:
        dict: {'rmse': float, 'mae': float, 'mse': float, 'r2': float}
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    # R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }


def dm_test(actual: np.ndarray, pred1: np.ndarray, pred2: np.ndarray,
            h: int = 1, one_sided: str = 'less') -> Tuple[float, float]:
    """
    Diebold-Mariano检验

    比较两个预测模型的预测误差

    Args:
        actual: 真实值
        pred1: 模型1预测值
        pred2: 模型2预测值
        h: 预测步长
        one_sided: 单侧检验方向
            - 'less': 检验模型1误差是否小于模型2
            - 'greater': 检验模型1误差是否大于模型2
            - 'two-sided': 双侧检验

    Returns:
        (dm_stat, p_value)
    """
    # 计算误差
    e1 = actual - pred1
    e2 = actual - pred2

    # 损失函数：平方误差
    d = e1 ** 2 - e2 ** 2

    # 计算均值和方差
    d_mean = np.mean(d)
    d_var = np.var(d, ddof=1)
    T = len(d)

    # 计算DM统计量
    dm_stat = d_mean / np.sqrt(d_var / T)

    # 计算p值
    if one_sided == 'less':
        p_value = norm.cdf(dm_stat)
    elif one_sided == 'greater':
        p_value = 1 - norm.cdf(dm_stat)
    else:
        p_value = 2 * (1 - norm.cdf(abs(dm_stat)))

    return dm_stat, p_value


def compute_confidence_interval(y_true: np.ndarray, y_pred: np.ndarray,
                                alpha: float = 0.05) -> Dict[str, float]:
    """
    计算预测误差的置信区间

    Args:
        y_true: 真实值
        y_pred: 预测值
        alpha: 显著性水平

    Returns:
        dict: {'lower': float, 'upper': float, 'mean': float}
    """
    from scipy.stats import t

    errors = y_true - y_pred
    mean_error = np.mean(errors)
    std_error = np.std(errors, ddof=1)
    n = len(errors)

    t_value = t.ppf(1 - alpha / 2, n - 1)
    margin = t_value * std_error / np.sqrt(n)

    return {
        'lower': mean_error - margin,
        'upper': mean_error + margin,
        'mean': mean_error
    }


def calculate_improvement(metrics1: Dict, metrics2: Dict) -> Dict[str, float]:
    """
    计算模型2相对于模型1的提升百分比

    Args:
        metrics1: 基准模型指标
        metrics2: 对比模型指标

    Returns:
        dict: 各指标提升百分比
    """
    improvement = {}
    for key in metrics1.keys():
        if key in metrics2:
            improvement[key] = (metrics1[key] - metrics2[key]) / metrics1[key] * 100

    return improvement