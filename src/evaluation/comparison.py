"""
模型对比工具
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from .metrics import calculate_metrics, dm_test


def compare_models(y_true: np.ndarray,
                   predictions: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    对比多个模型的预测性能

    Args:
        y_true: 真实值
        predictions: {'model_name': y_pred}

    Returns:
        DataFrame: 各模型的评估指标，按RMSE升序排列
    """
    results = []

    for name, y_pred in predictions.items():
        metrics = calculate_metrics(y_true, y_pred)
        results.append({
            'model': name,
            'rmse': metrics['rmse'],
            'mae': metrics['mae'],
            'mse': metrics['mse'],
            'r2': metrics['r2']
        })

    df = pd.DataFrame(results)
    df = df.sort_values('rmse').reset_index(drop=True)

    return df


def significance_test(y_true: np.ndarray,
                      predictions: Dict[str, np.ndarray],
                      baseline_model: Optional[str] = None) -> pd.DataFrame:
    """
    对所有模型进行两两DM检验，与最优模型对比

    Args:
        y_true: 真实值
        predictions: {'model_name': y_pred}
        baseline_model: 指定基准模型，如果None则使用RMSE最优模型

    Returns:
        DataFrame: 检验结果
    """
    # 确定最优模型（按RMSE）
    if baseline_model is None:
        best_model = min(predictions.keys(),
                         key=lambda k: np.sqrt(np.mean((y_true - predictions[k]) ** 2)))
        baseline_model = best_model

    results = []

    for name, y_pred in predictions.items():
        if name == baseline_model:
            continue

        dm_stat, p_value = dm_test(y_true, predictions[baseline_model], y_pred, one_sided='less')

        # 判断显著性
        if p_value < 0.01:
            significance = '***'  # p < 0.01
        elif p_value < 0.05:
            significance = '**'   # p < 0.05
        elif p_value < 0.1:
            significance = '*'    # p < 0.1
        else:
            significance = ''

        results.append({
            'comparison': f'{baseline_model} vs {name}',
            'baseline': baseline_model,
            'target': name,
            'dm_stat': dm_stat,
            'p_value': p_value,
            'significant_05': p_value < 0.05,
            'significant_01': p_value < 0.01,
            'significance': significance,
            'conclusion': '优于' if dm_stat < 0 else '劣于'
        })

    return pd.DataFrame(results)


def full_pairwise_test(y_true: np.ndarray,
                       predictions: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    对所有模型进行两两DM检验（全配对）

    Args:
        y_true: 真实值
        predictions: {'model_name': y_pred}

    Returns:
        DataFrame: 所有配对检验结果
    """
    model_names = list(predictions.keys())
    results = []

    for i, name1 in enumerate(model_names):
        for j, name2 in enumerate(model_names):
            if i >= j:
                continue

            y_pred1 = predictions[name1]
            y_pred2 = predictions[name2]

            # 检验模型1是否优于模型2
            dm_stat, p_value = dm_test(y_true, y_pred1, y_pred2, one_sided='less')

            results.append({
                'model_1': name1,
                'model_2': name2,
                'dm_stat': dm_stat,
                'p_value': p_value,
                'model1_better_than_model2': p_value < 0.05,
                'better_model': name1 if p_value < 0.05 else (name2 if (1 - p_value) < 0.05 else 'tie')
            })

    return pd.DataFrame(results)


def format_results_table(results_df: pd.DataFrame) -> str:
    """
    格式化结果表格为LaTeX格式（用于论文）

    Args:
        results_df: compare_models 返回的DataFrame

    Returns:
        str: LaTeX表格代码
    """
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{各模型预测性能对比}",
        "\\label{tab:model_comparison}",
        "\\begin{tabular}{lcccc}",
        "\\hline",
        "模型 & RMSE & MAE & MSE & R² \\\\",
        "\\hline"
    ]

    for _, row in results_df.iterrows():
        lines.append(
            f"{row['model']} & {row['rmse']:.4f} & {row['mae']:.4f} & "
            f"{row['mse']:.4f} & {row['r2']:.4f} \\\\"
        )

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def format_significance_table(significance_df: pd.DataFrame) -> str:
    """
    格式化显著性检验表格为LaTeX格式（用于论文）

    Args:
        significance_df: significance_test 返回的DataFrame

    Returns:
        str: LaTeX表格代码
    """
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Diebold-Mariano检验结果}",
        "\\label{tab:dm_test}",
        "\\begin{tabular}{lccc}",
        "\\hline",
        "对比 & DM统计量 & p值 & 显著性 \\\\",
        "\\hline"
    ]

    for _, row in significance_df.iterrows():
        sig_mark = ""
        if row['p_value'] < 0.01:
            sig_mark = "***"
        elif row['p_value'] < 0.05:
            sig_mark = "**"
        elif row['p_value'] < 0.1:
            sig_mark = "*"

        lines.append(
            f"{row['comparison']} & {row['dm_stat']:.4f} & "
            f"{row['p_value']:.4f} & {sig_mark} \\\\"
        )

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)