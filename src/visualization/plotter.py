"""
绘图工具
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


def plot_prediction(y_true: np.ndarray, y_pred: np.ndarray,
                    title: str = "Prediction Results",
                    save_path: Optional[str] = None,
                    show: bool = True) -> None:
    """
    绘制单模型预测结果

    Args:
        y_true: 真实值
        y_pred: 预测值
        title: 图表标题
        save_path: 保存路径
        show: 是否显示
    """
    plt.figure(figsize=(12, 5))

    plt.plot(y_true, label='实际值', marker='o', markersize=4, linewidth=1.5, color='black')
    plt.plot(y_pred, label='预测值', alpha=0.7, marker='v', markersize=4, linewidth=1.5)

    plt.xlabel('时间步')
    plt.ylabel('发病率')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_cross_province(results: Dict[str, Dict],
                        save_path: Optional[str] = None,
                        show: bool = True,
                        start_date: str = '2017-09') -> None:
    """
    绘制跨省验证结果

    Args:
        results: {'province': {'y_true': array, 'y_pred': array, 'rmse': float, 'indices': array}}
        save_path: 保存路径
        show: 是否显示
        start_date: 起始日期，格式 'YYYY-MM'，默认 '2017-09'（28个月到2019年12月）
    """
    n_provinces = len(results)
    fig, axes = plt.subplots(n_provinces, 1, figsize=(12, 5 * n_provinces))

    if n_provinces == 1:
        axes = [axes]

    # 生成日期标签
    start_year, start_month = map(int, start_date.split('-'))

    for i, (name, res) in enumerate(results.items()):
        ax = axes[i]

        n_points = len(res['y_true'])

        # 生成日期标签
        dates = []
        for j in range(n_points):
            year = start_year + (start_month + j - 1) // 12
            month = (start_month + j - 1) % 12 + 1
            dates.append(f"{year}-{month:02d}")

        # 设置x轴刻度位置（每隔3个月显示一个标签，避免拥挤）
        tick_positions = list(range(0, n_points, 3))
        tick_labels = [dates[k] for k in tick_positions]

        # 绘制实际值和预测值
        ax.plot(range(n_points), res['y_true'], label='实际值', marker='o',
                markersize=4, linewidth=1.5, color='black')
        ax.plot(range(n_points), res['y_pred'], label='预测值', alpha=0.7, marker='v',
                markersize=4, linewidth=1.5, color='#2ecc71')

        ax.set_xlabel('日期')
        ax.set_ylabel('发病率')
        ax.set_title(f'{name} (RMSE={res["rmse"]:.4f})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 设置x轴刻度
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right')

        # 设置x轴范围，让图表更美观
        ax.set_xlim(-0.5, n_points - 0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_comparison(y_true: np.ndarray,
                    predictions: Dict[str, np.ndarray],
                    title: str = "模型预测对比",
                    save_path: Optional[str] = None,
                    show: bool = True,
                    start_date: Optional[str] = None) -> None:
    """
    绘制多模型预测对比图

    Args:
        y_true: 真实值
        predictions: {'model_name': y_pred}
        title: 图表标题
        save_path: 保存路径
        show: 是否显示
        start_date: 起始日期，格式 'YYYY-MM'，如 '2018-01'
    """
    plt.figure(figsize=(14, 6))

    n_points = len(y_true)

    # 生成日期标签
    if start_date is not None:
        start_year, start_month = map(int, start_date.split('-'))
        dates = []
        for i in range(n_points):
            year = start_year + (start_month + i - 1) // 12
            month = (start_month + i - 1) % 12 + 1
            dates.append(f"{year}-{month:02d}")

        # 每隔3个月显示一个标签
        tick_positions = list(range(0, n_points, 3))
        tick_labels = [dates[k] for k in tick_positions]
        x_labels = dates
    else:
        x_labels = range(n_points)
        tick_positions = list(range(0, n_points, max(1, n_points // 10)))
        tick_labels = tick_positions

    # 绘制实际值
    plt.plot(x_labels, y_true, label='实际值', marker='o',
             markersize=4, linewidth=2, color='black')

    # 绘制各模型预测值
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']
    for i, (name, y_pred) in enumerate(predictions.items()):
        plt.plot(x_labels, y_pred, label=name, alpha=0.8,
                 marker='v', markersize=3, linewidth=1.5,
                 color=colors[i % len(colors)])

    plt.xlabel('日期' if start_date else '时间步')
    plt.ylabel('发病率')
    plt.title(title)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)

    # 设置x轴刻度
    if start_date:
        plt.xticks(tick_positions, tick_labels, rotation=45, ha='right')
    else:
        plt.xticks(tick_positions, tick_labels)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
def plot_cross_province(results: Dict[str, Dict],
                        save_path: Optional[str] = None,
                        show: bool = True) -> None:
    """
    绘制跨省验证结果

    Args:
        results: {'province': {'y_true': array, 'y_pred': array, 'rmse': float}}
        save_path: 保存路径
        show: 是否显示
    """
    n_provinces = len(results)
    fig, axes = plt.subplots(n_provinces, 1, figsize=(12, 5 * n_provinces))

    if n_provinces == 1:
        axes = [axes]

    for i, (name, res) in enumerate(results.items()):
        ax = axes[i]
        ax.plot(res['y_true'], label='实际值', marker='o', markersize=4,
                linewidth=1.5, color='black')
        ax.plot(res['y_pred'], label='预测值', alpha=0.7, marker='v',
                markersize=4, linewidth=1.5)
        ax.set_xlabel('时间步')
        ax.set_ylabel('发病率')
        ax.set_title(f'{name} (RMSE={res["rmse"]:.4f})')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_training_history(history: dict,
                          title: str = "Training History",
                          save_path: Optional[str] = None,
                          show: bool = True) -> None:
    """
    绘制训练历史

    Args:
        history: {'loss': [], 'val_loss': []}
        title: 图表标题
        save_path: 保存路径
        show: 是否显示
    """
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.get('loss', []), label='训练损失')
    plt.plot(history.get('val_loss', []), label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('训练损失')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    if 'r2' in history:
        plt.plot(history.get('r2', []), label='训练 R²')
        plt.plot(history.get('val_r2', []), label='验证 R²')
        plt.xlabel('Epoch')
        plt.ylabel('R²')
        plt.legend()
        plt.title('R² 分数')
        plt.grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_error_distribution(y_true: np.ndarray,
                            predictions: Dict[str, np.ndarray],
                            save_path: Optional[str] = None,
                            show: bool = True) -> None:
    """
    绘制误差分布箱线图

    Args:
        y_true: 真实值
        predictions: {'model_name': y_pred}
        save_path: 保存路径
        show: 是否显示
    """
    plt.figure(figsize=(10, 6))

    errors = []
    labels = []

    for name, y_pred in predictions.items():
        error = y_true - y_pred
        errors.append(error)
        labels.append(name)

    bp = plt.boxplot(errors, labels=labels, patch_artist=True)

    # 设置颜色
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']
    for patch, color in zip(bp['boxes'], colors[:len(labels)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    plt.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    plt.ylabel('预测误差')
    plt.title('各模型预测误差分布对比')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_feature_importance(importance_df: pd.DataFrame,
                            top_n: int = 20,
                            title: str = "特征重要性",
                            save_path: Optional[str] = None,
                            show: bool = True) -> None:
    """
    绘制特征重要性（用于LightGBM等树模型）

    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        top_n: 显示前N个特征
        title: 图表标题
        save_path: 保存路径
        show: 是否显示
    """
    df = importance_df.head(top_n).sort_values('importance', ascending=True)

    plt.figure(figsize=(10, 8))
    plt.barh(df['feature'], df['importance'], color='steelblue')
    plt.xlabel('重要性')
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()