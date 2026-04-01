"""
长期预测专项评估指标计算
- MAPE (平均绝对百分比误差)
- 趋势准确率 (方向判断准确率)
- 累积误差 (误差随步长变化)
- 峰值比较和拐点检测
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False


def calculate_mape(y_true, y_pred):
    """
    计算平均绝对百分比误差 (MAPE)
    
    参数:
        y_true: 真实值
        y_pred: 预测值
    
    返回:
        mape: 平均绝对百分比误差 (%)
    """
    # 避免除零，当真实值接近0时使用小常数
    y_true_safe = np.where(np.abs(y_true) < 1e-6, 1e-6, y_true)
    mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100
    return mape


def calculate_trend_accuracy(y_true, y_pred):
    """
    计算趋势准确率
    
    趋势准确率 = 预测方向与实际方向一致的步数占比
    
    参数:
        y_true: 真实值序列
        y_pred: 预测值序列
    
    返回:
        accuracy: 趋势准确率 (%)
        correct_count: 正确预测趋势的次数
        total_count: 总趋势次数
    """
    if len(y_true) < 2:
        return 0, 0, 0
    
    # 计算实际变化方向
    true_direction = np.sign(np.diff(y_true))
    # 计算预测变化方向
    pred_direction = np.sign(np.diff(y_pred))
    
    # 统计方向一致的数量
    correct = np.sum(true_direction == pred_direction)
    total = len(true_direction)
    accuracy = correct / total * 100
    
    return accuracy, correct, total


def calculate_cumulative_relative_error(y_true, y_pred):
    """计算累积相对误差（百分比）"""
    absolute_errors = np.abs(y_true - y_pred)
    cumulative_abs_errors = np.cumsum(absolute_errors)
    cumulative_true = np.cumsum(y_true)
    cumulative_relative_errors = cumulative_abs_errors / cumulative_true * 100
    return cumulative_relative_errors

def calculate_stage_cumulative_error(y_true, y_pred, n_stages=4):
    """将28个月分为n_stage个阶段，计算各阶段累积误差"""
    n_months = len(y_true)
    stage_length = n_months // n_stages
    stage_errors = []
    for i in range(n_stages):
        start = i * stage_length
        end = (i + 1) * stage_length if i < n_stages - 1 else n_months
        stage_abs_errors = np.abs(y_true[start:end] - y_pred[start:end])
        stage_errors.append(np.sum(stage_abs_errors))
    return stage_errors

def calculate_all_metrics(y_true, y_pred, model_name):
    """
    计算所有评估指标
    
    参数:
        y_true: 真实值
        y_pred: 预测值
        model_name: 模型名称
    
    返回:
        dict: 包含所有指标
    """
    # 基础指标
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    mape = calculate_mape(y_true, y_pred)
    
    # 长期预测专项指标
    trend_acc, correct, total = calculate_trend_accuracy(y_true, y_pred)
    cumulative_errors = calculate_cumulative_relative_error(y_true, y_pred)
    
    return {
        'model': model_name,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'trend_accuracy': trend_acc,
        'trend_correct': correct,
        'trend_total': total,
        'cumulative_errors': cumulative_errors,
        'y_true': y_true,
        'y_pred': y_pred
    }


def plot_results(results_dict, save_path=None):
    """
    绘制评估结果图
    
    参数:
        results_dict: {model_name: metrics_dict}
        save_path: 保存路径
    """
    # 从字典中获取模型列表
    models = list(results_dict.keys())
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 图1: 趋势准确率
    ax1 = axes[0]
    trend_acc = [results_dict[m]['trend_accuracy'] for m in models]
    colors = ['#2ecc71' if m == 'LightGBM' else '#e74c3c' for m in models]
    bars = ax1.bar(models, trend_acc, color=colors)
    ax1.set_xlabel('模型')
    ax1.set_ylabel('趋势准确率 (%)')
    ax1.set_title('趋势准确率 (方向判断)')
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='随机水平')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar in bars:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=8)
    
    ax2 = axes[1]
    for model_name in models:
        y_true = results_dict[model_name]['y_true']
        y_pred = results_dict[model_name]['y_pred']
        cum_rel_error = calculate_cumulative_relative_error(y_true, y_pred)
        ax2.plot(range(1, len(cum_rel_error) + 1),cum_rel_error, 
                 label=model_name, linewidth=1.5)
    ax2.set_xlabel('预测步数')
    ax2.set_ylabel('累积相对误差 (%)')
    ax2.set_title('累积相对误差图')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=15, color='red', linestyle='--', linewidth=0.8, alpha=0.5)  # 20%参考线
    ax2.axhline(y=25, color='red', linestyle='--', linewidth=0.8, alpha=0.5)  # 50%参考线
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存至: {save_path}")
    plt.show()
    
def calculate_peak_error(y_true, y_pred):
    """
    计算峰值误差（峰值大小误差和峰值位置误差）
    
    参数:
        y_true: 真实值序列
        y_pred: 预测值序列
    
    返回:
        dict: 包含峰值相关指标
    """
    # 找到实际峰值位置（第一个最大值）
    actual_peak_idx = np.argmax(y_true)
    actual_peak_value = y_true[actual_peak_idx]
    
    # 找到预测峰值位置
    pred_peak_idx = np.argmax(y_pred)
    pred_peak_value = y_pred[pred_peak_idx]
    
    # 计算峰值大小误差
    peak_magnitude_error = np.abs(actual_peak_value - pred_peak_value)
    peak_magnitude_error_pct = (peak_magnitude_error / actual_peak_value) * 100 if actual_peak_value > 0 else 0
    
    # 计算峰值位置误差（月数）
    peak_position_error = pred_peak_idx - actual_peak_idx
    
    return {
        'peak_magnitude_error': peak_magnitude_error,
        'peak_magnitude_error_pct': peak_magnitude_error_pct,
        'peak_position_error': peak_position_error,
        'actual_peak_idx': actual_peak_idx,
        'pred_peak_idx': pred_peak_idx,
        'actual_peak_value': actual_peak_value,
        'pred_peak_value': pred_peak_value
    }


def calculate_turning_point_delay(y_true, y_pred, threshold_pct=5):
    """
    计算拐点延迟（上升拐点和下降拐点）
    
    拐点定义为趋势发生变化的点（从上升到下降或从下降到上升）
    
    参数:
        y_true: 真实值序列
        y_pred: 预测值序列
        threshold_pct: 判断趋势变化的幅度阈值（%），过滤微小波动
    
    返回:
        dict: 包含拐点相关指标
    """
    n = len(y_true)
    if n < 3:
        return {
            'rise_turning_point_delay': None,
            'fall_turning_point_delay': None,
            'actual_rise_idx': None,
            'pred_rise_idx': None,
            'actual_fall_idx': None,
            'pred_fall_idx': None
        }
    
    # 计算实际序列的趋势变化
    actual_direction = np.sign(np.diff(y_true))
    
    # 找出趋势变化点（方向改变的位置）
    actual_rise_idx = None
    actual_fall_idx = None
    
    for i in range(1, len(actual_direction)):
        if actual_direction[i] != actual_direction[i-1]:
            # 检查变化幅度是否超过阈值
            if i+1 < len(y_true) and y_true[i] > 0:
                change_magnitude = abs(y_true[i+1] - y_true[i]) / y_true[i] * 100
            else:
                change_magnitude = 100  # 默认通过阈值
                
            if change_magnitude >= threshold_pct:
                # 判断拐点类型
                if actual_direction[i-1] == -1 and actual_direction[i] == 1:
                    # 上升拐点：从下降到上升
                    if actual_rise_idx is None:
                        actual_rise_idx = i  # 记录第一个上升拐点
                elif actual_direction[i-1] == 1 and actual_direction[i] == -1:
                    # 下降拐点：从上升到下降
                    if actual_fall_idx is None:
                        actual_fall_idx = i  # 记录第一个下降拐点
    
    # 计算预测序列的趋势变化
    pred_direction = np.sign(np.diff(y_pred))
    pred_rise_idx = None
    pred_fall_idx = None
    
    for i in range(1, len(pred_direction)):
        if pred_direction[i] != pred_direction[i-1]:
            if i+1 < len(y_pred) and abs(y_pred[i]) > 0:
                change_magnitude = abs(y_pred[i+1] - y_pred[i]) / abs(y_pred[i]) * 100
            else:
                change_magnitude = 100
                
            if change_magnitude >= threshold_pct:
                if pred_direction[i-1] == -1 and pred_direction[i] == 1:
                    if pred_rise_idx is None:
                        pred_rise_idx = i
                elif pred_direction[i-1] == 1 and pred_direction[i] == -1:
                    if pred_fall_idx is None:
                        pred_fall_idx = i
    
    # 计算延迟（月数）
    rise_delay = pred_rise_idx - actual_rise_idx if (actual_rise_idx is not None and pred_rise_idx is not None) else None
    fall_delay = pred_fall_idx - actual_fall_idx if (actual_fall_idx is not None and pred_fall_idx is not None) else None
    
    return {
        'rise_turning_point_delay': rise_delay,
        'fall_turning_point_delay': fall_delay,
        'actual_rise_idx': actual_rise_idx,
        'pred_rise_idx': pred_rise_idx,
        'actual_fall_idx': actual_fall_idx,
        'pred_fall_idx': pred_fall_idx
    }


def calculate_peak_window_hit(y_true, y_pred, window_size=3):
    """
    计算峰值窗口命中率（预测峰值是否落在实际峰值附近窗口内）
    
    参数:
        y_true: 真实值序列
        y_pred: 预测值序列
        window_size: 窗口大小（月数），例如window_size=3表示实际峰值前后3个月内
    
    返回:
        dict: 包含窗口命中信息
    """
    actual_peak_idx = np.argmax(y_true)
    pred_peak_idx = np.argmax(y_pred)
    
    # 计算窗口范围
    window_start = max(0, actual_peak_idx - window_size)
    window_end = min(len(y_true) - 1, actual_peak_idx + window_size)
    
    hit = window_start <= pred_peak_idx <= window_end
    
    return {
        'peak_window_hit': hit,
        'window_range': (window_start, window_end),
        'actual_peak_idx': actual_peak_idx,
        'pred_peak_idx': pred_peak_idx
    }


def calculate_all_metrics_enhanced(y_true, y_pred, model_name):
    """
    计算所有评估指标（增强版，包含峰值误差和拐点延迟）
    
    参数:
        y_true: 真实值
        y_pred: 预测值
        model_name: 模型名称
    
    返回:
        dict: 包含所有指标
    """
    # 基础指标
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    mape = calculate_mape(y_true, y_pred)
    
    # 长期预测专项指标
    trend_acc, correct, total = calculate_trend_accuracy(y_true, y_pred)
    cumulative_errors = calculate_cumulative_relative_error(y_true, y_pred)
    
    # 新增：峰值误差和拐点延迟
    peak_metrics = calculate_peak_error(y_true, y_pred)
    turning_point_metrics = calculate_turning_point_delay(y_true, y_pred)
    peak_window_metrics = calculate_peak_window_hit(y_true, y_pred)
    
    # 合并所有指标
    return {
        'model': model_name,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'trend_accuracy': trend_acc,
        'trend_correct': correct,
        'trend_total': total,
        'cumulative_errors': cumulative_errors,
        'y_true': y_true,
        'y_pred': y_pred,
        # 峰值指标
        'peak_magnitude_error': peak_metrics['peak_magnitude_error'],
        'peak_magnitude_error_pct': peak_metrics['peak_magnitude_error_pct'],
        'peak_position_error': peak_metrics['peak_position_error'],
        'actual_peak_idx': peak_metrics['actual_peak_idx'],
        'pred_peak_idx': peak_metrics['pred_peak_idx'],
        'actual_peak_value': peak_metrics['actual_peak_value'],
        'pred_peak_value': peak_metrics['pred_peak_value'],
        # 窗口命中指标
        'peak_window_hit': peak_window_metrics['peak_window_hit'],
        'window_range': peak_window_metrics['window_range'],
        # 拐点指标
        'rise_turning_point_delay': turning_point_metrics['rise_turning_point_delay'],
        'fall_turning_point_delay': turning_point_metrics['fall_turning_point_delay'],
        'actual_rise_idx': turning_point_metrics['actual_rise_idx'],
        'pred_rise_idx': turning_point_metrics['pred_rise_idx'],
        'actual_fall_idx': turning_point_metrics['actual_fall_idx'],
        'pred_fall_idx': turning_point_metrics['pred_fall_idx']
    }


def print_enhanced_results(results_dict):
    """
    打印增强版评估结果
    """
    print("\n" + "="*110)
    print("模型长期预测性能评估（含峰值误差与拐点延迟）")
    print("="*110)
    print(f"{'模型':<12} {'RMSE':<8} {'MAE':<8} {'MAPE(%)':<10} {'趋势准确率(%)':<12} {'峰值大小误差(%)':<12} {'峰值位置误差(月)':<12} {'峰值窗口命中':<10} {'上升拐点延迟':<10} {'下降拐点延迟':<10}")
    print("-"*110)
    
    for name, metrics in results_dict.items():
        # 处理拐点延迟显示
        rise_delay = metrics['rise_turning_point_delay']
        fall_delay = metrics['fall_turning_point_delay']
        rise_display = f"{rise_delay}" if rise_delay is not None else "N/A"
        fall_display = f"{fall_delay}" if fall_delay is not None else "N/A"
        
        print(f"{name:<12} {metrics['rmse']:<8.2f} {metrics['mae']:<8.2f} "
              f"{metrics['mape']:<10.2f} {metrics['trend_accuracy']:<12.2f} "
              f"{metrics['peak_magnitude_error_pct']:<12.2f} {metrics['peak_position_error']:<12} "
              f"{'✓' if metrics['peak_window_hit'] else '✗':<10} {rise_display:<10} {fall_display:<10}")
    
    print("="*110)


def plot_peak_analysis(results_dict, model_name, save_path=None):
    """
    绘制特定模型的峰值分析图（显示实际峰值与预测峰值位置）
    
    参数:
        results_dict: 包含各模型结果的字典
        model_name: 要绘制的模型名称
        save_path: 保存路径
    """
    if model_name not in results_dict:
        print(f"模型 {model_name} 不存在")
        return
    
    metrics = results_dict[model_name]
    y_true = metrics['y_true']
    y_pred = metrics['y_pred']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 绘制实际值和预测值
    months = range(1, len(y_true) + 1)
    ax.plot(months, y_true, 'b-', linewidth=2, label='实际值', marker='o', markersize=4)
    ax.plot(months, y_pred, 'r--', linewidth=2, label='预测值', marker='s', markersize=4)
    
    # 标记实际峰值
    actual_peak_idx = metrics['actual_peak_idx']
    ax.scatter(actual_peak_idx + 1, y_true[actual_peak_idx], 
               color='blue', s=150, zorder=5, marker='*', 
               label=f'实际峰值 ({actual_peak_idx+1}月, {y_true[actual_peak_idx]:.2f})')
    
    # 标记预测峰值
    pred_peak_idx = metrics['pred_peak_idx']
    ax.scatter(pred_peak_idx + 1, y_pred[pred_peak_idx], 
               color='red', s=150, zorder=5, marker='*',
               label=f'预测峰值 ({pred_peak_idx+1}月, {y_pred[pred_peak_idx]:.2f})')
    
    # 标记上升拐点
    actual_rise = metrics.get('actual_rise_idx')
    pred_rise = metrics.get('pred_rise_idx')
    if actual_rise is not None:
        ax.axvline(x=actual_rise + 1, color='green', linestyle='--', alpha=0.7, 
                   label=f'实际上升拐点 ({actual_rise+1}月)')
    if pred_rise is not None:
        ax.axvline(x=pred_rise + 1, color='lime', linestyle=':', alpha=0.7,
                   label=f'预测上升拐点 ({pred_rise+1}月)')
    
    # 标记下降拐点
    actual_fall = metrics.get('actual_fall_idx')
    pred_fall = metrics.get('pred_fall_idx')
    if actual_fall is not None:
        ax.axvline(x=actual_fall + 1, color='orange', linestyle='--', alpha=0.7,
                   label=f'实际下降拐点 ({actual_fall+1}月)')
    if pred_fall is not None:
        ax.axvline(x=pred_fall + 1, color='gold', linestyle=':', alpha=0.7,
                   label=f'预测下降拐点 ({pred_fall+1}月)')
    
    ax.set_xlabel('预测步数（月）')
    ax.set_ylabel('发病率 (1/10万)')
    ax.set_title(f'{model_name} 模型峰值与拐点分析')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"峰值分析图已保存至: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    # 读取数据
    df = pd.read_excel('../DATA/data_enhance.xlsx', sheet_name='Sheet1')
    
    # 提取真实值和预测值
    y_true = df['y_ture'].values
    
    models = {
        'LightGBM': df['LightGBM'].values,
        'CNN-BiLSTM': df['CNN-BiLSTM'].values,
        'LSTM': df['LSTM'].values,
        'N-BEATS': df['NBEATS'].values,
        'ARIMA': df['ARIMA'].values,
        'Prophet': df['Prophet'].values
    }
    
    # 计算各模型指标RMSE MAE MAPE
    results = {}
    for name, y_pred in models.items():
        results[name] = calculate_all_metrics(y_true, y_pred, name)
    
    for name, metrics in results.items():
        print(f"{name:<12} {metrics['rmse']:<8.4f} {metrics['mae']:<8.4f} "
              f"{metrics['mape']:<10.2f} {metrics['trend_accuracy']:<12.2f}")
    
    # 绘制趋势准确率、累积相对误差结果图
    plot_results(results, save_path='./long_term_metrics.png')
    
    results_enhanced = {}
    
    for name, y_pred in models.items():
        results_enhanced[name] = calculate_all_metrics_enhanced(y_true, y_pred, name)
    
    # 打印长期预测性能结果
    print_enhanced_results(results_enhanced)
    
    # 为 LightGBM 绘制峰值分析图
    plot_peak_analysis(results_enhanced, 'LightGBM', save_path='./lightgbm_peak_analysis.png')
    
    # 为 CNN-BiLSTM 绘制峰值分析图
    plot_peak_analysis(results_enhanced, 'CNN-BiLSTM', save_path='./cnn_bilstm_peak_analysis.png')