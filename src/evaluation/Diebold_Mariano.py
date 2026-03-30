"""
Diebold-Mariano 检验（手动实现）
比较 CNN-BiLSTM 与其他模型的预测误差
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


# ========== DM 检验手动实现 ==========
def dm_test(actual, pred1, pred2, h=1, one_sided='less'):
    """
    Diebold-Mariano 检验

    参数:
        actual: 真实值
        pred1: 模型1的预测值
        pred2: 模型2的预测值
        h: 预测步长（通常为1）
        one_sided: 'less' 表示检验 pred1 误差是否小于 pred2
                   'greater' 表示检验 pred1 误差是否大于 pred2

    返回:
        dm_stat: DM统计量
        p_value: p值
    """
    # 计算误差
    e1 = actual - pred1
    e2 = actual - pred2

    # 损失函数：平方误差
    d = e1 ** 2 - e2 ** 2

    # 计算均值和方差
    d_mean = np.mean(d)
    d_var = np.var(d, ddof=1)

    # 计算DM统计量
    T = len(d)
    dm_stat = d_mean / np.sqrt(d_var / T)

    # 计算p值
    from scipy.stats import norm
    if one_sided == 'less':
        p_value = norm.cdf(dm_stat)
    elif one_sided == 'greater':
        p_value = 1 - norm.cdf(dm_stat)
    else:
        p_value = 2 * (1 - norm.cdf(abs(dm_stat)))

    return dm_stat, p_value


# ========== 1. 读取数据 ==========
df = pd.read_csv('./Diebold_Mariano.csv')

# 查看数据
print("数据预览:")
print(df.head())
print(f"\n数据形状: {df.shape}")

# 提取列
y_true = df['text_data'].values
y_lightgbm = df['LIGHTGBM'].values
y_lstm = df['LSTM'].values
y_NBEATS = df['N-BEATS'].values
y_cnn_bilstm = df['CNN-BiLSTM'].values
y_arima = df['ARIMA'].values
y_prophet = df['prophet'].values


# ========== 2. 计算误差 ==========
error_lightgbm = y_true - y_lightgbm
error_lstm = y_true - y_lstm
error_NBEATS = y_true - y_NBEATS
error_cnn_bilstm = y_true - y_cnn_bilstm
error_arima = y_true - y_arima
error_prophet = y_true - y_prophet
# ========== 3. 计算评估指标 ==========
print("\n" + "=" * 60)
print("模型性能对比")
print("=" * 60)

models = {
    'CNN-BiLSTM': y_cnn_bilstm,
    'LSTM': y_lstm,
    'LightGBM':y_lightgbm,
    'N-BEATS': y_NBEATS
}

results = {}
for name, y_pred in models.items():
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    results[name] = {'RMSE': rmse, 'MAE': mae}

# 按 RMSE 排序
sorted_results = sorted(results.items(), key=lambda x: x[1]['RMSE'])

print(f"\n{'模型':<15} {'RMSE':<12} {'MAE':<12}")
print("-" * 40)
for name, metrics in sorted_results:
    print(f"{name:<15} {metrics['RMSE']:<12.4f} {metrics['MAE']:<12.4f}")

# ========== 4. Diebold-Mariano 检验 ==========
print("\n" + "=" * 60)
print("Diebold-Mariano 检验结果")
print("=" * 60)
print("检验假设: CNN-BiLSTM 的预测误差是否显著小于对比模型")
print("-" * 60)


dm_stat_lstm, p_value_lstm = dm_test(y_true, y_lightgbm, y_lstm, h=1, one_sided='less')
print(f"\nLightGBM vs LSTM:")
print(f"  DM统计量: {dm_stat_lstm:.4f}")
print(f"  p值: {p_value_lstm:.4f}")
if p_value_lstm < 0.05:
    print(f"  ✅ 结论: LightGBM 显著优于 LSTM (p={p_value_lstm:.4f} < 0.05)")
else:
    print(f"  ⚠️ 结论: 差异不显著 (p={p_value_lstm:.4f} ≥ 0.05)")


dm_stat_cnn, p_value_cnn = dm_test(y_true, y_lightgbm, y_cnn_bilstm, h=1, one_sided='less')
print(f"\nLightGBM vs CNN-BiLSTM:")
print(f"  DM统计量: {dm_stat_cnn:.4f}")
print(f"  p值: {p_value_cnn:.4f}")
if p_value_cnn < 0.05:
    print(f"  ✅ 结论: LightGBM 显著优于 CNN-BiLSTM (p={p_value_cnn:.4f} < 0.05)")
else:
    print(f"  ⚠️ 结论: 差异不显著 (p={p_value_cnn:.4f} ≥ 0.05)")


dm_stat_NBEATS, p_value_NBEATS = dm_test(y_true, y_lightgbm, y_NBEATS, h=1, one_sided='less')
print(f"\nLightGBM vs N-BEATS:")
print(f"  DM统计量: {dm_stat_NBEATS:.4f}")
print(f"  p值: {p_value_NBEATS:.4f}")
if p_value_NBEATS < 0.05:
    print(f"  ✅ 结论: LightGBM 显著优于 N-BEATS (p={p_value_NBEATS:.4f} < 0.05)")
else:
    print(f"  ⚠️ 结论: 差异不显著 (p={p_value_NBEATS:.4f} ≥ 0.05)")

dm_stat_arima, p_value_arima = dm_test(y_true, y_lightgbm, y_arima, h=1, one_sided='less')
print(f"\nLightGBM vs ARIMA:")
print(f"  DM统计量: {dm_stat_arima:.4f}")
print(f"  p值: {p_value_arima:.4f}")
if p_value_arima < 0.05:
    print(f"  ✅ 结论: LightGBM 显著优于 ARIMA (p={p_value_arima:.4f} < 0.05)")
else:
    print(f"  ⚠️ 结论: 差异不显著 (p={p_value_arima:.4f} ≥ 0.05)")
    
dm_stat_prophet, p_value_prophet = dm_test(y_true, y_lightgbm, y_prophet, h=1, one_sided='less')
print(f"\nLightGBM vs prophet:")
print(f"  DM统计量: {dm_stat_prophet:.4f}")
print(f"  p值: {p_value_prophet:.4f}")
if p_value_prophet < 0.05:
    print(f"  ✅ 结论: LightGBM 显著优于 prophet (p={p_value_prophet:.4f} < 0.05)")
else:
    print(f"  ⚠️ 结论: 差异不显著 (p={p_value_prophet:.4f} ≥ 0.05)")
    
# ========== 5. 计算提升幅度 ==========
print("\n" + "=" * 60)
print("提升幅度计算")
print("=" * 60)

cnn_rmse = results['LightGBM']['RMSE']
cnn_mae = results['LightGBM']['MAE']

for name, metrics in results.items():
    if name != 'LightGBM':
        rmse_improve = (metrics['RMSE'] - cnn_rmse) / metrics['RMSE'] * 100
        mae_improve = (metrics['MAE'] - cnn_mae) / metrics['MAE'] * 100
        print(f"\n{name} → LightGBM :")
        print(f"  RMSE: {metrics['RMSE']:.4f} → {cnn_rmse:.4f} (提升 {rmse_improve:.1f}%)")
        print(f"  MAE:  {metrics['MAE']:.4f} → {cnn_mae:.4f} (提升 {mae_improve:.1f}%)")


