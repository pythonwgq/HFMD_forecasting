"""
Diebold-Mariano 检验 - 对比数据增强前后模型性能
比较每个模型在数据增强前后的预测误差是否有统计学差异
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
        actual: 真实值（可选，用于比较两个预测值）
        pred1: 模型1的预测值（增强前）
        pred2: 模型2的预测值（增强后）
        h: 预测步长
        one_sided: 'less' 表示检验 pred1 误差是否小于 pred2（增强前是否更好）
                   'greater' 表示检验 pred1 误差是否大于 pred2（增强后是否更好）

    返回:
        dm_stat: DM统计量
        p_value: p值
    """
    # 计算误差（如果actual为None，则直接比较pred1和pred2的误差）
    if actual is not None:
        e1 = actual - pred1
        e2 = actual - pred2
    else:
        e1 = pred1
        e2 = pred2

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
df = pd.read_csv('./enhance_unenhance.csv')

print("=" * 60)
print("数据增强前后模型性能对比分析")
print("=" * 60)

print("\n数据预览:")
print(df.head())
print(f"\n数据形状: {df.shape}")

# 提取真实值
y_true = df['true_date'].values

# 定义模型列表
models = {
    'LSTM': {
        'unenhance': df['lstm_unenhance'].values,
        'enhance': df['lstm_enhance'].values
    },
    'CNN-BiLSTM': {
        'unenhance': df['cnn-bilstm_unenhance'].values,
        'enhance': df['cnn-bilstm_enhance'].values
    },
    'LightGBM': {
        'unenhance': df['lightGBM_unenhance'].values,
        'enhance': df['lightGBM_enhance'].values
    },
    'N-BEATS': {
        'unenhance': df['NBEATS_unenhance'].values,
        'enhance': df['NBEATS_enhance'].values
    }
}

# ========== 2. 计算评估指标 ==========
print("\n" + "=" * 60)
print("模型性能对比（增强前 vs 增强后）")
print("=" * 60)

results = {}

for model_name, preds in models.items():
    y_unenhance = preds['unenhance']
    y_enhance = preds['enhance']

    # 计算增强前指标
    rmse_un = np.sqrt(mean_squared_error(y_true, y_unenhance))
    mae_un = mean_absolute_error(y_true, y_unenhance)

    # 计算增强后指标
    rmse_en = np.sqrt(mean_squared_error(y_true, y_enhance))
    mae_en = mean_absolute_error(y_true, y_enhance)

    # 计算提升幅度
    rmse_improve = (rmse_un - rmse_en) / rmse_un * 100
    mae_improve = (mae_un - mae_en) / mae_un * 100

    results[model_name] = {
        'unenhance': {'RMSE': rmse_un, 'MAE': mae_un},
        'enhance': {'RMSE': rmse_en, 'MAE': mae_en},
        'improve': {'RMSE': rmse_improve, 'MAE': mae_improve}
    }

# 打印性能对比表格
print(f"\n{'模型':<15} {'增强前RMSE':<12} {'增强后RMSE':<12} {'提升幅度':<12}")
print("-" * 55)
for model_name, res in results.items():
    rmse_un = res['unenhance']['RMSE']
    rmse_en = res['enhance']['RMSE']
    improve = res['improve']['RMSE']
    print(f"{model_name:<15} {rmse_un:<12.4f} {rmse_en:<12.4f} {improve:<11.1f}%")

print(f"\n{'模型':<15} {'增强前MAE':<12} {'增强后MAE':<12} {'提升幅度':<12}")
print("-" * 55)
for model_name, res in results.items():
    mae_un = res['unenhance']['MAE']
    mae_en = res['enhance']['MAE']
    improve = res['improve']['MAE']
    print(f"{model_name:<15} {mae_un:<12.4f} {mae_en:<12.4f} {improve:<11.1f}%")

# ========== 3. Diebold-Mariano 检验 ==========
print("\n" + "=" * 60)
print("Diebold-Mariano 检验结果")
print("=" * 60)
print("检验假设: 增强后模型的预测误差是否显著小于增强前")
print("-" * 60)

dm_results = []

for model_name, preds in models.items():
    y_unenhance = preds['unenhance']
    y_enhance = preds['enhance']

    # 检验增强后是否优于增强前（增强后误差更小）
    dm_stat, p_value = dm_test(y_true, y_unenhance, y_enhance, h=1, one_sided='greater')

    dm_results.append({
        '模型': model_name,
        'DM统计量': dm_stat,
        'p值': p_value,
        '是否显著(p<0.05)': '是' if p_value < 0.05 else '否',
        '显著性水平': '***' if p_value < 0.01 else ('**' if p_value < 0.05 else ('*' if p_value < 0.1 else ''))
    })

    print(f"\n{model_name}:")
    print(f"  DM统计量: {dm_stat:.4f}")
    print(f"  p值: {p_value:.4f}")
    if p_value < 0.05:
        print(f"  ✅ 结论: 数据增强显著提升 {model_name} 性能 (p={p_value:.4f} < 0.05)")
    else:
        print(f"  ⚠️ 结论: 数据增强对 {model_name} 提升不显著 (p={p_value:.4f} ≥ 0.05)")

# ========== 4. 可视化 ==========

# 图1: 增强前后RMSE对比条形图
plt.figure(figsize=(12, 6))
models_names = list(results.keys())
x = np.arange(len(models_names))
width = 0.35

rmse_un = [results[m]['unenhance']['RMSE'] for m in models_names]
rmse_en = [results[m]['enhance']['RMSE'] for m in models_names]

bars1 = plt.bar(x - width / 2, rmse_un, width, label='增强前', color='#e74c3c', alpha=0.8)
bars2 = plt.bar(x + width / 2, rmse_en, width, label='增强后', color='#2ecc71', alpha=0.8)

plt.xlabel('模型')
plt.ylabel('RMSE')
plt.title('数据增强前后各模型RMSE对比')
plt.xticks(x, models_names)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for i, (un, en) in enumerate(zip(rmse_un, rmse_en)):
    plt.text(i - width / 2, un + 0.1, f'{un:.2f}', ha='center', va='bottom', fontsize=9)
    plt.text(i + width / 2, en + 0.1, f'{en:.2f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()







# 图4: 预测曲线对比（LightGBM为例，最优模型）
plt.figure(figsize=(14, 6))
plt.plot(y_true, label='实际值', marker='o', markersize=4, linewidth=2, color='black')
plt.plot(models['LightGBM']['unenhance'], label='LightGBM (增强前)', alpha=0.7, marker='v', markersize=4, linewidth=1.5)
plt.plot(models['LightGBM']['enhance'], label='LightGBM (增强后)', alpha=0.7, marker='s', markersize=4, linewidth=1.5)
plt.xlabel('时间步')
plt.ylabel('发病率')
plt.title('LightGBM 数据增强前后预测效果对比')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
"""
# ========== 5. 导出结果 ==========
# 保存性能对比结果
perf_df = pd.DataFrame({
    '模型': models_names,
    '增强前RMSE': [results[m]['unenhance']['RMSE'] for m in models_names],
    '增强后RMSE': [results[m]['enhance']['RMSE'] for m in models_names],
    'RMSE提升(%)': [results[m]['improve']['RMSE'] for m in models_names],
    '增强前MAE': [results[m]['unenhance']['MAE'] for m in models_names],
    '增强后MAE': [results[m]['enhance']['MAE'] for m in models_names],
    'MAE提升(%)': [results[m]['improve']['MAE'] for m in models_names]
})
perf_df.to_csv('../results/enhance_comparison_performance.csv', index=False, encoding='utf-8-sig')

# 保存DM检验结果
dm_df = pd.DataFrame(dm_results)
dm_df.to_csv('../results/enhance_comparison_dm.csv', index=False, encoding='utf-8-sig')

print("\n" + "=" * 60)
print("结果已保存至:")
print("  - ../results/enhance_comparison_performance.csv")
print("  - ../results/enhance_comparison_dm.csv")
print("=" * 60)
"""
# ========== 6. 打印汇总表格（可直接用于论文） ==========
print("\n" + "=" * 60)
print("汇总结果（可直接用于论文）")
print("=" * 60)
print(f"\n{'模型':<15} {'增强前RMSE':<12} {'增强后RMSE':<12} {'DM统计量':<12} {'p值':<12}")
print("-" * 65)
for i, model_name in enumerate(models_names):
    rmse_un = results[model_name]['unenhance']['RMSE']
    rmse_en = results[model_name]['enhance']['RMSE']
    dm_stat = dm_results[i]['DM统计量']
    p_value = dm_results[i]['p值']
    print(f"{model_name:<15} {rmse_un:<12.4f} {rmse_en:<12.4f} {dm_stat:<12.4f} {p_value:<12.4f}")