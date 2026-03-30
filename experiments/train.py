"""
云南模型训练实验
"""
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

from src.utils.seeds import set_global_seed
from src.data.loader import DataLoader
from src.data.processor import DataProcessor
from src.data.augmenter import DataAugmenter
from src.models import (
    ARIMAModel, ProphetModel, LSTMModel,
    CNNBiLSTMModel, LightGBMModel, NBeatsModel
)
from src.evaluation import compare_models, significance_test
from src.evaluation.metrics import calculate_metrics
from src.visualization import plot_comparison, plot_error_distribution


def load_config(config_path: str = '../config/config.yaml') -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def train_and_evaluate(config: dict) -> tuple:
    """训练并评估所有模型"""

    # 1. 设置随机种子
    set_global_seed(config['random_seed'])

    # 2. 加载数据
    print("\n" + "=" * 60)
    print("1. 数据加载")
    print("=" * 60)

    loader = DataLoader(config)
    yunnan_data = loader.load_yunnan()
    print(f"  数据长度: {len(yunnan_data)}")

    # 3. 数据预处理
    print("\n" + "=" * 60)
    print("2. 数据预处理")
    print("=" * 60)

    processor = DataProcessor(config)

    # 获取原始训练和测试数据
    train_raw, test_raw, _, _ = processor.split_train_test(yunnan_data)

    # 生成滑动窗口数据（用于深度学习模型）
    X_train, y_train = processor.create_sequences(train_raw, config['window_size'])
    X_test, y_test = processor.create_sequences(test_raw, config['window_size'])

    print(f"  滑动窗口训练样本: {len(X_train)}")
    print(f"  滑动窗口测试样本: {len(X_test)}")
    print(f"  原始训练序列长度: {len(train_raw)}")
    print(f"  原始测试序列长度: {len(test_raw)}")

    # 4. 数据增强（只对深度学习模型）
    # 保存原始滑动窗口数据，用于 ARIMA 和 Prophet
    X_train_original = X_train.copy()
    y_train_original = y_train.copy()

    if config['augmentation']['enabled']:
        print("\n" + "=" * 60)
        print("3. 数据增强（仅用于深度学习模型）")
        print("=" * 60)

        augmenter = DataAugmenter(config)
        X_train_aug, y_train_aug = augmenter.augment(train_raw, X_train, y_train)
        print(f"  增强后训练样本: {len(X_train_aug)}")
    else:
        X_train_aug = X_train
        y_train_aug = y_train
    time.sleep(5)
    # 5. 训练各模型
    print("\n" + "=" * 60)
    print("4. 模型训练与评估")
    print("=" * 60)

    results = {}
    predictions = {}

    # 定义哪些模型使用增强数据，哪些使用原始数据
    models_use_augmentation = ['lstm', 'cnn_bilstm', 'lightgbm', 'nbeats']
    models_use_original = ['arima', 'prophet']

    model_classes = {
        'arima': ARIMAModel,
        'prophet': ProphetModel,
        'lstm': LSTMModel,
        'cnn_bilstm': CNNBiLSTMModel,
        'lightgbm': LightGBMModel,
        'nbeats': NBeatsModel
    }

    for name, ModelClass in model_classes.items():
        model_config = config['models'].get(name, {})
        if not model_config.get('enabled', True):
            print(f"\n  跳过 {name.upper()} 模型（未启用）")
            continue

        print(f"\n  训练 {name.upper()} 模型...")

        try:
            model = ModelClass(model_config)

            # 根据模型类型选择训练数据
            if name in models_use_original:
                # ARIMA 和 Prophet 使用原始序列
                print(f"    使用原始序列训练（无数据增强）")
                model.fit(train_raw, None)
            elif name in models_use_augmentation:
                # 深度学习模型使用增强后的滑动窗口数据
                if config['augmentation']['enabled']:
                    print(f"    使用增强后的滑动窗口数据训练")
                else:
                    print(f"    使用原始滑动窗口数据训练")
                model.fit(X_train_aug, y_train_aug)
            else:
                # 默认使用滑动窗口
                model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            predictions[name.upper()] = y_pred
            results[name.upper()] = model

            metrics = calculate_metrics(y_test, y_pred)
            print(f"    RMSE: {metrics['rmse']:.4f}")
            print(f"    MAE: {metrics['mae']:.4f}")
            print(f"    R²: {metrics['r2']:.4f}")

        except Exception as e:
            print(f"    ❌ 训练失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 检查是否有成功训练的模型
    if not predictions:
        print("\n❌ 没有成功训练的模型，程序退出")
        return None, None, None

    # 6. 模型对比
    print("\n" + "=" * 60)
    print("5. 模型对比")
    print("=" * 60)

    comparison_df = compare_models(y_test, predictions)
    print("\n", comparison_df.to_string())

    # 7. 统计检验
    print("\n" + "=" * 60)
    print("6. 统计检验 (Diebold-Mariano)")
    print("=" * 60)

    significance_df = significance_test(y_test, predictions)
    print("\n", significance_df.to_string())

    # 8. 可视化
    print("\n" + "=" * 60)
    print("7. 结果可视化")
    print("=" * 60)

    # 创建结果目录
    results_dir = '../results/experiment_results/'
    os.makedirs(results_dir, exist_ok=True)

    # 绘制预测对比图
    plot_comparison(
        y_test, predictions,
        title="各模型预测结果对比",
        save_path=f'{results_dir}model_comparison.png',
        show=True
    )

    # 绘制误差分布图
    plot_error_distribution(
        y_test, predictions,
        save_path=f'{results_dir}error_distribution.png',
        show=True
    )

    # 9. 保存结果
    print("\n" + "=" * 60)
    print("8. 保存结果")
    print("=" * 60)

    # 保存评估结果表格
    comparison_df.to_csv(f'{results_dir}model_comparison.csv', index=False)
    significance_df.to_csv(f'{results_dir}significance_test.csv', index=False)

    # 保存预测值
    for name, y_pred in predictions.items():
        model_dir = f'{results_dir}{name}/'
        os.makedirs(model_dir, exist_ok=True)
        np.savetxt(f'{model_dir}pred_{name}.csv', y_pred, delimiter=',')

    # 保存真实值
    np.savetxt(f'{results_dir}y_true.csv', y_test, delimiter=',')

    print(f"\n  结果已保存至: {results_dir}")

    # 10. 保存最优模型
    best_model_name = comparison_df.iloc[0]['model']
    best_model = results.get(best_model_name)
    if best_model:
        os.makedirs('../saved_models', exist_ok=True)

        # 根据模型类型选择保存格式
        if best_model_name in ['LSTM', 'CNN-BILSTM', 'N-BEATS']:
            save_path = f'../saved_models/best_model_{best_model_name}.keras'
        else:
            save_path = f'../saved_models/best_model_{best_model_name}.pkl'

        try:
            best_model.save(save_path)
            print(f"\n  最优模型已保存: {save_path}")
        except Exception as e:
            print(f"\n  保存模型失败: {e}")

    return results, y_test, predictions


def main():
    """主函数"""
    config = load_config()
    results, y_test, predictions = train_and_evaluate(config)

    if results is None:
        print("\n❌ 实验失败，没有成功训练的模型")
        return

    print("\n" + "=" * 60)
    print("实验完成！")
    print("=" * 60)

    # 打印最优模型
    if predictions:
        from src.evaluation.metrics import calculate_metrics
        best_model = min(predictions.keys(),
                         key=lambda k: calculate_metrics(y_test, predictions[k])['rmse'])
        print(f"\n🏆 最优模型: {best_model}")


if __name__ == '__main__':
    main()