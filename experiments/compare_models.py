"""
模型对比实验 - 综合对比所有模型
"""
import sys
import os

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
from src.visualization import plot_comparison, plot_error_distribution


def load_config(config_path: str = '../config/config.yaml') -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def run_comparison_experiment(config: dict, use_augmentation: bool = True) -> dict:
    """
    运行模型对比实验

    Args:
        config: 配置
        use_augmentation: 是否使用数据增强

    Returns:
        实验结果
    """

    set_global_seed(config['random_seed'])

    # 加载数据
    loader = DataLoader(config)
    yunnan_data = loader.load_yunnan()

    # 预处理
    processor = DataProcessor(config)
    X_train, y_train, X_test, y_test = processor.get_train_test_sequences(yunnan_data)

    # 数据增强
    if use_augmentation and config['augmentation']['enabled']:
        augmenter = DataAugmenter(config)
        train_raw, _, _, _ = processor.split_train_test(yunnan_data)
        X_train, y_train = augmenter.augment(train_raw, X_train, y_train)

    # 训练所有模型
    results = {}
    predictions = {}

    model_configs = {
        'ARIMA': (ARIMAModel, config['models']['arima']),
        'Prophet': (ProphetModel, config['models']['prophet']),
        'LSTM': (LSTMModel, config['models']['lstm']),
        'CNN-BiLSTM': (CNNBiLSTMModel, config['models']['cnn_bilstm']),
        'LightGBM': (LightGBMModel, config['models']['lightgbm']),
        'N-BEATS': (NBeatsModel, config['models']['nbeats'])
    }

    for name, (ModelClass, model_config) in model_configs.items():
        if not model_config.get('enabled', True):
            continue

        print(f"\n训练 {name}...")

        try:
            model = ModelClass(model_config)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            predictions[name] = y_pred
            results[name] = model

        except Exception as e:
            print(f"  ❌ 失败: {e}")
            continue

    return {
        'y_test': y_test,
        'predictions': predictions,
        'results': results
    }


def main():
    """主函数"""
    config = load_config()

    print("=" * 60)
    print("模型对比实验")
    print("=" * 60)

    # 运行实验
    experiment = run_comparison_experiment(config, use_augmentation=True)

    y_test = experiment['y_test']
    predictions = experiment['predictions']

    # 模型对比
    print("\n" + "=" * 60)
    print("模型性能对比")
    print("=" * 60)

    comparison_df = compare_models(y_test, predictions)
    print("\n", comparison_df.to_string())

    # 统计检验
    print("\n" + "=" * 60)
    print("统计检验 (Diebold-Mariano)")
    print("=" * 60)

    significance_df = significance_test(y_test, predictions)
    print("\n", significance_df.to_string())

    # 可视化
    os.makedirs('../results', exist_ok=True)

    plot_comparison(y_test, predictions, save_path='../results/comparison.png')
    plot_error_distribution(y_test, predictions, save_path='../results/errors.png')

    # 保存结果
    comparison_df.to_csv('../results/comparison_results.csv', index=False)
    significance_df.to_csv('../results/significance_results.csv', index=False)

    print("\n" + "=" * 60)
    print("实验完成！结果已保存至 ../results/")
    print("=" * 60)


if __name__ == '__main__':
    main()