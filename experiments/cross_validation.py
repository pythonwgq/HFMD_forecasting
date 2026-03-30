"""
跨省验证实验 - 支持零样本迁移和微调两种模式
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import numpy as np
import pandas as pd
import warnings
import lightgbm as lgb
import torch
import torch.optim as optim

warnings.filterwarnings('ignore')

from src.utils.seeds import set_global_seed
from src.data.loader import DataLoader
from src.data.processor import DataProcessor
from src.models import NBeatsModel, LSTMModel, LightGBMModel
from src.evaluation import calculate_metrics
from src.visualization import plot_cross_province


def load_config(config_path: str = '../config/config.yaml') -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def train_baseline_model(config: dict, data: np.ndarray, model_name: str = 'lightgbm'):
    """
    在云南数据上训练基准模型

    Args:
        config: 配置
        data: 云南原始数据
        model_name: 模型名称 ('lightgbm', 'lstm', 'nbeats')

    Returns:
        训练好的模型
    """
    processor = DataProcessor(config)
    X_train, y_train, _, _ = processor.get_train_test_sequences(data)

    # 选择模型
    if model_name.lower() == 'nbeats':
        from src.models import NBeatsModel
        model = NBeatsModel(config['models']['nbeats'])
    elif model_name.lower() == 'lstm':
        from src.models import LSTMModel
        model = LSTMModel(config['models']['lstm'])
    elif model_name.lower() == 'lightgbm':
        from src.models import LightGBMModel
        model = LightGBMModel(config['models']['lightgbm'])
    else:
        raise ValueError(f"不支持的模型: {model_name}")

    model.fit(X_train, y_train)
    return model


def finetune_model(model, X_train, y_train, X_val=None, y_val=None,
                   learning_rate=0.05, epochs=50):
    """
    对模型进行微调

    Args:
        model: 预训练模型
        X_train, y_train: 微调训练数据
        X_val, y_val: 验证数据（可选）
        learning_rate: 学习率
        epochs: 微调轮数
    """
    print(f"  开始微调 (学习率={learning_rate}, 轮数={epochs})...")
    print(f"  🔧 X_train shape: {X_train.shape if X_train is not None else 'None'}")
    print(f"  🔧 y_train shape: {y_train.shape if y_train is not None else 'None'}")

    # 根据模型类型选择微调方式
    model_name = model.name.upper()

    if model_name == 'LIGHTGBM':
        # LightGBM 微调：最简单的方式
        X_train_feat = model._extract_features(X_train)

        if X_val is not None:
            X_val_feat = model._extract_features(X_val)
            eval_set = [(X_val_feat, y_val)]
        else:
            eval_set = None

        # 直接调用 fit，不传任何额外参数
        model.model.fit(
            X_train_feat, y_train,
            eval_set=eval_set,
            init_model=model.model,
            callbacks=[lgb.early_stopping(20)] if eval_set else None
        )

        print(f"  微调完成，树数量: {model.model.n_estimators_}")

    elif model_name in ['LSTM', 'CNN-BILSTM']:
        # 深度学习微调：降低学习率继续训练
        from tensorflow.keras.optimizers import Adam

        # 获取原始学习率
        if hasattr(model.model.optimizer, 'learning_rate'):
            original_lr = float(model.model.optimizer.learning_rate.numpy())
        else:
            original_lr = 0.001

        # 编译模型（使用新学习率）
        model.model.compile(optimizer=Adam(learning_rate=float(learning_rate)), loss='mse')

        # 准备数据
        X_train_3d = X_train.reshape(-1, X_train.shape[1], 1)

        # 微调训练
        if X_val is not None:
            X_val_3d = X_val.reshape(-1, X_val.shape[1], 1)
            model.model.fit(
                X_train_3d, y_train,
                epochs=epochs,
                batch_size=model.batch_size,
                validation_data=(X_val_3d, y_val),
                verbose=0
            )
        else:
            model.model.fit(
                X_train_3d, y_train,
                epochs=epochs,
                batch_size=model.batch_size,
                validation_split=0.1,
                verbose=0
            )

        # 恢复原学习率
        model.model.compile(optimizer=Adam(learning_rate=original_lr), loss='mse')

    elif model_name == 'N-BEATS':
        # N-BEATS 微调：降低学习率继续训练
        import torch
        import torch.optim as optim

        # 调整优化器
        optimizer = optim.Adam(model.model.parameters(), lr=learning_rate)
        criterion = torch.nn.MSELoss()

        # 准备数据
        X_tensor = torch.FloatTensor(X_train).to(model.device)
        y_tensor = torch.FloatTensor(y_train).reshape(-1, 1).to(model.device)

        # 微调训练
        model.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            y_pred = model.model(X_tensor)
            loss = criterion(y_pred, y_tensor)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 20 == 0:
                print(f"    Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}")

    print(f"  微调完成")


def cross_validate(config: dict, model_name: str = 'lightgbm',
                   finetune: bool = False,
                   finetune_ratio: float = 0.3) -> dict:
    """
    跨省验证

    Args:
        config: 配置
        model_name: 使用的模型名称
        finetune: 是否进行微调
        finetune_ratio: 微调使用训练集比例（从目标省份训练集中取）

    Returns:
        results: 各省验证结果
    """

    # 1. 设置随机种子
    set_global_seed(config['random_seed'])

    # 2. 加载数据
    print("\n" + "=" * 60)
    print("1. 数据加载")
    print("=" * 60)

    loader = DataLoader(config)
    yunnan_data = loader.load_yunnan()

    # 3. 训练云南模型
    print("\n" + "=" * 60)
    print(f"2. 训练云南模型 ({model_name.upper()})")
    print("=" * 60)

    model = train_baseline_model(config, yunnan_data, model_name)
    print(f"  模型训练完成")

    # 4. 跨省验证
    print("\n" + "=" * 60)
    print(f"3. 跨省验证 (模式: {'微调' if finetune else '零样本迁移'})")
    print("=" * 60)

    provinces = {
        'Guangdong': loader.load_guangdong,
        'Shandong': loader.load_shandong,
        'Beijing': loader.load_beijing
    }

    processor = DataProcessor(config)
    results = {}

    for name, load_func in provinces.items():
        print(f"\n  {name}...")

        # 加载省份数据
        province_data = load_func()

        # 划分训练集和测试集
        train_raw, test_raw, _, test_indices = processor.split_train_test(province_data)

        # 生成滑动窗口数据
        X_test, y_test = processor.create_sequences(test_raw, config['window_size'])
        X_train_prov, y_train_prov = processor.create_sequences(train_raw, config['window_size'])

        if len(X_test) == 0:
            print(f"    测试集为空，跳过")
            continue

        # 初始化变量
        original_weights = None

        # 微调模式：使用部分本地数据微调
        if finetune and len(X_train_prov) > 0:
            # 取部分数据用于微调
            finetune_size = int(len(X_train_prov) * finetune_ratio)
            X_finetune = X_train_prov[:finetune_size]
            y_finetune = y_train_prov[:finetune_size]

            # 剩余部分用于验证（可选）
            X_val = X_train_prov[finetune_size:] if finetune_size < len(X_train_prov) else None
            y_val = y_train_prov[finetune_size:] if finetune_size < len(y_train_prov) else None

            print(f"    微调样本数: {len(X_finetune)}")
            if X_val is not None:
                print(f"    验证样本数: {len(X_val)}")

            # 保存原始模型权重（用于恢复）
            if hasattr(model, 'model') and hasattr(model.model, 'get_weights'):
                try:
                    original_weights = model.model.get_weights()
                except:
                    original_weights = None
            else:
                original_weights = None

            # 微调
            finetune_model(model, X_finetune, y_finetune, X_val, y_val,
                           learning_rate=0.05 if model_name == 'lightgbm' else 0.0005,
                           epochs=50)

        # 预测
        y_pred = model.predict(X_test)

        # 计算指标
        metrics = calculate_metrics(y_test, y_pred)

        results[name] = {
            'rmse': metrics['rmse'],
            'mae': metrics['mae'],
            'r2': metrics['r2'],
            'y_true': y_test,
            'y_pred': y_pred,
            'indices': test_indices[config['window_size']:],
            'n_samples': len(X_test)
        }

        print(f"    RMSE: {metrics['rmse']:.4f}")
        print(f"    MAE: {metrics['mae']:.4f}")
        print(f"    R²: {metrics['r2']:.4f}")



    return results


def compare_models_cross_validation(config: dict, finetune: bool = False) -> pd.DataFrame:
    """
    对比多个模型在跨省验证中的表现
    """
    models = ['lightgbm', 'lstm', 'nbeats']
    all_results = {}

    for model_name in models:
        print(f"\n{'=' * 60}")
        print(f"验证模型: {model_name.upper()}")
        print(f"{'=' * 60}")

        # 检查模型是否启用
        if not config['models'].get(model_name, {}).get('enabled', True):
            print(f"  模型 {model_name.upper()} 未启用，跳过")
            continue

        results = cross_validate(config, model_name, finetune=finetune)
        all_results[model_name] = results

    # 汇总结果
    summary = []
    for model_name, results in all_results.items():
        for province, metrics in results.items():
            summary.append({
                'model': model_name.upper(),
                'province': province,
                'rmse': metrics['rmse'],
                'mae': metrics['mae'],
                'r2': metrics['r2']
            })

    summary_df = pd.DataFrame(summary)

    # 保存
    os.makedirs('../results', exist_ok=True)
    suffix = '_finetuned' if finetune else '_zeroshot'
    summary_df.to_csv(f'../results/cross_validation_summary{suffix}.csv', index=False)

    # 打印汇总表格
    print("\n" + "=" * 60)
    print(f"跨省验证汇总 (模式: {'微调' if finetune else '零样本迁移'})")
    print("=" * 60)
    print(summary_df.to_string(index=False))

    return summary_df


def main():
    """主函数"""
    config = load_config()

    # 选择验证模式
    print("\n" + "=" * 60)
    print("跨省验证实验")
    print("=" * 60)
    print("请选择验证模式:")
    print("  1. 零样本迁移（直接预测，不微调）")
    print("  2. 微调模式（用目标省份部分数据微调）")
    print("  3. 对比两种模式")

    choice = input("请输入选项 (1/2/3): ").strip()

    # 获取模型选择
    model_choices = {
        '1': 'lightgbm',
        '2': 'lstm',
        '3': 'nbeats',
        '4': 'all'
    }

    print("\n请选择模型:")
    print("  1. LightGBM (最优模型)")
    print("  2. LSTM")
    print("  3. N-BEATS")
    print("  4. 全部对比")

    model_choice = input("请输入选项 (1/2/3/4): ").strip()

    if choice == '1':
        # 零样本迁移
        if model_choice == '4':
            summary = compare_models_cross_validation(config, finetune=False)
        else:
            model_name = model_choices.get(model_choice, 'lightgbm')
            results = cross_validate(config, model_name, finetune=False)
            plot_cross_province(results, save_path='../results/cross_province_zeroshot.png')

    elif choice == '2':
        # 微调模式
        finetune_ratio_input = input("请输入微调数据比例 (默认0.3，即30%): ").strip()
        finetune_ratio = float(finetune_ratio_input) if finetune_ratio_input else 0.3

        if model_choice == '4':
            summary = compare_models_cross_validation(config, finetune=True)
        else:
            model_name = model_choices.get(model_choice, 'lightgbm')
            results = cross_validate(config, model_name, finetune=True, finetune_ratio=finetune_ratio)
            plot_cross_province(results, save_path='../results/cross_province_finetuned.png')

    elif choice == '3':
        # 对比两种模式
        print("\n" + "=" * 60)
        print("对比零样本迁移 vs 微调")
        print("=" * 60)

        if model_choice == '4':
            model_name = 'lightgbm'  # 默认用最优模型对比
        else:
            model_name = model_choices.get(model_choice, 'lightgbm')

        print(f"\n使用模型: {model_name.upper()}")

        # 零样本迁移
        print("\n--- 零样本迁移 ---")
        results_zero = cross_validate(config, model_name, finetune=False)

        # 微调模式
        print("\n--- 微调模式 ---")
        results_finetune = cross_validate(config, model_name, finetune=True, finetune_ratio=0.3)

        # 对比结果
        print("\n" + "=" * 60)
        print("对比结果")
        print("=" * 60)
        print(f"{'省份':<12} {'零样本RMSE':<12} {'微调后RMSE':<12} {'提升幅度':<12}")
        print("-" * 50)

        for province in results_zero.keys():
            if province in results_finetune:
                rmse_zero = results_zero[province]['rmse']
                rmse_fine = results_finetune[province]['rmse']
                improvement = (rmse_zero - rmse_fine) / rmse_zero * 100
                print(f"{province:<12} {rmse_zero:<12.4f} {rmse_fine:<12.4f} {improvement:<11.1f}%")

    print("\n" + "=" * 60)
    print("跨省验证实验完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()