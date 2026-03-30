"""
手足口病发病率预测工具 - 基于预训练LightGBM模型
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import joblib
import os
import sys
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


# ========== 特征工程函数（直接复制，避免导入问题）==========
def build_features(X: np.ndarray, y: np.ndarray = None, window_size: int = 6):
    """精简版特征工程"""
    n_samples = X.shape[0]
    features = []

    for i in range(n_samples):
        row = X[i]
        feat = {}

        # 1. 统计特征
        feat['mean'] = np.mean(row)
        feat['std'] = np.std(row)
        feat['max'] = np.max(row)
        feat['min'] = np.min(row)

        # 2. 趋势特征
        feat['slope'] = (row[-1] - row[0]) / window_size
        feat['trend_ratio'] = row[-1] / (row[0] + 1e-6)

        # 3. 近期特征
        feat['last_1'] = row[-1]
        feat['last_2'] = row[-2] if window_size >= 2 else row[-1]
        feat['last_3'] = row[-3] if window_size >= 3 else row[-1]
        feat['ma_3'] = np.mean(row[-3:]) if window_size >= 3 else row[-1]

        # 4. 波动特征
        feat['volatility'] = np.std(row)
        feat['volatility_recent'] = np.std(row[-3:]) if window_size >= 3 else np.std(row)

        # 5. 变化率特征
        feat['change_1_2'] = (row[-1] - row[-2]) / (row[-2] + 1e-6) if window_size >= 2 else 0
        feat['mean_change'] = np.mean(np.diff(row[-3:])) if window_size >= 3 else 0

        # 6. 峰值特征
        feat['is_peak'] = 1 if row[-1] == np.max(row) else 0
        feat['peak_ratio'] = row[-1] / (np.max(row) + 1e-6)

        # 7. 滞后值
        feat['lag_2'] = row[-2] if window_size >= 2 else row[-1]
        feat['lag_3'] = row[-3] if window_size >= 3 else row[-1]

        features.append(feat)

    X_feat = pd.DataFrame(features)
    X_feat = X_feat.fillna(0)
    X_feat = X_feat.replace([np.inf, -np.inf], 0)

    if y is not None:
        return X_feat, y
    return X_feat, None


class HFMDPredictor:
    """手足口病发病率预测器"""

    def __init__(self, model_path: str, window_size: int = 3):
        self.model = joblib.load(model_path)
        self.window_size = window_size
        self.feature_columns = None

        if hasattr(self.model, 'feature_name_in_'):
            self.feature_columns = self.model.feature_name_in_

        print(f"✅ 模型加载成功")
        print(f"   窗口大小: {window_size} 个月")

    def extract_features(self, X: np.ndarray) -> pd.DataFrame:
        """从滑动窗口数据提取特征"""
        X_feat, _ = build_features(X, window_size=self.window_size)
        return X_feat

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        加载本地数据，自动检测编码

        Args:
            file_path: CSV文件路径，需包含 date 和 rate 列

        Returns:
            DataFrame with date and rate
        """
        # 尝试多种编码
        encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'gb18030', 'latin-1']
        df = None
        used_encoding = None

        for enc in encodings:
            try:
                df = pd.read_csv(file_path, encoding=enc)
                used_encoding = enc
                break
            except (UnicodeDecodeError, UnicodeError):
                continue

        if df is None:
            raise ValueError(f"无法读取文件，尝试的编码: {encodings}")

        print(f"   文件编码: {used_encoding}")

        # 解析日期
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.sort_values('date', inplace=True)
        else:
            raise ValueError("数据文件必须包含 'date' 列")

        # 获取发病率列
        if 'rate' in df.columns:
            df['rate'] = df['rate'].astype(float)
        else:
            # 尝试其他常见列名
            rate_cols = ['incidence', '发病率', 'value', 'y', 'incidence_rate']
            for col in rate_cols:
                if col in df.columns:
                    df['rate'] = df[col].astype(float)
                    break
            else:
                raise ValueError("数据文件必须包含 'rate' 列")

        print(f"✅ 数据加载成功")
        print(f"   时间范围: {df['date'].min()} ~ {df['date'].max()}")
        print(f"   数据条数: {len(df)}")

        return df


    def create_sequences(self, data: np.ndarray) -> tuple:
        """
        生成滑动窗口序列

        Args:
            data: 一维时间序列

        Returns:
            X: 输入窗口 (n_samples, window_size)
            y: 目标值 (n_samples,)
        """
        X, y = [], []
        for i in range(len(data) - self.window_size):
            X.append(data[i:i + self.window_size])
            y.append(data[i + self.window_size])
        return np.array(X), np.array(y)

    def evaluate(self, df: pd.DataFrame) -> dict:
        """
        评估模型在本地数据上的性能

        Args:
            df: 本地数据

        Returns:
            metrics: {'rmse': float, 'mae': float, 'r2': float}
        """
        data = df['rate'].values

        if len(data) < self.window_size + 1:
            print(f"⚠️ 数据不足，需要至少 {self.window_size + 1} 条数据，当前 {len(data)} 条")
            return None

        # 生成序列
        X, y_true = self.create_sequences(data)

        if len(X) == 0:
            print("⚠️ 无法生成有效的测试样本")
            return None

        # 提取特征
        X_feat = self.extract_features(X)

        # 确保特征列顺序一致
        if self.feature_columns is not None:
            X_feat = X_feat[self.feature_columns]

        # 预测
        y_pred = self.model.predict(X_feat)

        # 计算指标
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        # 计算R²
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))

        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'y_true': y_true,
            'y_pred': y_pred,
            'n_samples': len(X)
        }

    def predict_future(self, df: pd.DataFrame, months: int = 6) -> dict:
        """
        预测未来发病率

        Args:
            df: 历史数据
            months: 预测月数（1-12）

        Returns:
            predictions: 预测结果
        """
        if months < 1 or months > 12:
            raise ValueError("预测月数应在 1-12 之间")

        data = df['rate'].values
        dates = df['date'].values

        if len(data) < self.window_size:
            raise ValueError(f"历史数据不足，需要至少 {self.window_size} 条数据")

        # 使用最后 window_size 个点作为起点
        last_window = data[-self.window_size:].copy()
        predictions = []
        future_dates = []

        # 获取最后日期（转换为 pandas Timestamp 便于处理）
        last_date = pd.to_datetime(dates[-1])

        for i in range(months):
            # 计算下个月日期
            next_date = self._add_months(last_date, i + 1)
            future_dates.append(next_date)

            # 提取特征
            X = last_window.reshape(1, -1)
            X_feat = self.extract_features(X)

            if self.feature_columns is not None:
                X_feat = X_feat[self.feature_columns]

            # 预测
            pred = self.model.predict(X_feat)[0]
            predictions.append(max(0, pred))  # 预测值不能为负

            # 更新窗口（滑动）
            last_window = np.append(last_window[1:], pred)

        return {
            'dates': future_dates,
            'predictions': predictions,
            'last_actual': data[-self.window_size:],
            'last_dates': pd.to_datetime(dates[-self.window_size:])
        }

    def _add_months(self, date, months: int):
        """日期增加月份（支持 numpy.datetime64 和 datetime）"""
        # 转换为 Python datetime
        if isinstance(date, np.datetime64):
            date = pd.to_datetime(date).to_pydatetime()
        elif hasattr(date, 'to_pydatetime'):
            date = date.to_pydatetime()

        year = date.year + (date.month + months - 1) // 12
        month = (date.month + months - 1) % 12 + 1
        day = min(date.day, [31, 29 if year % 4 == 0 else 28, 31, 30, 31, 30,
                             31, 31, 30, 31, 30, 31][month - 1])
        return datetime(year, month, day)

    def plot_evaluation(self, metrics: dict, save_path: str = None):
        """绘制评估结果"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 图1：预测 vs 实际
        ax1 = axes[0]
        y_true = metrics['y_true']
        y_pred = metrics['y_pred']
        ax1.plot(y_true, label='实际值', marker='o', markersize=4, linewidth=1.5, color='black')
        ax1.plot(y_pred, label='预测值', alpha=0.7, marker='v', markersize=4, linewidth=1.5, color='#2ecc71')
        ax1.set_xlabel('时间步')
        ax1.set_ylabel('发病率 (1/10万)')
        ax1.set_title(f'模型评估 (RMSE={metrics["rmse"]:.2f}, MAE={metrics["mae"]:.2f})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 图2：误差分布
        ax2 = axes[1]
        errors = y_true - y_pred
        ax2.hist(errors, bins=15, edgecolor='black', alpha=0.7, color='steelblue')
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
        ax2.set_xlabel('预测误差')
        ax2.set_ylabel('频次')
        ax2.set_title(f'误差分布 (均值={errors.mean():.2f}, 标准差={errors.std():.2f})')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_prediction(self, future: dict, save_path: str = None):
        """绘制预测结果"""
        plt.figure(figsize=(12, 6))

        # 历史数据
        last_dates = future['last_dates']
        last_actual = future['last_actual']
        plt.plot(last_dates, last_actual, label='历史数据', marker='o',
                 markersize=5, linewidth=1.5, color='black')

        # 预测数据
        pred_dates = future['dates']
        pred_values = future['predictions']
        plt.plot(pred_dates, pred_values, label='预测值', marker='v',
                 markersize=5, linewidth=1.5, color='#e74c3c', linestyle='--')

        plt.xlabel('日期')
        plt.ylabel('发病率 (1/10万)')
        plt.title(f'未来 {len(pred_dates)} 个月手足口病发病率预测')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def save_predictions(self, future: dict, output_path: str):
        """保存预测结果到CSV"""
        df = pd.DataFrame({
            '预测月份': [d.strftime('%Y-%m') for d in future['dates']],
            '预测发病率': [f'{p:.2f}' for p in future['predictions']]
        })
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"✅ 预测结果已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='手足口病发病率预测工具')
    parser.add_argument('--data', '-d', type=str,default="./data.csv",
                        help='本地数据文件路径（CSV格式，包含date和rate列）')
    parser.add_argument('--model', '-m', type=str, default='../saved_models/best_model_LightGBM.pkl',
                        help='预训练模型路径（默认: ../saved_models/best_model_LightGBM.pkl）')
    parser.add_argument('--months', '-p', type=int, default=3,
                        help='预测月数（1-12，默认: 3）')
    parser.add_argument('--window', '-w', type=int, default=3,
                        help='滑动窗口大小（用几个月预测下一个月，默认: 3）')
    parser.add_argument('--output', '-o', type=str, default='./predictions.csv',
                        help='预测结果输出路径（默认: ./predictions.csv）')
    parser.add_argument('--no_eval', action='store_true',
                        help='跳过模型评估（仅预测）')

    args = parser.parse_args()

    # 检查文件是否存在
    if not os.path.exists(args.data):
        print(f"❌ 数据文件不存在: {args.data}")
        sys.exit(1)

    if not os.path.exists(args.model):
        print(f"❌ 模型文件不存在: {args.model}")
        print(f"   请先训练模型或指定正确的模型路径")
        sys.exit(1)

    print("=" * 60)
    print("手足口病发病率预测工具")
    print("=" * 60)

    # 初始化预测器
    predictor = HFMDPredictor(args.model, window_size=args.window)

    # 加载数据
    df = predictor.load_data(args.data)
    # 检查数据量
    min_required = args.window + 6
    if len(df) < min_required:
        print(f"⚠️ 数据量较少（{len(df)}条），建议至少 {min_required} 条数据以获得可靠的评估结果")
        print(f"   当前数据只能生成 {len(df) - args.window} 个评估样本")
    # 评估模型性能
    if not args.no_eval and len(df) >= args.window + 1:
        print("\n" + "=" * 60)
        print("1. 模型评估")
        print("=" * 60)

        metrics = predictor.evaluate(df)
        if metrics:
            print(f"   样本数: {metrics['n_samples']}")
            print(f"   RMSE: {metrics['rmse']:.4f}")
            print(f"   MAE: {metrics['mae']:.4f}")
            print(f"   R²: {metrics['r2']:.4f}")
            predictor.plot_evaluation(metrics, save_path='./evaluation.png')

    # 预测未来
    print("\n" + "=" * 60)
    print(f"2. 未来 {args.months} 个月预测")
    print("=" * 60)

    future = predictor.predict_future(df, months=args.months)

    print("\n预测结果:")
    print("-" * 40)
    print(f"{'月份':<12} {'预测发病率 (1/10万)':<20}")
    print("-" * 40)
    for date, pred in zip(future['dates'], future['predictions']):
        print(f"{date.strftime('%Y-%m'):<12} {pred:<20.2f}")
    print("-" * 40)

    # 绘制预测图
    predictor.plot_prediction(future, save_path='./prediction.png')

    # 保存结果
    predictor.save_predictions(future, args.output)

    print("\n" + "=" * 60)
    print("✅ 预测完成！")
    print(f"   评估图: ./evaluation.png")
    print(f"   预测图: ./prediction.png")
    print(f"   预测结果: {args.output}")
    print("=" * 60)


if __name__ == '__main__':
    main()