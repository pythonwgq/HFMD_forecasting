# HFMD_forecasting
本项目针对短时序手足口病发病率预测中数据稀缺、模型评估不严谨等问题，系统比较了6种主流预测模型，提出基于数据增强的LightGBM预测方法，并实现了跨区域泛化验证。项目提供了完整的训练代码、预训练模型及轻量级预测API，可直接使用。
## 项目结构
```
HFMD_Prediction/
├── config/                      # 配置文件
│   └── config.yaml              # 模型参数、数据路径等配置
├── data/                        # 数据目录
│   ├── raw/                     # 原始数据（需自行申请）
│   │   ├── yunnan.csv
│   │   ├── guangdong.csv
│   │   ├── shandong.csv
│   │   └── beijing.csv
│   └── processed/               # 处理后数据（自动生成）
├── src/                         # 源代码
│   ├── data/                    # 数据处理
│   │   ├── loader.py            # 数据加载
│   │   ├── processor.py         # 数据预处理
│   │   ├── augmenter.py         # 数据增强
│   │   └── features.py          # 特征工程
│   ├── models/                  # 模型实现
│   │   ├── base.py              # 模型基类
│   │   ├── arima.py
│   │   ├── prophet.py
│   │   ├── lstm.py
│   │   ├── cnn_bilstm.py
│   │   ├── lightgbm.py
│   │   └── nbeats.py
│   ├── evaluation/              # 评估模块
│   │   ├── metrics.py           # 评估指标
│   │   └── comparison.py        # 模型对比
│   └── visualization/           # 可视化
│       └── plotter.py
├── experiments/                 # 实验脚本
│   ├── train.py                 # 云南模型训练
│   └── cross_validation.py      # 跨省验证
├── saved_models/                # 保存的模型
│   └── best_model_LightGBM.pkl  # 最优模型（LightGBM）
├── use_model/                   # 预测工具
│   └── predict_with_model.py    # 独立预测脚本
├── requirements.txt             # 依赖包
└── README.md                    # 本文件
```
