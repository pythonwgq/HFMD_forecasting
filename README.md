# HFMD_forecasting
本项目针对短时序手足口病发病率预测中数据稀缺、模型评估不严谨等问题，系统比较了6种主流预测模型，探讨了基于[TSAUG框架](https://tsaug.readthedocs.io/en/stable/)数据增强的LightGBM预测方法，并实现了跨区域泛化验证。项目提供了完整的训练代码、预训练模型及轻量级预测API，可直接使用。

---

### 项目结构
```txt
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
│   │   └── enhanced_features.py  # 特征工程
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
│   ├── train.py                 # 模型训练
│   └── cross_validation.py      # 跨省验证
├── saved_models/                # 保存的模型
│   └── best_model_LightGBM.pkl  # 最优模型（LightGBM）
├── use_model/                   # 预测工具
│   └── predict_with_model.py    # 独立预测脚本
    └── data.csv                 #目标区域数据
├── requirements.txt             # 依赖包
└── README.md                    # 本文件
```

---

### 调用模型，并运用本地数据，执行本区域的预测
1. 准备一份你的本区域手足口病发病率月度数据，您要预测未来三个月，请至少包含预测步+1的数据，即至少有近期4个月的数据。为了更好评估模型性能，推荐要有6个月以上的历史数据。文件类型需为csv格式，且列名需为"date"和"rate",将文件放在use_model文件夹下。

```csv
date,        rate
2005-01-01,  5.2
2005-02-01,  4.8
```
2. 调用模型代码文件在use_model/predict_with_model.py，有几个参数需要修改和确定：
   |参数 |	说明 |	默认值 |
   |-----| ----- | ------|
   | --data	| 本地数据文件路径	| ./data.csv |
   | --model	 | 模型路径	 | ../saved_models/best_model_LightGBM.pkl |
   | --months	| 预测月数（1-12）|	6 |
   | --window	| 滑动窗口大小,基于未来几个月预测下一个月 |	3 |
   | --output	| 预测结果保存路径	| ./predictions.csv |
   | --no_eval	| 跳过模型评估 |False |

3. 运行predict_with_model.py，开始调用模型并执行预测，模型性能评估和预测结果会可视化输出，并将结果文件保存至use_model目录下。


```
cd ../use_model
python predict_with_model.py --data ./your_data.csv --months 6 --window 3
```

---

### 可以用你的数据，进行新的模型训练

1. 你的数据需放在data/raw目录下，文件类型需为csv格式，且列名需为"date"和"rate"。
2. 调整各个模型的参数，参数配置文件在config/config.yaml。
3. 模型训练主文件在experiments/train.py:

```
cd experiments
python train.py
```


4. 训练结果将保存至results目录下，模型保存至saved_model目录下。
5. 第一次训练会保存增强的数据至data/processed目录下（pkl格式），后续训练调用该数据。如果该增强数据质量不高，将文件删除，重新训练即可重新执行增强策略。
6. 运行experiments/cross_validation.py执行跨区域验证，目标区域的文件同样需要放在data/raw目录下，文件类型需为csv格式，且列名需为"date"和"rate"。
## 论文信息

> 韦国清，李存仙，段云权等. 数据增强下短时序手足口病预测的多模型比较与跨区域验证[J]. 杂志，日期（审稿中）.

如您使用本代码，请引用以上论文。

## 联系方式
邮箱：1793295683@qq.com


---

### License

本项目采用 MIT 许可证。欢迎使用、修改和分享。本研究所用原始数据来源于[中国疾病预防控制中心公共卫生科学数据中心](https://www.phsciencedata.cn/)，包含云南、山东、广东、北京四省市2005—2019年手足口病月度发病率数据。原始数据不提供，如需请自行申请。

