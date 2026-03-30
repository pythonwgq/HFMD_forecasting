# HFMD_forecasting
本项目针对短时序手足口病发病率预测中数据稀缺、模型评估不严谨等问题，系统比较了6种主流预测模型，提出基于数据增强的LightGBM预测方法，并实现了跨区域泛化验证。项目提供了完整的训练代码、预训练模型及轻量级预测API，可直接使用。
项目结构见“项目目录.txt”。

---

### 调用模型，并运用本地数据，执行本区域的预测
1. 准备一份你的本区域手足口病发病率月度数据，您要预测未来三个月，请至少包含预测步+1的数据，即至少有近期4个月的数据。为了更好评估模型性能，推荐要有6个月以上的历史数据。文件类型需为csv格式，且列名需为"date"和"rate",将文件放在use_model文件夹下。
```
csv
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
1.你的数据需放在data/raw目录下，文件类型需为csv格式，且列名需为"date"和"rate"。
2.调整各个模型的参数，参数配置文件在config/config.yaml。
3.模型训练主文件在experiments/train.py:

```
cd experiments
python train.py
```
4.训练结果将保存至results目录下，模型保存至saved_model目录下。
5.第一次训练会保存增强的数据至data/processed目录下（pkl格式），后续训练调用该数据。如果该增强数据质量不高，将文件删除，重新训练即可重新执行增强策略。
6.运行experiments/cross_validation.py执行跨区域验证，目标区域的文件同样需要放在data/raw目录下，文件类型需为csv格式，且列名需为"date"和"rate"。

---
### 联系方式
如有问题，请联系：

作者：韦国清

邮箱：1784024931@qq.com

---

### License
本项目采用 MIT 许可证。欢迎使用、修改和分享。
