# 纽约出租车费用预测 - 多阶段复合抽样设计

## 📋 项目简介

这是一个**三层嵌套的混合抽样设计**项目，用于估计纽约出租车平均费用。设计结合了：

1. **时间分层抽样**（Stratified Sampling by Time Period）
2. **地理位置聚类抽样**（Cluster Sampling by Geographic Region）
3. **乘客人数分层抽样**（Stratified Sampling by Passenger Count）
4. **系统抽样**（Systematic Sampling）

## 🎯 设计亮点

- ✅ **理论完备**：四种经典抽样方法的有机结合
- ✅ **设计新颖**：三层嵌套在实际项目中少见
- ✅ **充分利用数据**：时间、空间、人群三个维度的信息
- ✅ **可对比验证**：与简单随机抽样对比，展示设计优势

## 📁 文件说明

- `multi_stage_sampling.py` - 多阶段复合抽样实现
- `compare_sampling_methods.py` - 不同抽样方法对比
- `抽样设计说明.md` - 详细的统计理论说明
- `train.csv` - 训练数据（需要Kaggle下载）

## 🚀 快速开始

### 1. 环境准备
请先下载数据集：https://www.kaggle.com/competitions/new-york-city-taxi-fare-prediction/data
```bash
pip install pandas numpy matplotlib seaborn scipy
```

### 2. 运行多阶段抽样

```python
from multi_stage_sampling import MultiStageSampling

# 初始化
sampler = MultiStageSampling(
    data_path='train.csv',
    sample_size=5000  # 总样本量
)

# 加载数据（可以先加载部分数据测试）
sampler.load_data(nrows=500000)  # 50万条记录

# 执行抽样设计
sampler.create_time_strata()           # 时间分层
sampler.create_geographic_clusters()   # 地理聚类
sampler.create_passenger_strata()      # 乘客分层
sampler.allocate_sample_size()         # 分配样本量
sampler.draw_sample()                  # 执行抽样

# 估计和报告
sampler.generate_report()

# 保存样本
sampler.final_sample.to_csv('sampled_data.csv', index=False)
```

### 3. 对比不同抽样方法

```python
from compare_sampling_methods import SamplingComparison

comparator = SamplingComparison(
    data_path='train.csv',
    sample_size=5000,
    nrows=500000
)

# 对比所有方法
comparison_df = comparator.compare_all_methods()
```

## 📊 输出结果

运行后会生成：

1. **控制台输出**：详细的抽样过程和估计结果
2. `sampled_data.csv`：最终抽取的样本数据
3. `sampling_comparison_results.csv`：不同方法的对比结果
4. `sampling_comparison.png`：可视化对比图

## 🔬 抽样设计结构

```
总体（所有出租车行程）
  │
  ├─ 第一层：时间分层
  │   └─ 按年份-季度（2009-Q1, 2009-Q2, ...）
  │
  ├─ 第二层：地理聚类
  │   └─ 10×10网格划分纽约市
  │
  ├─ 第三层：乘客分层
  │   └─ 1人、2人、3-4人、5人+
  │
  └─ 层内：系统抽样
      └─ 按时间排序后等距抽样
```

## 📈 统计公式

### 分层估计量

$$\bar{y}_{st} = \sum_{h=1}^{H} W_h \bar{y}_h$$

### 方差估计

$$\text{Var}(\bar{y}_{st}) = \sum_{h=1}^{H} W_h^2 \frac{s_h^2}{n_h} (1 - f_h)$$

### 设计效应

$$\text{Deff} = \frac{\text{Var}(\bar{y}_{design})}{\text{Var}(\bar{y}_{srs})}$$

## 📝 注意事项

1. **数据文件较大**：train.csv 可能超过200MB，建议先用部分数据测试
2. **内存占用**：完整数据可能占用较多内存，可根据机器配置调整 `nrows` 参数
3. **运行时间**：对比脚本会运行多次抽样，可能需要几分钟时间

## 🎓 学术引用
主要参考文献为：
- Lohr, S. L. (2019). Sampling: Design and Analysis (3rd ed.)
- Cochran, W. G. (1977). Sampling Techniques (3rd ed.)

## 📧 问题反馈

如有问题或建议，欢迎讨论！

