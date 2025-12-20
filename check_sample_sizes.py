"""检查SRS和多阶段复合抽样的实际样本量是否一致"""

import pandas as pd
import numpy as np
from multi_stage_sampling import MultiStageSampling

# 加载数据
data_path = 'train.csv'
sample_size = 1000
nrows = 500000

print("加载数据...")
sampler = MultiStageSampling(data_path, sample_size)
sampler.load_data(nrows=nrows)
data = sampler.data
true_mean = data['fare_amount'].mean()

print(f"\n真实总体均值: ${true_mean:.6f}")
print(f"总体大小: {len(data):,}")
print(f"目标样本量: {sample_size:,}\n")

# 1. 检查SRS的样本量
print("=" * 60)
print("【SRS】")
np.random.seed(114514)
srs_sample = data.sample(n=sample_size, random_state=0)
print(f"SRS实际样本量: {len(srs_sample):,}")
print(f"SRS样本均值: ${srs_sample['fare_amount'].mean():.2f}")

# 2. 检查多阶段复合抽样的样本量
print("\n" + "=" * 60)
print("【多阶段复合抽样】")
sampler2 = MultiStageSampling(data_path, sample_size)
sampler2.data = data.copy()

# 执行多阶段抽样
sampler2.create_time_strata(simplify=True)
sampler2.create_geographic_clusters(method='route', n_clusters=3)
sampler2.create_passenger_strata(simplify=True)
sampler2.allocate_sample_size()
sampler2.draw_sample()

if sampler2.final_sample is not None:
    print(f"多阶段复合实际样本量: {len(sampler2.final_sample):,}")
    print(f"多阶段复合样本均值: ${sampler2.final_sample['fare_amount'].mean():.2f}")
    print(f"\n样本量差异: {len(sampler2.final_sample) - sample_size:,}")
    
    # 检查分配信息
    if hasattr(sampler2, 'allocation_info'):
        actual_allocated = sampler2.allocation_info['allocation'].sum()
        print(f"分配的总样本量: {actual_allocated:,}")
        print(f"有效单元数: {(sampler2.allocation_info['allocation'] > 0).sum()}")
        
        # 显示分配详情
        print("\n分配详情（前10个单元）:")
        print(sampler2.allocation_info[sampler2.allocation_info['allocation'] > 0].head(10)[['N_h', 'allocation']])
else:
    print("多阶段复合抽样失败")

# 3. 多次试验检查样本量稳定性
print("\n" + "=" * 60)
print("【多次试验检查样本量稳定性】")
np.random.seed(114514)
srs_sizes = []
multi_sizes = []

for i in range(10):
    # SRS
    srs = data.sample(n=sample_size, random_state=i)
    srs_sizes.append(len(srs))
    
    # 多阶段复合
    sampler3 = MultiStageSampling(data_path, sample_size)
    sampler3.data = data.copy()
    sampler3.create_time_strata(simplify=True)
    sampler3.create_geographic_clusters(method='route', n_clusters=3)
    sampler3.create_passenger_strata(simplify=True)
    sampler3.allocate_sample_size()
    sampler3.draw_sample()
    if sampler3.final_sample is not None:
        multi_sizes.append(len(sampler3.final_sample))

print(f"SRS样本量: 平均={np.mean(srs_sizes):.1f}, 标准差={np.std(srs_sizes):.1f}")
print(f"多阶段复合样本量: 平均={np.mean(multi_sizes):.1f}, 标准差={np.std(multi_sizes):.1f}")
print(f"样本量差异: {np.mean(multi_sizes) - sample_size:.1f}")

