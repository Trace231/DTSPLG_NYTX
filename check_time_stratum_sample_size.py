"""统计多阶段抽样中每个时间层的样本量"""
import pandas as pd

# 读取样本文件
df = pd.read_csv('sampled_data_method1.csv')

# 统计每个时间层的样本量
time_stratum_counts = df.groupby('time_stratum').size().sort_index()

print("=" * 60)
print("Time Stratum Sample Sizes")
print("=" * 60)
for idx, val in time_stratum_counts.items():
    print(f"{idx}: {val}")

print(f"\nTotal sample size: {len(df)}")
print(f"Number of time strata: {len(time_stratum_counts)}")
print(f"Mean per stratum: {time_stratum_counts.mean():.1f}")
print(f"Min: {time_stratum_counts.min()}")
print(f"Max: {time_stratum_counts.max()}")

