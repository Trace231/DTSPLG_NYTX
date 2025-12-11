"""
快速演示脚本：展示多阶段复合抽样的核心功能
适合快速测试和理解抽样设计
"""

from multi_stage_sampling import MultiStageSampling
import pandas as pd

def quick_demo():
    """快速演示"""
    print("="*80)
    print("多阶段复合抽样设计 - 快速演示")
    print("="*80)
    print("\n【设计要素】")
    print("1. 时间分层（年份-季度）")
    print("2. 地理聚类（10×10网格）")
    print("3. 乘客人数分层（1人、2人、3-4人、5人+）")
    print("4. 系统抽样（层内等距抽样）")
    print("\n" + "-"*80 + "\n")
    
    # 初始化
    sampler = MultiStageSampling(
        data_path='train.csv',
        sample_size=3000  # 使用较小的样本量便于演示
    )
    
    # 加载数据（使用较小数据集演示）
    print("正在加载数据（使用10万条记录进行演示）...")
    sampler.load_data(nrows=100000)
    
    print(f"\n数据概览：")
    print(f"- 总记录数: {len(sampler.data):,}")
    print(f"- 平均车费: ${sampler.data['fare_amount'].mean():.2f}")
    print(f"- 车费标准差: ${sampler.data['fare_amount'].std():.2f}")
    print(f"- 时间范围: {sampler.data['pickup_datetime'].min()} 至 {sampler.data['pickup_datetime'].max()}")
    
    # 创建分层
    print("\n" + "="*80)
    print("步骤1：创建时间分层")
    print("="*80)
    sampler.create_time_strata()
    
    print("\n" + "="*80)
    print("步骤2：创建地理聚类")
    print("="*80)
    sampler.create_geographic_clusters()
    
    print("\n" + "="*80)
    print("步骤3：创建乘客人数分层")
    print("="*80)
    sampler.create_passenger_strata()
    
    print("\n" + "="*80)
    print("步骤4：分配样本量")
    print("="*80)
    allocation = sampler.allocate_sample_size()
    
    print("\n" + "="*80)
    print("步骤5：执行抽样")
    print("="*80)
    sample = sampler.draw_sample()
    
    print("\n样本概览：")
    print(f"- 样本量: {len(sample):,}")
    print(f"- 样本平均车费: ${sample['fare_amount'].mean():.2f}")
    print(f"- 样本车费标准差: ${sample['fare_amount'].std():.2f}")
    
    # 估计结果
    print("\n" + "="*80)
    print("步骤6：估计总体均值")
    print("="*80)
    results = sampler.estimate_mean_fare()
    
    # 简要总结
    print("\n" + "="*80)
    print("【总结】")
    print("="*80)
    print(f"真实总体均值: ${results['true_mean']:.2f}")
    print(f"分层估计均值: ${results['stratified_mean']:.2f}")
    print(f"估计偏差: ${results['stratified_mean'] - results['true_mean']:.2f}")
    print(f"相对误差: {abs(results['stratified_mean'] - results['true_mean']) / results['true_mean'] * 100:.2f}%")
    print(f"设计效应 (Deff): {results['deff']:.4f}")
    
    if results['deff'] < 1:
        print("\n✓ 多阶段分层聚类抽样比简单随机抽样更高效！")
        efficiency_gain = (1 - results['deff']) * 100
        print(f"  效率提升: {efficiency_gain:.2f}%")
    else:
        print("\n⚠ 注意：设计效应大于1，可能由于聚类效应导致方差增加")
        print("  但设计的优势在于：操作成本低、充分利用数据结构")
    
    # 保存结果
    sample.to_csv('demo_sampled_data.csv', index=False)
    print("\n✓ 样本已保存至: demo_sampled_data.csv")
    
    return sampler, results


if __name__ == "__main__":
    try:
        sampler, results = quick_demo()
        print("\n" + "="*80)
        print("演示完成！")
        print("="*80)
    except Exception as e:
        print(f"\n错误: {e}")
        print("\n提示：")
        print("1. 确保 train.csv 文件在当前目录")
        print("2. 如果文件太大，可以先用部分数据测试（已在代码中设置 nrows=100000）")
        print("3. 检查是否安装了所需库：pandas, numpy")

