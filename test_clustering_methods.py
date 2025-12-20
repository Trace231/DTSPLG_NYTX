"""
测试不同的地理聚类方法
对比：网格划分 vs 距离聚类 vs 路线相似性聚类
"""

from multi_stage_sampling import MultiStageSampling
import pandas as pd

def test_clustering_methods():
    """测试不同的聚类方法"""
    print("="*80)
    print("测试地理聚类方法")
    print("="*80)
    
    # 初始化
    sampler = MultiStageSampling(
        data_path='train.csv',
        sample_size=2000
    )
    
    # 加载数据（使用较小数据集进行快速测试）
    print("\n加载数据...")
    sampler.load_data(nrows=100000)  # 10万条记录
    
    # 创建时间分层和乘客分层（这些对所有方法都相同）
    sampler.create_time_strata()
    sampler.create_passenger_strata()
    
    # 测试三种聚类方法
    methods = [
        ('grid', '网格划分（原始方法）'),
        ('distance', '距离聚类（K-means，基于pickup位置）'),
        ('route', '路线相似性聚类（考虑pickup和dropoff位置）')
    ]
    
    results = {}
    
    for method, description in methods:
        print("\n" + "="*80)
        print(f"测试方法: {description}")
        print("="*80)
        
        # 创建地理聚类
        cluster_info = sampler.create_geographic_clusters(
            method=method,
            n_clusters=30  # 使用30个聚类进行测试
        )
        
        # 计算聚类质量指标
        within_var = cluster_info['std_fare'].mean()
        between_var = cluster_info['avg_fare'].std()
        ratio = between_var / within_var if within_var > 0 else 0
        
        results[method] = {
            'n_clusters': len(cluster_info),
            'avg_cluster_size': cluster_info['cluster_size'].mean(),
            'within_variance': within_var,
            'between_variance': between_var,
            'ratio': ratio
        }
        
        print(f"\n聚类质量总结:")
        print(f"  聚类数: {len(cluster_info)}")
        print(f"  平均聚类大小: {cluster_info['cluster_size'].mean():.0f}")
        print(f"  层内平均标准差: {within_var:.2f}")
        print(f"  层间标准差: {between_var:.2f}")
        print(f"  层间/层内比率: {ratio:.2f} (越大越好)")
    
    # 对比结果
    print("\n" + "="*80)
    print("方法对比总结")
    print("="*80)
    print(f"{'方法':<20} {'聚类数':<10} {'平均大小':<12} {'层内方差':<12} {'层间/层内比率':<15}")
    print("-" * 80)
    for method, description in methods:
        r = results[method]
        print(f"{description:<20} {r['n_clusters']:<10} {r['avg_cluster_size']:<12.0f} {r['within_variance']:<12.2f} {r['ratio']:<15.2f}")
    
    # 找出最佳方法（层间/层内比率最大）
    best_method = max(results.keys(), key=lambda k: results[k]['ratio'])
    print(f"\n最佳聚类方法（层间/层内比率最大）: {best_method}")
    print("  说明：层间/层内比率越大，说明聚类效果越好（层间差异大，层内差异小）")
    
    return results

if __name__ == "__main__":
    results = test_clustering_methods()

