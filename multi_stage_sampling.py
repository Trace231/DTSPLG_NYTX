"""
多阶段复合抽样设计：估计纽约出租车平均费用
设计要素：
1. 按时间区间分层（Stratified Sampling by Time Period）
2. 按距离范围分层（Stratified Sampling by Distance Range）- 替代地理聚类
3. 按乘客人数分层（Stratified Sampling by Passenger Count）
4. 结合系统抽样（Systematic Sampling）

这是一个三层嵌套的混合抽样设计
注意：第二层已从地理聚类改为距离范围分层，更简单且对车费预测更有意义
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
from tqdm import tqdm
from scipy.stats import norm
import argparse
warnings.filterwarnings('ignore')


class MultiStageSampling:
    """多阶段复合抽样设计类"""
    
    def __init__(self, data_path='train.csv', sample_size=5000):
        """
        初始化抽样设计
        
        Parameters:
        -----------
        data_path : str
            数据文件路径
        sample_size : int
            总样本量
        """
        self.data_path = data_path
        self.target_sample_size = sample_size
        self.data = None
        self.stratified_data = None
        self.final_sample = None
        
    def load_data(self, nrows=None):
        """加载数据（可选择只加载部分数据用于演示）"""
        print("正在加载数据...")
        if nrows:
            self.data = pd.read_csv(self.data_path, nrows=nrows)
        else:
            # 对于大数据集，可以先加载一部分进行分析
            self.data = pd.read_csv(self.data_path, nrows=1000000)  # 先加载100万行作为演示
        
        # 转换时间列
        self.data['pickup_datetime'] = pd.to_datetime(self.data['pickup_datetime'])
        
        # 数据预处理
        self.data = self.data.dropna(subset=['fare_amount', 'pickup_longitude', 
                                              'pickup_latitude', 'passenger_count'])
        
        # 过滤异常值
        self.data = self.data[
            (self.data['fare_amount'] > 0) & 
            (self.data['fare_amount'] < 500) &  # 合理的车费上限
            (self.data['pickup_longitude'] > -75) & 
            (self.data['pickup_longitude'] < -73) &
            (self.data['pickup_latitude'] > 40) & 
            (self.data['pickup_latitude'] < 41)
        ]
        
        print(f"数据加载完成，有效记录数: {len(self.data):,}")
        return self.data
    
    def create_time_strata(self, simplify=False):
        """
        第一层：按时间区间分层
        
        将数据按年份或季度分层，可以根据数据的时间分布灵活调整
        
        Parameters:
        -----------
        simplify : bool, default=False
            如果为True，只按年份分层（不按季度），减少层数
        """
        print("\n=== 第一层：创建时间分层 ===")
        self.data['year'] = self.data['pickup_datetime'].dt.year
        if simplify:
            # 简化模式：只按年份分层
            self.data['time_stratum'] = self.data['year'].astype(str)
            print("使用简化模式：只按年份分层")
        else:
            # 完整模式：按年份-季度分层
            self.data['quarter'] = self.data['pickup_datetime'].dt.quarter
            self.data['time_stratum'] = self.data['year'].astype(str) + '-Q' + self.data['quarter'].astype(str)
        
        # 统计各层信息
        stratum_info = self.data.groupby('time_stratum').agg({
            'fare_amount': 'count',
            'key': 'count'
        }).rename(columns={'fare_amount': 'N_h', 'key': 'count'})
        
        print("\n时间分层统计：")
        print(stratum_info)
        print(f"\n总层数: {len(stratum_info)}")
        
        return stratum_info
    
    def create_geographic_clusters(self, method='distance', n_clusters=50, use_route_similarity=False):
        """
        第二层：按距离范围分层（替代地理聚类）
        
        使用pickup和dropoff之间的距离，将行程分为若干距离范围（如：短途、中途、长途）
        这种方法比地理聚类更简单，且对车费预测更有意义
        
        Parameters:
        -----------
        method : str, default='distance'
            保留此参数以兼容现有代码，但实际使用距离范围分层
        n_clusters : int, default=50
            距离范围的数量（将被转换为距离分位数）
        use_route_similarity : bool, default=False
            保留此参数以兼容现有代码，但实际不使用
        """
        print("\n=== 第二层：创建距离范围分层（替代地理聚类） ===")
        print(f"将使用距离范围替代地理聚类，距离范围数: {n_clusters}")
        
        # 计算路线距离（Haversine距离）
        def haversine_distance_vectorized(lon1, lat1, lon2, lat2):
            """向量化计算两点间的Haversine距离（公里）"""
            R = 6371  # 地球半径（公里）
            dlat = np.radians(lat2 - lat1)
            dlon = np.radians(lon2 - lon1)
            a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
            return R * c
        
        # 检查必要的列是否存在
        required_cols = ['pickup_longitude', 'pickup_latitude', 
                        'dropoff_longitude', 'dropoff_latitude']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"缺少必要的列: {missing_cols}")
        
        # 检查并处理NaN值
        valid_mask = ~(
            self.data[required_cols].isna().any(axis=1)
        )
        n_invalid = (~valid_mask).sum()
        if n_invalid > 0:
            print(f"警告：发现 {n_invalid} 条记录包含NaN值，将在计算距离时排除这些记录")
        
        if valid_mask.sum() == 0:
            raise ValueError("所有记录的经纬度都包含NaN值，无法计算距离")
        
        # 向量化计算路线距离
        print("正在计算pickup和dropoff之间的距离...")
        route_distances = np.full(len(self.data), np.nan)
        route_distances[valid_mask] = haversine_distance_vectorized(
            self.data.loc[valid_mask, 'pickup_longitude'].values,
            self.data.loc[valid_mask, 'pickup_latitude'].values,
            self.data.loc[valid_mask, 'dropoff_longitude'].values,
            self.data.loc[valid_mask, 'dropoff_latitude'].values
        )
        
        # 将距离保存到数据中（可选，用于后续分析）
        self.data['route_distance_km'] = route_distances
        
        # 根据距离分位数创建距离范围
        # 使用分位数确保每个范围的大小大致相等
        valid_distances = route_distances[valid_mask]
        
        if n_clusters <= 1:
            # 如果只有1个或更少的范围，使用简单的阈值
            distance_thresholds = [0, np.inf]
            distance_labels = ['Distance_All']
        else:
            # 计算分位数作为阈值
            percentiles = np.linspace(0, 100, n_clusters + 1)
            distance_thresholds = np.percentile(valid_distances, percentiles)
            # 确保第一个阈值是0，最后一个阈值是inf
            distance_thresholds[0] = 0
            distance_thresholds[-1] = np.inf
            
            # 创建距离范围标签
            distance_labels = []
            for i in range(len(distance_thresholds) - 1):
                if i == 0:
                    label = f"Distance_0_{distance_thresholds[i+1]:.2f}km"
                elif i == len(distance_thresholds) - 2:
                    label = f"Distance_{distance_thresholds[i]:.2f}km_plus"
                else:
                    label = f"Distance_{distance_thresholds[i]:.2f}_{distance_thresholds[i+1]:.2f}km"
                distance_labels.append(label)
        
        print(f"\n距离范围阈值（公里）:")
        for i in range(len(distance_thresholds) - 1):
            print(f"  范围 {i+1}: {distance_thresholds[i]:.2f} - {distance_thresholds[i+1]:.2f} km")
        
        # 为每条记录分配距离范围
        def assign_distance_range(distance):
            """根据距离值分配距离范围"""
            if np.isnan(distance):
                return "Distance_NaN"
            for i in range(len(distance_thresholds) - 1):
                if distance_thresholds[i] <= distance < distance_thresholds[i+1]:
                    return distance_labels[i]
            # 如果距离等于最后一个阈值（inf），分配到最后一个范围
            return distance_labels[-1]
        
        self.data['geo_cluster'] = [assign_distance_range(d) for d in route_distances]
        
        # 保存阈值信息（可选，用于后续分析）
        self.distance_thresholds = distance_thresholds
        self.distance_labels = distance_labels
        
        # 统计距离范围信息
        cluster_info = self.data.groupby('geo_cluster').agg({
            'fare_amount': ['count', 'mean', 'std'],
            'route_distance_km': ['mean', 'min', 'max']
        }).round(2)
        cluster_info.columns = ['cluster_size', 'avg_fare', 'std_fare', 
                               'avg_distance', 'min_distance', 'max_distance']
        
        print(f"\n距离范围分层统计:")
        print(f"总范围数: {len(cluster_info)}")
        print(f"平均每个范围大小: {cluster_info['cluster_size'].mean():.0f}")
        print(f"范围大小标准差: {cluster_info['cluster_size'].std():.0f}")
        print(f"平均车费标准差（层内方差）: {cluster_info['std_fare'].mean():.2f}")
        print("\n各距离范围详细信息:")
        print(cluster_info.sort_values('avg_distance'))
        
        # 计算分层质量指标
        within_cluster_variance = cluster_info['std_fare'].mean()
        between_cluster_variance = cluster_info['avg_fare'].std()
        print(f"\n分层质量指标:")
        print(f"  层内平均标准差: {within_cluster_variance:.2f}")
        print(f"  层间标准差: {between_cluster_variance:.2f}")
        if within_cluster_variance > 0:
            print(f"  层间/层内比率: {between_cluster_variance/within_cluster_variance:.2f} (越大越好)")
        
        # 显示距离与车费的关系
        print(f"\n距离与车费关系:")
        distance_fare_corr = self.data[['route_distance_km', 'fare_amount']].corr().iloc[0, 1]
        print(f"  距离与车费的相关系数: {distance_fare_corr:.3f}")
        
        return cluster_info
    
    def create_passenger_strata(self, simplify=False):
        """
        第三层：按乘客人数分层
        
        将乘客人数分为若干层（例如：1人、2人、3-4人、5人及以上）
        
        Parameters:
        -----------
        simplify : bool, default=False
            如果为True，只分2层（1人 vs 2+人），减少层数
        """
        print("\n=== 第三层：创建乘客人数分层 ===")
        
        if simplify:
            # 简化模式：只分2层（1人 vs 2+人）
            def assign_passenger_stratum(passenger_count):
                """将乘客人数映射到分层（简化模式：2层）"""
                if passenger_count == 1:
                    return '1_passenger'
                else:
                    return '2+_passengers'
            print("使用简化模式：只分2层（1人 vs 2+人）")
        else:
            # 完整模式：分4层
            def assign_passenger_stratum(passenger_count):
                """将乘客人数映射到分层（完整模式：4层）"""
                if passenger_count == 1:
                    return '1_passenger'
                elif passenger_count == 2:
                    return '2_passengers'
                elif passenger_count <= 4:
                    return '3-4_passengers'
                else:
                    return '5+_passengers'
        
        self.data['passenger_stratum'] = self.data['passenger_count'].apply(assign_passenger_stratum)
        
        # 统计分层信息
        passenger_info = self.data.groupby('passenger_stratum').agg({
            'fare_amount': ['count', 'mean', 'std'],
            'passenger_count': 'mean'
        }).round(2)
        passenger_info.columns = ['count', 'avg_fare', 'std_fare', 'avg_passengers']
        
        print("\n乘客人数分层统计：")
        print(passenger_info)
        
        return passenger_info
    
    def allocate_sample_size(self, method='neyman'):
        """
        样本量分配策略
        
        使用最优分配（Neyman分配）的思想，但需要平衡三层结构
        
        Parameters:
        -----------
        method : str, default='neyman'
            分配方法：'proportional'（比例分配）或 'neyman'（Neyman最优分配）
        """
        print("\n=== 样本量分配 ===")
        
        # 创建完整的分层聚类结构
        self.data['stratum_cluster_key'] = (
            self.data['time_stratum'] + '_' + 
            self.data['geo_cluster'] + '_' + 
            self.data['passenger_stratum']
        )
        
        # 统计每个层级组合的大小和方差
        allocation_info = self.data.groupby('stratum_cluster_key').agg({
            'fare_amount': ['count', 'mean', 'std'],
        })
        allocation_info.columns = ['N_h', 'mean_fare', 'std_fare']
        allocation_info['std_fare'] = allocation_info['std_fare'].fillna(0)
        
        # 只考虑有效的单元（N_h > 0）
        valid_units = allocation_info[allocation_info['N_h'] > 0].copy()
        n_valid_units = len(valid_units)
        
        if method == 'neyman':
            # Neyman最优分配：n_h = n * (N_h * σ_h) / Σ(N_h * σ_h)
            # 考虑有限总体修正
            total_N = valid_units['N_h'].sum()
            # 使用标准差进行最优分配
            valid_units['N_sigma'] = valid_units['N_h'] * valid_units['std_fare']
            total_N_sigma = valid_units['N_sigma'].sum()
            
            if total_N_sigma > 0:
                # 初始分配（连续值）
                valid_units['allocation_float'] = (
                    self.target_sample_size * valid_units['N_sigma'] / total_N_sigma
                )
            else:
                # 如果所有标准差为0，回退到比例分配
                valid_units['allocation_float'] = (
                    self.target_sample_size * valid_units['N_h'] / total_N
                )
            print("使用Neyman最优分配")
        else:
            # 比例分配：n_h = n * N_h / N
            total_N = valid_units['N_h'].sum()
            valid_units['allocation_float'] = (
                self.target_sample_size * valid_units['N_h'] / total_N
            )
            print("使用比例分配")
        
        # 先向下取整
        valid_units['allocation'] = valid_units['allocation_float'].astype(int)
        
        # 如果样本量足够大，确保每个有效单元至少分配到1个样本
        if self.target_sample_size >= n_valid_units:
            # 确保每个单元至少1个样本
            valid_units.loc[valid_units['allocation'] == 0, 'allocation'] = 1
        else:
            # 样本量不足，只保留分配数>=1的单元
            print(f"警告：样本量({self.target_sample_size})小于有效单元数({n_valid_units})，"
                  f"将只从较大的单元中抽样")
        
        # 调整总样本量，使其精确等于目标样本量
        actual_sample_size = valid_units['allocation'].sum()
        difference = self.target_sample_size - actual_sample_size
        
        if difference != 0:
            # 计算每个单元的分配误差（小数部分）
            valid_units['remainder'] = valid_units['allocation_float'] - valid_units['allocation']
            # 按误差大小排序，优先分配给误差大的单元
            valid_units = valid_units.sort_values('remainder', ascending=False)
            
            # 调整分配，使总样本量等于目标值
            if difference > 0:
                # 需要增加样本量
                for idx in valid_units.index[:difference]:
                    valid_units.loc[idx, 'allocation'] += 1
            else:
                # 需要减少样本量（从误差最小的单元开始）
                for idx in valid_units.index[difference:]:
                    if valid_units.loc[idx, 'allocation'] > 1:
                        valid_units.loc[idx, 'allocation'] -= 1
                        difference += 1
                        if difference == 0:
                            break
                # 如果还有剩余，从分配数最大的单元减少
                if difference < 0:
                    for idx in valid_units.nlargest(-difference, 'allocation').index:
                        if valid_units.loc[idx, 'allocation'] > 1:
                            valid_units.loc[idx, 'allocation'] -= 1
                            difference += 1
                            if difference == 0:
                                break
        
        # 将结果合并回allocation_info
        allocation_info['allocation'] = 0
        allocation_info.loc[valid_units.index, 'allocation'] = valid_units['allocation']
        allocation_info['allocation'] = allocation_info['allocation'].astype(int)
        
        # 验证总样本量
        actual_sample_size = allocation_info['allocation'].sum()
        print(f"目标样本量: {self.target_sample_size:,}")
        print(f"实际分配样本量: {actual_sample_size:,}")
        if actual_sample_size != self.target_sample_size:
            print(f"警告：实际分配样本量({actual_sample_size})与目标样本量({self.target_sample_size})不一致")
        print(f"\n分配统计：")
        print(f"有效单元数: {(allocation_info['allocation'] > 0).sum()}")
        print(f"平均每单元样本量: {allocation_info[allocation_info['allocation'] > 0]['allocation'].mean():.1f}")
        
        self.allocation_info = allocation_info
        return allocation_info
    
    def allocate_sample_size_with_estimated_proportions(self, first_stage_sample):
        """
        基于第一阶段抽样估计的比例进行样本量分配（双阶段抽样）
        
        Parameters:
        -----------
        first_stage_sample : pd.DataFrame
            第一阶段抽取的样本，用于估计各层的比例
            
        Returns:
        --------
        allocation_info : pd.DataFrame
            基于估计比例的样本量分配信息
        """
        print("\n=== 样本量分配（基于第一阶段估计比例） ===")
        
        # 确保stratum_cluster_key已存在
        if 'stratum_cluster_key' not in first_stage_sample.columns:
            first_stage_sample['stratum_cluster_key'] = (
                first_stage_sample['time_stratum'] + '_' + 
                first_stage_sample['geo_cluster'] + '_' + 
                first_stage_sample['passenger_stratum']
            )
        
        # 从第一阶段样本估计各层的比例和大小
        # 使用第一阶段样本统计每个层的大小
        first_stage_stats = first_stage_sample.groupby('stratum_cluster_key').agg({
            'fare_amount': ['count', 'mean', 'std'],
        })
        first_stage_stats.columns = ['n_h_phase1', 'mean_fare_phase1', 'std_fare_phase1']
        first_stage_stats['std_fare_phase1'] = first_stage_stats['std_fare_phase1'].fillna(0)
        
        # 估计总体中各层的比例
        n_phase1_total = first_stage_stats['n_h_phase1'].sum()
        first_stage_stats['estimated_proportion'] = (
            first_stage_stats['n_h_phase1'] / n_phase1_total
        )
        
        # 如果有总体大小信息，可以估计各层的绝对大小
        # 否则，直接使用比例进行分配
        N_total = len(self.data)  # 总体大小
        first_stage_stats['N_h_estimated'] = (
            first_stage_stats['estimated_proportion'] * N_total
        ).astype(int)
        
        # 基于估计的比例分配第二阶段样本量
        # 使用第二阶段的目标样本量（总样本量 - 第一阶段样本量）
        first_stage_size = len(first_stage_sample)
        second_stage_target = self.target_sample_size - first_stage_size
        
        # 按估计的比例分配
        first_stage_stats['allocation'] = (
            first_stage_stats['estimated_proportion'] * second_stage_target
        ).astype(int)
        
        # 确保每个单元至少分配到1个样本（如果第一阶段有观测）
        first_stage_stats.loc[
            (first_stage_stats['allocation'] == 0) & (first_stage_stats['n_h_phase1'] > 0),
            'allocation'
        ] = 1
        first_stage_stats.loc[first_stage_stats['n_h_phase1'] == 0, 'allocation'] = 0
        
        # 确保 allocation 列始终是整数类型
        first_stage_stats['allocation'] = first_stage_stats['allocation'].astype(int)
        
        # 添加真实值用于对比
        true_allocation_info = self.data.groupby('stratum_cluster_key').agg({
            'fare_amount': 'count'
        })
        true_allocation_info.columns = ['N_h_true']
        
        allocation_info = first_stage_stats.join(true_allocation_info, how='left')
        allocation_info['N_h_true'] = allocation_info['N_h_true'].fillna(0).astype(int)
        allocation_info['true_proportion'] = allocation_info['N_h_true'] / allocation_info['N_h_true'].sum()
        
        # 计算估计误差
        allocation_info['proportion_error'] = (
            allocation_info['estimated_proportion'] - allocation_info['true_proportion']
        )
        
        actual_sample_size = allocation_info['allocation'].sum()
        total_with_phase1 = actual_sample_size + first_stage_size
        
        print(f"第一阶段样本量: {first_stage_size:,}")
        print(f"第二阶段目标样本量: {second_stage_target:,}")
        print(f"第二阶段实际分配样本量: {actual_sample_size:,}")
        print(f"总样本量: {total_with_phase1:,}")
        print(f"\n分配统计：")
        print(f"有效单元数: {(allocation_info['allocation'] > 0).sum()}")
        print(f"平均每单元样本量: {allocation_info[allocation_info['allocation'] > 0]['allocation'].mean():.1f}")
        
        # 显示比例估计的准确性
        mae_proportion = allocation_info['proportion_error'].abs().mean()
        print(f"\n比例估计准确性（MAE）: {mae_proportion:.6f}")
        
        self.allocation_info_estimated = allocation_info
        return allocation_info
    
    def two_stage_sampling(self, first_stage_size=1000):
        """
        执行双阶段抽样：
        第一阶段：简单随机抽样估计各层比例
        第二阶段：基于估计比例进行分层抽样
        
        Parameters:
        -----------
        first_stage_size : int
            第一阶段样本量
            
        Returns:
        --------
        final_sample : pd.DataFrame
            最终的样本（第一阶段 + 第二阶段）
        """
        print("\n" + "="*80)
        print("双阶段抽样：第一阶段估计比例，第二阶段基于估计比例抽样")
        print("="*80)
        
        # 确保stratum_cluster_key已创建
        if 'stratum_cluster_key' not in self.data.columns:
            self.data['stratum_cluster_key'] = (
                self.data['time_stratum'] + '_' + 
                self.data['geo_cluster'] + '_' + 
                self.data['passenger_stratum']
            )
        
        # 第一阶段：简单随机抽样
        print(f"\n【第一阶段】简单随机抽样（样本量: {first_stage_size:,}）")
        np.random.seed(42)  # 固定随机种子以便重现
        first_stage_sample = self.data.sample(n=min(first_stage_size, len(self.data)), random_state=42)
        first_stage_sample = first_stage_sample.copy()
        
        print(f"第一阶段抽样完成，样本量: {len(first_stage_sample):,}")
        
        # 基于第一阶段样本估计比例并分配第二阶段样本量
        allocation_info = self.allocate_sample_size_with_estimated_proportions(first_stage_sample)
        
        # 第二阶段：基于估计的比例进行分层抽样
        print(f"\n【第二阶段】基于估计比例的分层抽样")
        second_stage_samples = []
        
        # 计算需要抽样的单元数（用于进度条）
        valid_units = allocation_info[allocation_info['allocation'] > 0]
        total_units = len(valid_units)
        
        print(f"需要从 {total_units:,} 个分层聚类单元中抽样...")
        
        # 使用tqdm显示进度
        for stratum_key, row in tqdm(valid_units.iterrows(), 
                                     total=total_units,
                                     desc="第二阶段抽样",
                                     unit="单元"):
            n_sample = int(row['allocation'])
            if n_sample > 0:
                # 从剩余的总体中抽取（排除第一阶段已抽取的样本）
                unit_data = self.data[
                    (self.data['stratum_cluster_key'] == stratum_key) & 
                    (~self.data.index.isin(first_stage_sample.index))
                ].copy()
                
                if len(unit_data) > 0:
                    # 如果需要的样本数大于可用数据，则全部抽取
                    if n_sample >= len(unit_data):
                        sample_unit = unit_data
                    else:
                        # 系统抽样
                        unit_data = unit_data.sort_values('pickup_datetime')
                        k = len(unit_data) / n_sample
                        start = np.random.uniform(0, k)
                        indices = [int(start + i * k) for i in range(n_sample)]
                        indices = [idx for idx in indices if idx < len(unit_data)]
                        sample_unit = unit_data.iloc[indices].copy()
                    
                    second_stage_samples.append(sample_unit)
        
        second_stage_sample = pd.concat(second_stage_samples, ignore_index=True) if second_stage_samples else pd.DataFrame()
        
        # 合并第一阶段和第二阶段样本
        final_sample = pd.concat([first_stage_sample, second_stage_sample], ignore_index=True)
        
        print(f"第二阶段抽样完成，样本量: {len(second_stage_sample):,}")
        print(f"总样本量: {len(final_sample):,}")
        print(f"抽样比例: {len(final_sample) / len(self.data) * 100:.4f}%")
        
        self.final_sample_two_stage = final_sample
        self.first_stage_sample = first_stage_sample
        self.second_stage_sample = second_stage_sample
        
        return final_sample
    
    def systematic_sample_within_strata(self, stratum_key, n_sample):
        """
        在每个分层聚类单元内进行系统抽样
        
        Parameters:
        -----------
        stratum_key : str
            分层聚类单元标识
        n_sample : int
            该单元内要抽取的样本数
        """
        # 确保 n_sample 是整数类型
        n_sample = int(n_sample)
        
        unit_data = self.data[self.data['stratum_cluster_key'] == stratum_key].copy()
        N_unit = len(unit_data)
        
        if N_unit == 0 or n_sample == 0:
            return pd.DataFrame()
        
        if n_sample >= N_unit:
            return unit_data
        
        # 系统抽样：先按时间排序，然后等距抽样
        unit_data = unit_data.sort_values('pickup_datetime')
        
        # 计算抽样间隔
        k = N_unit / n_sample
        
        # 随机起点
        start = np.random.uniform(0, k)
        
        # 系统抽样索引
        indices = [int(start + i * k) for i in range(n_sample)]
        indices = [idx for idx in indices if idx < N_unit]  # 防止越界
        
        sampled = unit_data.iloc[indices].copy()
        return sampled
    
    def draw_sample(self):
        """执行多阶段抽样"""
        print("\n=== 执行多阶段复合抽样 ===")
        
        samples = []
        
        # 计算需要抽样的单元数（用于进度条）
        valid_units = self.allocation_info[self.allocation_info['allocation'] > 0]
        total_units = len(valid_units)
        
        print(f"需要从 {total_units:,} 个分层聚类单元中抽样...")
        
        # 使用tqdm显示进度
        for stratum_key, row in tqdm(valid_units.iterrows(), 
                                     total=total_units, 
                                     desc="抽样进度",
                                     unit="单元"):
            n_sample = int(row['allocation'])  # 确保是整数类型
            if n_sample > 0:
                sample_unit = self.systematic_sample_within_strata(stratum_key, n_sample)
                if len(sample_unit) > 0:
                    samples.append(sample_unit)
        
        self.final_sample = pd.concat(samples, ignore_index=True)
        
        print(f"\n抽样完成！")
        print(f"最终样本量: {len(self.final_sample):,}")
        print(f"抽样比例: {len(self.final_sample) / len(self.data) * 100:.4f}%")
        
        return self.final_sample
    
    def bootstrap_confidence_interval(self, n_bootstrap=1000, alpha=0.05, method='percentile'):
        """
        使用Bootstrap方法计算置信区间
        
        Parameters:
        -----------
        n_bootstrap : int
            Bootstrap重抽样次数
        alpha : float
            显著性水平，默认0.05对应95%置信区间
        method : str
            Bootstrap方法，可选：
            - 'percentile': 百分位数方法
            - 'bca': 偏差校正加速方法（需要更复杂的计算）
            
        Returns:
        --------
        dict : 包含置信区间下界、上界和方法名称
        """
        if self.final_sample is None:
            raise ValueError("请先执行抽样！")
        
        # 确保stratum_cluster_key已创建
        if 'stratum_cluster_key' not in self.final_sample.columns:
            self.final_sample['stratum_cluster_key'] = (
                self.final_sample['time_stratum'] + '_' + 
                self.final_sample['geo_cluster'] + '_' + 
                self.final_sample['passenger_stratum']
            )
        
        if self.allocation_info is None:
            raise ValueError("请先执行样本量分配！")
        
        N_total = len(self.data)
        bootstrap_means = []
        
        # 进行Bootstrap重抽样
        # 每次重抽样时，从每个层中按比例进行有放回抽样
        for _ in tqdm(range(n_bootstrap), desc="Bootstrap重抽样", unit="次"):
            bootstrap_sample_list = []
            
            # 对每个层进行有放回抽样
            for stratum_key, row in self.allocation_info.iterrows():
                if row['allocation'] > 0:
                    n_sample = int(row['allocation'])
                    if n_sample > 0:
                        # 从该层的样本中进行有放回抽样
                        stratum_sample = self.final_sample[
                            self.final_sample['stratum_cluster_key'] == stratum_key
                        ]
                        
                        if len(stratum_sample) > 0:
                            # 有放回抽样
                            bootstrap_stratum = stratum_sample.sample(
                                n=n_sample, 
                                replace=True, 
                                random_state=None  # 每次使用不同的随机状态
                            )
                            bootstrap_sample_list.append(bootstrap_stratum)
            
            if bootstrap_sample_list:
                bootstrap_sample = pd.concat(bootstrap_sample_list, ignore_index=True)
                
                # 计算分层估计的均值（与estimate_mean_fare中的方法一致）
                stratified_means = []
                for stratum_key, row in self.allocation_info.iterrows():
                    if row['allocation'] > 0:
                        stratum_data = self.data[self.data['stratum_cluster_key'] == stratum_key]
                        bootstrap_stratum = bootstrap_sample[
                            bootstrap_sample['stratum_cluster_key'] == stratum_key
                        ]
                        
                        if len(bootstrap_stratum) > 0:
                            N_h = len(stratum_data)
                            W_h = N_h / N_total  # 层权重
                            y_bar_h = bootstrap_stratum['fare_amount'].mean()
                            stratified_means.append(W_h * y_bar_h)
                
                if stratified_means:
                    bootstrap_mean = sum(stratified_means)
                    bootstrap_means.append(bootstrap_mean)
        
        if len(bootstrap_means) == 0:
            raise ValueError("Bootstrap重抽样失败，请检查数据！")
        
        bootstrap_means = np.array(bootstrap_means)
        
        # 根据方法计算置信区间
        if method == 'percentile':
            # 百分位数方法
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            ci_lower = np.percentile(bootstrap_means, lower_percentile)
            ci_upper = np.percentile(bootstrap_means, upper_percentile)
        elif method == 'bca':
            # 偏差校正加速方法（BCa）- 简化版本
            # 这里使用简化的BCa方法，更精确的BCa需要计算加速参数
            mean_estimate = bootstrap_means.mean()
            std_estimate = bootstrap_means.std()
            
            # 计算偏差校正
            z0 = np.sum(bootstrap_means < mean_estimate) / len(bootstrap_means)
            z0 = norm.ppf(z0) if z0 > 0 and z0 < 1 else 0
            
            # 简化的BCa（假设加速参数a=0）
            z_lower = norm.ppf(alpha / 2)
            z_upper = norm.ppf(1 - alpha / 2)
            
            # 调整后的百分位数
            adjusted_lower = norm.cdf(z0 + (z0 + z_lower))
            adjusted_upper = norm.cdf(z0 + (z0 + z_upper))
            
            ci_lower = np.percentile(bootstrap_means, adjusted_lower * 100)
            ci_upper = np.percentile(bootstrap_means, adjusted_upper * 100)
        else:
            raise ValueError(f"未知的Bootstrap方法: {method}")
        
        return {
            'lower': ci_lower,
            'upper': ci_upper,
            'method': method,
            'bootstrap_means': bootstrap_means
        }
    
    def estimate_mean_fare(self, enable_bootstrap=False, n_bootstrap=1000, bootstrap_alpha=0.05, verbose=True):
        """
        估计总体平均车费
        
        使用分层抽样的估计公式，但需要考虑多阶段结构
        
        Parameters:
        -----------
        enable_bootstrap : bool, default=False
            是否启用Bootstrap方法计算置信区间
        n_bootstrap : int, default=1000
            Bootstrap重抽样次数（仅在enable_bootstrap=True时使用）
        bootstrap_alpha : float, default=0.05
            Bootstrap显著性水平（仅在enable_bootstrap=True时使用）
        verbose : bool, default=True
            是否打印详细信息
        """
        if verbose:
            print("\n=== 估计总体平均车费 ===")
        
        if self.final_sample is None:
            raise ValueError("请先执行抽样！")
        
        # 总体大小
        N_total = len(self.data)
        
        # 方法1：简单估计（假设权重与抽样比例一致）
        sample_mean = self.final_sample['fare_amount'].mean()
        sample_std = self.final_sample['fare_amount'].std()
        sample_n = len(self.final_sample)
        
        # 简单随机抽样的标准误（作为基准）
        se_simple = sample_std / np.sqrt(sample_n)
        ci_95_lower = sample_mean - 1.96 * se_simple
        ci_95_upper = sample_mean + 1.96 * se_simple
        
        if verbose:
            print(f"\n【简单估计】（假设SRS）:")
            print(f"样本均值: ${sample_mean:.2f}")
            print(f"标准误: ${se_simple:.2f}")
            print(f"95%置信区间: [${ci_95_lower:.2f}, ${ci_95_upper:.2f}]")
        
        # 方法2：分层估计（考虑三层结构）
        # 使用分层抽样估计公式：\bar{y}_st = \sum_{h} W_h \bar{y}_h
        stratified_estimates = []
        
        for stratum_key, row in self.allocation_info.iterrows():
            if row['allocation'] > 0:
                stratum_data = self.data[self.data['stratum_cluster_key'] == stratum_key]
                sample_stratum = self.final_sample[
                    self.final_sample['stratum_cluster_key'] == stratum_key
                ]
                
                if len(sample_stratum) > 0:
                    N_h = len(stratum_data)
                    W_h = N_h / N_total  # 层权重
                    y_bar_h = sample_stratum['fare_amount'].mean()
                    
                    stratified_estimates.append({
                        'stratum': stratum_key,
                        'W_h': W_h,
                        'y_bar_h': y_bar_h,
                        'N_h': N_h,
                        'n_h': len(sample_stratum)
                    })
        
        est_df = pd.DataFrame(stratified_estimates)
        y_bar_st = (est_df['W_h'] * est_df['y_bar_h']).sum()
        
        # 分层估计的方差（简化版本）
        var_components = []
        for _, row in est_df.iterrows():
            N_h = row['N_h']
            n_h = row['n_h']
            if n_h > 1:
                # 获取该层的样本标准差
                stratum_sample = self.final_sample[
                    self.final_sample['stratum_cluster_key'] == row['stratum']
                ]
                s_h = stratum_sample['fare_amount'].std()
                W_h = row['W_h']
                f_h = n_h / N_h if N_h > 0 else 0
                var_h = (W_h ** 2) * (s_h ** 2) / n_h * (1 - f_h)
                var_components.append(var_h)
        
        var_st = sum(var_components)
        se_st = np.sqrt(var_st)
        
        # 理论置信区间（基于正态分布）
        ci_st_lower_theoretical = y_bar_st - 1.96 * se_st
        ci_st_upper_theoretical = y_bar_st + 1.96 * se_st
        
        if verbose:
            print(f"\n【分层估计】（考虑三层结构）:")
            print(f"总体均值估计: ${y_bar_st:.2f}")
            print(f"标准误: ${se_st:.2f}")
            print(f"95%理论置信区间: [${ci_st_lower_theoretical:.2f}, ${ci_st_upper_theoretical:.2f}]")
        
        # Bootstrap置信区间（可选）
        ci_bootstrap = None
        bootstrap_covers = None
        if enable_bootstrap:
            if verbose:
                print("\n正在计算Bootstrap置信区间...")
            ci_bootstrap = self.bootstrap_confidence_interval(n_bootstrap=n_bootstrap, alpha=bootstrap_alpha)
            
            if verbose:
                print(f"95%Bootstrap置信区间: [${ci_bootstrap['lower']:.2f}, ${ci_bootstrap['upper']:.2f}]")
                print(f"Bootstrap方法: {ci_bootstrap['method']}")
        
        # 总体均值（真实值，用于比较）
        true_mean = self.data['fare_amount'].mean()
        if verbose:
            print(f"\n【真实总体均值】: ${true_mean:.2f}")
            print(f"估计偏差: ${y_bar_st - true_mean:.2f}")
            print(f"相对误差: {abs(y_bar_st - true_mean) / true_mean * 100:.2f}%")
        
        # 置信区间覆盖情况
        theoretical_covers = (ci_st_lower_theoretical <= true_mean <= ci_st_upper_theoretical)
        if enable_bootstrap and ci_bootstrap is not None:
            bootstrap_covers = (ci_bootstrap['lower'] <= true_mean <= ci_bootstrap['upper'])
            if verbose:
                print(f"\n置信区间覆盖情况（真实均值是否在区间内）:")
                print(f"  理论CI: {'✓ 覆盖' if theoretical_covers else '✗ 未覆盖'}")
                print(f"  Bootstrap CI: {'✓ 覆盖' if bootstrap_covers else '✗ 未覆盖'}")
        else:
            if verbose:
                print(f"\n置信区间覆盖情况（真实均值是否在区间内）:")
                print(f"  理论CI: {'✓ 覆盖' if theoretical_covers else '✗ 未覆盖'}")
        
        # 设计效应（Design Effect）
        deff = (se_st / se_simple) ** 2
        if verbose:
            print(f"\n设计效应 (Deff): {deff:.4f}")
            if deff < 1:
                print("✓ 分层抽样比简单随机抽样更高效！")
            elif deff > 1:
                print("⚠ 分层抽样的效率略低于简单随机抽样（可能是由于聚类效应）")
        
        result = {
            'simple_mean': sample_mean,
            'stratified_mean': y_bar_st,
            'true_mean': true_mean,
            'se_simple': se_simple,
            'se_stratified': se_st,
            'ci_95_lower_theoretical': ci_st_lower_theoretical,
            'ci_95_upper_theoretical': ci_st_upper_theoretical,
            'deff': deff
        }
        
        # 仅在启用Bootstrap时添加相关字段
        if enable_bootstrap and ci_bootstrap is not None:
            result['ci_95_lower_bootstrap'] = ci_bootstrap['lower']
            result['ci_95_upper_bootstrap'] = ci_bootstrap['upper']
        else:
            result['ci_95_lower_bootstrap'] = None
            result['ci_95_upper_bootstrap'] = None
        
        return result
    
    def compare_with_srs(self):
        """与简单随机抽样进行对比"""
        print("\n=== 与简单随机抽样对比 ===")
        
        # 简单随机抽样
        np.random.seed(42)
        srs_sample = self.data.sample(n=len(self.final_sample), random_state=42)
        srs_mean = srs_sample['fare_amount'].mean()
        srs_se = srs_sample['fare_amount'].std() / np.sqrt(len(srs_sample))
        
        # 我们的分层聚类抽样
        stratified_mean = self.estimate_mean_fare()['stratified_mean']
        
        true_mean = self.data['fare_amount'].mean()
        
        print(f"\n对比结果：")
        print(f"{'方法':<30} {'均值估计':<15} {'标准误':<15} {'偏差':<15}")
        print("-" * 75)
        print(f"{'简单随机抽样 (SRS)':<30} ${srs_mean:<14.2f} ${srs_se:<14.2f} ${abs(srs_mean-true_mean):<14.2f}")
        print(f"{'多阶段分层聚类抽样':<30} ${stratified_mean:<14.2f} ${self.estimate_mean_fare()['se_stratified']:<14.2f} ${abs(stratified_mean-true_mean):<14.2f}")
        print(f"{'真实总体均值':<30} ${true_mean:<14.2f} {'-':<15} {'0.00':<15}")
    
    def compare_allocation_methods(self, first_stage_size=1000):
        """
        对比两种样本量分配方法：
        1. 已知真实比例（使用总体真实数据）
        2. 基于第一阶段抽样估计的比例（双阶段抽样）
        
        Parameters:
        -----------
        first_stage_size : int
            双阶段抽样中第一阶段的样本量
        """
        print("\n" + "="*80)
        print("对比分析：已知真实比例 vs 第一阶段估计比例")
        print("="*80)
        
        # 方法1：已知真实比例（当前方法）
        print("\n【方法1】已知真实比例的样本量分配")
        print("-" * 80)
        if not hasattr(self, 'allocation_info') or self.allocation_info is None:
            self.allocate_sample_size()
        
        allocation_true = self.allocation_info.copy()
        
        # 执行方法1的抽样
        if self.final_sample is None:
            self.draw_sample()
        
        # 方法2：双阶段抽样（估计比例）
        print("\n【方法2】基于第一阶段估计比例的样本量分配（双阶段抽样）")
        print("-" * 80)
        two_stage_sample = self.two_stage_sampling(first_stage_size=first_stage_size)
        allocation_estimated = self.allocation_info_estimated.copy()
        
        # 对比两种方法的估计结果
        print("\n" + "="*80)
        print("估计结果对比")
        print("="*80)
        
        # 方法1的估计
        true_mean = self.data['fare_amount'].mean()
        
        # 计算两种方法的样本均值（简单估计）
        method1_mean = self.final_sample['fare_amount'].mean()
        method1_se = self.final_sample['fare_amount'].std() / np.sqrt(len(self.final_sample))
        method1_bias = abs(method1_mean - true_mean)
        method1_relative_error = method1_bias / true_mean * 100
        
        method2_mean = two_stage_sample['fare_amount'].mean()
        method2_se = two_stage_sample['fare_amount'].std() / np.sqrt(len(two_stage_sample))
        method2_bias = abs(method2_mean - true_mean)
        method2_relative_error = method2_bias / true_mean * 100
        
        print(f"\n{'指标':<25} {'方法1（已知比例）':<20} {'方法2（估计比例）':<20} {'差异':<15}")
        print("-" * 80)
        print(f"{'样本量':<25} {len(self.final_sample):<20,} {len(two_stage_sample):<20,} {len(two_stage_sample) - len(self.final_sample):<15,}")
        print(f"{'均值估计':<25} ${method1_mean:<19.2f} ${method2_mean:<19.2f} ${method2_mean - method1_mean:<14.2f}")
        print(f"{'标准误':<25} ${method1_se:<19.2f} ${method2_se:<19.2f} ${method2_se - method1_se:<14.2f}")
        print(f"{'绝对偏差':<25} ${method1_bias:<19.2f} ${method2_bias:<19.2f} ${method2_bias - method1_bias:<14.2f}")
        print(f"{'相对误差 (%)':<25} {method1_relative_error:<19.2f} {method2_relative_error:<19.2f} {method2_relative_error - method1_relative_error:<14.2f}")
        print(f"{'真实总体均值':<25} ${true_mean:<19.2f} {'-':<20} {'-':<15}")
        
        # 对比样本量分配的准确性
        print("\n" + "="*80)
        print("样本量分配准确性对比")
        print("="*80)
        
        # 合并两种分配方法的信息
        comparison_df = allocation_true[['N_h', 'allocation']].rename(
            columns={'N_h': 'N_h_true', 'allocation': 'allocation_true'}
        )
        
        if 'N_h_estimated' in allocation_estimated.columns:
            comparison_df = comparison_df.join(
                allocation_estimated[['N_h_estimated', 'allocation', 'estimated_proportion', 'true_proportion']],
                how='outer'
            )
            comparison_df = comparison_df.rename(columns={'allocation': 'allocation_estimated'})
            
            # 填充缺失值
            comparison_df = comparison_df.fillna(0)
            
            # 计算分配误差
            comparison_df['allocation_error'] = (
                comparison_df['allocation_estimated'] - comparison_df['allocation_true']
            )
            comparison_df['allocation_relative_error'] = np.where(
                comparison_df['allocation_true'] > 0,
                comparison_df['allocation_error'] / comparison_df['allocation_true'] * 100,
                0
            )
            
            print(f"\n有效单元对比：")
            print(f"  方法1（已知比例）有效单元数: {(comparison_df['allocation_true'] > 0).sum()}")
            print(f"  方法2（估计比例）有效单元数: {(comparison_df['allocation_estimated'] > 0).sum()}")
            
            # 只对比两者都有效的单元
            both_valid = comparison_df[
                (comparison_df['allocation_true'] > 0) & (comparison_df['allocation_estimated'] > 0)
            ]
            
            if len(both_valid) > 0:
                mae_allocation = both_valid['allocation_error'].abs().mean()
                mape_allocation = both_valid['allocation_relative_error'].abs().mean()
                
                print(f"\n分配误差统计（仅对比共同有效单元，共{len(both_valid)}个）：")
                print(f"  平均绝对误差 (MAE): {mae_allocation:.2f} 个样本")
                print(f"  平均绝对百分比误差 (MAPE): {mape_allocation:.2f}%")
        
        # 总结
        print("\n" + "="*80)
        print("结论")
        print("="*80)
        if method2_relative_error <= method1_relative_error * 1.1:  # 允许10%的容忍度
            print("✓ 双阶段抽样（估计比例）的估计精度与方法1（已知比例）相近")
            print("  说明第一阶段抽样能够较好地估计各层比例")
        else:
            print("⚠ 双阶段抽样（估计比例）的估计精度略低于方法1（已知比例）")
            print("  可能需要增加第一阶段样本量以提高比例估计的准确性")
        
        print(f"\n方法2的优势：")
        print("  - 更符合实际调查场景（通常不知道总体真实比例）")
        print("  - 可以通过调整第一阶段样本量平衡成本与精度")
        print("  - 适合探索性研究或动态调整抽样策略")
        
        return {
            'method1': {
                'sample_size': len(self.final_sample),
                'mean': method1_mean,
                'se': method1_se,
                'bias': method1_bias,
                'relative_error': method1_relative_error
            },
            'method2': {
                'sample_size': len(two_stage_sample),
                'mean': method2_mean,
                'se': method2_se,
                'bias': method2_bias,
                'relative_error': method2_relative_error
            },
            'true_mean': true_mean
        }
    
    def three_stage_sampling_by_features(self, first_stage_size=1000):
        """
        方法3：双阶段抽样，基于单独特征比例估计的分层抽样
        
        第一阶段：简单随机抽样，估计各个分层特征（time_stratum, geo_cluster, passenger_stratum）
                  各自的边际分布比例
        第二阶段：基于估计的特征比例，进行分层抽样
        
        Parameters:
        -----------
        first_stage_size : int
            第一阶段样本量
            
        Returns:
        --------
        final_sample : pd.DataFrame
            最终的样本（第一阶段 + 第二阶段）
        """
        print("\n" + "="*80)
        print("方法3：双阶段抽样（基于单独特征比例估计的分层抽样）")
        print("="*80)
        
        # 确保stratum_cluster_key已创建
        if 'stratum_cluster_key' not in self.data.columns:
            self.data['stratum_cluster_key'] = (
                self.data['time_stratum'] + '_' + 
                self.data['geo_cluster'] + '_' + 
                self.data['passenger_stratum']
            )
        
        # 第一阶段：简单随机抽样，估计各特征的边际分布
        print(f"\n【第一阶段】简单随机抽样（样本量: {first_stage_size:,}）")
        np.random.seed(43)  # 使用不同的随机种子以区分方法
        first_stage_sample = self.data.sample(n=min(first_stage_size, len(self.data)), random_state=43)
        first_stage_sample = first_stage_sample.copy()
        
        print(f"第一阶段抽样完成，样本量: {len(first_stage_sample):,}")
        
        # 估计各特征的边际分布比例
        print("\n【估计各特征的边际分布比例】")
        
        # 估计时间分层比例
        time_proportions = first_stage_sample['time_stratum'].value_counts(normalize=True).sort_index()
        print(f"\n时间分层估计比例（共{len(time_proportions)}层）：")
        for stratum, prop in time_proportions.items():
            print(f"  {stratum}: {prop:.4f} ({prop*100:.2f}%)")
        
        # 估计地理聚类比例
        geo_proportions = first_stage_sample['geo_cluster'].value_counts(normalize=True).sort_index()
        print(f"\n地理聚类估计比例（共{len(geo_proportions)}个聚类）：")
        # 只显示前10个和后10个
        if len(geo_proportions) > 20:
            for cluster, prop in list(geo_proportions.items())[:10]:
                print(f"  {cluster}: {prop:.4f} ({prop*100:.2f}%)")
            print(f"  ... (省略 {len(geo_proportions)-20} 个聚类) ...")
            for cluster, prop in list(geo_proportions.items())[-10:]:
                print(f"  {cluster}: {prop:.4f} ({prop*100:.2f}%)")
        else:
            for cluster, prop in geo_proportions.items():
                print(f"  {cluster}: {prop:.4f} ({prop*100:.2f}%)")
        
        # 估计乘客分层比例
        passenger_proportions = first_stage_sample['passenger_stratum'].value_counts(normalize=True).sort_index()
        print(f"\n乘客分层估计比例（共{len(passenger_proportions)}层）：")
        for stratum, prop in passenger_proportions.items():
            print(f"  {stratum}: {prop:.4f} ({prop*100:.2f}%)")
        
        # 基于估计的边际分布比例，计算各层的期望比例（假设特征独立）
        # 实际中，我们可以按每个特征的边际比例来分配样本量
        # 这里我们使用各特征边际比例的乘积来估计组合比例
        
        print("\n【第二阶段】基于估计特征比例的分层抽样")
        
        # 计算第二阶段目标样本量
        second_stage_target = self.target_sample_size - len(first_stage_sample)
        
        # 方法：基于三个特征的估计比例，对每个组合单元分配样本量
        # 组合比例 = 时间比例 × 地理比例 × 乘客比例（假设特征独立）
        # 从第一阶段样本中获取每个stratum_key对应的三个特征值
        stratum_features = first_stage_sample.groupby('stratum_cluster_key').agg({
            'time_stratum': 'first',
            'geo_cluster': 'first',
            'passenger_stratum': 'first'
        })
        
        allocation_info = pd.DataFrame(index=stratum_features.index)
        allocation_info['estimated_proportion'] = 0.0
        
        for stratum_key, row in stratum_features.iterrows():
            time_stratum = row['time_stratum']
            geo_cluster = row['geo_cluster']
            passenger_stratum = row['passenger_stratum']
            
            # 获取各特征的估计比例
            time_prop = time_proportions.get(time_stratum, 0.0)
            geo_prop = geo_proportions.get(geo_cluster, 0.0)
            passenger_prop = passenger_proportions.get(passenger_stratum, 0.0)
            
            # 组合比例（假设特征独立）
            allocation_info.loc[stratum_key, 'estimated_proportion'] = (
                time_prop * geo_prop * passenger_prop
            )
        
        # 同时考虑总体中可能存在的其他组合（在第一阶段没有出现的）
        # 为了完整，我们也计算总体中所有可能的组合
        all_strata = self.data.groupby('stratum_cluster_key').agg({
            'time_stratum': 'first',
            'geo_cluster': 'first',
            'passenger_stratum': 'first'
        })
        
        # 添加第一阶段没有出现的组合
        new_strata = all_strata[~all_strata.index.isin(allocation_info.index)]
        if len(new_strata) > 0:
            for stratum_key, row in new_strata.iterrows():
                time_stratum = row['time_stratum']
                geo_cluster = row['geo_cluster']
                passenger_stratum = row['passenger_stratum']
                
                time_prop = time_proportions.get(time_stratum, 0.0)
                geo_prop = geo_proportions.get(geo_cluster, 0.0)
                passenger_prop = passenger_proportions.get(passenger_stratum, 0.0)
                
                allocation_info.loc[stratum_key, 'estimated_proportion'] = (
                    time_prop * geo_prop * passenger_prop
                )
        
        # 归一化（因为可能有些组合在第一阶段没有出现）
        total_prop = allocation_info['estimated_proportion'].sum()
        if total_prop > 0:
            allocation_info['estimated_proportion'] = allocation_info['estimated_proportion'] / total_prop
        
        # 基于估计比例分配样本量
        allocation_info['allocation'] = (
            allocation_info['estimated_proportion'] * second_stage_target
        ).astype(int)
        
        # 确保每个在第一阶段出现的单元至少分配到1个样本
        first_stage_strata = set(first_stage_sample['stratum_cluster_key'].unique())
        for stratum_key in first_stage_strata:
            if stratum_key in allocation_info.index and allocation_info.loc[stratum_key, 'allocation'] == 0:
                allocation_info.loc[stratum_key, 'allocation'] = 1
        
        # 如果还有剩余样本量，按比例分配
        allocated_total = allocation_info['allocation'].sum()
        remaining = second_stage_target - allocated_total
        if remaining > 0:
            # 按估计比例分配剩余样本量
            valid_allocation = allocation_info[allocation_info['estimated_proportion'] > 0].copy()
            if len(valid_allocation) > 0:
                additional_allocation = (valid_allocation['estimated_proportion'] * remaining).astype(int)
                allocation_info.loc[valid_allocation.index, 'allocation'] += additional_allocation
                # 分配最后的余数
                final_remaining = second_stage_target - allocation_info['allocation'].sum()
                if final_remaining > 0 and len(valid_allocation) > 0:
                    # 分配给估计比例最大的前几个单元
                    top_strata = valid_allocation.nlargest(int(final_remaining), 'estimated_proportion').index
                    allocation_info.loc[top_strata, 'allocation'] += 1
        
        # 确保 allocation 列始终是整数类型
        allocation_info['allocation'] = allocation_info['allocation'].astype(int)
        
        # 添加真实值用于对比
        true_allocation_info = self.data.groupby('stratum_cluster_key').agg({
            'fare_amount': 'count'
        })
        true_allocation_info.columns = ['N_h_true']
        true_allocation_info['true_proportion'] = true_allocation_info['N_h_true'] / true_allocation_info['N_h_true'].sum()
        
        allocation_info = allocation_info.join(true_allocation_info, how='left')
        allocation_info['N_h_true'] = allocation_info['N_h_true'].fillna(0).astype(int)
        allocation_info['true_proportion'] = allocation_info['true_proportion'].fillna(0)
        
        # 计算估计误差
        allocation_info['proportion_error'] = (
            allocation_info['estimated_proportion'] - allocation_info['true_proportion']
        )
        
        actual_sample_size = allocation_info['allocation'].sum()
        total_with_phase1 = actual_sample_size + len(first_stage_sample)
        
        print(f"第一阶段样本量: {len(first_stage_sample):,}")
        print(f"第二阶段目标样本量: {second_stage_target:,}")
        print(f"第二阶段实际分配样本量: {actual_sample_size:,}")
        print(f"总样本量: {total_with_phase1:,}")
        print(f"\n分配统计：")
        print(f"有效单元数: {(allocation_info['allocation'] > 0).sum()}")
        if (allocation_info['allocation'] > 0).sum() > 0:
            print(f"平均每单元样本量: {allocation_info[allocation_info['allocation'] > 0]['allocation'].mean():.1f}")
        
        # 显示比例估计的准确性
        valid_errors = allocation_info[
            (allocation_info['estimated_proportion'] > 0) | (allocation_info['true_proportion'] > 0)
        ]
        if len(valid_errors) > 0:
            mae_proportion = valid_errors['proportion_error'].abs().mean()
            print(f"\n比例估计准确性（MAE）: {mae_proportion:.6f}")
        
        self.allocation_info_method3 = allocation_info
        self.time_proportions_est = time_proportions
        self.geo_proportions_est = geo_proportions
        self.passenger_proportions_est = passenger_proportions
        
        # 第二阶段：基于估计的比例进行分层抽样
        print(f"\n从 {(allocation_info['allocation'] > 0).sum():,} 个分层聚类单元中抽样...")
        second_stage_samples = []
        
        valid_units = allocation_info[allocation_info['allocation'] > 0]
        
        # 使用tqdm显示进度
        for stratum_key, row in tqdm(valid_units.iterrows(), 
                                     total=len(valid_units),
                                     desc="第二阶段抽样",
                                     unit="单元"):
            n_sample = int(row['allocation'])
            if n_sample > 0:
                # 从剩余的总体中抽取（排除第一阶段已抽取的样本）
                unit_data = self.data[
                    (self.data['stratum_cluster_key'] == stratum_key) & 
                    (~self.data.index.isin(first_stage_sample.index))
                ].copy()
                
                if len(unit_data) > 0:
                    # 如果需要的样本数大于可用数据，则全部抽取
                    if n_sample >= len(unit_data):
                        sample_unit = unit_data
                    else:
                        # 系统抽样
                        unit_data = unit_data.sort_values('pickup_datetime')
                        k = len(unit_data) / n_sample
                        start = np.random.uniform(0, k)
                        indices = [int(start + i * k) for i in range(n_sample)]
                        indices = [idx for idx in indices if idx < len(unit_data)]
                        sample_unit = unit_data.iloc[indices].copy()
                    
                    second_stage_samples.append(sample_unit)
        
        second_stage_sample = pd.concat(second_stage_samples, ignore_index=True) if second_stage_samples else pd.DataFrame()
        
        # 合并第一阶段和第二阶段样本
        final_sample = pd.concat([first_stage_sample, second_stage_sample], ignore_index=True)
        
        print(f"第二阶段抽样完成，样本量: {len(second_stage_sample):,}")
        print(f"总样本量: {len(final_sample):,}")
        print(f"抽样比例: {len(final_sample) / len(self.data) * 100:.4f}%")
        
        self.final_sample_method3 = final_sample
        self.first_stage_sample_method3 = first_stage_sample
        self.second_stage_sample_method3 = second_stage_sample
        
        return final_sample
    
    def generate_report(self, enable_bootstrap=False, n_bootstrap=1000, bootstrap_alpha=0.05):
        """
        生成抽样报告
        
        Parameters:
        -----------
        enable_bootstrap : bool, default=False
            是否启用Bootstrap方法计算置信区间
        n_bootstrap : int, default=1000
            Bootstrap重抽样次数（仅在enable_bootstrap=True时使用）
        bootstrap_alpha : float, default=0.05
            Bootstrap显著性水平（仅在enable_bootstrap=True时使用）
        """
        print("\n" + "="*80)
        print("多阶段复合抽样设计报告")
        print("="*80)
        print("\n【抽样设计】")
        print("1. 第一层：按时间区间分层（年份-季度）")
        print("2. 第二层：按地理位置聚类（网格划分）")
        print("3. 第三层：按乘客人数分层（1人、2人、3-4人、5人+）")
        print("4. 层内抽样：系统抽样（按时间排序后等距抽样）")
        print("\n【抽样优势】")
        print("✓ 时间分层：捕获时间趋势，提高估计精度")
        print("✓ 地理聚类：反映空间异质性，降低调查成本")
        print("✓ 乘客分层：考虑组间差异，减少估计方差")
        print("✓ 系统抽样：保证样本时间分布的均匀性")
        print("\n【统计意义】")
        print("这是一个三层嵌套的混合抽样设计，结合了：")
        print("- 分层抽样（Stratified Sampling）的方差降低优势")
        print("- 聚类抽样（Cluster Sampling）的成本效益优势")
        print("- 系统抽样（Systematic Sampling）的操作便利性")
        
        results = self.estimate_mean_fare(enable_bootstrap=enable_bootstrap, 
                                         n_bootstrap=n_bootstrap, 
                                         bootstrap_alpha=bootstrap_alpha)
        self.compare_with_srs()
        
        print("\n" + "="*80)
    
    def compare_three_methods(self, first_stage_size=1000):
        """
        对比三种抽样方法的估计结果
        
        Parameters:
        -----------
        first_stage_size : int
            方法2和方法3中第一阶段的样本量
            
        Returns:
        --------
        dict : 包含三种方法估计结果的字典
        """
        print("\n" + "="*80)
        print("三种抽样方法对比")
        print("="*80)
        
        true_mean = self.data['fare_amount'].mean()
        true_std = self.data['fare_amount'].std()
        
        # 方法1的结果
        method1_mean = self.final_sample['fare_amount'].mean()
        method1_std = self.final_sample['fare_amount'].std()
        method1_se = method1_std / np.sqrt(len(self.final_sample))
        method1_bias = abs(method1_mean - true_mean)
        method1_relative_error = method1_bias / true_mean * 100
        
        # 方法2的结果
        method2_mean = self.final_sample_two_stage['fare_amount'].mean()
        method2_std = self.final_sample_two_stage['fare_amount'].std()
        method2_se = method2_std / np.sqrt(len(self.final_sample_two_stage))
        method2_bias = abs(method2_mean - true_mean)
        method2_relative_error = method2_bias / true_mean * 100
        
        # 方法3的结果
        method3_mean = self.final_sample_method3['fare_amount'].mean()
        method3_std = self.final_sample_method3['fare_amount'].std()
        method3_se = method3_std / np.sqrt(len(self.final_sample_method3))
        method3_bias = abs(method3_mean - true_mean)
        method3_relative_error = method3_bias / true_mean * 100
        
        # 输出对比表
        print(f"\n{'指标':<25} {'方法1（已知比例）':<25} {'方法2（估计组合比例）':<30} {'方法3（估计特征比例）':<30}")
        print("-" * 115)
        print(f"{'样本量':<25} {len(self.final_sample):<25,} {len(self.final_sample_two_stage):<30,} {len(self.final_sample_method3):<30,}")
        print(f"{'均值估计':<25} ${method1_mean:<24.2f} ${method2_mean:<29.2f} ${method3_mean:<29.2f}")
        print(f"{'标准误':<25} ${method1_se:<24.2f} ${method2_se:<29.2f} ${method3_se:<29.2f}")
        print(f"{'绝对偏差':<25} ${method1_bias:<24.2f} ${method2_bias:<29.2f} ${method3_bias:<29.2f}")
        print(f"{'相对误差 (%)':<25} {method1_relative_error:<24.2f} {method2_relative_error:<29.2f} {method3_relative_error:<29.2f}")
        print(f"{'95% CI下限':<25} ${method1_mean-1.96*method1_se:<24.2f} ${method2_mean-1.96*method2_se:<29.2f} ${method3_mean-1.96*method3_se:<29.2f}")
        print(f"{'95% CI上限':<25} ${method1_mean+1.96*method1_se:<24.2f} ${method2_mean+1.96*method2_se:<29.2f} ${method3_mean+1.96*method3_se:<29.2f}")
        print(f"{'真实总体均值':<25} ${true_mean:<24.2f} {'-':<30} {'-':<30}")
        
        # 方法说明
        print("\n" + "="*80)
        print("方法说明")
        print("="*80)
        print("方法1（已知真实比例）：")
        print("  - 假设已知各层（时间×地理×乘客）的真实大小和比例")
        print("  - 直接基于真实比例进行样本量分配")
        print("  - 适合：理论分析、对比基准")
        print("  - 局限：实际调查中通常不知道总体真实比例")
        
        print("\n方法2（估计组合比例）：")
        print("  - 第一阶段：简单随机抽样，估计各层组合（时间×地理×乘客）的比例")
        print("  - 第二阶段：基于估计的组合比例进行分层抽样")
        print("  - 适合：总体结构未知，需要估计组合比例的场景")
        print("  - 优势：直接估计最终使用的组合层比例，理论上更精确")
        
        print("\n方法3（估计特征比例）：")
        print("  - 第一阶段：简单随机抽样，估计各特征（时间、地理、乘客）的边际分布")
        print("  - 第二阶段：基于特征边际比例的乘积（假设特征独立）进行分层抽样")
        print("  - 适合：特征独立或近似独立的场景")
        print("  - 优势：只需要估计少量参数（特征数），计算简单")
        print("  - 局限：假设特征独立可能不符合实际情况")
        
        # 找出最佳方法
        print("\n" + "="*80)
        print("结论")
        print("="*80)
        errors = {
            '方法1': method1_relative_error,
            '方法2': method2_relative_error,
            '方法3': method3_relative_error
        }
        best_method = min(errors, key=errors.get)
        print(f"✓ 估计精度最高：{best_method}（相对误差 {errors[best_method]:.2f}%）")
        
        if method2_relative_error <= method1_relative_error * 1.05:
            print("✓ 方法2（估计组合比例）与方法1（已知比例）精度相近")
            print("  说明第一阶段抽样能够较好地估计组合层比例")
        
        if method3_relative_error <= method1_relative_error * 1.1:
            print("✓ 方法3（估计特征比例）估计精度可接受")
            if method3_relative_error > method2_relative_error:
                print("  ⚠ 但如果特征之间存在相关性，方法2会更准确")
        
        return {
            'method1': {
                'sample_size': len(self.final_sample),
                'mean': method1_mean,
                'se': method1_se,
                'bias': method1_bias,
                'relative_error': method1_relative_error,
                'ci_lower': method1_mean - 1.96 * method1_se,
                'ci_upper': method1_mean + 1.96 * method1_se
            },
            'method2': {
                'sample_size': len(self.final_sample_two_stage),
                'mean': method2_mean,
                'se': method2_se,
                'bias': method2_bias,
                'relative_error': method2_relative_error,
                'ci_lower': method2_mean - 1.96 * method2_se,
                'ci_upper': method2_mean + 1.96 * method2_se
            },
            'method3': {
                'sample_size': len(self.final_sample_method3),
                'mean': method3_mean,
                'se': method3_se,
                'bias': method3_bias,
                'relative_error': method3_relative_error,
                'ci_lower': method3_mean - 1.96 * method3_se,
                'ci_upper': method3_mean + 1.96 * method3_se
            },
            'true_mean': true_mean
        }


def main():
    """主函数：执行完整的抽样流程"""
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='多阶段复合抽样设计：估计纽约出租车平均费用',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 默认运行（Bootstrap关闭）
  python multi_stage_sampling.py
  
  # 启用Bootstrap（使用默认参数）
  python multi_stage_sampling.py --enable-bootstrap
  
  # 启用Bootstrap并自定义参数
  python multi_stage_sampling.py --enable-bootstrap --n-bootstrap 2000 --bootstrap-alpha 0.01
  
  # 自定义数据路径和样本量
  python multi_stage_sampling.py --data-path train.csv --sample-size 10000 --enable-bootstrap
        """
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        default='train.csv',
        help='数据文件路径（默认: train.csv）'
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        default=5000,
        help='总样本量（默认: 5000）'
    )
    
    parser.add_argument(
        '--nrows',
        type=int,
        default=500000,
        help='加载数据的行数（默认: 500000）'
    )
    
    parser.add_argument(
        '--enable-bootstrap',
        action='store_true',
        help='启用Bootstrap方法计算置信区间（默认: False）'
    )
    
    parser.add_argument(
        '--n-bootstrap',
        type=int,
        default=1000,
        help='Bootstrap重抽样次数（仅在启用Bootstrap时有效，默认: 1000）'
    )
    
    parser.add_argument(
        '--bootstrap-alpha',
        type=float,
        default=0.05,
        help='Bootstrap显著性水平（仅在启用Bootstrap时有效，默认: 0.05）'
    )
    
    parser.add_argument(
        '--cluster-method',
        type=str,
        default='distance',
        choices=['grid', 'distance', 'route'],
        help='地理聚类方法：grid（网格）, distance（距离聚类）, route（路线相似性聚类，默认: distance）'
    )
    
    parser.add_argument(
        '--n-clusters',
        type=int,
        default=50,
        help='聚类数量（仅用于distance和route方法，默认: 50）'
    )
    
    args = parser.parse_args()
    
    # 初始化抽样设计
    sampler = MultiStageSampling(
        data_path=args.data_path,
        sample_size=args.sample_size
    )
    
    # 加载数据（可以先用部分数据测试）
    sampler.load_data(nrows=args.nrows)
    
    # 执行各层设计
    sampler.create_time_strata()
    sampler.create_geographic_clusters(method=args.cluster_method, n_clusters=args.n_clusters)
    sampler.create_passenger_strata()
    
    # ===== 方法1：已知真实比例的方法 =====
    print("\n" + "="*80)
    print("方法1：已知真实比例的样本量分配")
    print("="*80)
    
    # 分配样本量（使用真实总体数据）
    sampler.allocate_sample_size()
    
    # 执行抽样
    sampler.draw_sample()
    
    # 估计和报告（传递命令行参数）
    sampler.generate_report(
        enable_bootstrap=args.enable_bootstrap,
        n_bootstrap=args.n_bootstrap,
        bootstrap_alpha=args.bootstrap_alpha
    )
    
    # 保存样本
    sampler.final_sample.to_csv('sampled_data_method1.csv', index=False)
    print("\n方法1样本已保存至: sampled_data_method1.csv")
    
    # ===== 方法2：双阶段抽样（第一阶段估计比例）=====
    print("\n" + "="*80)
    print("方法2：双阶段抽样（第一阶段估计比例，第二阶段基于估计比例抽样）")
    print("="*80)
    
    # 执行双阶段抽样对比
    comparison_results = sampler.compare_allocation_methods(first_stage_size=1000)
    
    # 保存双阶段抽样样本
    sampler.final_sample_two_stage.to_csv('sampled_data_method2.csv', index=False)
    print("\n方法2样本已保存至: sampled_data_method2.csv")
    
    # ===== 方法3：双阶段抽样（基于单独特征比例估计）=====
    print("\n" + "="*80)
    print("方法3：双阶段抽样（基于单独特征比例估计的分层抽样）")
    print("="*80)
    
    # 执行方法3：基于单独特征比例估计的双阶段抽样
    sampler.three_stage_sampling_by_features(first_stage_size=1000)
    
    # 保存方法3的样本
    sampler.final_sample_method3.to_csv('sampled_data_method3.csv', index=False)
    print("\n方法3样本已保存至: sampled_data_method3.csv")
    
    # ===== 三种方法对比 =====
    comparison_results = sampler.compare_three_methods(first_stage_size=1000)
    
    print("\n" + "="*80)
    print("三种方法执行完成！")
    print("="*80)
    print("\n已保存的文件：")
    print("  - sampled_data_method1.csv: 方法1（已知真实比例）的样本")
    print("  - sampled_data_method2.csv: 方法2（估计组合比例）的样本")
    print("  - sampled_data_method3.csv: 方法3（估计特征比例）的样本")
    
    return sampler


if __name__ == "__main__":
    sampler = main()

