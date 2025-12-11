"""
多阶段复合抽样设计：估计纽约出租车平均费用
设计要素：
1. 按时间区间分层（Stratified Sampling by Time Period）
2. 按地理位置聚类（Cluster Sampling by Geographic Region）
3. 按乘客人数分层（Stratified Sampling by Passenger Count）
4. 结合系统抽样（Systematic Sampling）

这是一个三层嵌套的混合抽样设计
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
    
    def create_time_strata(self):
        """
        第一层：按时间区间分层
        
        将数据按年份或季度分层，可以根据数据的时间分布灵活调整
        """
        print("\n=== 第一层：创建时间分层 ===")
        self.data['year'] = self.data['pickup_datetime'].dt.year
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
    
    def create_geographic_clusters(self):
        """
        第二层：按地理位置聚类
        
        使用上车位置的经纬度，将纽约市划分为若干地理区域（聚类）
        方法：使用网格划分或K-means聚类
        """
        print("\n=== 第二层：创建地理聚类 ===")
        
        # 方法1：网格划分（更直观，适合解释）
        # 将纽约市划分为网格（例如：10x10的网格）
        grid_size = 10
        
        # 计算经纬度范围
        lon_min, lon_max = self.data['pickup_longitude'].min(), self.data['pickup_longitude'].max()
        lat_min, lat_max = self.data['pickup_latitude'].min(), self.data['pickup_latitude'].max()
        
        # 创建网格
        lon_step = (lon_max - lon_min) / grid_size
        lat_step = (lat_max - lat_min) / grid_size
        
        def assign_grid_cluster(row):
            """将经纬度映射到网格聚类"""
            lon_idx = int((row['pickup_longitude'] - lon_min) / lon_step)
            lat_idx = int((row['pickup_latitude'] - lat_min) / lat_step)
            # 防止超出边界
            lon_idx = min(lon_idx, grid_size - 1)
            lat_idx = min(lat_idx, grid_size - 1)
            return f"Grid_{lon_idx}_{lat_idx}"
        
        self.data['geo_cluster'] = self.data.apply(assign_grid_cluster, axis=1)
        
        # 统计聚类信息
        cluster_info = self.data.groupby('geo_cluster').agg({
            'fare_amount': ['count', 'mean'],
        }).round(2)
        cluster_info.columns = ['cluster_size', 'avg_fare']
        
        print(f"\n地理聚类统计（网格大小: {grid_size}x{grid_size}）:")
        print(f"总聚类数: {len(cluster_info)}")
        print(f"平均每个聚类大小: {cluster_info['cluster_size'].mean():.0f}")
        print("\n前10个最大聚类:")
        print(cluster_info.nlargest(10, 'cluster_size'))
        
        return cluster_info
    
    def create_passenger_strata(self):
        """
        第三层：按乘客人数分层
        
        将乘客人数分为若干层（例如：1人、2人、3-4人、5人及以上）
        """
        print("\n=== 第三层：创建乘客人数分层 ===")
        
        def assign_passenger_stratum(passenger_count):
            """将乘客人数映射到分层"""
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
    
    def allocate_sample_size(self):
        """
        样本量分配策略
        
        使用最优分配（Neyman分配）的思想，但需要平衡三层结构
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
        
        # 简化的比例分配（按层大小）
        # 更复杂的话可以用Neyman最优分配
        total_N = allocation_info['N_h'].sum()
        allocation_info['allocation'] = (
            allocation_info['N_h'] / total_N * self.target_sample_size
        ).astype(int)
        
        # 确保每个单元至少分配到1个样本（如果原大小>0）
        allocation_info.loc[allocation_info['allocation'] == 0, 'allocation'] = 1
        allocation_info.loc[allocation_info['N_h'] == 0, 'allocation'] = 0
        
        # 确保 allocation 列始终是整数类型
        allocation_info['allocation'] = allocation_info['allocation'].astype(int)
        
        # 调整总样本量（可能会略大于目标样本量）
        actual_sample_size = allocation_info['allocation'].sum()
        print(f"目标样本量: {self.target_sample_size:,}")
        print(f"实际分配样本量: {actual_sample_size:,}")
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
    
    def estimate_mean_fare(self, enable_bootstrap=False, n_bootstrap=1000, bootstrap_alpha=0.05):
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
        """
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
        
        print(f"\n【分层估计】（考虑三层结构）:")
        print(f"总体均值估计: ${y_bar_st:.2f}")
        print(f"标准误: ${se_st:.2f}")
        print(f"95%理论置信区间: [${ci_st_lower_theoretical:.2f}, ${ci_st_upper_theoretical:.2f}]")
        
        # Bootstrap置信区间（可选）
        ci_bootstrap = None
        bootstrap_covers = None
        if enable_bootstrap:
            print("\n正在计算Bootstrap置信区间...")
            ci_bootstrap = self.bootstrap_confidence_interval(n_bootstrap=n_bootstrap, alpha=bootstrap_alpha)
            
            print(f"95%Bootstrap置信区间: [${ci_bootstrap['lower']:.2f}, ${ci_bootstrap['upper']:.2f}]")
            print(f"Bootstrap方法: {ci_bootstrap['method']}")
        
        # 总体均值（真实值，用于比较）
        true_mean = self.data['fare_amount'].mean()
        print(f"\n【真实总体均值】: ${true_mean:.2f}")
        print(f"估计偏差: ${y_bar_st - true_mean:.2f}")
        print(f"相对误差: {abs(y_bar_st - true_mean) / true_mean * 100:.2f}%")
        
        # 置信区间覆盖情况
        theoretical_covers = (ci_st_lower_theoretical <= true_mean <= ci_st_upper_theoretical)
        if enable_bootstrap and ci_bootstrap is not None:
            bootstrap_covers = (ci_bootstrap['lower'] <= true_mean <= ci_bootstrap['upper'])
            print(f"\n置信区间覆盖情况（真实均值是否在区间内）:")
            print(f"  理论CI: {'✓ 覆盖' if theoretical_covers else '✗ 未覆盖'}")
            print(f"  Bootstrap CI: {'✓ 覆盖' if bootstrap_covers else '✗ 未覆盖'}")
        else:
            print(f"\n置信区间覆盖情况（真实均值是否在区间内）:")
            print(f"  理论CI: {'✓ 覆盖' if theoretical_covers else '✗ 未覆盖'}")
        
        # 设计效应（Design Effect）
        deff = (se_st / se_simple) ** 2
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
    sampler.create_geographic_clusters()
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
    
    return sampler


if __name__ == "__main__":
    sampler = main()

