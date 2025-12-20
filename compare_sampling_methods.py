"""
对比不同抽样方法的效果
包括：
1. 简单随机抽样 (SRS)
2. 时间分层抽样
3. 地理聚类抽样
4. 乘客人数分层抽样
5. 多阶段复合抽样（时间+地理+乘客+系统）
"""

import pandas as pd
import numpy as np
from multi_stage_sampling import MultiStageSampling
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class SamplingComparison:
    """抽样方法对比类"""
    
    def __init__(self, data_path='train.csv', sample_size=500, nrows=500000):
        self.data_path = data_path
        self.sample_size = sample_size
        self.data = None
        self.true_mean = None
        self.results = {}
        
        # 加载数据
        print("正在加载数据...")
        sampler = MultiStageSampling(data_path, sample_size)
        sampler.load_data(nrows=nrows)
        self.data = sampler.data
        self.true_mean = self.data['fare_amount'].mean()
        print(f"数据加载完成，真实总体均值: ${self.true_mean:.6f}\n")
    
    def simple_random_sampling(self, n_trials=100):
        """简单随机抽样（SRS）"""
        print("【方法1】简单随机抽样 (SRS)")
        print("=" * 60)
        
        np.random.seed(114514)
        means = []
        stds = []
        
        for i in range(n_trials):
            sample = self.data.sample(n=self.sample_size, random_state=i)
            means.append(sample['fare_amount'].mean())
            stds.append(sample['fare_amount'].std() / np.sqrt(self.sample_size))
        
        mean_est = np.mean(means)
        se_est = np.mean(stds)
        bias = mean_est - self.true_mean
        rmse = np.sqrt(np.mean((np.array(means) - self.true_mean)**2))
        
        self.results['SRS'] = {
            'mean': mean_est,
            'se': se_est,
            'bias': bias,
            'rmse': rmse,
            'means_trials': means
        }
        
        print(f"估计均值: ${mean_est:.2f}")
        print(f"标准误: ${se_est:.2f}")
        print(f"偏差: ${bias:.2f}")
        print(f"RMSE: ${rmse:.2f}")
        print(f"95%置信区间: [${mean_est - 1.96*se_est:.2f}, ${mean_est + 1.96*se_est:.2f}]\n")
        
        return self.results['SRS']
    
    def time_stratified_sampling(self, n_trials=100):
        """时间分层抽样"""
        print("【方法2】时间分层抽样")
        print("=" * 60)
        
        # 创建时间分层
        self.data['year'] = self.data['pickup_datetime'].dt.year
        self.data['quarter'] = self.data['pickup_datetime'].dt.quarter
        self.data['time_stratum'] = self.data['year'].astype(str) + '-Q' + self.data['quarter'].astype(str)
        
        # 将时间转换为数值（从某个基准时间开始的秒数）用于回归估计
        if 'time_numeric' not in self.data.columns:
            base_time = self.data['pickup_datetime'].min()
            self.data['time_numeric'] = (self.data['pickup_datetime'] - base_time).dt.total_seconds()
        
        stratum_info = self.data.groupby('time_stratum').size()
        N_total = len(self.data)
        
        # 计算总体中时间的均值（用于回归估计）
        X_bar_time = self.data['time_numeric'].mean()
        
        means = []
        stds = []
        means_reg = []  # 回归估计的均值
        stds_reg = []   # 回归估计的标准误
        
        np.random.seed(114514)
        for trial in range(n_trials):
            sample_list = []
            for stratum, N_h in stratum_info.items():
                # 比例分配
                n_h = max(1, int(N_h / N_total * self.sample_size))
                stratum_data = self.data[self.data['time_stratum'] == stratum]
                if len(stratum_data) >= n_h:
                    # 确保随机种子在有效范围内 (0 到 2^32 - 1)
                    seed = (trial * 1000 + abs(hash(stratum))) % (2**32)
                    sample_stratum = stratum_data.sample(n=n_h, random_state=seed)
                    sample_list.append(sample_stratum)
            
            if sample_list:
                sample = pd.concat(sample_list, ignore_index=True)
                # 分层估计
                y_bar_st = 0
                var_st = 0
                for stratum, N_h in stratum_info.items():
                    sample_stratum = sample[sample['time_stratum'] == stratum]
                    if len(sample_stratum) > 0:
                        W_h = N_h / N_total
                        y_bar_h = sample_stratum['fare_amount'].mean()
                        s_h = sample_stratum['fare_amount'].std()
                        n_h = len(sample_stratum)
                        y_bar_st += W_h * y_bar_h
                        if n_h > 1:
                            var_st += (W_h**2) * (s_h**2) / n_h * (1 - n_h/N_h)
                
                # 回归估计：使用时间作为辅助变量
                sample_times = sample['time_numeric'].values
                sample_fares = sample['fare_amount'].values
                valid_mask = ~(np.isnan(sample_times) | np.isnan(sample_fares))
                
                if np.sum(valid_mask) > 1:
                    x_bar = np.mean(sample_times[valid_mask])
                    # 计算回归系数 b = Cov(y, x) / Var(x)
                    cov_yx = np.cov(sample_fares[valid_mask], sample_times[valid_mask])[0, 1]
                    var_x = np.var(sample_times[valid_mask], ddof=1)
                    
                    if var_x > 0:
                        b = cov_yx / var_x
                        y_bar_reg = y_bar_st + b * (X_bar_time - x_bar)
                        
                        # 回归估计的方差
                        if var_x > 0 and np.var(sample_fares[valid_mask], ddof=1) > 0:
                            r = np.corrcoef(sample_fares[valid_mask], sample_times[valid_mask])[0, 1]
                            var_reg = var_st * (1 - r**2) if not np.isnan(r) else var_st
                        else:
                            var_reg = var_st
                    else:
                        y_bar_reg = y_bar_st
                        var_reg = var_st
                else:
                    y_bar_reg = y_bar_st
                    var_reg = var_st
                
                means.append(y_bar_st)
                stds.append(np.sqrt(var_st))
                means_reg.append(y_bar_reg)
                stds_reg.append(np.sqrt(var_reg))
        
        # 分层估计结果
        mean_est = np.mean(means)
        se_est = np.mean(stds)
        bias = mean_est - self.true_mean
        rmse = np.sqrt(np.mean((np.array(means) - self.true_mean)**2))
        
        # 回归估计结果
        mean_est_reg = np.mean(means_reg)
        se_est_reg = np.mean(stds_reg)
        bias_reg = mean_est_reg - self.true_mean
        rmse_reg = np.sqrt(np.mean((np.array(means_reg) - self.true_mean)**2))
        
        self.results['时间分层'] = {
            'mean': mean_est,
            'se': se_est,
            'bias': bias,
            'rmse': rmse,
            'means_trials': means
        }
        
        self.results['时间分层-回归估计'] = {
            'mean': mean_est_reg,
            'se': se_est_reg,
            'bias': bias_reg,
            'rmse': rmse_reg,
            'means_trials': means_reg
        }
        
        print(f"\n【分层估计】")
        print(f"估计均值: ${mean_est:.2f}")
        print(f"标准误: ${se_est:.2f}")
        print(f"偏差: ${bias:.2f}")
        print(f"RMSE: ${rmse:.2f}")
        print(f"95%置信区间: [${mean_est - 1.96*se_est:.2f}, ${mean_est + 1.96*se_est:.2f}]")
        
        print(f"\n【回归估计（使用时间作为辅助变量）】")
        print(f"估计均值: ${mean_est_reg:.2f}")
        print(f"标准误: ${se_est_reg:.2f}")
        print(f"偏差: ${bias_reg:.2f}")
        print(f"RMSE: ${rmse_reg:.2f}")
        print(f"95%置信区间: [${mean_est_reg - 1.96*se_est_reg:.2f}, ${mean_est_reg + 1.96*se_est_reg:.2f}]")
        print(f"效率提升: {((se_est - se_est_reg) / se_est * 100):.2f}%\n")
        
        return self.results['时间分层-回归估计']
    
    def geographic_cluster_sampling(self, n_trials=100):
        """地理聚类抽样（基于距离范围分层）"""
        print("【方法3】地理聚类抽样（基于距离范围）")
        print("=" * 60)
        
        # 使用距离范围分层而不是网格聚类
        # 计算pickup和dropoff之间的距离
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
        
        # 计算距离
        print("正在计算pickup和dropoff之间的距离...")
        valid_mask = ~(self.data[required_cols].isna().any(axis=1))
        route_distances = np.full(len(self.data), np.nan)
        route_distances[valid_mask] = haversine_distance_vectorized(
            self.data.loc[valid_mask, 'pickup_longitude'].values,
            self.data.loc[valid_mask, 'pickup_latitude'].values,
            self.data.loc[valid_mask, 'dropoff_longitude'].values,
            self.data.loc[valid_mask, 'dropoff_latitude'].values
        )
        
        # 根据样本量确定距离范围数
        # 使用较少的范围数以减少偏度
        n_distance_ranges = min(10, max(3, self.sample_size // 100))  # 根据样本量调整，最多10个范围
        
        # 根据距离分位数创建距离范围
        valid_distances = route_distances[valid_mask]
        if len(valid_distances) == 0:
            raise ValueError("无法计算有效的距离值")
        
        percentiles = np.linspace(0, 100, n_distance_ranges + 1)
        distance_thresholds = np.percentile(valid_distances, percentiles)
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
        
        # 为每条记录分配距离范围
        def assign_distance_range(distance):
            """根据距离值分配距离范围"""
            if np.isnan(distance):
                return "Distance_NaN"
            for i in range(len(distance_thresholds) - 1):
                if distance_thresholds[i] <= distance < distance_thresholds[i+1]:
                    return distance_labels[i]
            return distance_labels[-1]
        
        self.data['geo_cluster'] = [assign_distance_range(d) for d in route_distances]
        # 保存距离数据用于回归估计
        self.data['route_distance'] = route_distances
        cluster_info = self.data.groupby('geo_cluster').size()
        
        # 计算总体中距离的均值（用于回归估计）
        X_bar_distance = np.nanmean(route_distances)
        
        # 准备时间和乘客辅助变量（用于多元回归估计）
        if 'time_numeric' not in self.data.columns:
            base_time = self.data['pickup_datetime'].min()
            self.data['time_numeric'] = (self.data['pickup_datetime'] - base_time).dt.total_seconds()
        
        # 计算总体中时间和乘客的均值（用于多元回归估计）
        X_bar_time = self.data['time_numeric'].mean()
        X_bar_passenger = self.data['passenger_count'].mean()
        
        print(f"使用 {n_distance_ranges} 个距离范围进行分层")
        print(f"距离范围阈值（公里）:")
        for i in range(len(distance_thresholds) - 1):
            print(f"  范围 {i+1}: {distance_thresholds[i]:.2f} - {distance_thresholds[i+1]:.2f} km")
        print(f"总体距离均值: {X_bar_distance:.4f} km")
        print(f"总体时间均值: {X_bar_time:.2f} (秒)")
        print(f"总体乘客均值: {X_bar_passenger:.2f}")
        
        means = []
        stds = []
        means_reg = []  # 单变量回归估计的均值（使用距离）
        stds_reg = []   # 单变量回归估计的标准误
        means_multi_reg = []  # 多元回归估计的均值（使用时间和乘客）
        stds_multi_reg = []   # 多元回归估计的标准误
        N_total = len(self.data)
        
        # 使用分层抽样而不是聚类抽样（因为距离范围是分层，不是聚类）
        np.random.seed(114514)
        for trial in range(n_trials):
            sample_list = []
            for cluster, N_h in cluster_info.items():
                # 比例分配
                n_h = max(1, int(N_h / N_total * self.sample_size))
                cluster_data = self.data[self.data['geo_cluster'] == cluster]
                if len(cluster_data) >= n_h:
                    # 确保随机种子在有效范围内 (0 到 2^32 - 1)
                    seed = (trial * 1000 + abs(hash(cluster))) % (2**32)
                    sample_cluster = cluster_data.sample(n=n_h, random_state=seed)
                    sample_list.append(sample_cluster)
            
            if sample_list:
                sample = pd.concat(sample_list, ignore_index=True)
                # 分层估计（因为距离范围是分层）
                y_bar_st = 0
                var_st = 0
                for cluster, N_h in cluster_info.items():
                    sample_cluster = sample[sample['geo_cluster'] == cluster]
                    if len(sample_cluster) > 0:
                        W_h = N_h / N_total
                        y_bar_h = sample_cluster['fare_amount'].mean()
                        s_h = sample_cluster['fare_amount'].std()
                        n_h = len(sample_cluster)
                        y_bar_st += W_h * y_bar_h
                        if n_h > 1:
                            var_st += (W_h**2) * (s_h**2) / n_h * (1 - n_h/N_h)
                
                # 单变量回归估计：使用距离作为辅助变量
                # y_reg = y_st + b * (X_bar - x_bar)
                # 其中 b 是回归系数
                sample_distances = sample['route_distance'].values
                sample_fares = sample['fare_amount'].values
                valid_mask = ~(np.isnan(sample_distances) | np.isnan(sample_fares))
                
                if np.sum(valid_mask) > 1:
                    x_bar = np.mean(sample_distances[valid_mask])
                    # 计算回归系数 b = Cov(y, x) / Var(x)
                    cov_yx = np.cov(sample_fares[valid_mask], sample_distances[valid_mask])[0, 1]
                    var_x = np.var(sample_distances[valid_mask], ddof=1)
                    
                    if var_x > 0:
                        b = cov_yx / var_x
                        y_bar_reg = y_bar_st + b * (X_bar_distance - x_bar)
                        
                        # 回归估计的方差（简化估计）
                        # Var(y_reg) ≈ Var(y_st) * (1 - r^2)，其中r是相关系数
                        if var_x > 0 and np.var(sample_fares[valid_mask], ddof=1) > 0:
                            r = np.corrcoef(sample_fares[valid_mask], sample_distances[valid_mask])[0, 1]
                            var_reg = var_st * (1 - r**2) if not np.isnan(r) else var_st
                        else:
                            var_reg = var_st
                    else:
                        y_bar_reg = y_bar_st
                        var_reg = var_st
                else:
                    y_bar_reg = y_bar_st
                    var_reg = var_st
                
                # 多元回归估计：使用时间和乘客作为辅助变量
                # y_multi_reg = y_st + b1 * (X̄1 - x̄1) + b2 * (X̄2 - x̄2)
                sample_times = sample['time_numeric'].values
                sample_passengers = sample['passenger_count'].values
                valid_mask_multi = ~(np.isnan(sample_times) | np.isnan(sample_passengers) | np.isnan(sample_fares))
                
                if np.sum(valid_mask_multi) > 2:  # 至少需要3个点才能做多元回归
                    # 准备数据
                    y = sample_fares[valid_mask_multi]
                    X1 = sample_times[valid_mask_multi]
                    X2 = sample_passengers[valid_mask_multi]
                    
                    # 计算样本均值
                    x_bar_time = np.mean(X1)
                    x_bar_passenger = np.mean(X2)
                    
                    # 中心化数据
                    y_centered = y - np.mean(y)
                    X1_centered = X1 - x_bar_time
                    X2_centered = X2 - x_bar_passenger
                    
                    # 构建设计矩阵
                    X_design = np.column_stack([X1_centered, X2_centered])
                    
                    # 计算多元回归系数 b = (X'X)^(-1) X'y
                    try:
                        XtX = X_design.T @ X_design
                        if np.linalg.cond(XtX) < 1e12:  # 检查条件数，避免奇异矩阵
                            XtX_inv = np.linalg.inv(XtX)
                            Xty = X_design.T @ y_centered
                            b_coefs = XtX_inv @ Xty
                            
                            # 多元回归估计
                            y_bar_multi_reg = y_bar_st + b_coefs[0] * (X_bar_time - x_bar_time) + b_coefs[1] * (X_bar_passenger - x_bar_passenger)
                            
                            # 计算多元回归的R²来估计方差
                            y_pred = np.mean(y) + X_design @ b_coefs
                            ss_res = np.sum((y - y_pred)**2)
                            ss_tot = np.sum(y_centered**2)
                            if ss_tot > 0:
                                r_squared = 1 - ss_res / ss_tot
                                var_multi_reg = var_st * (1 - r_squared) if r_squared > 0 else var_st
                            else:
                                var_multi_reg = var_st
                        else:
                            y_bar_multi_reg = y_bar_st
                            var_multi_reg = var_st
                    except np.linalg.LinAlgError:
                        y_bar_multi_reg = y_bar_st
                        var_multi_reg = var_st
                else:
                    y_bar_multi_reg = y_bar_st
                    var_multi_reg = var_st
                
                means.append(y_bar_st)
                stds.append(np.sqrt(var_st))
                means_reg.append(y_bar_reg)
                stds_reg.append(np.sqrt(var_reg))
                means_multi_reg.append(y_bar_multi_reg)
                stds_multi_reg.append(np.sqrt(var_multi_reg))
        
        # 分层估计结果
        mean_est = np.mean(means)
        se_est = np.mean(stds)
        bias = mean_est - self.true_mean
        rmse = np.sqrt(np.mean((np.array(means) - self.true_mean)**2))
        
        # 单变量回归估计结果（使用距离）
        mean_est_reg = np.mean(means_reg)
        se_est_reg = np.mean(stds_reg)
        bias_reg = mean_est_reg - self.true_mean
        rmse_reg = np.sqrt(np.mean((np.array(means_reg) - self.true_mean)**2))
        
        # 多元回归估计结果（使用时间和乘客）
        mean_est_multi_reg = np.mean(means_multi_reg)
        se_est_multi_reg = np.mean(stds_multi_reg)
        bias_multi_reg = mean_est_multi_reg - self.true_mean
        rmse_multi_reg = np.sqrt(np.mean((np.array(means_multi_reg) - self.true_mean)**2))
        
        self.results['地理聚类'] = {
            'mean': mean_est,
            'se': se_est,
            'bias': bias,
            'rmse': rmse,
            'means_trials': means
        }
        
        self.results['地理聚类-复合回归估计(时间+乘客)'] = {
            'mean': mean_est_multi_reg,
            'se': se_est_multi_reg,
            'bias': bias_multi_reg,
            'rmse': rmse_multi_reg,
            'means_trials': means_multi_reg
        }
        
        print(f"\n【分层估计】")
        print(f"估计均值: ${mean_est:.2f}")
        print(f"标准误: ${se_est:.2f}")
        print(f"偏差: ${bias:.2f}")
        print(f"RMSE: ${rmse:.2f}")
        print(f"95%置信区间: [${mean_est - 1.96*se_est:.2f}, ${mean_est + 1.96*se_est:.2f}]")
        
        print(f"\n【复合回归估计（使用时间和乘客作为辅助变量）】")
        print(f"估计均值: ${mean_est_multi_reg:.2f}")
        print(f"标准误: ${se_est_multi_reg:.2f}")
        print(f"偏差: ${bias_multi_reg:.2f}")
        print(f"RMSE: ${rmse_multi_reg:.2f}")
        print(f"95%置信区间: [${mean_est_multi_reg - 1.96*se_est_multi_reg:.2f}, ${mean_est_multi_reg + 1.96*se_est_multi_reg:.2f}]")
        print(f"相对于分层估计的效率提升: {((se_est - se_est_multi_reg) / se_est * 100):.2f}%\n")
        
        return self.results['地理聚类-复合回归估计(时间+乘客)']
    
    def passenger_stratified_sampling(self, n_trials=100):
        """乘客人数分层抽样"""
        print("【方法4】乘客人数分层抽样")
        print("=" * 60)
        
        def assign_stratum(pc):
            if pc == 1:
                return '1_passenger'
            elif pc == 2:
                return '2_passengers'
            elif pc <= 4:
                return '3-4_passengers'
            else:
                return '5+_passengers'
        
        self.data['passenger_stratum'] = self.data['passenger_count'].apply(assign_stratum)
        stratum_info = self.data.groupby('passenger_stratum').size()
        N_total = len(self.data)
        
        # 计算总体中乘客人数的均值（用于回归估计）
        X_bar_passenger = self.data['passenger_count'].mean()
        
        means = []
        stds = []
        means_reg = []  # 回归估计的均值
        stds_reg = []   # 回归估计的标准误
        
        np.random.seed(114514)
        for trial in range(n_trials):
            sample_list = []
            for stratum, N_h in stratum_info.items():
                n_h = max(1, int(N_h / N_total * self.sample_size))
                stratum_data = self.data[self.data['passenger_stratum'] == stratum]
                if len(stratum_data) >= n_h:
                    # 确保随机种子在有效范围内 (0 到 2^32 - 1)
                    seed = (trial * 1000 + abs(hash(stratum))) % (2**32)
                    sample_stratum = stratum_data.sample(n=n_h, random_state=seed)
                    sample_list.append(sample_stratum)
            
            if sample_list:
                sample = pd.concat(sample_list, ignore_index=True)
                # 分层估计
                y_bar_st = 0
                var_st = 0
                for stratum, N_h in stratum_info.items():
                    sample_stratum = sample[sample['passenger_stratum'] == stratum]
                    if len(sample_stratum) > 0:
                        W_h = N_h / N_total
                        y_bar_h = sample_stratum['fare_amount'].mean()
                        s_h = sample_stratum['fare_amount'].std()
                        n_h = len(sample_stratum)
                        y_bar_st += W_h * y_bar_h
                        if n_h > 1:
                            var_st += (W_h**2) * (s_h**2) / n_h * (1 - n_h/N_h)
                
                # 回归估计：使用乘客人数作为辅助变量
                sample_passengers = sample['passenger_count'].values
                sample_fares = sample['fare_amount'].values
                valid_mask = ~(np.isnan(sample_passengers) | np.isnan(sample_fares))
                
                if np.sum(valid_mask) > 1:
                    x_bar = np.mean(sample_passengers[valid_mask])
                    # 计算回归系数 b = Cov(y, x) / Var(x)
                    cov_yx = np.cov(sample_fares[valid_mask], sample_passengers[valid_mask])[0, 1]
                    var_x = np.var(sample_passengers[valid_mask], ddof=1)
                    
                    if var_x > 0:
                        b = cov_yx / var_x
                        y_bar_reg = y_bar_st + b * (X_bar_passenger - x_bar)
                        
                        # 回归估计的方差
                        if var_x > 0 and np.var(sample_fares[valid_mask], ddof=1) > 0:
                            r = np.corrcoef(sample_fares[valid_mask], sample_passengers[valid_mask])[0, 1]
                            var_reg = var_st * (1 - r**2) if not np.isnan(r) else var_st
                        else:
                            var_reg = var_st
                    else:
                        y_bar_reg = y_bar_st
                        var_reg = var_st
                else:
                    y_bar_reg = y_bar_st
                    var_reg = var_st
                
                means.append(y_bar_st)
                stds.append(np.sqrt(var_st))
                means_reg.append(y_bar_reg)
                stds_reg.append(np.sqrt(var_reg))
        
        # 分层估计结果
        mean_est = np.mean(means)
        se_est = np.mean(stds)
        bias = mean_est - self.true_mean
        rmse = np.sqrt(np.mean((np.array(means) - self.true_mean)**2))
        
        # 回归估计结果
        mean_est_reg = np.mean(means_reg)
        se_est_reg = np.mean(stds_reg)
        bias_reg = mean_est_reg - self.true_mean
        rmse_reg = np.sqrt(np.mean((np.array(means_reg) - self.true_mean)**2))
        
        self.results['乘客分层'] = {
            'mean': mean_est,
            'se': se_est,
            'bias': bias,
            'rmse': rmse,
            'means_trials': means
        }
        
        self.results['乘客分层-回归估计'] = {
            'mean': mean_est_reg,
            'se': se_est_reg,
            'bias': bias_reg,
            'rmse': rmse_reg,
            'means_trials': means_reg
        }
        
        print(f"\n【分层估计】")
        print(f"估计均值: ${mean_est:.2f}")
        print(f"标准误: ${se_est:.2f}")
        print(f"偏差: ${bias:.2f}")
        print(f"RMSE: ${rmse:.2f}")
        print(f"95%置信区间: [${mean_est - 1.96*se_est:.2f}, ${mean_est + 1.96*se_est:.2f}]")
        
        print(f"\n【回归估计（使用乘客人数作为辅助变量）】")
        print(f"估计均值: ${mean_est_reg:.2f}")
        print(f"标准误: ${se_est_reg:.2f}")
        print(f"偏差: ${bias_reg:.2f}")
        print(f"RMSE: ${rmse_reg:.2f}")
        print(f"95%置信区间: [${mean_est_reg - 1.96*se_est_reg:.2f}, ${mean_est_reg + 1.96*se_est_reg:.2f}]")
        print(f"效率提升: {((se_est - se_est_reg) / se_est * 100):.2f}%\n")
        
        return self.results['乘客分层-回归估计']
    
    def multi_stage_sampling(self, cluster_method='distance', n_clusters=20, max_total_strata=30, n_trials=100):
        """
        多阶段复合抽样
        
        Parameters:
        -----------
        max_total_strata : int, default=30
            总分层单元数的上限
        n_trials : int, default=100
            试验次数，用于计算RMSE
        """
        print("【方法5】多阶段复合抽样（时间+地理+乘客+系统）")
        print("=" * 60)
        
        # 先创建时间分层（简化模式）以获取实际时间层数
        temp_sampler = MultiStageSampling(self.data_path, self.sample_size)
        temp_sampler.data = self.data.copy()
        temp_sampler.create_time_strata(simplify=True)
        actual_time_strata = len(temp_sampler.data['time_stratum'].unique())
        
        # 根据总层数限制计算最大距离层数
        # 时间层数 × 距离层数 × 乘客层数(2) ≤ max_total_strata
        max_distance_strata = max(1, int(max_total_strata / (actual_time_strata * 2)))
        
        # 如果计算出的距离层数小于用户指定的，使用计算出的值
        n_clusters = min(n_clusters, max_distance_strata)
        
        print(f"\n分层配置（总层数限制: {max_total_strata}）:")
        print(f"  时间层数: {actual_time_strata} (简化模式：只按年份)")
        print(f"  距离层数: {n_clusters} (限制后)")
        print(f"  乘客层数: 2 (简化模式：1人 vs 2+人)")
        print(f"  理论最大单元数: {actual_time_strata * n_clusters * 2}")
        print(f"  试验次数: {n_trials}")
        
        means = []
        stds = []
        final_sampler = None  # 保存最后一次成功的sampler用于保存样本
        
        np.random.seed(114514)
        for trial in range(n_trials):
            sampler = MultiStageSampling(self.data_path, self.sample_size)
            sampler.data = self.data.copy()  # 每次试验使用原始数据的副本
            
            # 执行多阶段抽样（使用简化模式）
            sampler.create_time_strata(simplify=True)
            sampler.create_geographic_clusters(method=cluster_method, n_clusters=n_clusters)
            sampler.create_passenger_strata(simplify=True)
            sampler.allocate_sample_size()
            sampler.draw_sample()
            
            if sampler.final_sample is not None and len(sampler.final_sample) > 0:
                # 估计（抑制详细输出）
                results = sampler.estimate_mean_fare(verbose=False)
                means.append(results['stratified_mean'])
                stds.append(results['se_stratified'])
                # 保存最后一次成功的sampler
                final_sampler = sampler
        
        if not means:
            raise ValueError("多阶段复合抽样未能生成有效样本")
        
        mean_est = np.mean(means)
        se_est = np.mean(stds)
        bias = mean_est - self.true_mean
        rmse = np.sqrt(np.mean((np.array(means) - self.true_mean)**2))
        
        self.results['多阶段复合'] = {
            'mean': mean_est,
            'se': se_est,
            'bias': bias,
            'rmse': rmse,
            'means_trials': means
        }
        
        # 保存最后一次试验的样本
        if final_sampler is not None and final_sampler.final_sample is not None:
            final_sampler.final_sample.to_csv('sampled_data_multistage_complex.csv', index=False)
            print(f"\n✓ 多阶段复合抽样样本已保存至: sampled_data_multistage_complex.csv")
            print(f"  样本量: {len(final_sampler.final_sample):,}")
        
        print(f"\n估计均值: ${mean_est:.2f}")
        print(f"标准误: ${se_est:.2f}")
        print(f"偏差: ${bias:.2f}")
        print(f"RMSE: ${rmse:.2f}")
        print(f"95%置信区间: [${mean_est - 1.96*se_est:.2f}, ${mean_est + 1.96*se_est:.2f}]\n")
        
        return self.results['多阶段复合']
    
    def compare_all_methods(self, cluster_method='distance', n_clusters=20, max_total_strata=30):
        """
        对比所有方法
        
        Parameters:
        -----------
        max_total_strata : int, default=30
            多阶段抽样的总分层单元数上限
        """
        print("\n" + "="*80)
        print("抽样方法对比总结")
        print("="*80)
        print(f"距离范围分层方法: {cluster_method} (距离范围数: {n_clusters})")
        print(f"多阶段抽样总层数限制: {max_total_strata}")
        
        # 执行所有方法
        n_trials = 50  # 统一试验次数
        self.simple_random_sampling(n_trials=n_trials)
        self.time_stratified_sampling(n_trials=n_trials)
        self.geographic_cluster_sampling(n_trials=n_trials)
        self.passenger_stratified_sampling(n_trials=n_trials)
        self.multi_stage_sampling(cluster_method=cluster_method, n_clusters=n_clusters, max_total_strata=max_total_strata, n_trials=n_trials)
        
        # 创建对比表
        comparison_df = pd.DataFrame({
            '方法': list(self.results.keys()),
            '估计均值': [r['mean'] for r in self.results.values()],
            '标准误': [r['se'] for r in self.results.values()],
            '偏差': [r['bias'] for r in self.results.values()],
            'RMSE': [r['rmse'] for r in self.results.values()]
        })
        
        comparison_df['相对误差(%)'] = (comparison_df['RMSE'] / self.true_mean * 100).round(2)
        comparison_df['效率提升(%)'] = (
            (comparison_df.loc[0, '标准误'] - comparison_df['标准误']) / comparison_df.loc[0, '标准误'] * 100
        ).round(2)
        
        print("\n【对比表】")
        print(comparison_df.to_string(index=False))
        
        print(f"\n真实总体均值: ${self.true_mean:.6f}")
        print(f"样本量: {self.sample_size:,}")
        
        # 可视化对比
        self.plot_comparison()
        
        return comparison_df
    
    def plot_comparison(self):
        """可视化对比结果"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        methods = list(self.results.keys())
        means = [r['mean'] for r in self.results.values()]
        ses = [r['se'] for r in self.results.values()]
        biases = [r['bias'] for r in self.results.values()]
        
        # 图1: 估计均值对比
        axes[0, 0].barh(methods, means, color='skyblue', alpha=0.7)
        axes[0, 0].axvline(self.true_mean, color='red', linestyle='--', linewidth=2, label=f'真实均值 ${self.true_mean:.6f}')
        axes[0, 0].set_xlabel('估计均值 ($)')
        axes[0, 0].set_title('各方法估计均值对比')
        axes[0, 0].legend()
        axes[0, 0].grid(axis='x', alpha=0.3)
        
        # 图2: 标准误对比
        axes[0, 1].barh(methods, ses, color='lightcoral', alpha=0.7)
        axes[0, 1].set_xlabel('标准误 ($)')
        axes[0, 1].set_title('各方法标准误对比（越小越好）')
        axes[0, 1].grid(axis='x', alpha=0.3)
        
        # 图3: 偏差对比
        axes[1, 0].barh(methods, biases, color='lightgreen', alpha=0.7)
        axes[1, 0].axvline(0, color='black', linestyle='-', linewidth=1)
        axes[1, 0].set_xlabel('偏差 ($)')
        axes[1, 0].set_title('各方法偏差对比（越接近0越好）')
        axes[1, 0].grid(axis='x', alpha=0.3)
        
        # 图4: 置信区间对比
        for i, method in enumerate(methods):
            mean = means[i]
            se = ses[i]
            axes[1, 1].errorbar([i], [mean], yerr=1.96*se, 
                              fmt='o', capsize=5, label=method, markersize=8)
        axes[1, 1].axhline(self.true_mean, color='red', linestyle='--', 
                          linewidth=2, label=f'真实均值 ${self.true_mean:.6f}')
        axes[1, 1].set_xticks(range(len(methods)))
        axes[1, 1].set_xticklabels(methods, rotation=45, ha='right')
        axes[1, 1].set_ylabel('车费 ($)')
        axes[1, 1].set_title('各方法95%置信区间对比')
        axes[1, 1].legend(loc='best', fontsize=8)
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('sampling_comparison.png', dpi=300, bbox_inches='tight')
        print("\n对比图已保存至: sampling_comparison.png")
        plt.close()


def main():
    """主函数"""
    import sys
    
    # 默认使用距离聚类，可以通过命令行参数修改
    cluster_method = 'route'  # 可选: 'distance', 'route', 'grid'
    
    # 样本量设置
    sample_size = 300
    
    if len(sys.argv) > 1:
        cluster_method = sys.argv[1]  # 第一个参数：聚类方法
    if len(sys.argv) > 2:
        sample_size = int(sys.argv[2])  # 第二个参数：样本量
    if len(sys.argv) > 3:
        n_clusters = int(sys.argv[3])  # 第三个参数：聚类数（可选）
    else:
        # 根据样本量和总层数限制（30）自动调整聚类数
        # 使用简化模式：时间只按年份（约3-5层），乘客只分2层
        # 理论最大单元数 = 时间层数 × n_clusters（距离） × 2（乘客）
        # 假设时间层数为4（保守估计），则：4 × n_clusters × 2 ≤ 30
        # 因此：n_clusters ≤ 30 / 8 = 3.75，即最多3层
        
        # 但考虑到样本量，也需要合理分配
        # 策略：
        # - 如果样本量 < 200：只使用2个距离层（短途/长途）
        # - 如果样本量 < 500：使用2-3个距离层
        # - 如果样本量 >= 500：使用3个距离层（受总层数限制）
        if sample_size < 200:
            n_clusters = 2  # 只分短途和长途
        elif sample_size < 500:
            n_clusters = 3  # 最多3层（受总层数限制）
        else:
            n_clusters = 3  # 最多3层（受总层数限制）
        
        # 假设时间层数为4，乘客层数为2
        estimated_time_strata = 4
        passenger_strata = 2
        total_units = estimated_time_strata * n_clusters * passenger_strata
        print(f"根据样本量 {sample_size} 和总层数限制（30）自动计算：")
        print(f"  距离层数 = {n_clusters}")
        print(f"  估计时间层数 = {estimated_time_strata} (简化模式：只按年份)")
        print(f"  乘客层数 = {passenger_strata} (简化模式：1人 vs 2+人)")
        print(f"  理论最大单元数 = {total_units} (≤ 30)")
    
    print(f"使用距离范围分层方法: {cluster_method}, 样本量: {sample_size}, 距离范围数: {n_clusters}")
    
    comparator = SamplingComparison(
        data_path='train.csv',
        sample_size=sample_size,
        nrows=500000  # 使用50万条记录进行对比
    )
    
    comparison_df = comparator.compare_all_methods(cluster_method=cluster_method, n_clusters=n_clusters, max_total_strata=30)
    comparison_df.to_csv('sampling_comparison_results.csv', index=False, encoding='utf-8-sig')
    print("\n对比结果已保存至: sampling_comparison_results.csv")


if __name__ == "__main__":
    main()

