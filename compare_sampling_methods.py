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
    
    def __init__(self, data_path='train.csv', sample_size=5000, nrows=500000):
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
        print(f"数据加载完成，真实总体均值: ${self.true_mean:.2f}\n")
    
    def simple_random_sampling(self, n_trials=100):
        """简单随机抽样（SRS）"""
        print("【方法1】简单随机抽样 (SRS)")
        print("=" * 60)
        
        np.random.seed(42)
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
        
        stratum_info = self.data.groupby('time_stratum').size()
        N_total = len(self.data)
        
        means = []
        stds = []
        
        np.random.seed(42)
        for trial in range(n_trials):
            sample_list = []
            for stratum, N_h in stratum_info.items():
                # 比例分配
                n_h = max(1, int(N_h / N_total * self.sample_size))
                stratum_data = self.data[self.data['time_stratum'] == stratum]
                if len(stratum_data) >= n_h:
                    sample_stratum = stratum_data.sample(n=n_h, random_state=trial*1000+hash(stratum))
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
                
                means.append(y_bar_st)
                stds.append(np.sqrt(var_st))
        
        mean_est = np.mean(means)
        se_est = np.mean(stds)
        bias = mean_est - self.true_mean
        rmse = np.sqrt(np.mean((np.array(means) - self.true_mean)**2))
        
        self.results['时间分层'] = {
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
        
        return self.results['时间分层']
    
    def geographic_cluster_sampling(self, n_trials=100):
        """地理聚类抽样"""
        print("【方法3】地理聚类抽样")
        print("=" * 60)
        
        # 创建地理聚类（简化版，使用网格）
        grid_size = 10
        lon_min, lon_max = self.data['pickup_longitude'].min(), self.data['pickup_longitude'].max()
        lat_min, lat_max = self.data['pickup_latitude'].min(), self.data['pickup_latitude'].max()
        lon_step = (lon_max - lon_min) / grid_size
        lat_step = (lat_max - lat_min) / grid_size
        
        def assign_cluster(row):
            lon_idx = int((row['pickup_longitude'] - lon_min) / lon_step)
            lat_idx = int((row['pickup_latitude'] - lat_min) / lat_step)
            lon_idx = min(lon_idx, grid_size - 1)
            lat_idx = min(lat_idx, grid_size - 1)
            return f"Grid_{lon_idx}_{lat_idx}"
        
        self.data['geo_cluster'] = self.data.apply(assign_cluster, axis=1)
        cluster_info = self.data.groupby('geo_cluster').size()
        
        means = []
        stds = []
        N_total = len(self.data)
        n_clusters_to_sample = min(20, len(cluster_info))  # 抽20个聚类
        
        np.random.seed(42)
        for trial in range(n_trials):
            # 随机抽聚类
            selected_clusters = np.random.choice(cluster_info.index, size=n_clusters_to_sample, replace=False)
            
            sample_list = []
            for cluster in selected_clusters:
                cluster_data = self.data[self.data['geo_cluster'] == cluster]
                # 在每个聚类内抽相同数量的样本
                n_per_cluster = max(1, self.sample_size // n_clusters_to_sample)
                if len(cluster_data) >= n_per_cluster:
                    sample_cluster = cluster_data.sample(n=n_per_cluster, random_state=trial*1000+hash(cluster))
                    sample_list.append(sample_cluster)
            
            if sample_list:
                sample = pd.concat(sample_list, ignore_index=True)
                # 聚类抽样估计（简化版）
                mean_cluster = sample['fare_amount'].mean()
                std_cluster = sample['fare_amount'].std() / np.sqrt(len(sample))
                
                means.append(mean_cluster)
                stds.append(std_cluster)
        
        mean_est = np.mean(means)
        se_est = np.mean(stds)
        bias = mean_est - self.true_mean
        rmse = np.sqrt(np.mean((np.array(means) - self.true_mean)**2))
        
        self.results['地理聚类'] = {
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
        
        return self.results['地理聚类']
    
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
        
        means = []
        stds = []
        
        np.random.seed(42)
        for trial in range(n_trials):
            sample_list = []
            for stratum, N_h in stratum_info.items():
                n_h = max(1, int(N_h / N_total * self.sample_size))
                stratum_data = self.data[self.data['passenger_stratum'] == stratum]
                if len(stratum_data) >= n_h:
                    sample_stratum = stratum_data.sample(n=n_h, random_state=trial*1000+hash(stratum))
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
                
                means.append(y_bar_st)
                stds.append(np.sqrt(var_st))
        
        mean_est = np.mean(means)
        se_est = np.mean(stds)
        bias = mean_est - self.true_mean
        rmse = np.sqrt(np.mean((np.array(means) - self.true_mean)**2))
        
        self.results['乘客分层'] = {
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
        
        return self.results['乘客分层']
    
    def multi_stage_sampling(self):
        """多阶段复合抽样"""
        print("【方法5】多阶段复合抽样（时间+地理+乘客+系统）")
        print("=" * 60)
        
        sampler = MultiStageSampling(self.data_path, self.sample_size)
        sampler.data = self.data  # 使用已加载的数据
        
        # 执行多阶段抽样
        sampler.create_time_strata()
        sampler.create_geographic_clusters()
        sampler.create_passenger_strata()
        sampler.allocate_sample_size()
        sampler.draw_sample()
        
        # 估计
        results = sampler.estimate_mean_fare()
        
        self.results['多阶段复合'] = {
            'mean': results['stratified_mean'],
            'se': results['se_stratified'],
            'bias': results['stratified_mean'] - self.true_mean,
            'rmse': abs(results['stratified_mean'] - self.true_mean),  # 简化
            'deff': results['deff']
        }
        
        return self.results['多阶段复合']
    
    def compare_all_methods(self):
        """对比所有方法"""
        print("\n" + "="*80)
        print("抽样方法对比总结")
        print("="*80)
        
        # 执行所有方法
        self.simple_random_sampling(n_trials=50)
        self.time_stratified_sampling(n_trials=50)
        self.geographic_cluster_sampling(n_trials=50)
        self.passenger_stratified_sampling(n_trials=50)
        self.multi_stage_sampling()
        
        # 创建对比表
        comparison_df = pd.DataFrame({
            '方法': list(self.results.keys()),
            '估计均值': [r['mean'] for r in self.results.values()],
            '标准误': [r['se'] for r in self.results.values()],
            '偏差': [r['bias'] for r in self.results.values()],
            'RMSE': [r['rmse'] for r in self.results.values()]
        })
        
        comparison_df['相对误差(%)'] = (comparison_df['偏差'].abs() / self.true_mean * 100).round(2)
        comparison_df['效率提升(%)'] = (
            (comparison_df.loc[0, 'se'] - comparison_df['se']) / comparison_df.loc[0, 'se'] * 100
        ).round(2)
        
        print("\n【对比表】")
        print(comparison_df.to_string(index=False))
        
        print(f"\n真实总体均值: ${self.true_mean:.2f}")
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
        axes[0, 0].axvline(self.true_mean, color='red', linestyle='--', linewidth=2, label=f'真实均值 ${self.true_mean:.2f}')
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
                          linewidth=2, label=f'真实均值 ${self.true_mean:.2f}')
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
    comparator = SamplingComparison(
        data_path='train.csv',
        sample_size=5000,
        nrows=500000  # 使用50万条记录进行对比
    )
    
    comparison_df = comparator.compare_all_methods()
    comparison_df.to_csv('sampling_comparison_results.csv', index=False, encoding='utf-8-sig')
    print("\n对比结果已保存至: sampling_comparison_results.csv")


if __name__ == "__main__":
    main()

