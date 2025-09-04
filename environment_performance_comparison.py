#!/usr/bin/env python3
"""
Environment Performance Comparison Analysis

This script provides comprehensive analysis of A3C global vs individual worker 
performance across different environments using action log CSV files.

Usage:
    python environment_performance_comparison.py
    
CSV files expected:
    - a3c_global_env{0-4}_actions.csv
    - individual_w{0-4}_env{0-4}_actions.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
import re
from typing import Dict, List, Tuple, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

class EnvironmentPerformanceAnalyzer:
    """Main class for analyzing performance across environments"""
    
    def __init__(self, data_dir: str = "."):
        self.data_dir = Path(data_dir)
        self.raw_data = {}
        self.episode_metrics = {}
        self.comparison_results = {}
        
    def load_data(self) -> None:
        """Load all CSV action files"""
        print("Loading CSV files...")
        
        # A3C Global files
        for env_id in range(5):
            filepath = self.data_dir / f"a3c_global_env{env_id}_actions.csv"
            if filepath.exists():
                self.raw_data[f"a3c_global_env{env_id}"] = pd.read_csv(filepath)
                print(f"✓ Loaded {filepath.name}")
            else:
                print(f"✗ Missing {filepath.name}")
        
        # Individual worker files
        for worker_id in range(5):
            for env_id in range(5):
                filepath = self.data_dir / f"individual_w{worker_id}_env{env_id}_actions.csv"
                if filepath.exists():
                    self.raw_data[f"individual_w{worker_id}_env{env_id}"] = pd.read_csv(filepath)
                    print(f"✓ Loaded {filepath.name}")
                else:
                    print(f"✗ Missing {filepath.name}")
        
        print(f"\nLoaded {len(self.raw_data)} CSV files")
    
    def calculate_episode_metrics(self) -> None:
        """Calculate episode-level performance metrics"""
        print("Calculating episode metrics...")
        
        for key, df in self.raw_data.items():
            # Group by episode and calculate metrics
            episode_stats = df.groupby('episode').agg({
                'reward': ['sum', 'mean', 'count'],
                'step': 'max'
            }).round(2)
            
            # Flatten column names
            episode_stats.columns = ['total_reward', 'avg_reward', 'num_steps', 'episode_length']
            episode_stats['episode_length'] += 1  # step is 0-indexed
            
            self.episode_metrics[key] = episode_stats.reset_index()
        
        print(f"Calculated metrics for {len(self.episode_metrics)} models")
    
    def perform_comparison_analysis(self) -> None:
        """Perform statistical comparison between A3C global and individual workers"""
        print("Performing comparison analysis...")
        
        results = []
        
        for env_id in range(5):
            env_results = {'environment': env_id}
            
            # Get A3C global performance
            a3c_key = f"a3c_global_env{env_id}"
            if a3c_key in self.episode_metrics:
                a3c_rewards = self.episode_metrics[a3c_key]['total_reward'].values
                env_results['a3c_mean'] = np.mean(a3c_rewards)
                env_results['a3c_std'] = np.std(a3c_rewards)
                env_results['a3c_episodes'] = len(a3c_rewards)
            else:
                env_results.update({'a3c_mean': np.nan, 'a3c_std': np.nan, 'a3c_episodes': 0})
            
            # Get individual worker performance
            individual_rewards = []
            individual_means = []
            
            for worker_id in range(5):
                ind_key = f"individual_w{worker_id}_env{env_id}"
                if ind_key in self.episode_metrics:
                    worker_rewards = self.episode_metrics[ind_key]['total_reward'].values
                    individual_rewards.extend(worker_rewards)
                    individual_means.append(np.mean(worker_rewards))
                    env_results[f'individual_w{worker_id}_mean'] = np.mean(worker_rewards)
                    env_results[f'individual_w{worker_id}_std'] = np.std(worker_rewards)
                else:
                    env_results[f'individual_w{worker_id}_mean'] = np.nan
                    env_results[f'individual_w{worker_id}_std'] = np.nan
            
            if individual_rewards:
                env_results['individual_combined_mean'] = np.mean(individual_rewards)
                env_results['individual_combined_std'] = np.std(individual_rewards)
                env_results['individual_avg_mean'] = np.mean(individual_means)
                env_results['best_individual_mean'] = np.max(individual_means)
                env_results['worst_individual_mean'] = np.min(individual_means)
                
                # Statistical test (if A3C data available)
                if not np.isnan(env_results['a3c_mean']) and len(a3c_rewards) > 1:
                    t_stat, p_value = stats.ttest_ind(a3c_rewards, individual_rewards)
                    env_results['t_statistic'] = t_stat
                    env_results['p_value'] = p_value
                    env_results['significant'] = p_value < 0.05
                    
                    # Calculate improvement percentage
                    improvement = ((env_results['a3c_mean'] - env_results['individual_combined_mean']) / 
                                 abs(env_results['individual_combined_mean']) * 100)
                    env_results['improvement_pct'] = improvement
                else:
                    env_results.update({
                        't_statistic': np.nan, 'p_value': np.nan, 
                        'significant': False, 'improvement_pct': np.nan
                    })
            else:
                env_results.update({
                    'individual_combined_mean': np.nan, 'individual_combined_std': np.nan,
                    'individual_avg_mean': np.nan, 'best_individual_mean': np.nan,
                    'worst_individual_mean': np.nan, 't_statistic': np.nan,
                    'p_value': np.nan, 'significant': False, 'improvement_pct': np.nan
                })
            
            results.append(env_results)
        
        self.comparison_results = pd.DataFrame(results)
        print("Comparison analysis completed")
    
    def plot_environment_comparison(self, save_plots: bool = True) -> None:
        """Create bar chart comparison across environments"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Prepare data for plotting
        envs = self.comparison_results['environment'].values
        a3c_means = self.comparison_results['a3c_mean'].values
        a3c_stds = self.comparison_results['a3c_std'].values
        ind_means = self.comparison_results['individual_combined_mean'].values
        ind_stds = self.comparison_results['individual_combined_std'].values
        
        x = np.arange(len(envs))
        width = 0.35
        
        # Plot 1: Mean performance with error bars
        ax1.bar(x - width/2, a3c_means, width, yerr=a3c_stds, label='A3C Global', 
                alpha=0.8, capsize=5)
        ax1.bar(x + width/2, ind_means, width, yerr=ind_stds, label='Individual Combined', 
                alpha=0.8, capsize=5)
        
        ax1.set_xlabel('Environment ID')
        ax1.set_ylabel('Average Total Reward')
        ax1.set_title('A3C Global vs Individual Workers - Average Performance')
        ax1.set_xticks(x)
        ax1.set_xticklabels(envs)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Improvement percentage
        improvements = self.comparison_results['improvement_pct'].values
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        
        ax2.bar(envs, improvements, color=colors, alpha=0.7)
        ax2.set_xlabel('Environment ID')
        ax2.set_ylabel('Improvement (%)')
        ax2.set_title('A3C Global Improvement over Individual Workers')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(improvements):
            if not np.isnan(v):
                ax2.text(i, v + (1 if v > 0 else -1), f'{v:.1f}%', 
                        ha='center', va='bottom' if v > 0 else 'top')
        
        plt.tight_layout()
        if save_plots:
            plt.savefig('environment_comparison.png', dpi=300, bbox_inches='tight')
            print("Saved: environment_comparison.png")
        plt.show()
    
    def plot_episode_curves(self, save_plots: bool = True) -> None:
        """Plot episode-by-episode performance curves"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for env_id in range(5):
            ax = axes[env_id]
            
            # Plot A3C global
            a3c_key = f"a3c_global_env{env_id}"
            if a3c_key in self.episode_metrics:
                data = self.episode_metrics[a3c_key]
                ax.plot(data['episode'], data['total_reward'], 
                       label='A3C Global', linewidth=2, alpha=0.8)
            
            # Plot individual workers
            for worker_id in range(5):
                ind_key = f"individual_w{worker_id}_env{env_id}"
                if ind_key in self.episode_metrics:
                    data = self.episode_metrics[ind_key]
                    ax.plot(data['episode'], data['total_reward'], 
                           label=f'Worker {worker_id}', alpha=0.6, linewidth=1)
            
            ax.set_xlabel('Episode')
            ax.set_ylabel('Total Reward')
            ax.set_title(f'Environment {env_id}')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplot
        axes[5].remove()
        
        plt.tight_layout()
        if save_plots:
            plt.savefig('episode_curves.png', dpi=300, bbox_inches='tight')
            print("Saved: episode_curves.png")
        plt.show()
    
    def plot_performance_heatmap(self, save_plots: bool = True) -> None:
        """Create heatmap showing performance matrix"""
        # Prepare data for heatmap
        models = ['A3C Global'] + [f'Worker {i}' for i in range(5)]
        environments = list(range(5))
        
        performance_matrix = np.full((len(models), len(environments)), np.nan)
        
        for env_idx, env_id in enumerate(environments):
            # A3C Global
            if not np.isnan(self.comparison_results.iloc[env_id]['a3c_mean']):
                performance_matrix[0, env_idx] = self.comparison_results.iloc[env_id]['a3c_mean']
            
            # Individual workers
            for worker_id in range(5):
                worker_mean = self.comparison_results.iloc[env_id][f'individual_w{worker_id}_mean']
                if not np.isnan(worker_mean):
                    performance_matrix[worker_id + 1, env_idx] = worker_mean
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(performance_matrix, 
                   xticklabels=[f'Env {i}' for i in environments],
                   yticklabels=models,
                   annot=True, fmt='.1f', cmap='RdYlGn',
                   center=np.nanmean(performance_matrix),
                   cbar_kws={'label': 'Average Total Reward'})
        
        plt.title('Performance Heatmap: Models × Environments')
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('performance_heatmap.png', dpi=300, bbox_inches='tight')
            print("Saved: performance_heatmap.png")
        plt.show()
    
    def plot_distribution_analysis(self, save_plots: bool = True) -> None:
        """Create box plots for reward distribution analysis"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for env_id in range(5):
            ax = axes[env_id]
            
            plot_data = []
            labels = []
            
            # A3C global data
            a3c_key = f"a3c_global_env{env_id}"
            if a3c_key in self.episode_metrics:
                plot_data.append(self.episode_metrics[a3c_key]['total_reward'].values)
                labels.append('A3C Global')
            
            # Individual worker data (combined)
            individual_combined = []
            for worker_id in range(5):
                ind_key = f"individual_w{worker_id}_env{env_id}"
                if ind_key in self.episode_metrics:
                    individual_combined.extend(
                        self.episode_metrics[ind_key]['total_reward'].values
                    )
            
            if individual_combined:
                plot_data.append(individual_combined)
                labels.append('Individual Combined')
            
            if plot_data:
                ax.boxplot(plot_data, labels=labels, patch_artist=True)
                ax.set_title(f'Environment {env_id}')
                ax.set_ylabel('Total Reward')
                ax.grid(True, alpha=0.3)
        
        # Remove empty subplot
        axes[5].remove()
        
        plt.tight_layout()
        if save_plots:
            plt.savefig('distribution_analysis.png', dpi=300, bbox_inches='tight')
            print("Saved: distribution_analysis.png")
        plt.show()
    
    def generate_detailed_report(self, save_csv: bool = True) -> None:
        """Generate comprehensive performance report"""
        print("\n" + "="*80)
        print("ENVIRONMENT PERFORMANCE COMPARISON REPORT")
        print("="*80)
        
        for _, row in self.comparison_results.iterrows():
            env_id = int(row['environment'])
            print(f"\n🌍 ENVIRONMENT {env_id}")
            print("-" * 50)
            
            if not np.isnan(row['a3c_mean']):
                print(f"A3C Global:")
                print(f"  Mean Reward: {row['a3c_mean']:.2f} ± {row['a3c_std']:.2f}")
                print(f"  Episodes: {int(row['a3c_episodes'])}")
            else:
                print("A3C Global: No data available")
            
            if not np.isnan(row['individual_combined_mean']):
                print(f"\nIndividual Workers:")
                print(f"  Combined Mean: {row['individual_combined_mean']:.2f} ± {row['individual_combined_std']:.2f}")
                print(f"  Average of Means: {row['individual_avg_mean']:.2f}")
                print(f"  Best Worker: {row['best_individual_mean']:.2f}")
                print(f"  Worst Worker: {row['worst_individual_mean']:.2f}")
                
                # Individual worker details
                for worker_id in range(5):
                    worker_mean = row[f'individual_w{worker_id}_mean']
                    worker_std = row[f'individual_w{worker_id}_std']
                    if not np.isnan(worker_mean):
                        print(f"    Worker {worker_id}: {worker_mean:.2f} ± {worker_std:.2f}")
            else:
                print("\nIndividual Workers: No data available")
            
            # Statistical significance
            if not np.isnan(row['p_value']):
                print(f"\n📊 Statistical Analysis:")
                print(f"  T-statistic: {row['t_statistic']:.3f}")
                print(f"  P-value: {row['p_value']:.3f}")
                print(f"  Significant: {'Yes' if row['significant'] else 'No'} (α=0.05)")
                print(f"  A3C Improvement: {row['improvement_pct']:.1f}%")
                
                if row['improvement_pct'] > 0:
                    print("  📈 A3C Global outperforms Individual Workers")
                else:
                    print("  📉 Individual Workers outperform A3C Global")
        
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        
        # Overall comparison
        valid_improvements = self.comparison_results['improvement_pct'].dropna()
        if len(valid_improvements) > 0:
            print(f"Average A3C Improvement: {valid_improvements.mean():.1f}%")
            print(f"Environments where A3C wins: {(valid_improvements > 0).sum()}/{len(valid_improvements)}")
            print(f"Best A3C improvement: {valid_improvements.max():.1f}%")
            print(f"Worst A3C performance: {valid_improvements.min():.1f}%")
        
        significant_results = self.comparison_results[self.comparison_results['significant'] == True]
        print(f"Statistically significant differences: {len(significant_results)}/{len(self.comparison_results)}")
        
        if save_csv:
            self.comparison_results.to_csv('performance_comparison_results.csv', index=False)
            print(f"\n💾 Detailed results saved to: performance_comparison_results.csv")
    
    def run_complete_analysis(self, save_outputs: bool = True) -> None:
        """Run complete analysis pipeline"""
        self.load_data()
        if not self.raw_data:
            print("❌ No data files found. Please ensure CSV files are in the current directory.")
            return
        
        self.calculate_episode_metrics()
        self.perform_comparison_analysis()
        
        print("\nGenerating visualizations...")
        self.plot_environment_comparison(save_outputs)
        self.plot_episode_curves(save_outputs)
        self.plot_performance_heatmap(save_outputs)
        self.plot_distribution_analysis(save_outputs)
        
        print("\nGenerating report...")
        self.generate_detailed_report(save_outputs)
        
        print(f"\n✅ Analysis complete! Generated {'files saved' if save_outputs else 'visualization complete'}")

def main():
    """Main execution function"""
    analyzer = EnvironmentPerformanceAnalyzer()
    analyzer.run_complete_analysis(save_outputs=True)

if __name__ == "__main__":
    main()