"""
Ablation Study 결과 분석 및 시각화 스크립트
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import argparse

class AblationAnalyzer:
    def __init__(self, study_dir):
        self.study_dir = Path(study_dir)

        # Load data
        self.raw_df = pd.read_csv(self.study_dir / 'raw_results.csv')
        self.summary_df = pd.read_csv(self.study_dir / 'summary_results.csv', index_col=0)

        if (self.study_dir / 'baseline_comparison.csv').exists():
            self.comparison_df = pd.read_csv(self.study_dir / 'baseline_comparison.csv')
        else:
            self.comparison_df = None

        print(f"[INFO] Loaded results from {study_dir}")
        print(f"  Total experiments: {len(self.raw_df)}")
        print(f"  Unique configurations: {self.raw_df['config_name'].nunique()}")

    def plot_performance_comparison(self, top_n=10, save_path=None):
        """성능 비교 바 차트"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Sort by A3C performance
        sorted_df = self.summary_df.sort_values('a3c_final_reward_mean', ascending=True).tail(top_n)

        configs = sorted_df.index
        y_pos = np.arange(len(configs))

        # A3C performance
        ax1 = axes[0]
        a3c_means = sorted_df['a3c_final_reward_mean']
        a3c_stds = sorted_df['a3c_final_reward_std']

        bars = ax1.barh(y_pos, a3c_means, xerr=a3c_stds,
                       color='#3498db', alpha=0.7, capsize=5)

        # Highlight baseline
        if 'baseline' in configs:
            baseline_idx = list(configs).index('baseline')
            bars[baseline_idx].set_color('#e74c3c')
            bars[baseline_idx].set_alpha(1.0)

        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(configs, fontsize=9)
        ax1.set_xlabel('A3C Final Reward (Mean ± Std)', fontsize=12)
        ax1.set_title('Top Configurations (A3C)', fontsize=14, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)

        # Individual performance
        ax2 = axes[1]
        ind_means = sorted_df['individual_final_reward_mean']
        ind_stds = sorted_df['individual_final_reward_std']

        bars = ax2.barh(y_pos, ind_means, xerr=ind_stds,
                       color='#2ecc71', alpha=0.7, capsize=5)

        if 'baseline' in configs:
            bars[baseline_idx].set_color('#e74c3c')
            bars[baseline_idx].set_alpha(1.0)

        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(configs, fontsize=9)
        ax2.set_xlabel('Individual Final Reward (Mean ± Std)', fontsize=12)
        ax2.set_title('Top Configurations (Individual)', fontsize=14, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=180, bbox_inches='tight')
            print(f"[Saved] {save_path}")
        else:
            plt.savefig(self.study_dir / 'performance_comparison.png', dpi=180, bbox_inches='tight')
            print(f"[Saved] {self.study_dir / 'performance_comparison.png'}")

        plt.show()

    def plot_baseline_comparison(self, save_path=None):
        """Baseline 대비 성능 변화"""
        if self.comparison_df is None:
            print("[WARNING] No baseline comparison data available")
            return

        # Sort by A3C delta percentage
        sorted_comp = self.comparison_df.sort_values('a3c_delta_pct', ascending=True)

        # Exclude baseline itself
        sorted_comp = sorted_comp[sorted_comp['config'] != 'baseline']

        fig, ax = plt.subplots(figsize=(12, 8))

        configs = sorted_comp['config']
        y_pos = np.arange(len(configs))
        deltas = sorted_comp['a3c_delta_pct']

        colors = ['#2ecc71' if d > 0 else '#e74c3c' for d in deltas]

        bars = ax.barh(y_pos, deltas, color=colors, alpha=0.7)

        ax.axvline(x=0, color='black', linestyle='-', linewidth=2)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(configs, fontsize=9)
        ax.set_xlabel('Performance Change vs Baseline (%)', fontsize=12)
        ax.set_title('Ablation Study: Change from Baseline (A3C)',
                    fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, (bar, delta) in enumerate(zip(bars, deltas)):
            x_pos = delta + (1 if delta > 0 else -1)
            ax.text(x_pos, i, f'{delta:+.1f}%',
                   va='center', ha='left' if delta > 0 else 'right',
                   fontsize=8)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=180, bbox_inches='tight')
            print(f"[Saved] {save_path}")
        else:
            plt.savefig(self.study_dir / 'baseline_comparison.png', dpi=180, bbox_inches='tight')
            print(f"[Saved] {self.study_dir / 'baseline_comparison.png'}")

        plt.show()

    def plot_metrics_heatmap(self, save_path=None):
        """메트릭 히트맵"""
        # Select metrics
        metrics = [
            'a3c_final_reward_mean',
            'a3c_final_value_loss_mean',
            'a3c_final_policy_loss_mean',
            'individual_final_reward_mean',
            'individual_final_value_loss_mean',
        ]

        available_metrics = [m for m in metrics if m in self.summary_df.columns]

        if not available_metrics:
            print("[WARNING] No metrics available for heatmap")
            return

        # Create heatmap data
        heatmap_data = self.summary_df[available_metrics].copy()

        # Normalize each metric to [0, 1]
        for col in heatmap_data.columns:
            min_val = heatmap_data[col].min()
            max_val = heatmap_data[col].max()
            if max_val > min_val:
                # For losses, invert (lower is better)
                if 'loss' in col:
                    heatmap_data[col] = 1 - (heatmap_data[col] - min_val) / (max_val - min_val)
                else:
                    heatmap_data[col] = (heatmap_data[col] - min_val) / (max_val - min_val)

        # Rename columns for display
        display_names = {
            'a3c_final_reward_mean': 'A3C Reward',
            'a3c_final_value_loss_mean': 'A3C Value Loss',
            'a3c_final_policy_loss_mean': 'A3C Policy Loss',
            'individual_final_reward_mean': 'Ind Reward',
            'individual_final_value_loss_mean': 'Ind Value Loss',
        }

        heatmap_data.rename(columns=display_names, inplace=True)

        fig, ax = plt.subplots(figsize=(10, 12))

        sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn',
                   center=0.5, vmin=0, vmax=1, cbar_kws={'label': 'Normalized Score'},
                   ax=ax, linewidths=0.5)

        ax.set_title('Ablation Study Metrics Heatmap\n(Green=Good, Red=Bad)',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylabel('Configuration', fontsize=12)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=180, bbox_inches='tight')
            print(f"[Saved] {save_path}")
        else:
            plt.savefig(self.study_dir / 'metrics_heatmap.png', dpi=180, bbox_inches='tight')
            print(f"[Saved] {self.study_dir / 'metrics_heatmap.png'}")

        plt.show()

    def statistical_analysis(self):
        """통계적 유의성 분석"""
        print("\n" + "="*80)
        print("STATISTICAL SIGNIFICANCE ANALYSIS")
        print("="*80)

        if 'baseline' not in self.raw_df['config_name'].values:
            print("[WARNING] No baseline results found for comparison")
            return

        baseline_rewards = self.raw_df[self.raw_df['config_name'] == 'baseline']['a3c_final_reward']

        results = []

        for config in self.raw_df['config_name'].unique():
            if config == 'baseline':
                continue

            config_rewards = self.raw_df[self.raw_df['config_name'] == config]['a3c_final_reward']

            if len(config_rewards) < 2:
                continue

            # t-test
            t_stat, p_value = stats.ttest_ind(baseline_rewards, config_rewards)

            # Effect size (Cohen's d)
            mean_diff = config_rewards.mean() - baseline_rewards.mean()
            pooled_std = np.sqrt((baseline_rewards.std()**2 + config_rewards.std()**2) / 2)
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

            results.append({
                'config': config,
                'baseline_mean': baseline_rewards.mean(),
                'config_mean': config_rewards.mean(),
                'mean_diff': mean_diff,
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'significant': p_value < 0.05,
            })

        stats_df = pd.DataFrame(results)
        stats_df = stats_df.sort_values('p_value')

        # Save
        stats_csv = self.study_dir / 'statistical_analysis.csv'
        stats_df.to_csv(stats_csv, index=False)
        print(f"\n[Saved] {stats_csv}")

        # Print significant results
        print("\nStatistically Significant Differences (p < 0.05):")
        significant = stats_df[stats_df['significant']]

        if len(significant) > 0:
            for _, row in significant.iterrows():
                direction = "better" if row['mean_diff'] > 0 else "worse"
                print(f"\n  {row['config']}:")
                print(f"    Mean difference: {row['mean_diff']:+.3f} ({direction} than baseline)")
                print(f"    p-value: {row['p_value']:.4f}")
                print(f"    Cohen's d: {row['cohens_d']:.3f}")
        else:
            print("  None found.")

        return stats_df

    def generate_paper_table(self):
        """논문용 표 생성"""
        print("\n" + "="*80)
        print("PAPER TABLE (LaTeX Format)")
        print("="*80)

        # Select top configurations
        top_configs = self.summary_df.sort_values('a3c_final_reward_mean', ascending=False).head(10)

        print("\n\\begin{table}[h]")
        print("\\centering")
        print("\\caption{Ablation Study Results}")
        print("\\label{tab:ablation}")
        print("\\begin{tabular}{lcccc}")
        print("\\hline")
        print("Configuration & A3C Reward & Individual Reward & $\\Delta$ vs Baseline & p-value \\\\")
        print("\\hline")

        baseline_reward = self.summary_df.loc['baseline', 'a3c_final_reward_mean'] if 'baseline' in self.summary_df.index else None

        for config in top_configs.index:
            row = top_configs.loc[config]
            a3c_mean = row['a3c_final_reward_mean']
            a3c_std = row['a3c_final_reward_std']
            ind_mean = row['individual_final_reward_mean']
            ind_std = row['individual_final_reward_std']

            if baseline_reward and config != 'baseline':
                delta_pct = 100 * (a3c_mean - baseline_reward) / baseline_reward
                delta_str = f"{delta_pct:+.1f}\\%"
            else:
                delta_str = "-"

            print(f"{config} & {a3c_mean:.2f}$\\pm${a3c_std:.2f} & "
                  f"{ind_mean:.2f}$\\pm${ind_std:.2f} & {delta_str} & - \\\\")

        print("\\hline")
        print("\\end{tabular}")
        print("\\end{table}")

    def generate_all_plots(self):
        """모든 시각화 생성"""
        print("\n" + "="*80)
        print("GENERATING ALL VISUALIZATIONS")
        print("="*80)

        self.plot_performance_comparison()
        self.plot_baseline_comparison()
        self.plot_metrics_heatmap()

        print("\n[INFO] All visualizations generated!")

def main():
    parser = argparse.ArgumentParser(description='Analyze ablation study results')
    parser.add_argument('study_dir', type=str,
                       help='Path to ablation study directory (e.g., ablation_results/study_20251029_120000)')
    parser.add_argument('--plots', action='store_true',
                       help='Generate all plots')
    parser.add_argument('--stats', action='store_true',
                       help='Run statistical analysis')
    parser.add_argument('--table', action='store_true',
                       help='Generate LaTeX table for paper')
    parser.add_argument('--all', action='store_true',
                       help='Run all analyses')

    args = parser.parse_args()

    analyzer = AblationAnalyzer(args.study_dir)

    if args.all:
        analyzer.generate_all_plots()
        analyzer.statistical_analysis()
        analyzer.generate_paper_table()
    else:
        if args.plots:
            analyzer.generate_all_plots()
        if args.stats:
            analyzer.statistical_analysis()
        if args.table:
            analyzer.generate_paper_table()

        # Default: show summary
        if not (args.plots or args.stats or args.table):
            print("\n[INFO] Use --plots, --stats, --table, or --all to run analyses")
            print("\nSummary of top 5 configurations:")
            print(analyzer.summary_df.sort_values('a3c_final_reward_mean', ascending=False).head(5))

if __name__ == "__main__":
    main()
