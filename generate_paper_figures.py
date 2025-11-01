#!/usr/bin/env python3
"""
Generate publication-quality figures for ablation study paper
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Set publication style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("colorblind")
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 16

# Output directory
output_dir = Path('paper_figures')
output_dir.mkdir(exist_ok=True)

# Data from generalization summary
data = {
    'Configuration': ['Baseline\n(RNN+LN, 5w)', 'Few Workers\n(3)', 'Many Workers\n(10)',
                     'No RNN\n(LN only)', 'No LayerNorm\n(RNN only)'],
    'A3C': [49.57, 44.13, 50.17, 52.94, 50.58],
    'Individual': [38.22, 43.19, 42.95, 46.76, 39.58],
    'Gap_Pct': [29.7, 2.2, 16.8, 13.2, 27.8],
    'A3C_Std': [14.35, 8.50, 13.04, 19.31, 18.27],
    'Ind_Std': [16.24, 14.95, 13.71, 10.14, 17.97],
    'A3C_Worst': [31.72, 32.27, 29.39, 32.18, 30.29],
    'Ind_Worst': [1.25, 4.97, 2.69, 29.11, 0.00]
}

df = pd.DataFrame(data)

# ============================================================================
# Figure 1: Main Result - Worker Count Impact
# ============================================================================
def plot_worker_impact():
    fig, ax = plt.subplots(figsize=(10, 6))

    worker_data = df[df['Configuration'].str.contains('Workers|Baseline')]
    x_labels = ['3 Workers', '5 Workers\n(Baseline)', '10 Workers']
    x_pos = np.arange(len(x_labels))
    gaps = [2.2, 29.7, 16.8]

    colors = ['#d62728', '#2ca02c', '#ff7f0e']  # Red, Green, Orange
    bars = ax.bar(x_pos, gaps, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Highlight the optimal
    bars[1].set_edgecolor('black')
    bars[1].set_linewidth(3)

    ax.set_ylabel('A3C Advantage over Individual (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Workers', fontsize=14, fontweight='bold')
    ax.set_title('Impact of Worker Diversity on A3C Performance', fontsize=16, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0, 35)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for i, (bar, gap) in enumerate(zip(bars, gaps)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{gap:.1f}%',
                ha='center', va='bottom', fontsize=14, fontweight='bold')

    # Add annotation for optimal
    ax.annotate('Optimal', xy=(1, 29.7), xytext=(1, 33),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'),
                fontsize=12, fontweight='bold', ha='center')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_worker_impact.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig1_worker_impact.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'fig1_worker_impact.pdf'}")
    plt.close()

# ============================================================================
# Figure 2: Performance Comparison with Error Bars
# ============================================================================
def plot_performance_comparison():
    fig, ax = plt.subplots(figsize=(12, 6))

    x_pos = np.arange(len(df))
    width = 0.35

    # Plot bars with error bars
    bars1 = ax.bar(x_pos - width/2, df['A3C'], width,
                   yerr=df['A3C_Std'], label='A3C',
                   color='#1f77b4', alpha=0.8, capsize=5)
    bars2 = ax.bar(x_pos + width/2, df['Individual'], width,
                   yerr=df['Ind_Std'], label='Individual',
                   color='#ff7f0e', alpha=0.8, capsize=5)

    ax.set_ylabel('Generalization Score', fontsize=14, fontweight='bold')
    ax.set_xlabel('Configuration', fontsize=14, fontweight='bold')
    ax.set_title('Generalization Performance: A3C vs Individual Learning', fontsize=16, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df['Configuration'], rotation=15, ha='right')
    ax.legend(loc='upper left', fontsize=12)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 75)

    # Highlight baseline
    bars1[0].set_edgecolor('black')
    bars1[0].set_linewidth(2.5)
    bars2[0].set_edgecolor('black')
    bars2[0].set_linewidth(2.5)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_performance_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig2_performance_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'fig2_performance_comparison.pdf'}")
    plt.close()

# ============================================================================
# Figure 3: Worst-Case Performance (Robustness)
# ============================================================================
def plot_worst_case():
    fig, ax = plt.subplots(figsize=(12, 6))

    x_pos = np.arange(len(df))
    width = 0.35

    bars1 = ax.bar(x_pos - width/2, df['A3C_Worst'], width,
                   label='A3C Worst-Case', color='#1f77b4', alpha=0.8)
    bars2 = ax.bar(x_pos + width/2, df['Ind_Worst'], width,
                   label='Individual Worst-Case', color='#ff7f0e', alpha=0.8)

    ax.set_ylabel('Worst-Case Performance', fontsize=14, fontweight='bold')
    ax.set_xlabel('Configuration', fontsize=14, fontweight='bold')
    ax.set_title('Robustness Comparison: A3C Prevents Catastrophic Failures', fontsize=16, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df['Configuration'], rotation=15, ha='right')
    ax.legend(loc='upper left', fontsize=12)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 40)

    # Add horizontal line at safe threshold
    ax.axhline(y=20, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Safety Threshold')

    # Highlight catastrophic failures
    for i, (a3c_worst, ind_worst) in enumerate(zip(df['A3C_Worst'], df['Ind_Worst'])):
        if ind_worst < 10:
            bars2[i].set_color('red')
            bars2[i].set_alpha(1.0)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_worst_case.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig3_worst_case.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'fig3_worst_case.pdf'}")
    plt.close()

# ============================================================================
# Figure 4: Component Contribution Analysis
# ============================================================================
def plot_component_contribution():
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate contributions (impact on gap)
    baseline_gap = 29.7

    contributions = {
        'Worker Diversity\n(5→3)': baseline_gap - 2.2,  # 27.5%
        'RNN': baseline_gap - 13.2,  # 16.5%
        'LayerNorm': baseline_gap - 27.8,  # 1.9%
    }

    components = list(contributions.keys())
    values = list(contributions.values())
    colors = ['#2ca02c', '#1f77b4', '#ff7f0e']

    bars = ax.barh(components, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    ax.set_xlabel('Contribution to A3C Advantage (%)', fontsize=14, fontweight='bold')
    ax.set_title('Component Contribution Analysis', fontsize=16, fontweight='bold')
    ax.set_xlim(0, 30)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    # Add value labels
    for bar, val in zip(bars, values):
        width = bar.get_width()
        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%',
                ha='left', va='center', fontsize=14, fontweight='bold')

    # Add percentage of total
    total = sum(values)
    for bar, val in zip(bars, values):
        pct = (val / baseline_gap) * 100
        ax.text(1, bar.get_y() + bar.get_height()/2,
                f'({pct:.0f}%)',
                ha='left', va='center', fontsize=10, style='italic')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig4_component_contribution.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig4_component_contribution.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'fig4_component_contribution.pdf'}")
    plt.close()

# ============================================================================
# Figure 5: Gap Percentage Comparison
# ============================================================================
def plot_gap_comparison():
    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort by gap percentage
    sorted_df = df.sort_values('Gap_Pct', ascending=True)

    colors = ['#d62728' if x < 10 else '#ff7f0e' if x < 20 else '#2ca02c'
              for x in sorted_df['Gap_Pct']]

    bars = ax.barh(sorted_df['Configuration'], sorted_df['Gap_Pct'],
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Highlight baseline
    baseline_idx = list(sorted_df['Configuration']).index('Baseline\n(RNN+LN, 5w)')
    bars[baseline_idx].set_edgecolor('black')
    bars[baseline_idx].set_linewidth(3)

    ax.set_xlabel('A3C Advantage over Individual (%)', fontsize=14, fontweight='bold')
    ax.set_title('A3C Performance Advantage Across Configurations', fontsize=16, fontweight='bold')
    ax.set_xlim(0, 35)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    # Add value labels
    for bar, gap in zip(bars, sorted_df['Gap_Pct']):
        width = bar.get_width()
        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                f'{gap:.1f}%',
                ha='left', va='center', fontsize=12, fontweight='bold')

    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#d62728', alpha=0.8, label='Weak (<10%)'),
        Patch(facecolor='#ff7f0e', alpha=0.8, label='Moderate (10-20%)'),
        Patch(facecolor='#2ca02c', alpha=0.8, label='Strong (>20%)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig5_gap_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig5_gap_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'fig5_gap_comparison.pdf'}")
    plt.close()

# ============================================================================
# Generate LaTeX Table
# ============================================================================
def generate_latex_table():
    latex_file = output_dir / 'table1_results.tex'

    latex = r"""\begin{table}[t]
\centering
\caption{Generalization Performance Comparison of A3C Configurations}
\label{tab:ablation_results}
\begin{tabular}{lcccccc}
\toprule
\textbf{Configuration} & \textbf{A3C} & \textbf{Individual} & \textbf{Gap} & \textbf{Gap \%} & \textbf{A3C Worst} & \textbf{Ind Worst} \\
\midrule
"""

    for _, row in df.iterrows():
        config = row['Configuration'].replace('\n', ' ')
        latex += f"{config} & "
        latex += f"{row['A3C']:.2f} $\pm$ {row['A3C_Std']:.2f} & "
        latex += f"{row['Individual']:.2f} $\pm$ {row['Ind_Std']:.2f} & "
        latex += f"+{row['A3C'] - row['Individual']:.2f} & "
        latex += f"\\textbf{{{row['Gap_Pct']:.1f}\%}} & "
        latex += f"{row['A3C_Worst']:.2f} & "
        latex += f"{row['Ind_Worst']:.2f} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Performance measured on velocity sweep (5-100 km/h, 9 velocities, 100 episodes per velocity).
\item Baseline configuration: 5 workers, RNN with LayerNorm, hidden dimension 128.
\item \textbf{Gap \%}: Percentage improvement of A3C over Individual learning.
\end{tablenotes}
\end{table}
"""

    with open(latex_file, 'w') as f:
        f.write(latex)

    print(f"✓ Saved: {latex_file}")

# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':
    print("=" * 80)
    print("Generating Paper Figures")
    print("=" * 80)
    print()

    plot_worker_impact()
    plot_performance_comparison()
    plot_worst_case()
    plot_component_contribution()
    plot_gap_comparison()
    generate_latex_table()

    print()
    print("=" * 80)
    print(f"✓ All figures saved to: {output_dir}/")
    print("=" * 80)
    print()
    print("Generated files:")
    print("  - fig1_worker_impact.pdf (Main result)")
    print("  - fig2_performance_comparison.pdf")
    print("  - fig3_worst_case.pdf (Robustness)")
    print("  - fig4_component_contribution.pdf")
    print("  - fig5_gap_comparison.pdf")
    print("  - table1_results.tex (LaTeX table)")
    print()
    print("Recommended for paper:")
    print("  - Figure 1: Worker Impact (Main result)")
    print("  - Figure 3: Worst-Case (Robustness)")
    print("  - Figure 4: Component Contribution")
    print("  - Table 1: Complete Results")
