#!/usr/bin/env python3
"""
Generate Paper-Ready Tables and Figures
Creates LaTeX tables and publication-quality figures from ablation results
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

def generate_latex_table_training(summary_csv, output_dir):
    """Generate LaTeX table for training performance"""

    df = pd.read_csv(summary_csv)

    latex_lines = []
    latex_lines.append(r"\begin{table}[t]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\caption{Training Performance Comparison: High-Priority Ablations}")
    latex_lines.append(r"\label{tab:ablation_training}")
    latex_lines.append(r"\begin{tabular}{lccc}")
    latex_lines.append(r"\toprule")
    latex_lines.append(r"Configuration & A3C & Individual & Gap \\")
    latex_lines.append(r"\midrule")

    for _, row in df.iterrows():
        config = row['Description'] if row['Description'] else row['Ablation']
        a3c = f"{row['A3C_Mean']:.2f} $\\pm$ {row['A3C_Std']:.2f}"
        individual = f"{row['Individual_Mean']:.2f} $\\pm$ {row['Individual_Std']:.2f}"
        gap = f"{row['Gap']:+.2f} ({row['Gap_Pct']:+.1f}\\%)"

        latex_lines.append(f"{config} & {a3c} & {individual} & {gap} \\\\")

    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\end{table}")

    latex_file = Path(output_dir) / "table_training_performance.tex"
    with open(latex_file, 'w') as f:
        f.write('\n'.join(latex_lines))

    print(f"Generated LaTeX table: {latex_file}")
    return latex_file


def generate_bar_chart(summary_csv, output_dir):
    """Generate bar chart comparing A3C vs Individual for each ablation"""

    df = pd.read_csv(summary_csv)

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(df))
    width = 0.35

    # A3C bars
    ax.bar(x - width/2, df['A3C_Mean'], width, yerr=df['A3C_Std'],
           label='A3C', capsize=5, alpha=0.8)

    # Individual bars
    ax.bar(x + width/2, df['Individual_Mean'], width, yerr=df['Individual_Std'],
           label='Individual', capsize=5, alpha=0.8)

    # Labels and formatting
    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Mean Episode Reward', fontsize=12)
    ax.set_title('Training Performance: A3C vs Individual Learning', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([desc if desc else abl for desc, abl in
                        zip(df['Description'], df['Ablation'])],
                       rotation=15, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    output_file = Path(output_dir) / "training_performance_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_file.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Generated bar chart: {output_file}")

    plt.close()
    return output_file


def generate_gap_analysis(summary_csv, output_dir):
    """Generate gap analysis chart showing A3C advantage"""

    df = pd.read_csv(summary_csv)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate relative gap compared to baseline
    baseline_gap = df.iloc[0]['Gap']
    relative_gaps = [(row['Gap'] / baseline_gap) * 100 for _, row in df.iterrows()]

    colors = ['green' if gap >= 100 else 'orange' if gap >= 50 else 'red'
              for gap in relative_gaps]

    bars = ax.barh(range(len(df)), relative_gaps, color=colors, alpha=0.7)

    # Add baseline reference line
    ax.axvline(x=100, color='black', linestyle='--', linewidth=2, label='Baseline')

    # Labels
    ax.set_xlabel('A3C Advantage (% of Baseline)', fontsize=12)
    ax.set_ylabel('Configuration', fontsize=12)
    ax.set_title('A3C Advantage Across Ablations', fontsize=14, fontweight='bold')
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([desc if desc else abl for desc, abl in
                        zip(df['Description'], df['Ablation'])])

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, relative_gaps)):
        ax.text(val + 2, i, f'{val:.1f}%', va='center', fontsize=10)

    ax.legend(fontsize=11)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    output_file = Path(output_dir) / "a3c_advantage_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_file.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Generated gap analysis chart: {output_file}")

    plt.close()
    return output_file


def generate_summary_report(summary_csv, output_dir):
    """Generate markdown summary report"""

    df = pd.read_csv(summary_csv)

    lines = []
    lines.append("# High-Priority Ablation Study Results")
    lines.append("")
    lines.append(f"**Analysis Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## Training Performance Summary")
    lines.append("")
    lines.append("| Configuration | A3C | Individual | Gap | Gap % |")
    lines.append("|---------------|-----|------------|-----|-------|")

    for _, row in df.iterrows():
        config = row['Description'] if row['Description'] else row['Ablation']
        a3c = f"{row['A3C_Mean']:.2f} ± {row['A3C_Std']:.2f}"
        individual = f"{row['Individual_Mean']:.2f} ± {row['Individual_Std']:.2f}"
        gap = f"{row['Gap']:+.2f}"
        gap_pct = f"{row['Gap_Pct']:+.1f}%"

        lines.append(f"| {config} | {a3c} | {individual} | {gap} | {gap_pct} |")

    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## Key Findings")
    lines.append("")

    baseline_gap = df.iloc[0]['Gap']

    for idx, row in df.iloc[1:].iterrows():
        ablation_gap = row['Gap']
        gap_reduction = baseline_gap - ablation_gap
        gap_reduction_pct = (gap_reduction / baseline_gap) * 100

        lines.append(f"### {row['Description']}")
        lines.append("")
        lines.append(f"- **A3C Performance**: {row['A3C_Mean']:.2f} ± {row['A3C_Std']:.2f}")
        lines.append(f"- **Individual Performance**: {row['Individual_Mean']:.2f} ± {row['Individual_Std']:.2f}")
        lines.append(f"- **Gap**: {ablation_gap:+.2f} ({row['Gap_Pct']:+.1f}%)")
        lines.append(f"- **Gap Reduction vs Baseline**: {gap_reduction:.2f} ({gap_reduction_pct:.1f}%)")
        lines.append("")

        if gap_reduction > baseline_gap * 0.5:
            lines.append("**Impact**: ⚠️ CRITICAL - This component is essential for A3C's advantage")
        elif gap_reduction > baseline_gap * 0.25:
            lines.append("**Impact**: ⚡ IMPORTANT - This component significantly contributes to A3C's advantage")
        else:
            lines.append("**Impact**: ℹ️ Minor - This component has limited impact on A3C's advantage")

        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Conclusions")
    lines.append("")
    lines.append("Based on the ablation study results, the following components are ranked by importance:")
    lines.append("")

    # Rank by gap reduction
    ablation_impacts = []
    for idx, row in df.iloc[1:].iterrows():
        gap_reduction = baseline_gap - row['Gap']
        gap_reduction_pct = (gap_reduction / baseline_gap) * 100
        ablation_impacts.append({
            'name': row['Description'],
            'gap_reduction': gap_reduction,
            'gap_reduction_pct': gap_reduction_pct
        })

    ablation_impacts.sort(key=lambda x: x['gap_reduction'], reverse=True)

    for i, impact in enumerate(ablation_impacts, 1):
        lines.append(f"{i}. **{impact['name']}**: "
                    f"{impact['gap_reduction']:.2f} reduction ({impact['gap_reduction_pct']:.1f}%)")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Next Steps")
    lines.append("")
    lines.append("1. Run generalization tests for all ablations")
    lines.append("2. Compare generalization performance (velocity sweep)")
    lines.append("3. Generate final paper-ready figures and tables")
    lines.append("4. Write discussion section highlighting critical components")

    report_file = Path(output_dir) / "ABLATION_SUMMARY_REPORT.md"
    with open(report_file, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Generated summary report: {report_file}")
    return report_file


def main():
    analysis_dir = "ablation_results/analysis"
    paper_dir = Path(analysis_dir) / "paper_tables"
    os.makedirs(paper_dir, exist_ok=True)

    summary_csv = Path(analysis_dir) / "high_priority_training_summary.csv"

    if not summary_csv.exists():
        print(f"Error: Summary CSV not found: {summary_csv}")
        print("Please run analyze_high_priority_ablations.py first")
        return 1

    print("="*80)
    print("Generating Paper-Ready Tables and Figures")
    print("="*80)
    print()

    # Generate outputs
    generate_latex_table_training(summary_csv, paper_dir)
    generate_bar_chart(summary_csv, paper_dir)
    generate_gap_analysis(summary_csv, paper_dir)
    generate_summary_report(summary_csv, analysis_dir)

    print()
    print("="*80)
    print("All outputs generated successfully!")
    print(f"Output directory: {paper_dir}")
    print("="*80)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
