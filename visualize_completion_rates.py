"""Completion Rates Visualization - Generates 3 key charts."""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from data_wrangling import load_csv_data, subset_by_province, subset_by_level, merge_schools_population

def clean_numeric(series): return series.str.replace(',', '').astype(float) if series.dtype == 'object' else series

def calculate_completion_metrics(merged_df):
    df = merged_df[merged_df['_merge'] == 'both'].copy()
    for col in ['Total*', 'At School/ Learning Institution', 'Left School/ Learning Institution After Completion', 'Left School/ Learning Institution Before Completion', 'Never Been to School/ Learning Institution']:
        if col in df.columns: df[col] = clean_numeric(df[col])
    df['Completed_Rate'] = (df['Left School/ Learning Institution After Completion'] / df['Total*']) * 100
    return df

def create_completion_rates_chart(metrics_df, output_file='completion_rates_comparison.png'):
    df = metrics_df.copy()
    df['Total_Pop'], df['Completed'] = clean_numeric(df['Total*']), clean_numeric(df['Left School/ Learning Institution After Completion'])
    df['Completion_Rate'] = (df['Completed'] / df['Total_Pop']) * 100
    df = df.sort_values('Completion_Rate', ascending=True)
    fig, ax = plt.subplots(figsize=(12, 8))
    divisions, rates = df['Division'].tolist(), df['Completion_Rate'].tolist()
    colors = ['#2ecc71' if r >= 50 else '#f39c12' if r >= 45 else '#e74c3c' for r in rates]
    bars = ax.barh(divisions, rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    for i, (bar, rate, div) in enumerate(zip(bars, rates, divisions)):
        comp, total = df[df['Division'] == div]['Completed'].iloc[0], df[df['Division'] == div]['Total_Pop'].iloc[0]
        ax.text(rate + 1, i, f'{rate:.1f}%', va='center', fontweight='bold', fontsize=11)
        ax.text(-2, i, f'({comp:,.0f}/{total:,.0f})', va='center', ha='right', fontsize=9, style='italic', color='gray')
    avg = df['Completion_Rate'].mean()
    from matplotlib.patches import Patch
    ax.axvline(avg, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Average: {avg:.1f}%')
    ax.legend(handles=[Patch(facecolor='#2ecc71', alpha=0.8, label='≥50% (Good)'), Patch(facecolor='#f39c12', alpha=0.8, label='45-50% (Moderate)'), Patch(facecolor='#e74c3c', alpha=0.8, label='<45% (Needs Improvement)'), plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2, label=f'Average: {avg:.1f}%')], loc='lower right', framealpha=0.9, fontsize=10)
    ax.set_xlabel('Completion Rate (%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Division', fontsize=13, fontweight='bold')
    ax.set_title('Primary School Completion Rates by Division\n(Direct Percentage Comparison)', fontsize=15, fontweight='bold', pad=20)
    ax.set_xlim(0, max(rates) * 1.2)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.text(0.02, 0.98, 'Numbers in (): Completed/Total Population', transform=ax.transAxes, fontsize=9, style='italic', verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def create_schools_per_capita_charts(metrics_df):
    df = metrics_df.copy()
    df['Total_Pop'] = clean_numeric(df['Total*'])
    df['Schools_per_10k'], df['Students_per_School'] = (df['School_Count'] / df['Total_Pop']) * 10000, df['Total_Enrollment'] / df['School_Count']
    df['Completed_Rate'] = (clean_numeric(df['Left School/ Learning Institution After Completion']) / df['Total_Pop']) * 100
    df = df.sort_values('Schools_per_10k', ascending=True)
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Schools per Capita Analysis - Nairobi Divisions', fontsize=16, fontweight='bold', y=0.995)
    divisions, colors = df['Division'].tolist(), sns.color_palette("viridis", len(df))
    bars1 = axes[0, 0].barh(divisions, df['Schools_per_10k'], color=colors)
    for i, (bar, val) in enumerate(zip(bars1, df['Schools_per_10k'])): axes[0, 0].text(val, i, f' {val:.2f}', va='center', fontweight='bold', fontsize=10)
    axes[0, 0].axvline(df['Schools_per_10k'].mean(), color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Average: {df["Schools_per_10k"].mean():.2f}')
    axes[0, 0].legend(loc='lower right')
    axes[0, 0].set_xlabel('Number of Schools per 10,000 People', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Division', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('School Density: Schools per 10,000 People', fontsize=13, fontweight='bold')
    axes[0, 0].grid(axis='x', alpha=0.3)
    scatter = axes[0, 1].scatter(df['Total_Pop'], df['Schools_per_10k'], s=df['School_Count']*10, c=range(len(df)), cmap='viridis', alpha=0.7, edgecolors='black', linewidth=1.5)
    for i, row in df.iterrows(): axes[0, 1].annotate(row['Division'], (row['Total_Pop'], row['Schools_per_10k']), xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')
    axes[0, 1].set_xlabel('Total Population', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Schools per 10,000 People', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('School Density vs Population Size\n(Bubble size = Number of Schools)', fontsize=13, fontweight='bold')
    axes[0, 1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}K' if x >= 1000 else f'{x:.0f}'))
    axes[0, 1].grid(True, alpha=0.3)
    bars3 = axes[1, 0].barh(divisions, df['Students_per_School'], color=colors)
    for i, (bar, val) in enumerate(zip(bars3, df['Students_per_School'])): axes[1, 0].text(val, i, f' {val:.0f}', va='center', fontweight='bold', fontsize=10)
    axes[1, 0].axvline(df['Students_per_School'].mean(), color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Average: {df["Students_per_School"].mean():.0f}')
    axes[1, 0].legend(loc='lower right')
    axes[1, 0].set_xlabel('Number of Students per School', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Division', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('School Capacity: Students per School', fontsize=13, fontweight='bold')
    axes[1, 0].grid(axis='x', alpha=0.3)
    axes[1, 1].scatter(df['Schools_per_10k'], df['Completed_Rate'], s=df['School_Count']*10, c=range(len(df)), cmap='plasma', alpha=0.7, edgecolors='black', linewidth=1.5)
    slope, intercept, r_value, _, _ = stats.linregress(df['Schools_per_10k'], df['Completed_Rate'])
    x_line = np.linspace(df['Schools_per_10k'].min(), df['Schools_per_10k'].max(), 100)
    axes[1, 1].plot(x_line, slope * x_line + intercept, 'r--', alpha=0.7, linewidth=2, label=f'R² = {r_value**2:.3f}')
    for i, row in df.iterrows(): axes[1, 1].annotate(row['Division'], (row['Schools_per_10k'], row['Completed_Rate']), xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')
    axes[1, 1].set_xlabel('Schools per 10,000 People', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Completion Rate (%)', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('School Density vs Completion Rate\n(Size = Number of Schools, Line = Correlation)', fontsize=13, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('schools_per_capita_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_grouped_waterfall(metrics_df, output_file='waterfall_all_divisions_grouped.png'):
    divisions = metrics_df['Division'].tolist()
    n_divs, width = len(divisions), 0.85 / len(divisions)
    division_colors = {div: sns.color_palette("husl", n_divs)[i] for i, div in enumerate(divisions)}
    fig, ax = plt.subplots(figsize=(16, 10))
    x = np.arange(5)
    total_pops = [float(str(metrics_df['Total*'].iloc[i]).replace(',', '')) for i in range(len(metrics_df))]
    never = [float(str(metrics_df['Never Been to School/ Learning Institution'].iloc[i]).replace(',', '')) for i in range(len(metrics_df))]
    dropped = [float(str(metrics_df['Left School/ Learning Institution Before Completion'].iloc[i]).replace(',', '')) for i in range(len(metrics_df))]
    completed = [float(str(metrics_df['Left School/ Learning Institution After Completion'].iloc[i]).replace(',', '')) for i in range(len(metrics_df))]
    in_school = [float(str(metrics_df['At School/ Learning Institution'].iloc[i]).replace(',', '')) for i in range(len(metrics_df))]
    for i, div in enumerate(divisions):
        offset = (i - n_divs/2 + 0.5) * width
        x_pos = x + offset
        total_pop, never_val, dropped_val, comp_val, in_school_val = total_pops[i], never[i], dropped[i], completed[i], in_school[i]
        current_bottom = total_pop
        ax.bar(x_pos[0], total_pop, width, color=division_colors[div], edgecolor='black', linewidth=1.2, label=div if i == 0 else '', alpha=0.8)
        ax.bar(x_pos[1], never_val, width, bottom=current_bottom - never_val, color=division_colors[div], edgecolor='black', linewidth=1.2, alpha=0.8)
        current_bottom -= never_val
        ax.bar(x_pos[2], dropped_val, width, bottom=current_bottom - dropped_val, color=division_colors[div], edgecolor='black', linewidth=1.2, alpha=0.8)
        ax.bar(x_pos[3], comp_val, width, color=division_colors[div], edgecolor='black', linewidth=1.2, alpha=0.8)
        ax.bar(x_pos[4], in_school_val, width, color=division_colors[div], edgecolor='black', linewidth=1.2, alpha=0.8)
        ax.text(x_pos[0], total_pop + max(total_pops) * 0.02, div, ha='center', va='bottom', fontsize=9, fontweight='bold', rotation=90)
    ax.set_xlabel('Education Status', fontsize=13, fontweight='bold')
    ax.set_ylabel('Population', fontsize=13, fontweight='bold')
    ax.set_title('Education Status Waterfall Chart - All Divisions Comparison', fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(['Total Population', 'Never Attended', 'Dropped Out', 'Completed', 'Currently in School'], fontsize=11)
    ax.legend(loc='upper left', framealpha=0.9, ncol=2, fontsize=9)
    ax.set_ylim(0, max(total_pops) * 1.2)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}K' if x >= 1000 else f'{x:.0f}'))
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    school_data = subset_by_level(subset_by_province(load_csv_data('kenya_primary_schools.csv'), "NAIROBI"), "PRIMARY SCHOOL")
    pop_data = load_csv_data('distribution-of-population-age-3-years-and-above-by-school-attendance.csv')
    pop_data = pop_data[pop_data['County/ Sub-County'].isin(['NAIROBI CITY', 'DAGORETTI', 'EMBAKASI', 'KAMUKUNJI', 'KASARANI', 'KIBRA', "LANG'ATA", 'MAKADARA', 'MATHARE', 'STAREHE', 'WESTLANDS'])]
    merged_data = merge_schools_population(school_data, pop_data)
    metrics_df = calculate_completion_metrics(merged_data)
    create_completion_rates_chart(metrics_df)
    create_schools_per_capita_charts(metrics_df)
    create_grouped_waterfall(metrics_df)

if __name__ == "__main__":
    main()
