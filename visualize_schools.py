"""School Distribution Visualization for Nairobi - Generates static analysis chart."""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_nairobi_schools(file_path='kenya_primary_schools.csv'):
    df = pd.read_csv(file_path)
    df = df[(df["Province"] == "NAIROBI") & (df["Level_"] == "PRIMARY SCHOOL")].copy()
    df = df.dropna(subset=['Latitude', 'Longitude'])
    return df

def create_division_statistics(df):
    division_stats = df.groupby('Division').agg({'FID': 'count', 'Latitude': ['mean', 'std'], 'Longitude': ['mean', 'std']}).reset_index()
    division_stats.columns = ['Division', 'School_Count', 'Lat_Mean', 'Lat_Std', 'Lon_Mean', 'Lon_Std']
    division_stats = division_stats.sort_values('School_Count', ascending=False)
    return division_stats

def create_static_visualizations(df, stats_df):
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Nairobi Schools Distribution by Division', fontsize=16, fontweight='bold')
    
    # 1. Bar chart of school counts by division
    bars = axes[0, 0].barh(stats_df['Division'], stats_df['School_Count'], color=sns.color_palette("husl", len(stats_df)))
    axes[0, 0].set_xlabel('Number of Schools', fontsize=12)
    axes[0, 0].set_ylabel('Division', fontsize=12)
    axes[0, 0].set_title('Total Schools per Division', fontsize=13, fontweight='bold')
    axes[0, 0].grid(axis='x', alpha=0.3)
    for i, bar in enumerate(bars):
        width = bar.get_width()
        axes[0, 0].text(width, bar.get_y() + bar.get_height()/2, f' {int(width)}', ha='left', va='center', fontweight='bold')
    
    # 2. Scatter plot of schools colored by division
    divisions = sorted(df['Division'].unique())
    colors = sns.color_palette("husl", len(divisions))
    division_color_map = {div: colors[i] for i, div in enumerate(divisions)}
    for division in divisions:
        division_data = df[df['Division'] == division]
        axes[0, 1].scatter(division_data['Longitude'], division_data['Latitude'], label=division, alpha=0.6, s=50, color=division_color_map[division])
    axes[0, 1].set_xlabel('Longitude', fontsize=12)
    axes[0, 1].set_ylabel('Latitude', fontsize=12)
    axes[0, 1].set_title('Geographic Distribution of Schools', fontsize=13, fontweight='bold')
    axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Density plot (2D histogram)
    hb = axes[1, 0].hexbin(df['Longitude'], df['Latitude'], gridsize=20, cmap='YlOrRd', mincnt=1)
    axes[1, 0].set_xlabel('Longitude', fontsize=12)
    axes[1, 0].set_ylabel('Latitude', fontsize=12)
    axes[1, 0].set_title('School Density Heatmap', fontsize=13, fontweight='bold')
    plt.colorbar(hb, ax=axes[1, 0], label='Number of Schools')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Pie chart of school distribution
    wedges, texts, autotexts = axes[1, 1].pie(stats_df['School_Count'], labels=stats_df['Division'], autopct='%1.1f%%', startangle=90, colors=sns.color_palette("husl", len(stats_df)))
    axes[1, 1].set_title('Percentage Distribution by Division', fontsize=13, fontweight='bold')
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(9)
    
    plt.tight_layout()
    plt.savefig('nairobi_schools_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    nairobi_schools = load_nairobi_schools()
    division_stats = create_division_statistics(nairobi_schools)
    create_static_visualizations(nairobi_schools, division_stats)

if __name__ == "__main__":
    main()
