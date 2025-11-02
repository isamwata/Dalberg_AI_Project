"""
Data Wrangling Script
Functional approach for data loading and processing
"""

import pandas as pd


def load_csv_data(file_path: str) -> pd.DataFrame:
    """
    Load a CSV file and return it as a pandas DataFrame.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame containing the CSV data
    """
    return pd.read_csv(file_path)


def display_dataframe_info(df: pd.DataFrame, title: str, num_rows: int = 5) -> None:
    """
    Display information about a DataFrame.
    
    Args:
        df: DataFrame to display
        title: Title for the display section
        num_rows: Number of rows to display (default: 5)
    """
    print(f"=== {title} ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst {num_rows} rows:")
    print(df.head(num_rows))
    print("\n" + "="*50 + "\n")


def subset_by_province(df: pd.DataFrame, province: str, province_col: str = "Province") -> pd.DataFrame:
    """
    Subset a DataFrame by province.
    
    Args:
        df: DataFrame to subset
        province: Name of the province to filter
        province_col: Name of the province column (default: "Province")
        
    Returns:
        Filtered DataFrame
    """
    return df[df[province_col] == province]


def subset_by_level(df: pd.DataFrame, level: str, level_col: str = "Level_") -> pd.DataFrame:
    """
    Subset a DataFrame by school level.
    
    Args:
        df: DataFrame to subset
        level: Name of the school level to filter (e.g., "PRIMARY SCHOOL")
        level_col: Name of the level column (default: "Level_")
        
    Returns:
        Filtered DataFrame
    """
    return df[df[level_col] == level]


def get_nairobi_subcounties() -> list:
    """
    Get list of all official Nairobi subcounties/constituencies.
    Includes NAIROBI CITY (total) and all 17 official subcounties.
    
    Returns:
        List of Nairobi subcounty names
    """
    return [
        "NAIROBI CITY",    # Total for Nairobi City County
        "WESTLANDS",
        "DAGORETTI NORTH",
        "DAGORETTI SOUTH",
        "LANG'ATA",
        "KIBRA",
        "ROYSAMBU",
        "KASARANI",
        "RUARAKA",
        "EMBAKASI NORTH",
        "EMBAKASI CENTRAL",
        "EMBAKASI EAST",
        "EMBAKASI WEST",
        "EMBAKASI SOUTH",
        "MAKADARA",
        "KAMUKUNJI",
        "STAREHE",
        "MATHARE"
    ]


def subset_by_subcounties(df: pd.DataFrame, subcounties: list, subcounty_col: str = "County/ Sub-County") -> pd.DataFrame:
    """
    Subset a DataFrame by subcounties.
    
    Args:
        df: DataFrame to subset
        subcounties: List of subcounty names to filter
        subcounty_col: Name of the subcounty column (default: "County/ Sub-County")
        
    Returns:
        Filtered DataFrame
    """
    return df[df[subcounty_col].isin(subcounties)]


def normalize_division_name(division: str) -> str:
    """
    Normalize division/subcounty names to match between datasets.
    
    Args:
        division: Division name from school data
        
    Returns:
        Normalized name that matches population data
    """
    name_mapping = {
        "KIBERA": "KIBRA",  # Schools use KIBERA, population uses KIBRA
    }
    return name_mapping.get(division, division)


def merge_schools_population(school_df: pd.DataFrame, pop_df: pd.DataFrame, 
                             school_division_col: str = "Division",
                             pop_subcounty_col: str = "County/ Sub-County") -> pd.DataFrame:
    """
    Merge school data with population data based on Division = Subcounty.
    
    Args:
        school_df: DataFrame with school data (should have Division column)
        pop_df: DataFrame with population data (should have County/ Sub-County column)
        school_division_col: Name of division column in school data
        pop_subcounty_col: Name of subcounty column in population data
        
    Returns:
        Merged DataFrame with school statistics aggregated by division
    """
    # Normalize division names in school data
    school_df = school_df.copy()
    school_df['Normalized_Division'] = school_df[school_division_col].apply(normalize_division_name)
    
    # Aggregate school data by division
    school_stats = school_df.groupby('Normalized_Division').agg({
        'FID': 'count',  # Number of schools
        'TotalEnrol': 'sum',  # Total enrollment
        'TotalBoys': 'sum',
        'TotalGirls': 'sum',
        'No_Classrm': 'sum',
        'TeachersTo': 'sum',
        'Latitude': 'mean',  # Centroid
        'Longitude': 'mean'
    }).reset_index()
    
    school_stats.columns = ['Division', 'School_Count', 'Total_Enrollment', 
                           'Total_Boys', 'Total_Girls', 'Total_Classrooms', 
                           'Total_Teachers', 'Latitude', 'Longitude']
    
    # Merge with population data
    merged_df = school_stats.merge(
        pop_df,
        left_on='Division',
        right_on=pop_subcounty_col,
        how='outer',
        indicator=True
    )
    
    return merged_df


def display_column_details(df: pd.DataFrame, title: str = "Column Details") -> None:
    """
    Display detailed information about columns in a DataFrame.
    
    Args:
        df: DataFrame to analyze
        title: Title for the display section
    """
    print(f"=== {title} ===")
    print(f"\nTotal Columns: {len(df.columns)}")
    print(f"\nColumn Names:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")
    
    print(f"\n{'='*50}")
    print("Sample Data for Each Column:")
    print(f"{'='*50}\n")
    
    # Display the full DataFrame with all columns visible
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    print(df.to_string())
    
    print(f"\n{'='*50}")
    print("Column Data Types:")
    print(df.dtypes)
    print(f"\n{'='*50}\n")


def main():
    """Main function to orchestrate data loading and processing."""
    # Load the CSV files
    school_data = load_csv_data('kenya_primary_schools.csv')
    pop_data = load_csv_data('distribution-of-population-age-3-years-and-above-by-school-attendance.csv')
    
    # Display initial data info
    display_dataframe_info(school_data, "School Data Info")
    display_dataframe_info(pop_data, "Population Data Info")
    
    # Subset school data to NAIROBI province
    school_data = subset_by_province(school_data, "NAIROBI")
    
    # Subset to Primary schools only
    school_data = subset_by_level(school_data, "PRIMARY SCHOOL")
    
    print("=== School Data (NAIROBI Primary Schools only) ===")
    print(f"Shape: {school_data.shape}")
    print(f"Number of primary schools in Nairobi: {len(school_data)}")
    print("\nFirst few rows:")
    print(school_data.head())
    print("\n" + "="*50 + "\n")
    
    # Subset population data to Nairobi county and all subcounties
    nairobi_subcounties = get_nairobi_subcounties()
    pop_data = subset_by_subcounties(pop_data, nairobi_subcounties)
    
    print("=== Population Data (Nairobi City and Subcounties) ===")
    print(f"Shape: {pop_data.shape}")
    print(f"Number of rows for Nairobi: {len(pop_data)}")
    print(f"All columns preserved: {list(pop_data.columns)}")
    print("\nFirst few rows (all columns):")
    print(pop_data.head(20))
    
    # Display detailed column information for Nairobi population data
    display_column_details(pop_data, "Nairobi Subcounty Population Data - Column Details")
    
    # Merge school and population data
    print("\n" + "="*50)
    print("MERGING SCHOOL AND POPULATION DATA")
    print("="*50 + "\n")
    
    merged_data = merge_schools_population(school_data, pop_data)
    
    print("=== Merged Data (Schools + Population) ===")
    print(f"Shape: {merged_data.shape}")
    print(f"\nMatching Status:")
    print(merged_data['_merge'].value_counts())
    
    print("\n=== Merged Data Preview ===")
    # Show key columns
    key_cols = ['Division', 'County/ Sub-County', 'School_Count', 'Total_Enrollment', 
                'Total*', 'At School/ Learning Institution', '_merge']
    available_cols = [col for col in key_cols if col in merged_data.columns]
    print(merged_data[available_cols].to_string())
    
    print("\n=== Divisions with both School and Population Data ===")
    matched = merged_data[merged_data['_merge'] == 'both']
    print(f"Number of matched divisions: {len(matched)}")
    if len(matched) > 0:
        print(matched[['Division', 'School_Count', 'Total_Enrollment', 'Total*']].to_string())
    
    print("\n=== Divisions only in School Data ===")
    schools_only = merged_data[merged_data['_merge'] == 'left_only']
    print(f"Number of divisions: {len(schools_only)}")
    if len(schools_only) > 0:
        print(schools_only[['Division', 'School_Count', 'Total_Enrollment']].to_string())
    
    print("\n=== Subcounties only in Population Data ===")
    pop_only = merged_data[merged_data['_merge'] == 'right_only']
    print(f"Number of subcounties: {len(pop_only)}")
    if len(pop_only) > 0:
        print(pop_only[['County/ Sub-County', 'Total*']].to_string())
    
    # Display school attendance metrics per division
    print("\n" + "="*70)
    print("SCHOOL ATTENDANCE METRICS BY DIVISION")
    print("="*70 + "\n")
    
    attendance_cols = [
        'Division',
        'County/ Sub-County',
        'Total*',
        'At School/ Learning Institution',
        'Left School/ Learning Institution After Completion',
        'Left School/ Learning Institution Before Completion',
        'Never Been to School/ Learning Institution',
        'School_Count',
        'Total_Enrollment'
    ]
    
    # Filter for matched divisions only
    matched_data = merged_data[merged_data['_merge'] == 'both'].copy()
    
    if len(matched_data) > 0:
        # Select relevant columns that exist
        available_cols = [col for col in attendance_cols if col in matched_data.columns]
        attendance_summary = matched_data[available_cols].copy()
        
        # Sort by Division
        attendance_summary = attendance_summary.sort_values('Division')
        
        print("=== School Attendance Metrics for Matched Divisions ===")
        print(attendance_summary.to_string(index=False))
        
        # Calculate ratios and additional metrics
        print("\n" + "="*70)
        print("ADDITIONAL METRICS (Per Division)")
        print("="*70 + "\n")
        
        # Create summary with calculated metrics
        summary_df = matched_data[['Division']].copy()
        
        # Add population metrics if available
        if 'Total*' in matched_data.columns:
            summary_df['Total_Population'] = matched_data['Total*']
        if 'At School/ Learning Institution' in matched_data.columns:
            summary_df['At_School'] = matched_data['At School/ Learning Institution']
        if 'Left School/ Learning Institution After Completion' in matched_data.columns:
            summary_df['Left_After_Completion'] = matched_data['Left School/ Learning Institution After Completion']
        if 'Left School/ Learning Institution Before Completion' in matched_data.columns:
            summary_df['Left_Before_Completion'] = matched_data['Left School/ Learning Institution Before Completion']
        if 'Never Been to School/ Learning Institution' in matched_data.columns:
            summary_df['Never_Been_to_School'] = matched_data['Never Been to School/ Learning Institution']
        if 'School_Count' in matched_data.columns:
            summary_df['Number_of_Schools'] = matched_data['School_Count']
        if 'Total_Enrollment' in matched_data.columns:
            summary_df['Total_School_Enrollment'] = matched_data['Total_Enrollment']
        
        # Calculate percentages if population data is numeric
        print(summary_df.to_string(index=False))
        
        # Show detailed breakdown for each matched division
        print("\n" + "="*70)
        print("DETAILED BREAKDOWN BY DIVISION")
        print("="*70 + "\n")
        
        for idx, row in matched_data.iterrows():
            division = row['Division']
            print(f"\n--- {division} ---")
            print(f"Number of Schools: {row.get('School_Count', 'N/A')}")
            print(f"Total School Enrollment: {row.get('Total_Enrollment', 'N/A'):,.0f}" if pd.notna(row.get('Total_Enrollment')) else f"Total School Enrollment: N/A")
            print(f"\nPopulation Data:")
            print(f"  Total Population: {row.get('Total*', 'N/A')}")
            print(f"  At School/Learning Institution: {row.get('At School/ Learning Institution', 'N/A')}")
            print(f"  Left School After Completion: {row.get('Left School/ Learning Institution After Completion', 'N/A')}")
            print(f"  Left School Before Completion: {row.get('Left School/ Learning Institution Before Completion', 'N/A')}")
            print(f"  Never Been to School: {row.get('Never Been to School/ Learning Institution', 'N/A')}")
    else:
        print("No matched divisions found.")


if __name__ == "__main__":
    main()
