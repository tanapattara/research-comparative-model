import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import glob

def create_box_plot_by_year(plot_df, station_name, file_name):
    """Create a box and whisker plot grouped by year"""
    # Extract year from date and filter out missing/zero values
    plot_df['year'] = plot_df['date'].dt.year
    plot_df = plot_df[(plot_df['water_level'].notna()) & (plot_df['water_level'] != 0)]
    
    if len(plot_df) > 0:
        # Group by year
        years = sorted(plot_df['year'].unique())
        data_by_year = [plot_df[plot_df['year'] == year]['water_level'].tolist() for year in years]
        
        # Filter out empty years
        data_by_year_filtered = []
        years_filtered = []
        for i, year_data in enumerate(data_by_year):
            if len(year_data) > 0:
                data_by_year_filtered.append(year_data)
                years_filtered.append(years[i])
        
        if len(data_by_year_filtered) > 0:
            plt.figure(figsize=(14, 6))
            bp = plt.boxplot(data_by_year_filtered, vert=True, patch_artist=True,
                           boxprops=dict(facecolor='lightblue', alpha=0.7),
                           medianprops=dict(color='red', linewidth=2),
                           whiskerprops=dict(color='black', linewidth=1.5),
                           capprops=dict(color='black', linewidth=1.5))
            
            # Calculate and add median values as text labels
            medians = [np.median(year_data) for year_data in data_by_year_filtered]
            for i, (median_val, year_data) in enumerate(zip(medians, data_by_year_filtered)):
                # Get the position of the box (x position)
                x_pos = i + 1
                # Add text annotation at the median line
                plt.text(x_pos, median_val, f'{median_val:.2f}', 
                        ha='center', va='bottom', fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
            
            plt.xlabel('Year', fontsize=12)
            plt.ylabel('Water Level', fontsize=12)
            plt.title(f'Box and Whisker Plot by Year - {station_name} ({file_name})', fontsize=14, fontweight='bold')
            plt.xticks(range(1, len(years_filtered) + 1), years_filtered, rotation=45)
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            
            # Save the graph
            output_name = f"{file_name}_{station_name}_boxplot.png"
            plt.savefig(output_name, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved: {output_name}")

def create_box_plot_by_month(plot_df, station_name, file_name):
    """Create a box and whisker plot grouped by month"""
    # Extract month from date and filter out missing/zero values
    plot_df['month'] = plot_df['date'].dt.month
    plot_df = plot_df[(plot_df['water_level'].notna()) & (plot_df['water_level'] != 0)]
    
    if len(plot_df) > 0:
        # Group by month (1-12)
        months = sorted(plot_df['month'].unique())
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        data_by_month = [plot_df[plot_df['month'] == month]['water_level'].tolist() for month in months]
        
        # Filter out empty months
        data_by_month_filtered = []
        months_filtered = []
        month_names_filtered = []
        for i, month_data in enumerate(data_by_month):
            if len(month_data) > 0:
                data_by_month_filtered.append(month_data)
                months_filtered.append(months[i])
                month_names_filtered.append(month_names[months[i] - 1])
        
        if len(data_by_month_filtered) > 0:
            plt.figure(figsize=(14, 6))
            bp = plt.boxplot(data_by_month_filtered, vert=True, patch_artist=True,
                           boxprops=dict(facecolor='lightblue', alpha=0.7),
                           medianprops=dict(color='red', linewidth=2),
                           whiskerprops=dict(color='black', linewidth=1.5),
                           capprops=dict(color='black', linewidth=1.5))
            
            # Calculate and add median values as text labels
            medians = [np.median(month_data) for month_data in data_by_month_filtered]
            for i, (median_val, month_data) in enumerate(zip(medians, data_by_month_filtered)):
                # Get the position of the box (x position)
                x_pos = i + 1
                # Add text annotation at the median line
                plt.text(x_pos, median_val, f'{median_val:.2f}', 
                        ha='center', va='bottom', fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
            
            plt.xlabel('Month', fontsize=12)
            plt.ylabel('Water Level', fontsize=12)
            plt.title(f'Box and Whisker Plot by Month - {station_name} ({file_name})', fontsize=14, fontweight='bold')
            plt.xticks(range(1, len(month_names_filtered) + 1), month_names_filtered, rotation=45)
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            
            # Save the graph
            output_name = f"{file_name}_{station_name}_monthly_boxplot.png"
            plt.savefig(output_name, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved: {output_name}")

def process_file(file_path):
    """Process a CSV file and create a line graph and box plot"""
    file_name = Path(file_path).stem
    
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Check the format of the file
        if 'date' in df.columns and 'date_gmt' not in df.columns:
            # Format 1: data.csv format (date column with multiple station columns)
            # Create a graph for each station column
            date_col = 'date'
            df[date_col] = pd.to_datetime(df[date_col])
            
            # Get all columns except date
            station_columns = [col for col in df.columns if col != date_col]
            
            for station in station_columns:
                # Filter out rows with missing values
                plot_df = df[[date_col, station]].dropna()
                plot_df = plot_df.rename(columns={station: 'water_level'})
                
                if len(plot_df) > 0:
                    # Line graph
                    plt.figure(figsize=(12, 6))
                    plt.plot(plot_df[date_col], plot_df['water_level'], linewidth=1.5)
                    plt.xlabel('Date', fontsize=12)
                    plt.ylabel('Water Level', fontsize=12)
                    plt.title(f'Water Level Over Time - {station} ({file_name})', fontsize=14, fontweight='bold')
                    plt.grid(True, alpha=0.3)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    # Save the graph
                    output_name = f"{file_name}_{station}_graph.png"
                    plt.savefig(output_name, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"Saved: {output_name}")
                    
                    # Box plot by year
                    plot_df_box = plot_df.rename(columns={date_col: 'date'})
                    create_box_plot_by_year(plot_df_box, station, file_name)
                    
                    # Box plot by month
                    create_box_plot_by_month(plot_df_box, station, file_name)
        
        elif 'date_gmt' in df.columns:
            # Format 2: Individual station files (date_gmt with year columns)
            date_col = 'date_gmt'
            year_columns = [col for col in df.columns if col != date_col]
            
            # Reshape the data: convert from wide to long format
            plot_data = []
            for _, row in df.iterrows():
                date_str = row[date_col]
                # Parse the date (format: "1/1/2025")
                try:
                    # Try to parse the date
                    date_parts = date_str.split('/')
                    if len(date_parts) == 3:
                        month, day, year = date_parts
                        # For each year column, create a date entry
                        for year_col in year_columns:
                            year_val = int(year_col)
                            water_level = row[year_col]
                            if pd.notna(water_level) and water_level != 0:  # Skip missing or zero values
                                # Create full date
                                full_date = pd.to_datetime(f"{year_val}-{month.zfill(2)}-{day.zfill(2)}")
                                plot_data.append({
                                    'date': full_date,
                                    'water_level': water_level
                                })
                except:
                    continue
            
            if plot_data:
                plot_df = pd.DataFrame(plot_data)
                plot_df = plot_df.sort_values('date')
                
                # Line graph
                plt.figure(figsize=(12, 6))
                plt.plot(plot_df['date'], plot_df['water_level'], linewidth=1.5)
                plt.xlabel('Date', fontsize=12)
                plt.ylabel('Water Level', fontsize=12)
                plt.title(f'Water Level Over Time - {file_name}', fontsize=14, fontweight='bold')
                plt.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                # Save the graph
                output_name = f"{file_name}_graph.png"
                plt.savefig(output_name, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Saved: {output_name}")
                
                # Box plot by year
                create_box_plot_by_year(plot_df, file_name, file_name)
                
                # Box plot by month
                create_box_plot_by_month(plot_df, file_name, file_name)
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")

def main():
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    # Find all CSV files in the directory
    csv_files = glob.glob(str(script_dir / "*.csv"))
    
    print(f"Found {len(csv_files)} CSV file(s)")
    
    # Process each CSV file
    for csv_file in csv_files:
        print(f"\nProcessing: {Path(csv_file).name}")
        process_file(csv_file)
    
    print("\nAll graphs have been generated!")

if __name__ == "__main__":
    main()

