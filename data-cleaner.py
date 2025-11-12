#load data from data folder
import pandas as pd
import os
import re
from datetime import datetime

def process_file(file_path, output_path):
    """Process a single CSV file and transform it."""
    print(f"\n{'=' * 60}")
    print(f"Processing: {os.path.basename(file_path)}")
    print(f"{'=' * 60}")
    
    # Load data from backup folder
    data = pd.read_csv(file_path)
    print(f"Original data shape: {data.shape}")
    print(f"Original columns: {list(data.columns)}")
    
    # Convert date_gmt to datetime (handle invalid dates gracefully)
    data['date_gmt'] = pd.to_datetime(data['date_gmt'], errors='coerce', format='mixed')
    
    # Filter out rows with invalid dates
    data = data[data['date_gmt'].notna()].copy()
    print(f"After filtering invalid dates: {data.shape}")
    
    # Identify year columns (columns that match pattern like "1992.93", "2003.04", etc.)
    year_columns = []
    for col in data.columns:
        # Check if column name starts with 4 digits (year)
        if re.match(r'^\d{4}\.', col):
            year_columns.append(col)
    
    print(f"Found year columns: {year_columns}")
    
    # Create list to store new rows
    new_rows = []
    
    # For each row in the original data
    for idx, row in data.iterrows():
        original_date = row['date_gmt']
        
        # Skip if date is invalid
        if pd.isna(original_date):
            continue
            
        month = original_date.month
        day = original_date.day
        
        # For each year column, create a new row
        for year_col in year_columns:
            # Extract year from column name (first 4 digits)
            year = int(year_col.split('.')[0])
            
            # Only process years from 2006 to 2025
            if year < 2007 or year > 2024:
                continue
            
            # Get the value from the year column
            value = row[year_col]
            
            # Only create row if value is not NaN/null
            if pd.notna(value):
                try:
                    # Create new date with extracted year but same month and day
                    new_date = datetime(year, month, day)
                    
                    # Create new row
                    new_rows.append({
                        'date_gmt': new_date.strftime('%Y-%m-%d'),
                        'AVG': value
                    })
                except ValueError:
                    # Skip invalid dates (e.g., Feb 29 in non-leap year)
                    continue
    
    # Create new dataframe from the transformed rows
    transformed_data = pd.DataFrame(new_rows)
    
    # Sort by date_gmt
    transformed_data = transformed_data.sort_values('date_gmt').reset_index(drop=True)
    
    print(f"\nTransformed data shape: {transformed_data.shape}")
    print(f"\nFirst few rows:")
    print(transformed_data.head(10))
    print(f"\nLast few rows:")
    print(transformed_data.tail(10))
    
    # Save data to data folder
    transformed_data.to_csv(output_path, index=False)
    print(f"\nData saved to {output_path}")

def main():
    print("=" * 60)
    print("Data Cleaner - Processing all files from backup folder")
    print("=" * 60)
    
    # Get all CSV files from backup folder
    backup_folder = 'backup'
    data_folder = 'data'
    
    # Ensure data folder exists
    os.makedirs(data_folder, exist_ok=True)
    
    # Get all CSV files in backup folder
    csv_files = [f for f in os.listdir(backup_folder) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {backup_folder} folder")
        return
    
    print(f"\nFound {len(csv_files)} CSV file(s) to process:")
    for f in csv_files:
        print(f"  - {f}")
    
    # Process each file
    for csv_file in csv_files:
        input_path = os.path.join(backup_folder, csv_file)
        output_path = os.path.join(data_folder, csv_file)
        process_file(input_path, output_path)
    
    print(f"\n{'=' * 60}")
    print("All files processed successfully!")
    print(f"{'=' * 60}")

def merge_all_data(data_folder='data', output_file='data.csv'):
    """Merge all station CSV files into one table with columns: date_gmt, CSA, LUA, CKH, VIE, NON"""
    print(f"\n{'=' * 60}")
    print("Merging all station data into one table")
    print(f"{'=' * 60}")
    
    # Define the station files and their column names
    station_files = {
        'CSA.csv': 'CSA',
        'LUA.csv': 'LUA',
        'CKH.csv': 'CKH',
        'VIE.csv': 'VIE',
        'NON.csv': 'NON'
    }
    
    # Start with the first file to get the base structure
    merged_df = None
    
    for filename, station_name in station_files.items():
        file_path = os.path.join(data_folder, filename)
        
        if not os.path.exists(file_path):
            print(f"Warning: {filename} not found, skipping...")
            continue
        
        print(f"\nLoading {filename}...")
        df = pd.read_csv(file_path)
        
        # Convert date_gmt to datetime
        df['date_gmt'] = pd.to_datetime(df['date_gmt'], errors='coerce')
        
        # Rename AVG column to station name
        df = df.rename(columns={'AVG': station_name})
        
        # Select only date_gmt and station column
        df = df[['date_gmt', station_name]].copy()
        
        # Merge with existing dataframe
        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on='date_gmt', how='outer')
        
        print(f"  Shape: {df.shape}, Date range: {df['date_gmt'].min()} to {df['date_gmt'].max()}")
    
    if merged_df is None:
        print("No data files found to merge!")
        return
    
    # Sort by date_gmt
    merged_df = merged_df.sort_values('date_gmt').reset_index(drop=True)
    
    # Ensure column order: date_gmt, CSA, LUA, CKH, VIE, NON
    column_order = ['date_gmt', 'CSA', 'LUA', 'CKH', 'VIE', 'NON']
    # Only include columns that exist
    column_order = [col for col in column_order if col in merged_df.columns]
    merged_df = merged_df[column_order]
    
    # Save to data.csv
    output_path = os.path.join(data_folder, output_file)
    merged_df.to_csv(output_path, index=False)
    
    print(f"\n{'=' * 60}")
    print(f"Merged data saved to {output_path}")
    print(f"Final shape: {merged_df.shape}")
    print(f"Columns: {list(merged_df.columns)}")
    print(f"Date range: {merged_df['date_gmt'].min()} to {merged_df['date_gmt'].max()}")
    print(f"\nFirst few rows:")
    print(merged_df.head(10))
    print(f"\nLast few rows:")
    print(merged_df.tail(10))
    print(f"{'=' * 60}")
    
if __name__ == "__main__":
    main()
    # After processing all files, merge them into one table
    merge_all_data()