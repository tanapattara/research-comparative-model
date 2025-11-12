import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime

def transform_csv_to_long_format(file_path, station_code):
    """
    Transform a CSV file from wide format (years as columns) to long format (date and value).
    Each row in the original file becomes multiple rows, one for each year.
    
    Args:
        file_path: Path to the CSV file
        station_code: Station code (e.g., 'CKH', 'CSA', etc.)
    
    Returns:
        DataFrame with columns: date, value, station
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Get the date column (first column)
    date_col = df.columns[0]  # Should be 'date_gmt'
    
    # Get all year columns (all columns except the first one)
    # Filter to only include columns that can be converted to integers (years)
    year_columns = []
    for col in df.columns:
        if col != date_col:
            try:
                int(col)  # Try to convert to int to verify it's a year
                year_columns.append(col)
            except ValueError:
                continue  # Skip non-year columns
    
    if not year_columns:
        raise ValueError(f"No valid year columns found in {file_path}")
    
    # Melt the dataframe to long format
    df_long = pd.melt(
        df,
        id_vars=[date_col],
        value_vars=year_columns,
        var_name='year',
        value_name='value'
    )
    
    # Convert year to integer, handling any errors
    df_long['year'] = pd.to_numeric(df_long['year'], errors='coerce')
    df_long = df_long[df_long['year'].notna()]  # Remove rows with invalid years
    
    # Parse the date (format: day/month, e.g., "1/1/2025")
    # We need to combine the day/month from date_gmt with the year from the year column
    def create_full_date(row):
        date_str = str(row[date_col])
        year = int(row['year'])
        
        # Skip if date_str is the header or invalid
        if date_str == date_col or date_str == 'nan' or pd.isna(row[date_col]):
            return pd.NaT
        
        # Parse the date string (format: "day/month" or "day/month/year")
        try:
            parts = date_str.split('/')
            if len(parts) < 2:
                return pd.NaT
            
            day = int(parts[0])
            month = int(parts[1])
            
            # Create full date with the year from the year column
            # Handle invalid dates (e.g., Feb 29 in non-leap years)
            try:
                return pd.to_datetime(f"{year}-{month:02d}-{day:02d}", errors='coerce')
            except:
                return pd.NaT
        except (ValueError, IndexError):
            return pd.NaT
    
    df_long['date'] = df_long.apply(create_full_date, axis=1)
    
    # Remove rows with invalid dates or zero/null values
    df_long = df_long[df_long['date'].notna()]
    df_long = df_long[df_long['value'].notna()]
    df_long = df_long[df_long['value'] != 0]  # Remove zero values
    
    # Select and rename columns
    result = df_long[['date', 'value']].copy()
    result['station'] = station_code
    
    return result

def merge_all_stations(data_folder='data'):
    """
    Read all CSV files in the data folder, transform them, and merge into one dataframe.
    
    Args:
        data_folder: Path to folder containing CSV files
    
    Returns:
        DataFrame with dates as rows and station codes as columns
    """
    data_path = Path(data_folder)
    all_data = []
    
    # Get all CSV files in the data folder, excluding the merged output file
    csv_files = [f for f in data_path.glob('*.csv') if f.name != 'merged_data.csv']
    
    if not csv_files:
        print(f"No CSV files found in {data_folder}")
        return None
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Process each CSV file
    for csv_file in csv_files:
        # Extract station code from filename (e.g., 'CKH.csv' -> 'CKH')
        station_code = csv_file.stem
        
        print(f"Processing {station_code}...")
        
        try:
            # Transform the file to long format
            df_transformed = transform_csv_to_long_format(csv_file, station_code)
            
            # Add to list
            all_data.append(df_transformed)
            print(f"  - {station_code}: {len(df_transformed)} records")
            
        except Exception as e:
            print(f"  - Error processing {station_code}: {str(e)}")
            continue
    
    if not all_data:
        print("No data was successfully processed")
        return None
    
    # Combine all dataframes
    print("\nCombining all data...")
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Pivot to create final format: dates as rows, stations as columns
    final_df = combined_df.pivot_table(
        index='date',
        columns='station',
        values='value',
        aggfunc='first'  # In case of duplicates, take first value
    )
    
    # Reset index to make date a column
    final_df = final_df.reset_index()
    
    # Sort by date
    final_df = final_df.sort_values('date').reset_index(drop=True)
    
    # Reorder columns: date, CSA, LUA, CKH, VIE, NON
    desired_order = ['date', 'CSA', 'LUA', 'CKH', 'VIE', 'NON']
    # Only include columns that exist in the dataframe
    available_columns = [col for col in desired_order if col in final_df.columns]
    # Add any other columns that might exist but aren't in the desired order
    other_columns = [col for col in final_df.columns if col not in desired_order]
    final_df = final_df[available_columns + other_columns]
    
    print(f"\nFinal merged dataframe shape: {final_df.shape}")
    print(f"Date range: {final_df['date'].min()} to {final_df['date'].max()}")
    print(f"Stations: {', '.join([col for col in final_df.columns if col != 'date'])}")
    
    return final_df

def fill_missing_values(merged_df):
    """
    Fill missing values and zero values in the merged dataframe.
    Uses linear interpolation to fill gaps, creating smooth transitions between known values.
    Handles both NaN values and zero values as missing data.
    """
    # Make a copy to avoid modifying during iteration
    df = merged_df.copy()
    
    # Ensure date column is datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort by date to ensure proper interpolation
    df = df.sort_values('date').reset_index(drop=True)
    
    # Process each station column (skip 'date' column)
    station_columns = [col for col in df.columns if col != 'date']
    
    for station in station_columns:
        # Create a copy of the station column for processing
        station_data = df[station].copy()
        
        # Replace zero values with NaN so they can be interpolated
        zero_mask = (station_data == 0) | (station_data.isna())
        num_zeros = (station_data == 0).sum()
        num_nans = station_data.isna().sum()
        
        if num_zeros > 0 or num_nans > 0:
            # Mark zeros and NaNs as missing
            station_data[zero_mask] = np.nan
            
            # Use linear interpolation based on date index
            # Set date as index temporarily for interpolation
            temp_df = df[['date', station]].copy()
            temp_df[station] = station_data
            temp_df = temp_df.set_index('date')
            
            # Interpolate using linear method (works with datetime index)
            temp_df[station] = temp_df[station].interpolate(method='linear', limit_direction='both')
            
            # Fill remaining NaNs at the edges with forward/backward fill
            # Use ffill() and bfill() instead of deprecated fillna(method=...)
            temp_df[station] = temp_df[station].ffill().bfill()
            
            # Update the original dataframe
            df[station] = temp_df[station].values
            
            total_filled = num_zeros + num_nans
            print(f"  - {station}: Filled {total_filled} missing/zero values ({num_zeros} zeros, {num_nans} NaNs) using interpolation")
    
    return df

def main():
    """
    Main function to process all CSV files and create merged dataframe.
    """
    print("=" * 60)
    print("Data Processing: Transform and Merge CSV Files")
    print("=" * 60)
    
    # Process and merge all files
    merged_df = merge_all_stations('data')

    # check if some rows are missing or zero or nan in middle of date range
    # averaging the values of the same station on date befor and after the missing value
    # and replace the missing value with the average

    merged_df = fill_missing_values(merged_df)

    
    if merged_df is not None:
        # Save the merged dataframe
        output_path = 'data/data.csv'
        merged_df.to_csv(output_path, index=False)
        print(f"\nMerged data saved to: {output_path}")
        
        # Display first few rows
        print("\nFirst 10 rows of merged data:")
        print(merged_df.head(10))
        
        # Display summary statistics
        print("\nSummary statistics:")
        print(merged_df.describe())
        
        return merged_df
    else:
        print("Failed to process data")
        return None

if __name__ == "__main__":
    results = main()
