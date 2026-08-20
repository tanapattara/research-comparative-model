import json
import os
import pandas as pd
from pathlib import Path

def load_metrics_from_json(file_path):
    """Load metrics from a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def extract_metrics(data):
    """Extract metrics from the JSON data structure."""
    model_name = data.get('model_name', 'Unknown')
    
    # Extract timing information if available
    timing = data.get('timing', {})
    
    # Handle different structures
    if 'metrics' in data:
        # LSTM structure - single metrics
        metrics = data['metrics'].copy()
        metrics['model'] = model_name
        metrics['split'] = 'test'  # Assume it's test metrics
        # Add timing information
        if timing:
            metrics['start_time'] = timing.get('start_time', 'N/A')
            metrics['end_time'] = timing.get('end_time', 'N/A')
            metrics['duration_seconds'] = timing.get('duration_seconds', 'N/A')
            metrics['duration_minutes'] = timing.get('duration_minutes', 'N/A')
        return [metrics]
    elif 'test_metrics' in data or 'training_metrics' in data:
        # RF and SVM structure - separate training and test metrics
        results = []
        if 'training_metrics' in data:
            train_metrics = data['training_metrics'].copy()
            train_metrics['model'] = model_name
            train_metrics['split'] = 'training'
            # Add timing information
            if timing:
                train_metrics['start_time'] = timing.get('start_time', 'N/A')
                train_metrics['end_time'] = timing.get('end_time', 'N/A')
                train_metrics['duration_seconds'] = timing.get('duration_seconds', 'N/A')
                train_metrics['duration_minutes'] = timing.get('duration_minutes', 'N/A')
            results.append(train_metrics)
        if 'test_metrics' in data:
            test_metrics = data['test_metrics'].copy()
            test_metrics['model'] = model_name
            test_metrics['split'] = 'test'
            # Add timing information
            if timing:
                test_metrics['start_time'] = timing.get('start_time', 'N/A')
                test_metrics['end_time'] = timing.get('end_time', 'N/A')
                test_metrics['duration_seconds'] = timing.get('duration_seconds', 'N/A')
                test_metrics['duration_minutes'] = timing.get('duration_minutes', 'N/A')
            results.append(test_metrics)
        return results
    else:
        return []

def create_comparison_table(result_folder='result'):
    """Create a comparison table from all metrics JSON files."""
    result_path = Path(result_folder)
    
    # Find all JSON metrics files
    json_files = list(result_path.glob('*_metrics.json'))
    
    if not json_files:
        print(f"No metrics JSON files found in {result_folder} folder")
        return None
    
    all_metrics = []
    
    # Load metrics from each file
    for json_file in sorted(json_files):
        try:
            data = load_metrics_from_json(json_file)
            metrics_list = extract_metrics(data)
            all_metrics.extend(metrics_list)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue
    
    if not all_metrics:
        print("No metrics found in the JSON files")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(all_metrics)
    
    # Reorder columns to have model, split, and timing first
    priority_cols = ['model', 'split']
    timing_cols = ['start_time', 'end_time', 'duration_seconds', 'duration_minutes']
    other_cols = [col for col in df.columns if col not in priority_cols + timing_cols]
    
    # Build column order: model, split, timing, then others
    ordered_cols = []
    for col in priority_cols:
        if col in df.columns:
            ordered_cols.append(col)
    for col in timing_cols:
        if col in df.columns:
            ordered_cols.append(col)
    ordered_cols.extend(other_cols)
    
    df = df[ordered_cols]
    
    # Replace Infinity and NaN with more readable values
    df = df.replace([float('inf'), float('-inf')], 'Infinity')
    df = df.fillna('N/A')
    
    return df

def display_results_table(result_folder='result', show_all_splits=True):
    """Display a formatted comparison table of model results."""
    df = create_comparison_table(result_folder)
    
    if df is None:
        return
    
    print("\n" + "="*80)
    print("MODEL COMPARISON TABLE")
    print("="*80)
    
    if show_all_splits:
        # Show all splits (training and test)
        print("\nAll Metrics (Training and Test):")
        print("-"*80)
        print(df.to_string(index=False))
    else:
        # Show only test metrics
        if 'split' in df.columns:
            test_df = df[df['split'] == 'test'].copy()
            if not test_df.empty:
                test_df = test_df.drop('split', axis=1)
                print("\nTest Metrics Only:")
                print("-"*80)
                print(test_df.to_string(index=False))
            else:
                print("\nAll Metrics:")
                print("-"*80)
                print(df.to_string(index=False))
        else:
            print("\nAll Metrics:")
            print("-"*80)
            print(df.to_string(index=False))
    
    print("\n" + "="*80)
    
    return df

def save_results_table(result_folder='result', output_file='result/comparison_table.csv', show_all_splits=False):
    """Save the comparison table to a CSV file."""
    df = create_comparison_table(result_folder)
    
    if df is None:
        return None
    
    if show_all_splits:
        # Save all splits
        df.to_csv(output_file, index=False)
    else:
        # Save only test metrics
        if 'split' in df.columns:
            test_df = df[df['split'] == 'test'].copy()
            if not test_df.empty:
                test_df = test_df.drop('split', axis=1)
                test_df.to_csv(output_file, index=False)
                df = test_df
            else:
                df.to_csv(output_file, index=False)
        else:
            df.to_csv(output_file, index=False)
    
    print(f"\nComparison table saved to: {output_file}")
    return df

if __name__ == "__main__":
    # Display the comparison table
    df = display_results_table(result_folder='result', show_all_splits=False)
    
    # Also save to CSV
    if df is not None:
        save_results_table(result_folder='result', 
                          output_file='result/comparison_table.csv', 
                          show_all_splits=False)

