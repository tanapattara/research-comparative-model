import os
import sys
import warnings
import time
warnings.filterwarnings('ignore')

# Import all model modules
from lstm import main as lstm_main
from rf import main as rf_main
from svm import main as svm_main
from sarima import main as sarima_main
from gru import main as gru_main

def print_model_results(model_name, results):
    """Print formatted results for a model"""
    print("\n" + "=" * 80)
    print(f"RESULTS SUMMARY: {model_name.upper()}")
    print("=" * 80)
    
    if results is None:
        print(f"  ❌ {model_name.upper()} model failed to produce results")
        return
    
    # Print common metrics that most models have
    if 'test_mse' in results:
        print(f"\n  Test Set Metrics:")
        print(f"    - MSE:  {results.get('test_mse', 'N/A'):.4f}")
        print(f"    - RMSE: {results.get('test_rmse', 'N/A'):.4f}")
        print(f"    - MAE:  {results.get('test_mae', 'N/A'):.4f}")
        print(f"    - R²:   {results.get('test_r2', 'N/A'):.4f}")
        if 'test_accuracy' in results:
            print(f"    - Accuracy: {results.get('test_accuracy', 'N/A'):.2f}%")
    
    # LSTM specific (uses different key names)
    if 'mse' in results and 'test_mse' not in results:
        print(f"\n  Test Set Metrics:")
        print(f"    - MSE:  {results.get('mse', 'N/A'):.4f}")
        print(f"    - RMSE: {results.get('rmse', 'N/A'):.4f}")
        print(f"    - MAE:  {results.get('mae', 'N/A'):.4f}")
        print(f"    - R²:   {results.get('r2', 'N/A'):.4f}")
    
    # Print training metrics if available
    if 'train_mse' in results:
        print(f"\n  Training Set Metrics:")
        print(f"    - MSE:  {results.get('train_mse', 'N/A'):.4f}")
        print(f"    - RMSE: {results.get('train_rmse', 'N/A'):.4f}")
        print(f"    - MAE:  {results.get('train_mae', 'N/A'):.4f}")
        print(f"    - R²:   {results.get('train_r2', 'N/A'):.4f}")
        if 'train_accuracy' in results:
            print(f"    - Accuracy: {results.get('train_accuracy', 'N/A'):.2f}%")
    
    # Print feature importance for RF
    if 'feature_importance' in results:
        print(f"\n  Feature Importance:")
        for feature, importance in results['feature_importance'].items():
            print(f"    - {feature}: {importance:.4f} ({importance*100:.2f}%)")
    
    print("=" * 80)

def generate_result_markdown(all_results, durations):
    """Generate result.md file with comparison table"""
    result_dir = 'result'
    os.makedirs(result_dir, exist_ok=True)
    result_file = os.path.join(result_dir, 'result.md')
    
    # Prepare comparison data
    comparison_data = []
    for model_name, results in all_results.items():
        if results is None:
            comparison_data.append({
                'Model': model_name,
                'MSE': 'N/A',
                'RMSE': 'N/A',
                'MAE': 'N/A',
                'R²': 'N/A',
                'Accuracy': 'N/A',
                'Duration': 'N/A'
            })
        else:
            # Handle different result structures
            mse = None
            rmse = None
            mae = None
            r2 = None
            accuracy = None
            
            if 'test_mse' in results:
                mse = results['test_mse']
                rmse = results['test_rmse']
                mae = results['test_mae']
                r2 = results['test_r2']
                if 'test_accuracy' in results:
                    accuracy = results['test_accuracy']
                else:
                    accuracy = r2 * 100  # Calculate from R²
            elif 'mse' in results:
                mse = results['mse']
                rmse = results['rmse']
                mae = results['mae']
                r2 = results['r2']
                if 'accuracy' in results:
                    accuracy = results['accuracy']
                else:
                    accuracy = r2 * 100  # Calculate from R²
            
            duration_str = 'N/A'
            if model_name in durations and durations[model_name] is not None:
                duration_str = f"{durations[model_name]:.2f}s"
            
            comparison_data.append({
                'Model': model_name,
                'MSE': f"{mse:.4f}" if mse is not None else 'N/A',
                'RMSE': f"{rmse:.4f}" if rmse is not None else 'N/A',
                'MAE': f"{mae:.4f}" if mae is not None else 'N/A',
                'R²': f"{r2:.4f}" if r2 is not None else 'N/A',
                'Accuracy': f"{accuracy:.2f}%" if accuracy is not None else 'N/A',
                'Duration': duration_str
            })
    
    # Find best model for each metric
    metrics_data = {}
    for model_name, results in all_results.items():
        if results is None:
            continue
        
        if 'test_mse' in results:
            metrics_data[model_name] = {
                'mse': results['test_mse'],
                'rmse': results['test_rmse'],
                'mae': results['test_mae'],
                'r2': results['test_r2']
            }
        elif 'mse' in results:
            metrics_data[model_name] = {
                'mse': results['mse'],
                'rmse': results['rmse'],
                'mae': results['mae'],
                'r2': results['r2']
            }
    
    # Generate markdown content
    markdown_content = "# Model Comparison Summary\n\n"
    markdown_content += "This table compares the performance of all models when run through `app.py` on the test set.\n\n"
    markdown_content += "## Test Set Performance Comparison\n\n"
    markdown_content += "| Model | MSE    | RMSE   | MAE    | R²     | Accuracy | Duration |\n"
    markdown_content += "| ----- | ------ | ------ | ------ | ------ | -------- | -------- |\n"
    
    for row in comparison_data:
        markdown_content += f"| {row['Model']:<5} | {row['MSE']:<6} | {row['RMSE']:<6} | {row['MAE']:<6} | {row['R²']:<6} | {row['Accuracy']:<8} | {row['Duration']:<8} |\n"
    
    markdown_content += "\n## Best Model by Metric\n\n"
    
    if metrics_data:
        best_mse = min(metrics_data.items(), key=lambda x: x[1]['mse'])
        best_rmse = min(metrics_data.items(), key=lambda x: x[1]['rmse'])
        best_mae = min(metrics_data.items(), key=lambda x: x[1]['mae'])
        best_r2 = max(metrics_data.items(), key=lambda x: x[1]['r2'])
        
        markdown_content += f"- **Best MSE (Lower is Better):** {best_mse[0]} ({best_mse[1]['mse']:.4f})\n"
        markdown_content += f"- **Best RMSE (Lower is Better):** {best_rmse[0]} ({best_rmse[1]['rmse']:.4f})\n"
        markdown_content += f"- **Best MAE (Lower is Better):** {best_mae[0]} ({best_mae[1]['mae']:.4f})\n"
        markdown_content += f"- **Best R² (Higher is Better):** {best_r2[0]} ({best_r2[1]['r2']:.4f})\n"
        
        # Find best accuracy
        best_accuracy_model = None
        best_accuracy_value = -1
        for model_name, results in all_results.items():
            if results is None:
                continue
            if 'test_accuracy' in results:
                if results['test_accuracy'] > best_accuracy_value:
                    best_accuracy_value = results['test_accuracy']
                    best_accuracy_model = model_name
            elif 'accuracy' in results:
                if results['accuracy'] > best_accuracy_value:
                    best_accuracy_value = results['accuracy']
                    best_accuracy_model = model_name
            elif 'test_r2' in results:
                acc = results['test_r2'] * 100
                if acc > best_accuracy_value:
                    best_accuracy_value = acc
                    best_accuracy_model = model_name
            elif 'r2' in results:
                acc = results['r2'] * 100
                if acc > best_accuracy_value:
                    best_accuracy_value = acc
                    best_accuracy_model = model_name
        
        if best_accuracy_model:
            markdown_content += f"- **Best Accuracy (Higher is Better):** {best_accuracy_model} ({best_accuracy_value:.2f}%)\n"
        
        # Find fastest model
        fastest_model = None
        fastest_duration = float('inf')
        for model_name, duration in durations.items():
            if duration is not None and duration < fastest_duration:
                fastest_duration = duration
                fastest_model = model_name
        
        if fastest_model:
            markdown_content += f"- **Fastest Training:** {fastest_model} ({fastest_duration:.2f}s)\n"
    
    markdown_content += "\n## Summary\n\n"
    
    # Generate summary text
    if metrics_data:
        best_overall = best_r2[0]  # Use R² as overall best indicator
        markdown_content += f"The **{best_overall} model** performs best across most error metrics "
        markdown_content += f"(MSE: {best_mse[1]['mse']:.4f}, RMSE: {best_rmse[1]['rmse']:.4f}, MAE: {best_mae[1]['mae']:.4f}) "
        markdown_content += f"and achieves the highest R² score ({best_r2[1]['r2']:.4f}). "
        
        if fastest_model and fastest_model != best_overall:
            markdown_content += f"However, **{fastest_model}** offers the fastest training time ({fastest_duration:.2f}s) "
            markdown_content += "while still maintaining competitive performance. "
        
        # Check for deep learning models
        dl_models = [m for m in all_results.keys() if m in ['LSTM', 'GRU'] and all_results[m] is not None]
        if dl_models:
            markdown_content += "The deep learning models (LSTM and GRU) require significantly more training time "
            markdown_content += "but provide good accuracy."
    
    # Write to file
    with open(result_file, 'w') as f:
        f.write(markdown_content)
    
    print(f"\n✓ Comparison table saved to: {result_file}")
    return result_file

def main():
    """Main function to run all models and compare results"""
    print("\n" + "=" * 80)
    print("COMPARATIVE MODEL ANALYSIS")
    print("Water Level Prediction - Comparing LSTM, GRU, RF, SVM, and SARIMA Models")
    print("=" * 80)
    
    # Check if data file exists
    data_path = 'data/data.csv'
    if not os.path.exists(data_path):
        print(f"\n❌ Error: {data_path} not found!")
        print("Please ensure the data file exists in the data/ directory.")
        return
    
    print(f"\n✓ Data file found: {data_path}")
    print("\n" + "=" * 80)
    print("Running all models...")
    print("=" * 80)
    
    all_results = {}
    durations = {}
    
    # Run LSTM Model
    print("\n" + "=" * 80)
    print("1. Running LSTM Model...")
    print("=" * 80)
    try:
        start_time = time.time()
        lstm_results = lstm_main()
        duration = time.time() - start_time
        all_results['LSTM'] = lstm_results
        durations['LSTM'] = duration
        print("\n✓ LSTM model completed successfully")
    except Exception as e:
        print(f"\n❌ LSTM model failed: {str(e)}")
        all_results['LSTM'] = None
        durations['LSTM'] = None
    
    # Run GRU Model
    print("\n" + "=" * 80)
    print("2. Running GRU Model...")
    print("=" * 80)
    try:
        start_time = time.time()
        gru_results = gru_main()
        duration = time.time() - start_time
        all_results['GRU'] = gru_results
        durations['GRU'] = duration
        print("\n✓ GRU model completed successfully")
    except Exception as e:
        print(f"\n❌ GRU model failed: {str(e)}")
        all_results['GRU'] = None
        durations['GRU'] = None
    
    # Run Random Forest Model
    print("\n" + "=" * 80)
    print("3. Running Random Forest (RF) Model...")
    print("=" * 80)
    try:
        start_time = time.time()
        rf_results = rf_main()
        duration = time.time() - start_time
        all_results['RF'] = rf_results
        durations['RF'] = duration
        print("\n✓ Random Forest model completed successfully")
    except Exception as e:
        print(f"\n❌ Random Forest model failed: {str(e)}")
        all_results['RF'] = None
        durations['RF'] = None
    
    # Run SVM Model
    print("\n" + "=" * 80)
    print("4. Running Support Vector Machine (SVM) Model...")
    print("=" * 80)
    try:
        start_time = time.time()
        svm_results = svm_main()
        duration = time.time() - start_time
        all_results['SVM'] = svm_results
        durations['SVM'] = duration
        print("\n✓ SVM model completed successfully")
    except Exception as e:
        print(f"\n❌ SVM model failed: {str(e)}")
        all_results['SVM'] = None
        durations['SVM'] = None
    
    # # Run SARIMA Model
    # print("\n" + "=" * 80)
    # print("5. Running SARIMA Model...")
    # print("=" * 80)
    # try:
    #     sarima_results = sarima_main()
    #     all_results['SARIMA'] = sarima_results
    #     print("\n✓ SARIMA model completed successfully")
    # except Exception as e:
    #     print(f"\n❌ SARIMA model failed: {str(e)}")
    #     all_results['SARIMA'] = None
    
    # # Print summary of all results
    # print("\n" + "=" * 80)
    # print("COMPARATIVE RESULTS SUMMARY")
    # print("=" * 80)
    
    # Print individual model results
    for model_name, results in all_results.items():
        print_model_results(model_name, results)
    
    # Create comparison table
    print("\n" + "=" * 80)
    print("COMPARISON TABLE - Test Set Performance")
    print("=" * 80)
    
    # Prepare comparison data
    comparison_data = []
    for model_name, results in all_results.items():
        if results is None:
            comparison_data.append({
                'Model': model_name,
                'MSE': 'N/A',
                'RMSE': 'N/A',
                'MAE': 'N/A',
                'R²': 'N/A'
            })
        else:
            # Handle different result structures
            if 'test_mse' in results:
                comparison_data.append({
                    'Model': model_name,
                    'MSE': f"{results['test_mse']:.4f}",
                    'RMSE': f"{results['test_rmse']:.4f}",
                    'MAE': f"{results['test_mae']:.4f}",
                    'R²': f"{results['test_r2']:.4f}"
                })
            elif 'mse' in results:
                comparison_data.append({
                    'Model': model_name,
                    'MSE': f"{results['mse']:.4f}",
                    'RMSE': f"{results['rmse']:.4f}",
                    'MAE': f"{results['mae']:.4f}",
                    'R²': f"{results['r2']:.4f}"
                })
            else:
                comparison_data.append({
                    'Model': model_name,
                    'MSE': 'N/A',
                    'RMSE': 'N/A',
                    'MAE': 'N/A',
                    'R²': 'N/A'
                })
    
    # Print comparison table
    print(f"\n{'Model':<10} {'MSE':<12} {'RMSE':<12} {'MAE':<12} {'R²':<12}")
    print("-" * 60)
    for row in comparison_data:
        print(f"{row['Model']:<10} {row['MSE']:<12} {row['RMSE']:<12} {row['MAE']:<12} {row['R²']:<12}")
    
    # Find best model for each metric
    print("\n" + "=" * 80)
    print("BEST MODEL BY METRIC")
    print("=" * 80)
    
    # Extract numeric values for comparison
    metrics_data = {}
    for model_name, results in all_results.items():
        if results is None:
            continue
        
        if 'test_mse' in results:
            metrics_data[model_name] = {
                'mse': results['test_mse'],
                'rmse': results['test_rmse'],
                'mae': results['test_mae'],
                'r2': results['test_r2']
            }
        elif 'mse' in results:
            metrics_data[model_name] = {
                'mse': results['mse'],
                'rmse': results['rmse'],
                'mae': results['mae'],
                'r2': results['r2']
            }
    
    if metrics_data:
        # Best model (lower is better for MSE, RMSE, MAE; higher is better for R²)
        if metrics_data:
            best_mse = min(metrics_data.items(), key=lambda x: x[1]['mse'])
            best_rmse = min(metrics_data.items(), key=lambda x: x[1]['rmse'])
            best_mae = min(metrics_data.items(), key=lambda x: x[1]['mae'])
            best_r2 = max(metrics_data.items(), key=lambda x: x[1]['r2'])
            
            print(f"\n  Best MSE:  {best_mse[0]} ({best_mse[1]['mse']:.4f})")
            print(f"  Best RMSE: {best_rmse[0]} ({best_rmse[1]['rmse']:.4f})")
            print(f"  Best MAE:  {best_mae[0]} ({best_mae[1]['mae']:.4f})")
            print(f"  Best R²:   {best_r2[0]} ({best_r2[1]['r2']:.4f})")
    
    # Generate result.md file
    print("\n" + "=" * 80)
    print("Generating comparison table...")
    print("=" * 80)
    try:
        generate_result_markdown(all_results, durations)
    except Exception as e:
        print(f"\n❌ Failed to generate result.md: {str(e)}")
    
    print("\n" + "=" * 80)
    print("All models completed!")
    print("=" * 80 + "\n")
    
    return all_results

if __name__ == "__main__":
    results = main()

