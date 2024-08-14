import os
import pandas as pd

def save_results_to_csv(results, base_dir):
    """
    Save the simulation results to a CSV file.

    Parameters:
        results (list): List of dictionaries containing the simulation 
                        results.
        base_dir (str): Base directory where the CSV file will be saved.
        
    Returns:
        None
    """
    results_df = pd.DataFrame(results)                                      # Convert the processed results back to a DataFrame
    results_df.to_csv(os.path.join(base_dir, 'simulation_results.csv'),     # Save the results to a CSV file
                      index=False)
    print(f"Results saved to {os.path.join(base_dir, 
                                           'simulation_results.csv')}")
    