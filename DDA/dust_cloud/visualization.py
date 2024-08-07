import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import json
import proc_output_ddscat as proc_output_ddscat

def load_sample_parameters(simulation_directory):
    """
    Load the sample parameters from a file in the simulation 
    directory.
    
    """
    sample_file_path = os.path.join(
        simulation_directory, 'sample_parameters.json'
        )
    with open(sample_file_path, 'r') as f:
        sample = json.load(f)
    return sample

def load_samples(file_path):
    """Load samples from a file."""
    with open(file_path, "r") as f:
        samples = json.load(f)
    return samples

def process_results(base_dir):
    """
    Process the results from the simulation directories.

    Parameters:
        base_dir (str): Base directory where simulation data is stored.

    Returns:
        pd.DataFrame: A DataFrame containing the aggregated results.
    """
    original_directory = os.getcwd()  # Save the current directory to revert back to it later
    results = []
    samples = []  # To store loaded samples
    for i in range(1, 1000):  # Assuming 100 samples as in the original setup
        simulation_directory = os.path.join(base_dir, f"sim_{i}")
        if not os.path.exists(simulation_directory):
            continue

        sample = load_sample_parameters(simulation_directory)
        samples.append(sample)  # Collect the sample
        proc_output_ddscat.process_ddscat_result(simulation_directory, results, sample)  # Process the result

    os.chdir(original_directory)
    # Save the collected samples back to Generated_samples.json
    with open("Generated_samples.json", "w") as f:
        json.dump(samples, f, indent=4)

    return pd.concat(results, ignore_index=True)

def analyze_results(results_df):
    sns.pairplot(results_df, vars=['S_11', 'size_param', 'wavelength', 'Qsca', 'Qbk', 'Qpol'], hue='shape')
    plt.suptitle('Pairplot of Parameters and Results')
    plt.show()
    correlation_matrix = results_df[['S_11', 'size_param', 'wavelength', 'Qsca', 'Qbk', 'Qpol']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix of Parameters and Results')
    plt.show()

def plot_data(data_frames, labels):
    """
    Plots data from multiple DataFrames, assuming each DataFrame contains 
    'theta' and 'S_11' columns.

    Parameters:
    data_frames (list of pd.DataFrame): List of DataFrames to plot.
    labels (list of str): Labels for each DataFrame plot.
    """
    plt.figure(figsize=(10, 6))
    for df, label in zip(data_frames, labels):
        plt.plot(df[df["phi"] == 0]["theta"], df[df["phi"] == 0]["S_11"], 
                 label=label, alpha=0.7, markersize=5, linestyle='-', 
                 linewidth=1)
    plt.xlabel('Theta')
    plt.ylabel('S_11')
    plt.title('Comparison of S_11 Values')
    plt.legend()
    plt.show()


def plot_polar_data(data_frames, labels):
    """
    Plots data from multiple DataFrames on a polar plot, assuming 
    each DataFrame contains 
    'theta' and 'S_11' columns. Theta should be in degrees for plotting.

    Parameters:
    data_frames (list of pd.DataFrame): List of DataFrames to plot.
    labels (list of str): Labels for each DataFrame plot.
    """
    plt.figure(figsize=(8, 8))                                           # Adjust the figure size as needed
    ax = plt.subplot(111, polar=True)                                    # Create a polar subplot
    for df, label in zip(data_frames, labels):                           # Convert degrees to radians for polar plotting
        radians = np.deg2rad(df[df["phi"] == 0]["theta"])
        ax.plot(radians, df[df["phi"] == 0]["S_11"], label=label, 
                alpha=0.7, linestyle='-', linewidth=1)
    ax.set_theta_zero_location('N')                                        # This sets the 0 degrees to the North
    ax.set_theta_direction(-1)                                             # This sets the direction of degrees to clockwise
    ax.set_xlabel('Theta (radians)')
    ax.set_ylabel('S_11')
    ax.set_title('Polar Comparison of S_11 Values')
    ax.legend()
    plt.show()

def plot_mie_ddscat_comparison(ddscat_df, mie_df):
    plt.figure(figsize=(10, 6))
    for label, group in ddscat_df.groupby('shape'):
        plt.plot(group['theta'], group['S_11'], label=f'DDSCAT {label}', alpha=0.7, linestyle='-', linewidth=1)
    
    for label, group in mie_df.groupby(['radius', 'wavelength']):
        plt.plot(group['angle'], group['S_11'], label=f'Mie: r={label[0]}, λ={label[1]}', alpha=0.7, linestyle='--', linewidth=1)
    
    plt.xlabel('Theta (degrees)')
    plt.ylabel('S_11 / Phase Function')
    plt.title('Comparison of S_11 Values and Phase Function')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    for label, group in mie_df.groupby(['radius', 'wavelength']):
        interpolated_ddscat = np.interp(group['angle'], ddscat_df['theta'], ddscat_df['S_11'])
        error = interpolated_ddscat - group['S_11']
        plt.plot(group['angle'], error, label=f'Error: r={label[0]}, λ={label[1]}', alpha=0.7, linestyle='-', linewidth=1)
    
    plt.xlabel('Theta (degrees)')
    plt.ylabel('Error')
    plt.title('Error between DDSCAT S_11 and Mie Phase Function')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(ddscat_df['radius'], ddscat_df['Qsca'], 'o', label='DDSCAT', alpha=0.7)
    plt.plot(mie_df['radius'], mie_df['Qsca'], 'x', label='Mie', alpha=0.7)
    
    plt.xlabel('Radius')
    plt.ylabel('Qsca')
    plt.title('Comparison of Qsca Values')
    plt.legend()
    plt.show()
