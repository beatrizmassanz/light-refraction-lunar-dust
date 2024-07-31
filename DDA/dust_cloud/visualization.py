import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import glob
import json
import logging
import subprocess

def find_data_files(base_dir):
    """
    Finds all 'w000r000.avg' files within a specified base directory.

    Parameters:
    base_dir (str): The directory to search for 'w000r000.avg' files.

    Returns:
    tuple: A tuple containing two lists, one with the file paths and 
    another with corresponding labels derived from directory names.
    """
    data_files = []
    labels = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file == 'w000r000.avg':
                data_files.append(os.path.join(root, file))
                labels.append(root.split('/')[-1])                       # Use the directory name as the label
    return data_files, labels

def load_sample_parameters(simulation_directory):
    """Load the sample parameters from a file in the simulation directory."""
    sample_file_path = os.path.join(simulation_directory, 'sample_parameters.json')
    with open(sample_file_path, 'r') as f:
        sample = json.load(f)
    return sample

def load_samples(file_path):
    """Load samples from a file."""
    with open(file_path, "r") as f:
        samples = json.load(f)
    return samples

def normalize_s11(df, size_param, method='qsca'):
    """
    Normalize the S_11 values in the DataFrame using the specified normalization method.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the S_11 values.
        method (str): The normalization method. Options are 'albedo', 'one', '4pi', 'qsca', 'qext', 'bohren', 'wiscombe'.
        
    Returns:
        pd.DataFrame: The DataFrame with normalized S_11 values.
    """
    normalization_constant = None
    x = size_param

    if method == 'albedo':
        normalization_constant = x * np.sqrt(np.pi * df['Qbk'].iloc[0])
    elif method == 'one':
        normalization_constant = x * np.sqrt(df['Qsca'].iloc[0] * np.pi)
    elif method == '4pi':
        normalization_constant = x * np.sqrt(df['Qsca'].iloc[0] / 4)
    elif method == 'qsca':
        normalization_constant = x * np.sqrt(np.pi)
    elif method == 'qext':
        normalization_constant = x * np.sqrt(df['Qsca'].iloc[0] * np.pi / df['Qbk'].iloc[0])
    elif method == 'bohren':
        normalization_constant = 0.5
    elif method == 'wiscombe':
        normalization_constant = 1
    else:
        raise ValueError("Invalid normalization method. Choose from 'albedo', 'one', '4pi', 'qsca', 'qext', 'bohren', 'wiscombe'.")

    df['S_11'] /= normalization_constant**2
    print(f"Size parameter (x): {x}")
    print(f"Size parameter (x): {x}, Qsca: {df['Qsca'].iloc[0]}, Norm factor ddscat: {normalization_constant}")
    return df

def extract_data(file_path):
    """
    Extracts data from a file that matches a specific data structure 
    starting from a marker. The function reads a file line-by-line 
    until it finds a line containing a data start marker, then it 
    continues to read and parse each line into a dictionary until an 
    empty line is encountered. Additionally, it extracts Qsca, Qbk, 
    and Qpol values from the file.

    Parameters:
    file_path (str): The path to the file from which data is to be 
    extracted.

    Returns:
    pd.DataFrame: A DataFrame containing the extracted data, with 
    each column converted to float type.
    """
    columns_of_interest = ['theta', 'phi', 'Pol.', 'S_11', 'S_12', 'S_21', 'S_22', 'S_31', 'S_41']
    extracted_data = []
    qsca, qbk, qpol = None, None, None

    with open(file_path, 'r') as file:
        data_section_started = False
        mean_line_found = False
        for line in file:
            if "mean:" in line and not mean_line_found:  # Check if line contains mean Q values
                parts = line.split( )
                try:
                    qsca = float(parts[3])
                    qbk = float(parts[6])
                    print(f"qsca: {qsca}")
                    print(f"qbk: {qbk}")
                    mean_line_found = True
                except (IndexError, ValueError) as e:
                    print(f"Error extracting Qsca or Qbk from line: {line}")
                    continue
            if "Qpol=" in line:
                try:
                    qpol = float(line.split()[1])
                    print(f"qpol: {qpol}")
                except (IndexError, ValueError) as e:
                    print(f"Error extracting Qpol from line: {line}")
                    continue
            if not data_section_started:
                if "theta" in line and "phi" in line and "Pol." in line:  # Check if line contains header
                    data_section_started = True
                continue
            if line.strip() == "":
                break
            parts = line.split()
            if len(parts) == len(columns_of_interest):  # Check that we have the correct number of data elements
                try:
                    data_row = {col: float(parts[idx]) for idx, col in enumerate(columns_of_interest)}
                    extracted_data.append(data_row)
                except ValueError as e:
                    print(f"Skipping malformed line: {line}")
                    continue

    df = pd.DataFrame(extracted_data)
    df['Qsca'] = qsca
    df['Qbk'] = qbk
    df['Qpol'] = qpol
    return df

def process_ddscat_result(simulation_directory, results, sample):
    """Process a single result file and append it to the results list."""
    result_file = os.path.join(simulation_directory, 'w000r000.avg')
    if os.path.exists(result_file):
        try:
            df = extract_data(result_file)
            df['shape'] = sample['shape']
            df['size_param'] = sample['size_param']
            df['radius'] = sample['radius']
            df['wavelength'] = sample['wavelength']
            df = normalize_s11(df, sample['size_param'], method='qsca')  # Normalize using size parameter
            results.append(df)
        except ValueError as e:
            logging.error(f"Error processing file {result_file}: {e}")

def process_mie_result(mie_results, results, samples):
    """
    Process the Mie results and match them with the DDSCAT results.

    Parameters:
        mie_results (DataFrame): DataFrame containing the Mie calculation results.
        results (list): List to store the final processed results.
        samples (list): List of sample parameters used in the simulations.
    """
    for sample in samples:
        if sample["shape"] != "SPHERE":
            continue
        
        mie_sample_results = mie_results[(mie_results['radius'] == sample['radius']) & 
                                         (mie_results['wavelength'] == sample['wavelength'])]
        for _, mie_result in mie_sample_results.iterrows():
            result = {
                "theta": mie_result['angle'],
                "phi": 0,  # Assuming phi = 0 for simplicity
                "Pol.": 0,  # Assuming unpolarized light for simplicity
                "S_11": mie_result['S_11'],
                "S_12": None,  # Not calculated in this example
                "S_21": None,  # Not calculated in this example
                "S_22": None,  # Not calculated in this example
                "S_31": None,  # Not calculated in this example
                "S_41": None,  # Not calculated in this example
                "Qsca": mie_result['Qsca'],
                "Qbk": mie_result['Qback'],
                "Qpol": None,  # Not calculated in this example
                "shape": sample['shape'],
                "size_param": sample['size_param'],
                "radius": sample['radius'],
                "wavelength": sample['wavelength']
            }
            results.append(result)

    return results

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
        process_ddscat_result(simulation_directory, results, sample)  # Process the result

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
