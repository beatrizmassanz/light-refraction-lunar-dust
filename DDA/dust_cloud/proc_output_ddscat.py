import logging
import os
import numpy as np
import pandas as pd
import json

def process_existing_results(base_dir):
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
                labels.append(root.split('/')[-1])                     # Use the directory name as the label
    return data_files, labels


def process_ddscat_result(simulation_directory, results, sample, only_spheres= False):
    """Process a single result file and append it to the results list."""
    result_file = os.path.join(simulation_directory, 'w000r000.avg')
    if os.path.exists(result_file):
        try:
            df = extract_data(result_file)
            df['shape'] = sample['shape']
            df['size_param'] = sample['size_param']
            df['radius'] = sample['radius']
            df['wavelength'] = sample['wavelength']
            if only_spheres:
                df = normalize_s11(df, sample['size_param'], method='qsca')  # Normalize using size parameter
            results.append(df)
        except ValueError as e:
            logging.error(f"Error processing file {result_file}: {e}")

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



def normalize_s11(df, size_param, method='qsca'):
    """
    Normalize the S_11 values in the DataFrame using the specified 
    normalization method.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the S_11 values.
        method (str): The normalization method. Options are 
            'albedo', 'one', '4pi', 'qsca', 'qext', 'bohren', 
            'wiscombe'.
        
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
        raise ValueError("Invalid normalization method. Choose from \
                         'albedo', 'one', '4pi', 'qsca', 'qext', 'bohren', 'wiscombe'.")

    df['S_11'] /= normalization_constant**2
    print(f"Size parameter (x): {x}")
    print(f"Size parameter (x): {x}, Qsca: {df['Qsca'].iloc[0]}, \
          Norm factor ddscat: {normalization_constant}")
    return df
