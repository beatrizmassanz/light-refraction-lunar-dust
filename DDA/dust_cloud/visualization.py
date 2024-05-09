import subprocess
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def convert_to_vtk(simulation_directory, target_file_name):
    """
    Converts a DDSCAT shape file to a VTK file using vtrconvert.

    Parameters:
    simulation_directory (str): Path to the directory containing the input file 'target.out'.
    target_file_name (str): Base name for the output VTK file.
    """
    # Save the current directory to revert back to it later
    original_directory = os.getcwd()

    # Change to the directory containing the input file
    os.chdir(simulation_directory)
    
    # Relative path to the vtrconvert tool, assuming it's two directories up in 'src'
    vtrconvert_path = os.path.join('..', '..', 'src', 'vtrconvert')

    # Construct the vtrconvert command
    command = f"{vtrconvert_path} target.out {target_file_name}"
    
    # Execute the command
    try:
        output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
        print("Conversion successful:", output.decode())
    except subprocess.CalledProcessError as e:
        print("Error during conversion:", e.output.decode())

    # Change back to the original directory
    os.chdir(original_directory)

def extract_data(file_path):
    data_start_marker = "theta    phi    Pol.    S_11        S_12        S_21       S_22       S_31       S_41"
    extracted_data = []
    with open(file_path, 'r') as file:
        data_section_started = False
        for line in file:
            if data_start_marker in line:
                data_section_started = True
                continue
            if data_section_started and line.strip() == "":
                break
            if data_section_started:
                parts = line.split()
                data_row = {col: parts[idx] for idx, col in enumerate(columns_of_interest)}
                extracted_data.append(data_row)
    return pd.DataFrame(extracted_data).astype(float)

columns_of_interest = ['theta', 'phi', 'Pol.', 'S_11', 'S_12', 'S_21', 'S_22', 'S_31', 'S_41']

def plot_data(data_frames, labels):
    plt.figure(figsize=(10, 6))
    for df, label in zip(data_frames, labels):
        plt.plot(df[df["phi"] == 0]["theta"], df[df["phi"] == 0]["S_11"], label=label, alpha=0.7, markersize=5, linestyle='-', linewidth=1)
    plt.xlabel('Theta')
    plt.ylabel('S_11')
    plt.title('Comparison of S_11 Values')
    plt.legend()
    plt.show()

def find_data_files(base_dir):
    """
    Find all 'w000r000.avg' files in the base directory.
    """
    data_files = []
    labels = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file == 'w000r000.avg':
                data_files.append(os.path.join(root, file))
                labels.append(root.split('/')[-1])  # Use the directory name as the label
    return data_files, labels



