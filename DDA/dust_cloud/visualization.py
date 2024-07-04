import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def extract_data(file_path):
    """
    Extracts data from a file that matches a specific data structure 
    starting from a marker. The function reads a file line-by-line 
    until it finds a line containing a data start marker, then it 
    continues to read and parse each line into a dictionary until an 
    empty line is encountered.

    Parameters:
    file_path (str): The path to the file from which data is to be 
    extracted.

    Returns:
    pd.DataFrame: A DataFrame containing the extracted data, with 
    each column converted to float type.
    """
    data_start_marker = ("theta    phi    Pol.    S_11        "                     
                         "S_12       S_21       S_22       "
                         "S_31       S_41" )   
    columns_of_interest = ['theta', 'phi', 'Pol.', 'S_11', 'S_12',  
                           'S_21', 'S_22', 'S_31', 'S_41']
    extracted_data = []

    with open(file_path, 'r') as file:
        data_section_started = False
        for line in file:
            if not data_section_started:
                if "theta" in line and "phi" in line and "Pol." in line: # Check if line contains header, flexible about spaces
                    data_section_started = True
                continue
            if line.strip() == "":
                break
            parts = line.split()
            if len(parts) == len(columns_of_interest):                   # Check that we have the correct number of data elements
                data_row = {col: float(parts[idx]) for idx, col 
                            in enumerate(columns_of_interest)}
                extracted_data.append(data_row)
            else:
                print(f"Skipping malformed line: {line}")
    return pd.DataFrame(extracted_data)

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


def plot_particle_positions(positions_all):
    """
    Plot 2D positions of particles around the Moon, color-coded by particle type.

    Parameters:
        positions_all (list): A list of tuples containing positions and labels.
    """
    fig, ax = plt.subplots()
    for positions, label in positions_all:
        ax.scatter(positions[:, 0], positions[:, 1], label=label, alpha=0.7)
    ax.set_title('Particle Positions around the Moon')
    ax.set_xlabel('X Position (km)')
    ax.set_ylabel('Y Position (km)')
    ax.legend()
    plt.show()
