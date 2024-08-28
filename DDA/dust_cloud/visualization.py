import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

def plot_shape_counts(results_df):
    # Remove duplicates based on shape and size_param
    unique_samples_df = results_df.drop_duplicates(subset=['shape', 'size_param'])

    # Count the number of unique spheres and rectangular prisms
    shape_counts = unique_samples_df['shape'].value_counts()

    # Plot the counts
    plt.figure(figsize=(8, 6))
    plt.bar(shape_counts.index, shape_counts.values, color=['orange', 'blue'])
    
    plt.xlabel('Shape', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.title('Count of Unique Spheres and Rectangular Prisms', fontsize=16)
    plt.xticks(ticks=[0, 1], labels=['Spheres', 'Rectangular Prisms'], fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_size_param_distribution(results_df):
    """
    Plot the distribution of unique size parameters from the simulation input data.
    Mark the mean of the distribution with a vertical line.
    
    Parameters:
    results_df (pd.DataFrame): DataFrame containing the 'size_param' column.
    """
    unique_size_params = results_df['size_param'].unique()
    mean_size_param = unique_size_params.mean()  # Calculate the mean of the unique size parameters

    plt.figure(figsize=(10, 6))
    sns.histplot(unique_size_params, kde=True, color='purple', bins=30)
    plt.axvline(mean_size_param, color='orange', linestyle='--', linewidth=2, label=f'Mean: {mean_size_param:.2f}')
    plt.xlabel('Size Parameter x', fontsize=14)
    plt.ylabel('Number of Samples', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_wavelength_distribution(results_df):
    """
    Plot the distribution of unique wavelengths from the simulation input data.
    
    Parameters:
    results_df (pd.DataFrame): DataFrame containing the 'wavelength' column.
    """
    unique_wavelengths = results_df['wavelength'].unique()
    plt.figure(figsize=(10, 6))
    sns.histplot(unique_wavelengths, kde=True, color='green', bins=30)
    plt.xlabel('Wavelength (µm)', fontsize=14)
    plt.ylabel('Number of Samples', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_radius_distribution(results_df):
    """
    Plot the distribution of unique particle radii from the simulation input data.
    Mark the mean of the distribution with a vertical line.
    
    Parameters:
    results_df (pd.DataFrame): DataFrame containing the 'radius' column.
    """
    unique_radii = results_df['radius'].unique()
    mean_radius = unique_radii.mean()  # Calculate the mean of the unique radii

    plt.figure(figsize=(10, 6))
    sns.histplot(unique_radii, kde=True, color='blue', bins=30)
    plt.axvline(mean_radius, color='orange', linestyle='--', linewidth=2, label=f'Mean: {mean_radius:.2f} µm')
    plt.xlabel('Particle Radius (µm)', fontsize=14)
    plt.ylabel('Number of Samples', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_qsca_vs_size(results_df):
    plt.figure(figsize=(10, 6))

    # Combine data from both shapes into a single scatter plot
    sns.regplot(
        x=results_df['radius'],
        y=results_df['Qsca'],
        scatter_kws={'s': 20, 'alpha': 0.7, 'color': 'gray'},
        line_kws={'color': 'black'}
    )

    # Plot spheres with orange circles
    plt.scatter(
        results_df[results_df['shape'] == 'SPHERE']['radius'],
        results_df[results_df['shape'] == 'SPHERE']['Qsca'],
        color='orange', marker='o', s=20, label='Spheres', alpha=0.7
    )
    
    # Plot rectangular prisms with blue squares
    plt.scatter(
        results_df[results_df['shape'] == 'RCTGLPRSM']['radius'],
        results_df[results_df['shape'] == 'RCTGLPRSM']['Qsca'],
        color='blue', marker='s', s=20, label='Rectangular Prisms', alpha=0.7
    )

    plt.xlabel('Particle Size (radius)', fontsize=14)
    plt.ylabel(r'Q$_{sca}$', fontsize=14)  # Sca as subscript
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Function to plot Qsca vs Wavelength with distinction between shapes
def plot_qsca_vs_wavelength(results_df):
    plt.figure(figsize=(10, 6))

    # Plot spheres with orange circles
    plt.scatter(
        results_df[results_df['shape'] == 'SPHERE']['wavelength'],
        results_df[results_df['shape'] == 'SPHERE']['Qsca'],
        color='orange', marker='o', s=20, label='Spheres', alpha=0.7
    )
    
    # Plot rectangular prisms with blue squares
    plt.scatter(
        results_df[results_df['shape'] == 'RCTGLPRSM']['wavelength'],
        results_df[results_df['shape'] == 'RCTGLPRSM']['Qsca'],
        color='blue', marker='s', s=20, label='Rectangular Prisms', alpha=0.7
    )

    plt.xlabel('Wavelength', fontsize=14)
    plt.ylabel(r'Q$_{sca}$', fontsize=14)  # Sca as subscript
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Function to plot S11 (Forward Scattering) vs Particle Size
def plot_s11_vs_size_forward_scattering(results_df):
    plt.figure(figsize=(10, 6))

    # Filter DataFrame for forward scattering where theta = 0 degrees and phi = 0 degrees
    forward_df = results_df[(results_df['theta'] == 0) & (results_df['phi'] == 0)]

    # Plot spheres with orange circles
    plt.scatter(
        forward_df[forward_df['shape'] == 'SPHERE']['radius'],
        forward_df[forward_df['shape'] == 'SPHERE']['S_11'],
        color='orange', marker='o', s=20, label='Spheres', alpha=0.7
    )
    
    # Plot rectangular prisms with blue squares
    plt.scatter(
        forward_df[forward_df['shape'] == 'RCTGLPRSM']['radius'],
        forward_df[forward_df['shape'] == 'RCTGLPRSM']['S_11'],
        color='blue', marker='s', s=20, label='Rectangular Prisms', alpha=0.7
    )

    plt.xlabel('Particle Size (radius)', fontsize=14)
    plt.ylabel('S₁₁ (Forward Scattering, θ=0°)', fontsize=14, fontweight='normal')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Function to plot S11 (Forward Scattering) vs Wavelength
def plot_s11_vs_wavelength_forward_scattering(results_df):
    plt.figure(figsize=(10, 6))

    # Filter DataFrame for forward scattering where theta = 0 degrees and phi = 0 degrees
    forward_df = results_df[(results_df['theta'] == 0) & (results_df['phi'] == 0)]

    # Plot spheres with orange circles
    plt.scatter(
        forward_df[forward_df['shape'] == 'SPHERE']['wavelength'],
        forward_df[forward_df['shape'] == 'SPHERE']['S_11'],
        color='orange', marker='o', s=20, label='Spheres', alpha=0.7
    )
    
    # Plot rectangular prisms with blue squares
    plt.scatter(
        forward_df[forward_df['shape'] == 'RCTGLPRSM']['wavelength'],
        forward_df[forward_df['shape'] == 'RCTGLPRSM']['S_11'],
        color='blue', marker='s', s=20, label='Rectangular Prisms', alpha=0.7
    )

    plt.xlabel('Wavelength', fontsize=14)
    plt.ylabel('S₁₁ (Forward Scattering, θ=0°)', fontsize=14, fontweight='normal')  # Subscript for S_11
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Function for plotting Q_sca for individual size parameters
def plot_qsca_by_size(results_df):
    plt.figure(figsize=(10, 6))
    # Plot spheres with orange circles
    plt.scatter(
        results_df[results_df['shape'] == 'SPHERE']['size_param'],
        results_df[results_df['shape'] == 'SPHERE']['Qsca'],
        color='orange', marker='o', s=10, label='Spheres'
    )
    # Plot rectangular prisms with blue squares
    plt.scatter(
        results_df[results_df['shape'] == 'RCTGLPRSM']['size_param'],
        results_df[results_df['shape'] == 'RCTGLPRSM']['Qsca'],
        color='blue', marker='s', s=10, label='Rectangular Prisms'
    )
    plt.xlabel(r'Size Parameter $x$', fontsize=14)
    plt.ylabel(r'$Q_{sca}$', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Similar plots for Q_bk and Q_pol
def plot_qbk_by_size(results_df):
    plt.figure(figsize=(10, 6))
    # Plot spheres with orange circles
    plt.scatter(
        results_df[results_df['shape'] == 'SPHERE']['size_param'],
        results_df[results_df['shape'] == 'SPHERE']['Qbk'],
        color='orange', marker='o', s=10, label='Spheres'
    )
    # Plot rectangular prisms with blue squares
    plt.scatter(
        results_df[results_df['shape'] == 'RCTGLPRSM']['size_param'],
        results_df[results_df['shape'] == 'RCTGLPRSM']['Qbk'],
        color='blue', marker='s', s=10, label='Rectangular Prisms'
    )
    plt.xlabel(r'Size Parameter $x$', fontsize=14)
    plt.ylabel(r'$Q_{bk}$', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_qpol_by_size(results_df):
    plt.figure(figsize=(10, 6))
    # Plot spheres with orange circles
    plt.scatter(
        results_df[results_df['shape'] == 'SPHERE']['size_param'],
        results_df[results_df['shape'] == 'SPHERE']['Qpol'],
        color='orange', marker='o', s=10, label='Spheres'
    )
    # Plot rectangular prisms with blue squares
    plt.scatter(
        results_df[results_df['shape'] == 'RCTGLPRSM']['size_param'],
        results_df[results_df['shape'] == 'RCTGLPRSM']['Qpol'],
        color='blue', marker='s', s=10, label='Rectangular Prisms'
    )
    plt.xlabel(r'Size Parameter $x$', fontsize=14)
    plt.ylabel(r'$Q_{pol}$', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_s11_forward_scattering(results_df):
    """
    Plot S_11 values for forward scattering (θ = 0 degrees and φ = 0 degrees) across size parameters.
    
    Parameters:
    results_df (pd.DataFrame): DataFrame containing 'size_param', 'S_11', 'theta', 'phi', and 'shape' columns.
    """
    plt.figure(figsize=(10, 6))

    # Filter DataFrame for forward scattering where theta = 0 degrees and phi = 0 degrees
    forward_df = results_df[(results_df['theta'] == 0) & (results_df['phi'] == 0)]

    # Plot spherical particles with orange circles
    sph_df = forward_df[forward_df['shape'] == 'SPHERE']
    plt.scatter(sph_df['size_param'], sph_df['S_11'], color='darkorange', label='Sphere', marker='o', s=10)

    # Plot rectangular prisms with blue boxes
    rect_df = forward_df[forward_df['shape'] == 'RCTGLPRSM']
    plt.scatter(rect_df['size_param'], rect_df['S_11'], color='royalblue', label='Rectangular Prism', marker='s', s=10)

    plt.xlabel(r'Size Parameter $x$', fontsize=16, fontweight='normal')
    plt.ylabel(r'$S_{11}$ (Forward Scattering, $\theta=0^\circ$)', fontsize=16, fontweight='normal')
    plt.legend(fontsize=12, title='Shape', title_fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_s11_back_scattering(results_df):
    """
    Plot S_11 values for backscattering (θ = 180 degrees and φ = 0 degrees) across size parameters.
    
    Parameters:
    results_df (pd.DataFrame): DataFrame containing 'size_param', 'S_11', 'theta', 'phi', and 'shape' columns.
    """
    plt.figure(figsize=(10, 6))

    # Filter DataFrame for backscattering where theta = 180 degrees and phi = 0 degrees
    back_df = results_df[(results_df['theta'] == 180) & (results_df['phi'] == 0)]

    # Plot spherical particles with orange circles
    sph_df = back_df[back_df['shape'] == 'SPHERE']
    plt.scatter(sph_df['size_param'], sph_df['S_11'], color='darkorange', label='Sphere', marker='o', s=10)

    # Plot rectangular prisms with blue boxes
    rect_df = back_df[back_df['shape'] == 'RCTGLPRSM']
    plt.scatter(rect_df['size_param'], rect_df['S_11'], color='royalblue', label='Rectangular Prism', marker='s', s=10)

    plt.xlabel(r'Size Parameter $x$', fontsize=16, fontweight='normal')
    plt.ylabel(r'$S_{11}$ (Back Scattering, $\theta=180^\circ$)', fontsize=16, fontweight='normal')
    plt.legend(fontsize=12, title='Shape', title_fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_pol_vs_theta(results_df):
    """
    Plot the Pol. values against theta angles for different shapes.
    
    Parameters:
    results_df (pd.DataFrame): DataFrame containing 'theta', 'Pol.', and 'shape' columns.
    """
    plt.figure(figsize=(10, 6))

    # Plot spherical particles with orange circles
    sph_df = results_df[results_df['shape'] == 'SPHERE']
    plt.plot(sph_df['theta'], sph_df['Pol.'], color='orange', marker='o', linestyle='-', markersize=5, label='Sphere')

    # Plot rectangular prisms with blue squares
    rect_df = results_df[results_df['shape'] == 'RCTGLPRSM']
    plt.plot(rect_df['theta'], rect_df['Pol.'], color='blue', marker='s', linestyle='-', markersize=5, label='Rectangular Prism')

    # Add labels and legend
    plt.xlabel(r'$\theta$ (degrees)', fontsize=14)
    plt.ylabel(r'Pol.', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)  # Add grid with dashed lines and transparency
    plt.tight_layout()
    plt.show()

def plot_average_pol_vs_theta(results_df):
    """
    Plot the average Pol. values against theta angles for different shapes.
    
    Parameters:
    results_df (pd.DataFrame): DataFrame containing 'theta', 'Pol.', and 'shape' columns.
    """
    # Group by 'theta' and 'shape' and calculate the mean of 'Pol.'
    grouped_df = results_df.groupby(['theta', 'shape']).mean().reset_index()

    plt.figure(figsize=(10, 6))

    # Plot average Pol. for spherical particles with orange circles
    sph_df = grouped_df[grouped_df['shape'] == 'SPHERE']
    plt.plot(sph_df['theta'], sph_df['Pol.'], color='orange', marker='o', linestyle='-', markersize=5, label='Sphere')

    # Plot average Pol. for rectangular prisms with blue squares
    rect_df = grouped_df[grouped_df['shape'] == 'RCTGLPRSM']
    plt.plot(rect_df['theta'], rect_df['Pol.'], color='blue', marker='s', linestyle='-', markersize=5, label='Rectangular Prism')

    # Add labels and legend
    plt.xlabel(r'$\theta$ (degrees)', fontsize=14)
    plt.ylabel(r'Pol.', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)  # Add grid with dashed lines and transparency
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import pandas as pd

def plot_average_s11_vs_theta(results_df):
    """
    Plot the average S_11 values against theta angles for different shapes.
    
    Parameters:
    results_df (pd.DataFrame): DataFrame containing 'theta', 'S_11', and 'shape' columns.
    """
    # Group by 'theta' and 'shape' and calculate the mean of 'S_11'
    grouped_df = results_df.groupby(['theta', 'shape']).mean().reset_index()

    plt.figure(figsize=(10, 6))

    # Plot average S_11 for spherical particles with orange circles
    sph_df = grouped_df[grouped_df['shape'] == 'SPHERE']
    plt.plot(sph_df['theta'], sph_df['S_11'], color='orange', marker='o', linestyle='-', markersize=5, label='Sphere')

    # Plot average S_11 for rectangular prisms with blue squares
    rect_df = grouped_df[grouped_df['shape'] == 'RCTGLPRSM']
    plt.plot(rect_df['theta'], rect_df['S_11'], color='blue', marker='s', linestyle='-', markersize=5, label='Rectangular Prism')

    # Add labels and legend
    plt.xlabel(r'$\theta$ (degrees)', fontsize=14)
    plt.ylabel(r'S$_{11}$', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)  # Add grid with dashed lines and transparency
    plt.tight_layout()
    plt.show()


def plot_ddscat_correlation_results(results_df):
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
    # Set plot style for better aesthetics
    sns.set(style="whitegrid")

    # Create the figure
    plt.figure(figsize=(10, 6))

    # Loop over the data frames and plot them in orange
    for df, label in zip(data_frames, labels):
        plt.plot(df[df["phi"] == 0]["theta"], 
                 df[df["phi"] == 0]["S_11"], 
                 label=label, color='orange', 
                 alpha=0.8, linestyle='-', linewidth=2)

    # Set the title, labels, and legend
    plt.xlabel('Theta (degrees)', fontsize=14)
    plt.ylabel(r'S$_{11}$', fontsize=14)
    plt.legend(title="Samples", fontsize=11, title_fontsize=13)

    # Display the plot
    plt.tight_layout()
    plt.show()



def plot_polar_data(data_frames, labels):
    """
    Plots data from multiple DataFrames on a polar plot, assuming 
    each DataFrame contains 'theta' and 'S_11' columns.
    Theta should be in degrees for plotting.

    Parameters:
    data_frames (list of pd.DataFrame): List of DataFrames to plot.
    labels (list of str): Labels for each DataFrame plot.
    """
    # Set plot style for better aesthetics
    sns.set(style="whitegrid")

    # Create a larger polar plot
    plt.figure(figsize=(10, 10))

    # Create a polar subplot
    ax = plt.subplot(111, polar=True)
    
    # Loop over the data frames and plot them in orange
    for df, label in zip(data_frames, labels):
        # Convert theta from degrees to radians for polar plotting
        radians = np.deg2rad(df[df["phi"] == 0]["theta"])
        ax.plot(radians, df[df["phi"] == 0]["S_11"], 
                label=label, color='orange', 
                alpha=0.8, linestyle='-', linewidth=2)

    # Set 0 degrees to be at the top (north) and direction to clockwise
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    # Set the radial limits to focus on the central values and improve readability
    ax.set_ylim(0, np.percentile([df["S_11"].max() for df in data_frames], 95))  # Focus on the central 95% of the data

    # Customize grid and ticks for better readability
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)

    # Adjust the position and appearance of the legend
    ax.legend(title="Samples", title_fontsize='13', fontsize='11', 
              loc='upper right', bbox_to_anchor=(1.2, 1.05))

    # Display the plot with a tight layout for better spacing
    plt.tight_layout()
    plt.show()

'''

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
'''
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
