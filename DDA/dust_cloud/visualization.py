import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import gen_input as gen_input
import os
import scipy.stats as stats
import proc_output_ddscat as proc_output_ddscat


def plot_s11_for_selected_wavelengths(results_df):
    """
    Plot S_11 values against theta angles for particles with size parameters 
    close to specified values (0.5, 1, 1.5, 2, 2.5, 3, 3.5), considering only 
    data with phi = 0 degrees.

    Parameters:
    results_df (pd.DataFrame): DataFrame containing 'theta', 'S_11', 'shape', 'size_param', and 'phi' columns.
    """
    # Define the target size parameters
    target_size_params = [0.3, 0.4, 0.5, 0.6, 0.7]

    plt.figure(figsize=(10, 6))

    # Filter the DataFrame to only include data where phi = 0 degrees
    phi_zero_df = results_df[results_df['phi'] == 0.0]

    # Verify that filtering worked as expected
    print(f"Filtered DataFrame with phi=0:\n{phi_zero_df[['theta', 'phi', 'wavelength', 'S_11']].head()}")

    # Select and plot the S_11 vs. theta for each target size parameter
    for target in target_size_params:
        # Find the single closest size parameter to the target
        closest_size_param = phi_zero_df.iloc[(phi_zero_df['wavelength'] - target).abs().argsort()[:1]]['wavelength'].values[0]
        particle_df = phi_zero_df[phi_zero_df['wavelength'] == closest_size_param]

        # Ensure that only one unique particle is selected
        particle_df = particle_df.drop_duplicates(subset=['wavelength', 'theta'])

        # Sort by theta to ensure correct plotting
        particle_df = particle_df.sort_values('theta')

        # Get the shape of the particle for labeling
        shape = particle_df['shape'].iloc[0]

        # Plot with a label that includes both the size parameter and the shape
        plt.plot(particle_df['theta'], particle_df['S_11'], label=f'λ = {closest_size_param}, {shape}')

    # Add labels and legend
    plt.xlabel(r'$\theta$ (degrees)', fontsize=14)
    plt.ylabel(r'S$_{11}$', fontsize=14)
    plt.legend(title='Wavelength and Shape')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_pol_for_selected_wavelengths(results_df, tolerance=0.01):
    """
    Plot Pol. values against theta angles for particles with size parameters 
    close to specified values (0.5, 1, 1.5, 2, 2.5, 3, 3.5), considering only 
    data with phi = 0 degrees.

    Parameters:
    results_df (pd.DataFrame): DataFrame containing 'theta', 'Pol.', 'shape', 'size_param', and 'phi' columns.
    """
    # Define the target size parameters
    target_size_params = [0.3, 0.4, 0.5, 0.6, 0.7]

    plt.figure(figsize=(10, 6))

    # Filter the DataFrame to only include data where phi = 0 degrees
    phi_zero_df = results_df[results_df['phi'] == 0.0]

    # Verify that filtering worked as expected
    print(f"Filtered DataFrame with phi=0:\n{phi_zero_df[['theta', 'phi', 'wavelength', 'Pol.']].head()}")

    # Select and plot the Pol. vs. theta for each target size parameter
    for target in target_size_params:
        # Find the single closest size parameter to the target
        closest_size_param = phi_zero_df.iloc[(phi_zero_df['wavelength'] - target).abs().argsort()[:1]]['wavelength'].values[0]
        particle_df = phi_zero_df[phi_zero_df['wavelength'] == closest_size_param]

        # Ensure that only one unique particle is selected
        particle_df = particle_df.drop_duplicates(subset=['wavelength', 'theta'])

        # Sort by theta to ensure correct plotting
        particle_df = particle_df.sort_values('theta')

        # Get the shape of the particle for labeling
        shape = particle_df['shape'].iloc[0]

        # Plot with a label that includes both the size parameter and the shape
        plt.plot(particle_df['theta'], particle_df['Pol.'], label=f'λ = {closest_size_param}, {shape}')

    # Add labels and legend
    plt.xlabel(r'$\theta$ (degrees)', fontsize=14)
    plt.ylabel(r'Pol.', fontsize=14)
    plt.legend(title='Wavelength and Shape')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_s11_for_selected_size_params(results_df):
    """
    Plot S_11 values against theta angles for particles with size parameters 
    close to specified values (0.5, 1, 1.5, 2, 2.5, 3, 3.5), considering only 
    data with phi = 0 degrees.

    Parameters:
    results_df (pd.DataFrame): DataFrame containing 'theta', 'S_11', 'shape', 'size_param', and 'phi' columns.
    """
    # Define the target size parameters
    target_size_params = [0.5, 1, 1.5, 2, 2.5, 3]

    plt.figure(figsize=(10, 6))

    # Filter the DataFrame to only include data where phi = 0 degrees
    phi_zero_df = results_df[results_df['phi'] == 0.0]

    # Verify that filtering worked as expected
    print(f"Filtered DataFrame with phi=0:\n{phi_zero_df[['theta', 'phi', 'size_param', 'S_11']].head()}")

    # Select and plot the S_11 vs. theta for each target size parameter
    for target in target_size_params:
        # Find the single closest size parameter to the target
        closest_size_param = phi_zero_df.iloc[(phi_zero_df['size_param'] - target).abs().argsort()[:1]]['size_param'].values[0]
        particle_df = phi_zero_df[phi_zero_df['size_param'] == closest_size_param]

        # Ensure that only one unique particle is selected
        particle_df = particle_df.drop_duplicates(subset=['size_param', 'theta'])

        # Sort by theta to ensure correct plotting
        particle_df = particle_df.sort_values('theta')

        # Get the shape of the particle for labeling
        shape = particle_df['shape'].iloc[0]

        # Plot with a label that includes both the size parameter and the shape
        plt.plot(particle_df['theta'], particle_df['S_11'], label=f'x = {closest_size_param}, {shape}')

    # Add labels and legend
    plt.xlabel(r'$\theta$ (degrees)', fontsize=14)
    plt.ylabel(r'S$_{11}$', fontsize=14)
    plt.legend(title='Size Parameter and Shape')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_pol_for_selected_size_params(results_df):
    """
    Plot Pol. values against theta angles for particles with size parameters 
    close to specified values (0.5, 1, 1.5, 2, 2.5, 3, 3.5), considering only 
    data with phi = 0 degrees.

    Parameters:
    results_df (pd.DataFrame): DataFrame containing 'theta', 'Pol.', 'shape', 'size_param', and 'phi' columns.
    """
    # Define the target size parameters
    target_size_params = [0.5, 1, 1.5, 2, 2.5, 3, 3.5]

    plt.figure(figsize=(10, 6))

    # Filter the DataFrame to only include data where phi = 0 degrees
    phi_zero_df = results_df[results_df['phi'] == 0.0]

    # Verify that filtering worked as expected
    print(f"Filtered DataFrame with phi=0:\n{phi_zero_df[['theta', 'phi', 'size_param', 'Pol.']].head()}")

    # Select and plot the Pol. vs. theta for each target size parameter
    for target in target_size_params:
        # Find the single closest size parameter to the target
        closest_size_param = phi_zero_df.iloc[(phi_zero_df['size_param'] - target).abs().argsort()[:1]]['size_param'].values[0]
        particle_df = phi_zero_df[phi_zero_df['size_param'] == closest_size_param]

        # Ensure that only one unique particle is selected
        particle_df = particle_df.drop_duplicates(subset=['size_param', 'theta'])

        # Sort by theta to ensure correct plotting
        particle_df = particle_df.sort_values('theta')

        # Get the shape of the particle for labeling
        shape = particle_df['shape'].iloc[0]

        # Plot with a label that includes both the size parameter and the shape
        plt.plot(particle_df['theta'], particle_df['Pol.'], label=f'x = {closest_size_param}, {shape}')

    # Add labels and legend
    plt.xlabel(r'$\theta$ (degrees)', fontsize=14)
    plt.ylabel(r'Pol.', fontsize=14)
    plt.legend(title='Size Parameter and Shape')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_s11_for_selected_size_params(results_df):
    """
    Plot S_11 values against theta angles for particles with size parameters 
    close to specified values (0.5, 1, 1.5, 2, 2.5, 3, 3.5), considering only 
    data with phi = 0 degrees.

    Parameters:
    results_df (pd.DataFrame): DataFrame containing 'theta', 'S_11', 'shape', 'size_param', and 'phi' columns.
    """
    # Define the target size parameters
    target_size_params = [0.5, 1, 1.5, 2, 2.5, 3]

    plt.figure(figsize=(10, 6))

    # Filter the DataFrame to only include data where phi = 0 degrees
    phi_zero_df = results_df[results_df['phi'] == 0.0]

    # Verify that filtering worked as expected
    print(f"Filtered DataFrame with phi=0:\n{phi_zero_df[['theta', 'phi', 'size_param', 'S_11']].head()}")

    # Select and plot the S_11 vs. theta for each target size parameter
    for target in target_size_params:
        # Find the single closest size parameter to the target
        closest_size_param = phi_zero_df.iloc[(phi_zero_df['size_param'] - target).abs().argsort()[:1]]['size_param'].values[0]
        particle_df = phi_zero_df[phi_zero_df['size_param'] == closest_size_param]

        # Ensure that only one unique particle is selected
        particle_df = particle_df.drop_duplicates(subset=['size_param', 'theta'])

        # Sort by theta to ensure correct plotting
        particle_df = particle_df.sort_values('theta')

        # Get the shape of the particle for labeling
        shape = particle_df['shape'].iloc[0]

        # Plot with a label that includes both the size parameter and the shape
        plt.plot(particle_df['theta'], particle_df['S_11'], label=f'x = {closest_size_param}, {shape}')

    # Add labels and legend
    plt.xlabel(r'$\theta$ (degrees)', fontsize=14)
    plt.ylabel(r'S$_{11}$', fontsize=14)
    plt.legend(title='Size Parameter and Shape')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_s11_for_selected_size_params_polar(results_df):
    """
    Plot S_11 values against theta angles for particles with size parameters 
    close to specified values (0.5, 1, 1.5, 2, 2.5, 3, 3.5) on a polar plot, 
    considering only data with phi = 0 degrees.

    Parameters:
    results_df (pd.DataFrame): DataFrame containing 'theta', 'S_11', 'shape', 'size_param', and 'phi' columns.
    """
    # Define the target size parameters
    target_size_params = [0.5, 1, 1.5, 2, 2.5, 3]

    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111, polar=True)

    # Filter the DataFrame to only include data where phi = 0 degrees
    phi_zero_df = results_df[results_df['phi'] == 0.0]

    # Verify that filtering worked as expected
    print(f"Filtered DataFrame with phi=0:\n{phi_zero_df[['theta', 'phi', 'size_param', 'S_11']].head()}")

    # Select and plot the S_11 vs. theta for each target size parameter
    for target in target_size_params:
        # Find the single closest size parameter to the target
        closest_size_param = phi_zero_df.iloc[(phi_zero_df['size_param'] - target).abs().argsort()[:1]]['size_param'].values[0]
        particle_df = phi_zero_df[phi_zero_df['size_param'] == closest_size_param]

        # Ensure that only one unique particle is selected
        particle_df = particle_df.drop_duplicates(subset=['size_param', 'theta'])

        # Sort by theta to ensure correct plotting
        particle_df = particle_df.sort_values('theta')

        # Get the shape of the particle for labeling
        shape = particle_df['shape'].iloc[0]

        # Convert theta to radians for polar plotting
        theta_radians = np.deg2rad(particle_df['theta'])

        # Plot with a label that includes both the size parameter and the shape
        ax.plot(theta_radians, particle_df['S_11'], label=f'x = {closest_size_param}, {shape}')

    # Add labels and legend
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    plt.legend(title='Size Parameter and Shape', bbox_to_anchor=(1.05, 1.05))
    plt.tight_layout()
    plt.show()

def plot_qpol_vs_size_param_and_radius(results_df):
    """
    Plot the distribution of Qpol and size parameter values.
    The y-axis represents the number of samples.
    
    Parameters:
    results_df (pd.DataFrame): DataFrame containing 'size_param' and 'Qpol' columns.
    """
    # Ensure the Qpol values are absolute
    results_df['abs_Qpol'] = np.abs(results_df['Qpol'])

    # Drop duplicates based on size_param to ensure each sample is counted only once
    unique_samples = results_df.drop_duplicates(subset='size_param')

    # Calculate histograms for Qpol and size_param
    size_param_counts, size_param_bins = np.histogram(unique_samples['size_param'], bins=30)
    qpol_sum, _ = np.histogram(unique_samples['size_param'], bins=size_param_bins, weights=unique_samples['abs_Qpol'])

    plt.figure(figsize=(12, 6))

    # Plot the distribution of |Qpol| values
    plt.bar(size_param_bins[:-1], qpol_sum, width=np.diff(size_param_bins), color='red', alpha=0.5, label=r'|Q$_{pol}$|')

    # Plot the distribution of size parameters
    plt.bar(size_param_bins[:-1], size_param_counts, width=np.diff(size_param_bins), color='blue', alpha=0.5, label='Size Parameter')

    plt.xlabel('Size Parameter x', fontsize=14)
    plt.ylabel('Number of Samples', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_max_pol_angle_vs_size_param(results_df):
    """
    Plot the maximum polarization angle (theta angle for which Pol. is highest)
    with respect to the size parameter of each particle, distinguishing between spheres and rectangular prisms.
    
    Parameters:
    results_df (pd.DataFrame): DataFrame containing 'theta', 'Pol.', 'size_param', and 'shape' columns.
    """
    # Group by size_param and shape, and find the theta with maximum Pol. for each group
    max_pol_angles = results_df.loc[results_df.groupby(['size_param', 'shape'])['Pol.'].idxmax()]

    plt.figure(figsize=(10, 6))

    # Plot for spheres
    sph_df = max_pol_angles[max_pol_angles['shape'] == 'SPHERE']
    plt.scatter(sph_df['size_param'], sph_df['theta'], color='orange', marker='o', label='Spheres', alpha=0.7)

    # Plot for rectangular prisms
    rect_df = max_pol_angles[max_pol_angles['shape'] == 'RCTGLPRSM']
    plt.scatter(rect_df['size_param'], rect_df['theta'], color='blue', marker='s', label='Rectangular Prisms', alpha=0.7)

    # Add labels, legend, and grid
    plt.xlabel('Size Parameter x', fontsize=14)
    plt.ylabel('Maximum Polarization Angle (θ)', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
def plot_surface_area_to_volume_vs_s11_qsca(results_df, base_dir):
    """
    Plot the surface area to volume ratio against S_11 (forward scattering) and Q_sca (scattering efficiency)
    for both spheres and rectangular prisms.
    
    Parameters:
    results_df (pd.DataFrame): DataFrame containing 'theta', 'S_11', 'shape', and other columns.
    base_dir (str): Base directory where simulation data is stored.
    """
    # Load the samples from the JSON file
    samples = gen_input.load_samples(os.path.join(base_dir, "Generated_samples.json"))
    samples_df = pd.DataFrame(samples)

    # Merge results_df with samples_df on relevant columns
    merged_df = pd.merge(results_df, samples_df, on=['size_param', 'radius', 'wavelength'], how='inner')

    # Calculate the surface area to volume ratio
    merged_df['surface_area_to_volume'] = np.where(
        merged_df['shape_x'] == 'SPHERE',
        3 / merged_df['radius'],  # Surface area to volume ratio for spheres
        2 * (merged_df['x_length'] * merged_df['y_length'] + 
             merged_df['x_length'] * merged_df['z_length'] + 
             merged_df['y_length'] * merged_df['z_length']) / merged_df['volume']  # For rectangular prisms
    )

    plt.figure(figsize=(12, 6))

    # Plot Surface Area to Volume Ratio vs S_11 (Forward Scattering)
    plt.subplot(1, 2, 1)
    sph_df = merged_df[(merged_df['shape_x'] == 'SPHERE') & (merged_df['theta'] == 0)]
    rect_df = merged_df[(merged_df['shape_x'] == 'RCTGLPRSM') & (merged_df['theta'] == 0)]

    plt.scatter(sph_df['surface_area_to_volume'], sph_df['S_11'], color='orange', marker='o', label='Spheres', alpha=0.7)
    plt.scatter(rect_df['surface_area_to_volume'], rect_df['S_11'], color='blue', marker='s', label='Rectangular Prisms', alpha=0.7)
    plt.xlabel('Surface Area to Volume Ratio', fontsize=14)
    plt.ylabel(r'S$_{11}$ (Forward Scattering)', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Plot Surface Area to Volume Ratio vs Q_sca (Scattering Efficiency)
    plt.subplot(1, 2, 2)
    plt.scatter(sph_df['surface_area_to_volume'], sph_df['Qsca'], color='orange', marker='o', label='Spheres', alpha=0.7)
    plt.scatter(rect_df['surface_area_to_volume'], rect_df['Qsca'], color='blue', marker='s', label='Rectangular Prisms', alpha=0.7)
    plt.xlabel('Surface Area to Volume Ratio', fontsize=14)
    plt.ylabel(r'Q$_{sca}$ (Scattering Efficiency)', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

def plot_s11_vs_yz_area_forward_scattering(results_df, base_dir):
    """
    Plot the S_11 values for forward scattering (theta = 0) against the cross-sectional area 
    perpendicular to the incident light for rectangular prisms and the projected area for spheres.

    Parameters:
    results_df (pd.DataFrame): DataFrame containing 'theta', 'S_11', 'shape', and other columns.
    base_dir (str): Base directory where simulation data is stored.
    """
    # Load the samples from the JSON file
    samples = gen_input.load_samples(os.path.join(base_dir, "Generated_samples.json"))
    samples_df = pd.DataFrame(samples)

    # Merge results_df with samples_df on relevant columns
    merged_df = pd.merge(results_df, samples_df, on=['size_param', 'radius', 'wavelength'], how='inner')

    print(f"Samples DataFrame Columns: {samples_df.columns}")
    print(f"Results DataFrame Columns: {results_df.columns}")
    print(f"Merged DataFrame Columns: {merged_df.columns}")

    plt.figure(figsize=(10, 6))

    # Filter and calculate cross-sectional area for rectangular prisms
    rect_df = merged_df[(merged_df['shape_x'] == 'RCTGLPRSM') & (merged_df['theta'] == 0)]
    rect_cross_section = rect_df['y_length'] * rect_df['z_length']
    plt.scatter(rect_cross_section, rect_df['S_11'], color='blue', label='Rectangular Prisms', marker='s', alpha=0.7)

    # Filter and calculate projected area for spheres
    sph_df = merged_df[(merged_df['shape_x'] == 'SPHERE') & (merged_df['theta'] == 0)]
    sph_cross_section = np.pi * sph_df['radius']**2
    plt.scatter(sph_cross_section, sph_df['S_11'], color='orange', label='Spheres', marker='o', alpha=0.7)

    plt.xlabel('Cross-Sectional Area (Rectangular Prisms) / Projected Area (Spheres)', fontsize=14)
    plt.ylabel(r'S$_{11}$ (Forward Scattering)', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

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
    Plot the distribution of unique size parameters from the simulation input data separately
    for spheres and rectangular prisms. Mark the mean of each distribution with a vertical line.
    
    Parameters:
    results_df (pd.DataFrame): DataFrame containing the 'size_param' and 'shape' columns.
    """
    # Separate the data for spheres and rectangular prisms
    spheres_df = results_df[results_df['shape'] == 'SPHERE']
    prisms_df = results_df[results_df['shape'] == 'RCTGLPRSM']

    # Get unique size parameters and calculate the mean for spheres
    unique_size_params_spheres = spheres_df['size_param'].unique()
    mean_size_param_spheres = unique_size_params_spheres.mean()

    # Get unique size parameters and calculate the mean for prisms
    unique_size_params_prisms = prisms_df['size_param'].unique()
    mean_size_param_prisms = unique_size_params_prisms.mean()

    # Plot the distribution for spheres
    plt.figure(figsize=(10, 6))
    sns.histplot(unique_size_params_spheres, kde=True, color='orange', bins=30, label='Spheres')
    plt.axvline(mean_size_param_spheres, color='red', linestyle='--', linewidth=2, label=f'Spheres Mean: {mean_size_param_spheres:.2f}')
    plt.xlabel('Size Parameter x (Spheres)', fontsize=14)
    plt.ylabel('Number of Samples', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Plot the distribution for rectangular prisms
    plt.figure(figsize=(10, 6))
    sns.histplot(unique_size_params_prisms, kde=True, color='blue', bins=30, label='Rectangular Prisms')
    plt.axvline(mean_size_param_prisms, color='green', linestyle='--', linewidth=2, label=f'Prisms Mean: {mean_size_param_prisms:.2f}')
    plt.xlabel('Size Parameter x (Rectangular Prisms)', fontsize=14)
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
    plt.plot(sph_df['theta'], sph_df['Pol.'], color='orange', linestyle='-', label='Sphere')

    # Plot rectangular prisms with blue squares
    rect_df = results_df[results_df['shape'] == 'RCTGLPRSM']
    plt.plot(rect_df['theta'], rect_df['Pol.'], color='blue', linestyle='-', label='Rectangular Prism')

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
    plt.plot(sph_df['theta'], sph_df['Pol.'], color='orange', linestyle='-', label='Sphere')

    # Plot average Pol. for rectangular prisms with blue squares
    rect_df = grouped_df[grouped_df['shape'] == 'RCTGLPRSM']
    plt.plot(rect_df['theta'], rect_df['Pol.'], color='blue', linestyle='-', label='Rectangular Prism')

    # Add labels and legend
    plt.xlabel(r'$\theta$ (degrees)', fontsize=14)
    plt.ylabel(r'Pol.', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)  # Add grid with dashed lines and transparency
    plt.tight_layout()
    plt.show()

def plot_average_s11_vs_theta(results_df):
    """
    Plot the average S_11 values against theta angles for different shapes,
    only considering samples up to the maximum size parameter of the rectangular prisms.
    
    Parameters:
    results_df (pd.DataFrame): DataFrame containing 'theta', 'S_11', 'shape', and 'size_param' columns.
    """
    # Determine the maximum size parameter for rectangular prisms
    max_rect_size_param = results_df[results_df['shape'] == 'RCTGLPRSM']['size_param'].max()
    print(f"Max size parameter for rectangular prisms: {max_rect_size_param}")

    # Filter DataFrame to only include samples with size_param up to max_rect_size_param
    filtered_results_df = results_df[results_df['size_param'] <= max_rect_size_param]
    print(f"Number of samples after filtering: {len(filtered_results_df)}")
    print(f"Number of spherical samples after filtering: {len(filtered_results_df[filtered_results_df['shape'] == 'SPHERE'])}")
    print(f"Number of rectangular prism samples after filtering: {len(filtered_results_df[filtered_results_df['shape'] == 'RCTGLPRSM'])}")

    # Group by 'theta' and 'shape' and calculate the mean of 'S_11'
    grouped_df = filtered_results_df.groupby(['theta', 'shape']).mean().reset_index()

    plt.figure(figsize=(10, 6))

    # Plot average S_11 for spherical particles with orange circles
    sph_df = grouped_df[grouped_df['shape'] == 'SPHERE']
    plt.plot(sph_df['theta'], sph_df['S_11'], color='orange', linestyle='-', label='Sphere')

    # Plot average S_11 for rectangular prisms with blue squares
    rect_df = grouped_df[grouped_df['shape'] == 'RCTGLPRSM']
    plt.plot(rect_df['theta'], rect_df['S_11'], color='blue', linestyle='-', label='Rectangular Prism')

    # Add labels and legend
    plt.xlabel(r'$\theta$ (degrees)', fontsize=14)
    plt.ylabel(r'S$_{11}$', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)  # Add grid with dashed lines and transparency
    plt.tight_layout()
    plt.show()


def plot_ddscat_correlation_results(results_df):
    # Filter the DataFrame for forward scattering (theta = 0 degrees)
    forward_scattering_df = results_df[results_df['theta'] == 0]

    # Reorder the variables and include 'radius'
    variables = ['size_param', 'radius', 'wavelength', 'S_11', 'Qsca', 'Qbk', 'Qpol']

    # Define the custom palette for shapes
    palette = {'SPHERE': 'orange', 'RCTGLPRSM': 'blue'}

    # Pairplot with the specified variables, using the custom palette for shapes
    sns.pairplot(
        forward_scattering_df, 
        vars=variables, 
        hue='shape',
        palette=palette
    )

    # Set the title for the entire plot
    plt.suptitle('Pairplot of Input Parameters and Results (Forward Scattering)', y=1.02)
    plt.show()

    # Compute the correlation matrix and plot it
    correlation_matrix = forward_scattering_df[variables].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix of Input Parameters and Results (Forward Scattering)')
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

def plot_qsca_ratio_ddscat_mie(results_df, mie_df):
    """
    Plot the ratio of scattering efficiencies (Qsca DDSCAT / Qsca Mie) for each sample 
    with respect to the size parameter. Only include DDSCAT results with phi = 0.

    Parameters:
    results_df (pd.DataFrame): DataFrame containing DDSCAT results.
    mie_df (pd.DataFrame): DataFrame containing Mie results.
    """
    # Filter DDSCAT results for phi = 0
    ddscat_phi_zero_df = results_df[results_df['phi'] == 0]

    # Initialize lists to store size parameters and Qsca ratios
    size_params = []
    qsca_ratios = []

    # Loop through each sample in the DDSCAT results and compare with Mie results
    for _, ddscat_sample in ddscat_phi_zero_df.iterrows():
        # Find the corresponding Mie result for the same radius and wavelength
        mie_sample = mie_df[
            (mie_df['radius'] == ddscat_sample['radius']) &
            (mie_df['wavelength'] == ddscat_sample['wavelength'])
        ]
        
        if not mie_sample.empty:
            # Calculate the Qsca ratio (DDSCAT/Mie)
            qsca_ddscat = ddscat_sample['Qsca']
            qsca_mie = mie_sample.iloc[0]['Qsca']  # Take the first match (since all should be the same)
            qsca_ratio = qsca_ddscat / qsca_mie
            
            # Store the size parameter and Qsca ratio
            size_params.append(ddscat_sample['size_param'])
            qsca_ratios.append(qsca_ratio)

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.scatter(size_params, qsca_ratios, color='red', marker='o', s=30, alpha=0.7)
    
    # Add horizontal line at y = 1
    plt.axhline(y=1, color='black', linestyle='--', linewidth=1)
    
    plt.xlabel('Size Parameter (x)', fontsize=14)
    plt.ylabel('Q$_{sca}$ Ratio (DDSCAT / Mie)', fontsize=14)
    plt.title('Comparison of Scattering Efficiencies (DDSCAT vs. Mie)', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    plt.show()
