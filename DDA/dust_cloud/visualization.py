import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import gen_input as gen_input
import proc_output_ddscat as proc_output_ddscat

def plot_ddscat_correlation_results(results_df):
    """
    Generate and display a pairplot and a correlation matrix for input 
    parameters and DDSCAT simulation results, focusing on forward 
    scattering (theta = 0 degrees).

    Parameters:
        results_df (pd.DataFrame): DataFrame containing DDSCAT 
        results, including columns for 'theta', 'shape', 
        'size_param', 'radius', 
        'wavelength', 'S_11', 'Qsca', 'Qbk', and 'Qpol'.

    Plots:
        1. Pairplot of input parameters ('size_param', 'radius', 
        'wavelength') and results ('S_11', 'Qsca', 'Qbk', 'Qpol').
        2. Heatmap showing the correlation matrix of the same variables.
    """

    forward_scattering_df = results_df[results_df['theta'] == 0]
    variables = [
        'size_param', 'radius', 'wavelength', 
        'S_11', 'Qsca', 'Qbk', 'Qpol'
        ]
    palette = {'SPHERE': 'orange', 'RCTGLPRSM': 'blue'}

    sns.pairplot(
        forward_scattering_df, 
        vars=variables, 
        hue='shape',
        palette=palette
    )

    plt.suptitle(
        'Pairplot of Input Parameters and Results '
        '(Forward Scattering)',
        y=1.02
        )
    plt.show()

    correlation_matrix = forward_scattering_df[variables].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title(
        'Correlation Matrix of Input Parameters and Results'
        )
    plt.show()

def plot_shape_counts(results_df):
    """
    Plot the count of unique samples for each shape (spheres and 
    rectangular prisms) based on the 'size_param' attribute.

    Parameters:
        results_df (pd.DataFrame): DataFrame containing DDSCAT results, 
        including 'shape' and 'size_param' columns.

    Plots:
        A bar chart showing the count of unique spheres and rectangular 
        prisms in the dataset.
    """
    unique_samples_df = results_df.drop_duplicates(
        subset=['shape', 'size_param']
        )
    shape_counts = unique_samples_df['shape'].value_counts()

    plt.figure(figsize=(8, 6))
    plt.bar(
        shape_counts.index, 
        shape_counts.values, 
        color=['orange', 'blue']
        )
    
    plt.xlabel('Shape', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.title(
        'Count of Unique Spheres and Rectangular Prisms', 
        fontsize=16
        )
    plt.xticks(
        ticks=[0, 1], 
        labels=['Spheres', 'Rectangular Prisms'], 
        fontsize=12
        )
    plt.tight_layout()
    plt.show()

def plot_size_param_distribution(results_df):
    """
    Plot the distribution of unique size parameters from the simulation 
    input data separately for spheres and rectangular prisms. Mark the 
    mean of each distribution with a vertical line.
    
    Parameters:
    results_df (pd.DataFrame): DataFrame containing the 'size_param' 
        and 'shape' columns.
    """

    spheres_df = results_df[results_df['shape'] == 'SPHERE']
    prisms_df = results_df[results_df['shape'] == 'RCTGLPRSM']
    unique_size_params_spheres = spheres_df['size_param'].unique()
    mean_size_param_spheres = unique_size_params_spheres.mean()
    unique_size_params_prisms = prisms_df['size_param'].unique()
    mean_size_param_prisms = unique_size_params_prisms.mean()

    plt.figure(figsize=(10, 6))
    sns.histplot(
        unique_size_params_spheres, 
        kde=True, 
        color='orange', 
        bins=30, 
        label='Spheres'
        )
    plt.axvline(
        mean_size_param_spheres, 
        color='red', 
        linestyle='--', 
        linewidth=2
        )
    plt.xlabel('Size Parameter x (Spheres)', fontsize=14)
    plt.ylabel('Number of Samples', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.histplot(
        unique_size_params_prisms, 
        kde=True, 
        color='blue', 
        bins=30, 
        label='Rectangular Prisms'
        )
    plt.axvline(
        mean_size_param_prisms, 
        color='green', 
        linestyle='--', 
        linewidth=2, 
        label=f'Prisms Mean: {mean_size_param_prisms:.2f}'
        )
    plt.xlabel('Size Parameter x (Rectangular Prisms)', fontsize=14)
    plt.ylabel('Number of Samples', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_wavelength_distribution(results_df):
    """
    Plot the distribution of unique wavelengths from the simulation 
    input data.
    
    Parameters:
    results_df (pd.DataFrame): DataFrame containing the 'wavelength' 
        column.
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
    Plot the distribution of unique particle radii from the simulation 
    input data. Mark the mean of the distribution with a vertical line.
    
    Parameters:
    results_df (pd.DataFrame): DataFrame containing the 'radius' column.
    """
    unique_radii = results_df['radius'].unique()
    mean_radius = unique_radii.mean() 

    plt.figure(figsize=(10, 6))
    sns.histplot(unique_radii, kde=True, color='blue', bins=30)
    plt.axvline(
        mean_radius, 
        color='orange', 
        linestyle='--', 
        linewidth=2, 
        label=f'Mean: {mean_radius:.2f} µm'
        )
    plt.xlabel('Particle Radius (µm)', fontsize=14)
    plt.ylabel('Number of Samples', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_qsca_vs_size(results_df):
    """
    Plot the scattering efficiency (Qsca) as a function of particle size 
    (radius) for both spheres and rectangular prisms.

    Parameters:
        results_df (pd.DataFrame): DataFrame containing DDSCAT results, 
        including 'radius', 'Qsca', and 'shape' columns.

    Plots:
        A scatter plot and regression line showing Qsca versus particle 
        size, with different colors representing spheres and rectangular 
        prisms.
    """
    plt.figure(figsize=(10, 6))
    sns.regplot(
        x=results_df['radius'],
        y=results_df['Qsca'],
        scatter_kws={'s': 20, 'alpha': 0.7, 'color': 'gray'},
        line_kws={'color': 'black'}
    )
    plt.scatter(
        results_df[results_df['shape'] == 'SPHERE']['radius'],
        results_df[results_df['shape'] == 'SPHERE']['Qsca'],
        color='orange', marker='o', s=20, 
        label='Spheres', alpha=0.7
    ) 
    plt.scatter(
        results_df[results_df['shape'] == 'RCTGLPRSM']['radius'],
        results_df[results_df['shape'] == 'RCTGLPRSM']['Qsca'],
        color='blue', marker='s', s=20, 
        label='Rectangular Prisms', alpha=0.7
    )
    plt.xlabel('Particle Size (radius)', fontsize=14)
    plt.ylabel(r'Q$_{sca}$', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_qsca_vs_wavelength(results_df):
    """
    Plot the scattering efficiency (Qsca) as a function of wavelength for 
    both spheres and rectangular prisms.

    Parameters:
        results_df (pd.DataFrame): DataFrame containing DDSCAT results, 
        including 'wavelength', 'Qsca', and 'shape' columns.

    Plots:
        A scatter plot showing Qsca versus wavelength, with different colors 
        representing spheres and rectangular prisms.
    """
    plt.figure(figsize=(10, 6))

    plt.scatter(
        results_df[results_df['shape'] == 'SPHERE']['wavelength'],
        results_df[results_df['shape'] == 'SPHERE']['Qsca'],
        color='orange', marker='o', s=20, label='Spheres', alpha=0.7
    )
    
    plt.scatter(
        results_df[results_df['shape'] == 'RCTGLPRSM']['wavelength'],
        results_df[results_df['shape'] == 'RCTGLPRSM']['Qsca'],
        color='blue', marker='s', s=20, 
        label='Rectangular Prisms', alpha=0.7
    )

    plt.xlabel('Wavelength', fontsize=14)
    plt.ylabel(r'Q$_{sca}$', fontsize=14) 
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_s11_vs_size_forward_scattering(results_df):
    """
    Plot the S11 parameter (scattering intensity) in forward scattering 
    (theta = 0 degrees) as a function of particle size (radius) for both 
    spheres and rectangular prisms.

    Parameters:
        results_df (pd.DataFrame): DataFrame containing DDSCAT results, 
        including 'radius', 'S_11', 'shape', 'theta', and 'phi' columns.

    Plots:
        A scatter plot showing S11 in forward scattering versus particle 
        size, with different colors representing spheres and rectangular 
        prisms.
    """
    plt.figure(figsize=(10, 6))

    forward_df = results_df[
        (results_df['theta'] == 0) & 
        (results_df['phi'] == 0)
        ]
    plt.scatter(
        forward_df[forward_df['shape'] == 'SPHERE']['radius'],
        forward_df[forward_df['shape'] == 'SPHERE']['S_11'],
        color='orange', marker='o', s=20, label='Spheres', alpha=0.7
    )
    plt.scatter(
        forward_df[forward_df['shape'] == 'RCTGLPRSM']['radius'],
        forward_df[forward_df['shape'] == 'RCTGLPRSM']['S_11'],
        color='blue', marker='s', s=20, 
        label='Rectangular Prisms', alpha=0.7
    )

    plt.xlabel('Particle Size (radius)', fontsize=14)
    plt.ylabel(
        'S₁₁ (Forward Scattering, θ=0°)', fontsize=14, 
        fontweight='normal')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_s11_vs_wavelength_forward_scattering(results_df):
    """
    Plot the S11 parameter (scattering intensity) in forward scattering 
    (theta = 0 degrees) as a function of wavelength for both spheres 
    and rectangular prisms.

    Parameters:
        results_df (pd.DataFrame): DataFrame containing DDSCAT results, 
        including 'wavelength', 'S_11', 'shape', 'theta', and 'phi' columns.

    Plots:
        A scatter plot showing S11 in forward scattering versus wavelength, 
        with different colors representing spheres and rectangular prisms.
    """
    plt.figure(figsize=(10, 6))

    forward_df = results_df[
        (results_df['theta'] == 0) & 
        (results_df['phi'] == 0)
        ]
    plt.scatter(
        forward_df[forward_df['shape'] == 'SPHERE']['wavelength'],
        forward_df[forward_df['shape'] == 'SPHERE']['S_11'],
        color='orange', marker='o', s=20, 
        label='Spheres', alpha=0.7
    )
    plt.scatter(
        forward_df[forward_df['shape'] == 'RCTGLPRSM']['wavelength'],
        forward_df[forward_df['shape'] == 'RCTGLPRSM']['S_11'],
        color='blue', marker='s', s=20, 
        label='Rectangular Prisms', alpha=0.7
    )
    plt.xlabel('Wavelength', fontsize=14)
    plt.ylabel(
        'S₁₁ (Forward Scattering, θ=0°)', 
        fontsize=14, fontweight='normal'
        )
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_qsca_by_size(results_df):
    """
    Plot the scattering efficiency (Qsca) as a function of the size parameter 
    for both spheres and rectangular prisms.

    Parameters:
        results_df (pd.DataFrame): DataFrame containing DDSCAT results, 
        including 'size_param', 'Qsca', and 'shape' columns.

    Plots:
        A scatter plot showing Qsca versus size parameter, with different 
        colors representing spheres and rectangular prisms.
    """
    plt.figure(figsize=(10, 6))

    plt.scatter(
        results_df[results_df['shape'] == 'SPHERE']['size_param'],
        results_df[results_df['shape'] == 'SPHERE']['Qsca'],
        color='orange', marker='o', s=10, label='Spheres'
    )
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


def plot_qbk_by_size(results_df):
    """
    Plot the backscattering efficiency (Qbk) as a function of the size parameter 
    for both spheres and rectangular prisms.

    Parameters:
        results_df (pd.DataFrame): DataFrame containing DDSCAT results, 
        including 'size_param', 'Qbk', and 'shape' columns.

    Plots:
        A scatter plot showing Qbk versus size parameter, with different 
        colors representing spheres and rectangular prisms.
    """
    plt.figure(figsize=(10, 6))

    plt.scatter(
        results_df[results_df['shape'] == 'SPHERE']['size_param'],
        results_df[results_df['shape'] == 'SPHERE']['Qbk'],
        color='orange', marker='o', s=10, label='Spheres'
    )
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
    """
    Plot the polarization efficiency (Qpol) as a function of the size 
    parameter for both spheres and rectangular prisms.

    Parameters:
        results_df (pd.DataFrame): DataFrame containing DDSCAT results, 
        including 'size_param', 'Qpol', and 'shape' columns.

    Plots:
        A scatter plot showing Qpol versus size parameter, with different 
        colors representing spheres and rectangular prisms.
    """
    plt.figure(figsize=(10, 6))

    plt.scatter(
        results_df[results_df['shape'] == 'SPHERE']['size_param'],
        results_df[results_df['shape'] == 'SPHERE']['Qpol'],
        color='orange', marker='o', s=10, label='Spheres'
    )
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

def plot_qpol_vs_size_param_and_radius(results_df):
    """
    Plot the distribution of Qpol and size parameter values.
    The y-axis represents the number of samples.
    
    Parameters:
    results_df (pd.DataFrame): DataFrame containing 'size_param' 
        and 'Qpol' columns.
    """

    results_df['abs_Qpol'] = np.abs(results_df['Qpol'])
    unique_samples = results_df.drop_duplicates(subset='size_param')

    size_param_counts, size_param_bins = np.histogram(
        unique_samples['size_param'], 
        bins=30
        )
    qpol_sum, _ = np.histogram(
        unique_samples['size_param'], 
        bins=size_param_bins, 
        weights=unique_samples['abs_Qpol']
        )

    plt.figure(figsize=(12, 6))

    plt.bar(
        size_param_bins[:-1], qpol_sum, 
        width=np.diff(size_param_bins), 
        color='red', alpha=0.5, 
        label=r'|Q$_{pol}$|'
        )
    plt.bar(
        size_param_bins[:-1], size_param_counts, 
        width=np.diff(size_param_bins), 
        color='blue', alpha=0.5, 
        label='Size Parameter'
        )

    plt.xlabel('Size Parameter x', fontsize=14)
    plt.ylabel('Number of Samples', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_s11_forward_scattering(results_df):
    """
    Plot S_11 values for forward scattering (θ = 0 degrees and 
    φ = 0 degrees) across size parameters.
    
    Parameters:
    results_df (pd.DataFrame): DataFrame containing 'size_param', 'S_11', 
        'theta', 'phi', and 'shape' columns.
    """
    plt.figure(figsize=(10, 6))

    forward_df = results_df[
        (results_df['theta'] == 0) & 
        (results_df['phi'] == 0)
        ]

    sph_df = forward_df[forward_df['shape'] == 'SPHERE']
    plt.scatter(
        sph_df['size_param'], sph_df['S_11'], 
        color='orange', label='Sphere', 
        marker='o', s=10)

    rect_df = forward_df[forward_df['shape'] == 'RCTGLPRSM']
    plt.scatter(
        rect_df['size_param'], rect_df['S_11'], 
        color='blue', label='Rectangular Prism', 
        marker='s', s=10)

    plt.xlabel(r'Size Parameter $x$', fontsize=16, fontweight='normal')
    plt.ylabel(
        r'$S_{11}$ (Forward Scattering, $\theta=0^\circ$)', 
        fontsize=16, fontweight='normal')
    plt.legend(fontsize=12, title='Shape', title_fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_s11_back_scattering(results_df):
    """
    Plot S_11 values for backscattering (θ = 180 degrees and 
    φ = 0 degrees) across size parameters.
    
    Parameters:
    results_df (pd.DataFrame): DataFrame containing 'size_param', 'S_11', 
    'theta', 'phi', and 'shape' columns.
    """
    plt.figure(figsize=(10, 6))

    back_df = results_df[
        (results_df['theta'] == 180) & 
        (results_df['phi'] == 0)
        ]
    
    sph_df = back_df[back_df['shape'] == 'SPHERE']
    plt.scatter(
        sph_df['size_param'], sph_df['S_11'], 
        color='orange', label='Sphere', 
        marker='o', s=10)

    rect_df = back_df[back_df['shape'] == 'RCTGLPRSM']
    plt.scatter(
        rect_df['size_param'], rect_df['S_11'], 
        color='blue', label='Rectangular Prism', 
        marker='s', s=10)

    plt.xlabel(r'Size Parameter $x$', fontsize=16, fontweight='normal')
    plt.ylabel(
        r'$S_{11}$ (Back Scattering, $\theta=180^\circ$)', 
        fontsize=16, fontweight='normal')
    plt.legend(fontsize=12, title='Shape', title_fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_average_pol_vs_theta(results_df):
    """
    Plot the average Pol. values against theta angles for 
    different shapes.
    
    Parameters:
    results_df (pd.DataFrame): DataFrame containing 'theta', 'Pol.',
     and 'shape' columns.
    """

    grouped_df = results_df.groupby(
        ['theta', 'shape']
        ).mean().reset_index()

    plt.figure(figsize=(10, 6))

    sph_df = grouped_df[grouped_df['shape'] == 'SPHERE']
    plt.plot(
        sph_df['theta'], sph_df['Pol.'], 
        color='orange', linestyle='-', label='Sphere')

    rect_df = grouped_df[grouped_df['shape'] == 'RCTGLPRSM']
    plt.plot(
        rect_df['theta'], rect_df['Pol.'],
        color='blue', linestyle='-', label='Rectangular Prism')

    plt.xlabel(r'$\theta$ (degrees)', fontsize=14)
    plt.ylabel(r'Pol.', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_average_s11_vs_theta(results_df):
    """
    Plot the average S_11 values against theta angles for 
    different shapes, only considering samples up to the maximum 
    size parameter of the rectangular prisms.
    
    Parameters:
    results_df (pd.DataFrame): DataFrame containing 'theta', 'S_11', 
        'shape', and 'size_param' columns.
    """ 

    max_rect_size_param = results_df[
        results_df['shape'] == 'RCTGLPRSM'
        ]['size_param'].max()

    filtered_results_df = results_df[
        results_df['size_param'] <= max_rect_size_param
        ]

    grouped_df = filtered_results_df.groupby(
        ['theta', 'shape']
        ).mean().reset_index()

    plt.figure(figsize=(10, 6))

    sph_df = grouped_df[grouped_df['shape'] == 'SPHERE']
    plt.plot(
        sph_df['theta'], sph_df['S_11'], 
        color='orange', linestyle='-', label='Sphere')

    rect_df = grouped_df[grouped_df['shape'] == 'RCTGLPRSM']
    plt.plot(
        rect_df['theta'], rect_df['S_11'], 
        color='blue', linestyle='-', label='Rectangular Prism')

    plt.xlabel(r'$\theta$ (degrees)', fontsize=14)
    plt.ylabel(r'S$_{11}$', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_s11_for_selected_wavelengths(results_df):
    """
    Plot S_11 values against theta angles for particles with 
    size parameters close to specified values, considering only 
    data with phi = 0 degrees.

    Parameters:
    results_df (pd.DataFrame): DataFrame containing 'theta', 'S_11', 
    'shape', 'size_param', and 'phi' columns.
    """
    target_size_params = [0.3, 0.4, 0.5, 0.6, 0.7]                          # Define the target size parameters

    plt.figure(figsize=(10, 6))

    phi_zero_df = results_df[results_df['phi'] == 0.0]                      # Filter the DataFrame to phi = 0 degrees

    for target in target_size_params:                                       # Select and plot the S_11 vs. theta for each size
        closest_size_param = phi_zero_df.iloc[ \
            (phi_zero_df['wavelength'] - target).abs().argsort()[:1] \
                ]['wavelength'].values[0]
        particle_df = phi_zero_df[
            phi_zero_df['wavelength'] == closest_size_param
            ]
        particle_df = particle_df.drop_duplicates(
            subset=['wavelength', 'theta']
            )
        particle_df = particle_df.sort_values('theta')
        shape = particle_df['shape'].iloc[0]
        plt.plot(
            particle_df['theta'], 
            particle_df['S_11'], 
            label=f'λ = {closest_size_param}, {shape}'
            )

    plt.xlabel(r'$\theta$ (degrees)', fontsize=14)
    plt.ylabel(r'S$_{11}$', fontsize=14)
    plt.legend(title='Wavelength and Shape')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_pol_for_selected_wavelengths(results_df):
    """
    Plot Pol. values against theta angles for particles with size 
    parameters close to specified values, considering only 
    data with phi = 0 degrees.

    Parameters:
    results_df (pd.DataFrame): DataFrame containing 'theta', 'Pol.', 
        'shape', 'size_param', and 'phi' columns.
    """

    target_size_params = [0.3, 0.4, 0.5, 0.6, 0.7]                          # Define the target size parameters

    plt.figure(figsize=(10, 6))

    phi_zero_df = results_df[results_df['phi'] == 0.0]                      # Filter the DataFrame to phi = 0 degrees

    for target in target_size_params:                                       # Select and plot the Pol. vs. theta for each size

        closest_size_param = phi_zero_df.iloc[
            (phi_zero_df['wavelength'] - target).abs().argsort()[:1]
            ]['wavelength'].values[0]
        
        particle_df = phi_zero_df[
            phi_zero_df['wavelength'] == closest_size_param
            ]
        particle_df = particle_df.drop_duplicates(
            subset=['wavelength', 'theta']
            )
        particle_df = particle_df.sort_values('theta')
        shape = particle_df['shape'].iloc[0]
        plt.plot(
            particle_df['theta'], 
            particle_df['Pol.'], 
            label=f'λ = {closest_size_param}, {shape}'
            )

    # Add labels and legend
    plt.xlabel(r'$\theta$ (degrees)', fontsize=14)
    plt.ylabel(r'Pol.', fontsize=14)
    plt.legend(title='Wavelength and Shape')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_s11_for_selected_size_params(results_df):
    """
    Plot S_11 values against theta angles for particles with size 
    parameters close to specified values, considering only data 
    with phi = 0 degrees.

    Parameters:
    results_df (pd.DataFrame): DataFrame containing 'theta', 'S_11', 
    'shape', 'size_param', and 'phi' columns.
    """
    
    target_size_params = [0.5, 1, 1.5, 2, 2.5, 3]                           # Define the target size parameters

    plt.figure(figsize=(10, 6))

    phi_zero_df = results_df[results_df['phi'] == 0.0]                      # Filter the DataFrame to phi = 0 degrees

    for target in target_size_params:                                       # Select and plot the S_11 vs. theta for each size

        closest_size_param = phi_zero_df.iloc[
            (phi_zero_df['size_param'] - target).abs().argsort()[:1]
            ]['size_param'].values[0]

        particle_df = phi_zero_df[
            phi_zero_df['size_param'] == closest_size_param
            ]
        particle_df = particle_df.drop_duplicates(
            subset=['size_param', 'theta']
            )
        particle_df = particle_df.sort_values('theta')
        shape = particle_df['shape'].iloc[0]
        plt.plot(
            particle_df['theta'], 
            particle_df['S_11'], 
            label=f'x = {closest_size_param}, {shape}'
            )

    plt.xlabel(r'$\theta$ (degrees)', fontsize=14)
    plt.ylabel(r'S$_{11}$', fontsize=14)
    plt.legend(title='Size Parameter and Shape')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_pol_for_selected_size_params(results_df):
    """
    Plot Pol. values against theta angles for particles with size 
    parameters close to specified values, considering only data 
    with phi = 0 degrees.

    Parameters:
    results_df (pd.DataFrame): DataFrame containing 'theta', 'Pol.', 
    'shape', 'size_param', and 'phi' columns.
    """
    target_size_params = [0.5, 1, 1.5, 2, 2.5, 3, 3.5]                      # Define the target size parameters

    plt.figure(figsize=(10, 6))

    phi_zero_df = results_df[results_df['phi'] == 0.0]                      # Filter the DataFrame to phi = 0 degrees

    for target in target_size_params:                                       # Select and plot the Pol. vs. theta for each size
        closest_size_param = phi_zero_df.iloc[
            (phi_zero_df['size_param'] - target).abs().argsort()[:1]
            ]['size_param'].values[0]

        particle_df = phi_zero_df[
            phi_zero_df['size_param'] == closest_size_param
            ]
        particle_df = particle_df.drop_duplicates(
            subset=['size_param', 'theta']
            )
        particle_df = particle_df.sort_values('theta')

        shape = particle_df['shape'].iloc[0]

        plt.plot(
            particle_df['theta'], 
            particle_df['Pol.'], 
            label=f'x = {closest_size_param}, {shape}'
            )

    plt.xlabel(r'$\theta$ (degrees)', fontsize=14)
    plt.ylabel(r'Pol.', fontsize=14)
    plt.legend(title='Size Parameter and Shape')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_s11_for_selected_size_params(results_df):
    """
    Plot S_11 values against theta angles for particles with size 
    parameters close to specified values, considering only data 
    with phi = 0 degrees.

    Parameters:
    results_df (pd.DataFrame): DataFrame containing 'theta', 'S_11', 
    'shape', 'size_param', and 'phi' columns.
    """

    target_size_params = [0.5, 1, 1.5, 2, 2.5, 3]                           # Define the target size parameters

    plt.figure(figsize=(10, 6))

    phi_zero_df = results_df[results_df['phi'] == 0.0]                      # Filter the DataFrame to only phi = 0 degrees

    for target in target_size_params:                                       # Select and plot the S_11 vs. theta for each size

        closest_size_param = phi_zero_df.iloc[
            (phi_zero_df['size_param'] - target).abs().argsort()[:1]
            ]['size_param'].values[0]

        particle_df = phi_zero_df[
            phi_zero_df['size_param'] == closest_size_param
            ]
        particle_df = particle_df.drop_duplicates(
            subset=['size_param', 'theta']
            )
        particle_df = particle_df.sort_values('theta')

        shape = particle_df['shape'].iloc[0]

        plt.plot(
            particle_df['theta'], 
            particle_df['S_11'], 
            label=f'x = {closest_size_param}, {shape}'
            )

    plt.xlabel(r'$\theta$ (degrees)', fontsize=14)
    plt.ylabel(r'S$_{11}$', fontsize=14)
    plt.legend(title='Size Parameter and Shape')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_qsca_ratio_ddscat_mie(results_df, mie_df):
    """
    Plot the ratio of scattering efficiencies (Qsca DDSCAT / Qsca Mie) 
    for each sample with respect to the size parameter. 
    Only include DDSCAT results with phi = 0.

    Parameters:
    results_df (pd.DataFrame): DataFrame containing DDSCAT results.
    mie_df (pd.DataFrame): DataFrame containing Mie results.
    """
    ddscat_phi_zero_df = results_df[results_df['phi'] == 0]

    size_params = []
    qsca_ratios = []

    for _, ddscat_sample in ddscat_phi_zero_df.iterrows():
        mie_sample = mie_df[
            (mie_df['radius'] == ddscat_sample['radius']) &
            (mie_df['wavelength'] == ddscat_sample['wavelength'])
        ]
        
        if not mie_sample.empty:
            qsca_ddscat = ddscat_sample['Qsca']
            qsca_mie = mie_sample.iloc[0]['Qsca']
            qsca_ratio = qsca_ddscat / qsca_mie
            
            size_params.append(ddscat_sample['size_param'])
            qsca_ratios.append(qsca_ratio)

    plt.figure(figsize=(10, 6))
    plt.scatter(
        size_params, qsca_ratios, 
        color='red', marker='o', s=30, alpha=0.7)

    plt.axhline(y=1, color='black', linestyle='--', linewidth=1)
    
    plt.xlabel('Size Parameter (x)', fontsize=14)
    plt.ylabel('Q$_{sca}$ Ratio (DDSCAT / Mie)', fontsize=14)
    plt.title(
        'Comparison of Scattering Efficiencies (DDSCAT vs. Mie)', 
        fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    plt.show()
