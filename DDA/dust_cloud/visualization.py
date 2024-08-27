import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


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
