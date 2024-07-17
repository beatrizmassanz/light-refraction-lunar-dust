import subprocess
import os
import miepython as mie
import pandas as pd
import numpy as np

def run_ddscat(par_file_path):
    """
    Executes the DDSCAT simulation from the given directory.

    Parameters:
    par_file_path (str): Path to the directory containing the 
    'ddscat.par' configuration file.
    """
    original_directory = os.getcwd()                                   # Save the current directory to revert back to it later
    os.chdir(par_file_path)                                            # Change to the directory of DDSCAT configuration file
    ddscat_path = os.path.join('..','..', 'src', 'ddscat')             # Relative path to vtrconvert, two directories up in 'src'
    command = f"./{ddscat_path}"                                       # Construct the DDSCAT command

    try:                                                               # Execute the command
        print("Starting DDSCAT simulation...")
        subprocess.run(command, check=True)
        print("DDSCAT simulation completed successfully.")
    except subprocess.CalledProcessError as e:
        print("DDSCAT simulation failed:", e)
    finally:
        os.chdir(original_directory)                                   # Change back to the original directory

def convert_to_vtk(simulation_directory, target_file_name):
    """
    Converts a DDSCAT shape file to a VTK file using vtrconvert.

    Parameters:
    simulation_directory (str): Path to the directory containing the 
        input file 'target.out'.
    target_file_name (str): Base name for the output VTK file.
    """
    original_directory = os.getcwd()                                   # Save the current directory to revert back to it later
    os.chdir(simulation_directory)                                     # Change to the directory containing the input file
    vtrconvert_path = os.path.join('..', '..', 'src', 'vtrconvert')    # Relative path to vtrconvert, two directories up in 'src'
    command = f"{vtrconvert_path} target.out {target_file_name}"       # Construct the vtrconvert command

    try:                                                               # Execute the command
        output = subprocess.check_output(command, shell=True, 
                                         stderr=subprocess.STDOUT)
        print("Conversion successful:", output.decode())
    except subprocess.CalledProcessError as e:
        print("Error during conversion:", e.output.decode())
    finally:
        os.chdir(original_directory)                                   # Change back to the original directory

def mie_calculation(samples):
    """
    Perform Mie calculations for the given samples.
    
    Parameters:
        samples (list): List of sample parameters for Mie calculations.
        
    Returns:
        DataFrame: A DataFrame with Mie calculation results.
    """
    mie_results = []
    angles_degrees = np.arange(0, 360, 5)  # Generate angles from 0 to 355 degrees in 5 degree steps
    angles_cosine = np.cos(np.radians(angles_degrees))

    for sample in samples:
        if sample["shape"] != "SPHERE":
            continue
        
        m_particle = 1.56 + 0.003j  # Assuming a constant complex refractive index for simplicity
        wavelength = sample["wavelength"]
        radius = sample["radius"]
        
        x = 2 * np.pi * radius / wavelength
        qext, qsca, qback, g = mie.mie(m_particle, x)

        s1, s2 = mie.mie_S1_S2(m_particle, x, angles_cosine)
        s_11 = 0.5 * (np.abs(s1)**2 + np.abs(s2)**2)

        for angle, s11_value in zip(angles_degrees, s_11):
            mie_result = {
                "radius": radius,
                "wavelength": wavelength,
                "Qext": qext,
                "Qsca": qsca,
                "Qback": qback,
                "G": g,
                "angle": angle,
                "S_11": s11_value
            }
            mie_results.append(mie_result)
    
    mie_df = pd.DataFrame(mie_results)
    return mie_df




