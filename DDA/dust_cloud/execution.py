import subprocess
import os

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



