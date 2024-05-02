import subprocess
import os

def run_ddscat(par_file_path):
    """
    Executes the DDSCAT simulation from the given directory.

    Parameters:
    par_file_path (str): Path to the directory containing the 'ddscat.par' configuration file.
    """
    # Save the current directory to revert back to it later
    original_directory = os.getcwd()

    # Change to the directory containing the DDSCAT configuration file
    os.chdir(par_file_path)
    
    # Relative path to the ddscat executable, assuming it's two directories up in 'src'
    ddscat_path = os.path.join('..','..', 'src', 'ddscat')

    # Execute the DDSCAT command
    command = f"./{ddscat_path}"
    try:
        print("Starting DDSCAT simulation...")
        subprocess.run(command, check=True)
        print("DDSCAT simulation completed successfully.")
    except subprocess.CalledProcessError as e:
        print("DDSCAT simulation failed:", e)

    # Change back to the original directory
    os.chdir(original_directory)



