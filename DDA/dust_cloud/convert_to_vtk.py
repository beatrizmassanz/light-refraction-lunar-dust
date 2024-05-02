import subprocess
import os

def convert_to_vtk(target_file_path, target_file_name):
    """
    Converts a DDSCAT shape file to a VTK file using vtrconvert.

    Parameters:
    target_file_path (str): Path to the directory containing the input file 'target.out'.
    target_file_name (str): Base name for the output VTK file.
    """
    # Save the current directory to revert back to it later
    original_directory = os.getcwd()

    # Change to the directory containing the input file
    os.chdir(target_file_path)
    
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

