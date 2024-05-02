import input_generation as input_generation
import convert_to_vtk as convert_to_vtk
import run_ddscat as run_ddscat
import numpy as np
import os
import argparse

# ------------------------------------------------------------------------------------------------------
def main():
    # Generate dust cloud geometry in shape.dat for DDSCAT
    shape_file_name = "shape.dat"

    # Generate randomly distributed dipoles within a cubic volume
    num_dipoles = 1000
    cube_size = 10.0
    positions = np.random.uniform(low=-cube_size/2, high=cube_size/2, size=(num_dipoles, 3))
    orientations = np.random.uniform(low=-1.0, high=1.0, size=(num_dipoles, 3))
    orientations /= np.linalg.norm(orientations, axis=1)[:, np.newaxis] # Normalize orientations to get unit vectors

    # Specify the shape file path
    shape_file_path = os.path.join("input")
    os.makedirs(shape_file_path, exist_ok=True)

    # Execute gen_dust_cloud function
    shape_file_content = input_generation.generate_dust_cloud_data(num_dipoles, positions, orientations)
    input_generation.save_dust_cloud_info_to_file(shape_file_name, shape_file_content, shape_file_path)

    # Define the parameter dictionary
    params = {
        "par_file_name": "ddscat.par",
        "par_file_directory": "input",
        "shape": "FROM_FILE",
        "material_file": "astrosil",
        "vacuum_wavelengths": (0.5600, 0.5600),
        "wavelengths_count": 1,
        "aeff_range": (0.1, 0.1),
        "beta_params": (0, 0, 1),
        "theta_params": (0, 0, 1),
        "phi_params": (0, 0, 1),
        "plane1": (0, 0, 360, 5),  # phi, theta_min, theta_max, dtheta
        "plane2": (90, 0, 360, 5)  # phi, theta_min, theta_max, dtheta
    }

    # Specify the parameter file path
    par_file_path = os.path.join(params["par_file_directory"])
    os.makedirs(par_file_path, exist_ok=True)

    # Execute the ddscat.par generation function
    par_file_content = input_generation.generate_par(params)
    input_generation.save_param_info_to_file(params, par_file_content, par_file_path)

    # Execute ddscat program with provided inputs
    run_ddscat.run_ddscat(par_file_path)

    # Convert target.out file to vtk file to display in paraview
    target_file_name = "output"
    target_file_path = par_file_path
    convert_to_vtk.convert_to_vtk(target_file_path, target_file_name)

if __name__== "__main__":
    main()


