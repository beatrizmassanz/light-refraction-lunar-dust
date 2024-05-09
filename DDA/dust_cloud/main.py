import input_generation as input_generation
import visualization as visualization
import execution as execution
import numpy as np
import os
import logging
import argparse

def setup_logging():
    """Configure the logging for the application."""
    logging.basicConfig(level=logging.INFO, 
                        filename='simulation.log', 
                        filemode='a', 
                        format=
                        '%(asctime)s - %(levelname)s - %(message)s')
    
def run_simulation(sim_id, base_dir):
    """
    Run a simulation given a simulation ID and a base directory.
    
    Parameters:
        sim_id (int): The unique identifier for the simulation.
        base_dir (str): Directory where simulation outputs 
            will be stored.
    """   
    logging.basicConfig(level=logging.INFO)
    original_directory = os.getcwd()                                   # Save the current directory to revert back to it later
    simulation_directory = os.path.join(base_dir, f"sim_{sim_id}")     # Directory for this specific simulation
    os.makedirs(simulation_directory, exist_ok=True)
    logging.info(f"Starting simulation {sim_id} in {simulation_directory}")
    shape_file_name = "shape.dat"                                       # Generate dust cloud geometry in shape.dat for DDSCAT  
    num_dipoles = 1000                                                  # Generate randomly distributed dipoles in a cubic volume
    cube_size = 10.0
    positions, orientations = input_generation.random_cloud_gen(        # Generate dipole positions and orientations    
        num_dipoles, cube_size)
    logging.info(f"Simulation {sim_id} data generated in {simulation_directory}")
    shape_file_content = input_generation.generate_dust_cloud_data(     # Execute gen_dust_cloud function
        num_dipoles, positions, orientations)
    input_generation.save_dust_cloud_info_to_file(
        shape_file_name, shape_file_content, simulation_directory)
    params = {                                                          # Define the parameter dictionary
        "par_file_name": "ddscat.par",
        "shape": "FROM_FILE",
        "material_file": "astrosil",
        "vacuum_wavelengths": (0.5600, 0.5600),
        "wavelengths_count": 1,
        "aeff_range": (0.1, 0.1),
        "beta_params": (0, 0, 1),
        "theta_params": (0, 0, 1),
        "phi_params": (0, 0, 1),
        "plane1": (0, 0, 360, 5),                                       # phi, theta_min, theta_max, dtheta
        "plane2": (90, 0, 360, 5)                                       # phi, theta_min, theta_max, dtheta
    }
    par_file_content = input_generation.generate_par(params)            # Execute the ddscat.par generation function
    input_generation.save_param_info_to_file(
        params, par_file_content, simulation_directory)   
    execution.run_ddscat(simulation_directory)                          # Execute ddscat program with provided inputs
    target_file_name = "output"                                         # Convert target.out file to vtk file to display in paraview
    execution.convert_to_vtk(simulation_directory, target_file_name)
    logging.info(f"Simulation {sim_id} data generated and processed in {simulation_directory}")
    os.chdir(original_directory)                                        # Change back to the original directory


def main(sim_id, base_dir, num_simulations):
    """
    Execute multiple simulations and log the outcomes.

    Parameters:
        sim_id (int): Identifier for the simulation.
        num_simulations (int): Number of simulations to run.
        base_dir (str): Base directory where simulations 
            data will be stored.

    """
    base_dir = os.path.abspath(base_dir)                                # Convert to absolute path to avoid directory confusion
    for sim_id in range(1, num_simulations + 1):
        try:
            run_simulation(sim_id, base_dir)
            logging.info(f"Successfully completed simulation {sim_id}")
            file_paths, labels = visualization.find_data_files(base_dir)
            data_frames = [visualization.extract_data(fp) 
                           for fp in file_paths]
        except Exception as e:
            logging.error(f"Failed to complete simulation {sim_id}: {repr(e)}")
    visualization.plot_data(data_frames, labels)   


# Testing script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
                                     "Run a DDSCAT simulation.")
    parser.add_argument('--sim_id', type=int, default=1, 
                        help="Identifier for the simulation")
    parser.add_argument('--base_dir', type=str, default=os.getcwd(), 
                        help="Base directory for simulation output")
    parser.add_argument('--num_simulations', type=int, default=1, 
                        help="Number of simulations to run")
    args = parser.parse_args()
    setup_logging()
    main(args.sim_id, args.base_dir, args.num_simulations)
