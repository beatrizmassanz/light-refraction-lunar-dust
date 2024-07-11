import simulation_env as simulation_env
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
    
def run_simulation_random(sim_id, base_dir):
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


def main_random(sim_id, base_dir, num_simulations):
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
            run_simulation_random(sim_id, base_dir)
            logging.info(f"Successfully completed simulation {sim_id}")
            file_paths, labels = visualization.find_data_files(base_dir)
            data_frames = [visualization.extract_data(fp) 
                           for fp in file_paths]
        except Exception as e:
            logging.error(f"Failed to complete simulation {sim_id}: {repr(e)}")
    visualization.plot_data(data_frames, labels)   
    visualization.plot_polar_data(data_frames, labels)


def run_simulation(base_dir, geometry_settings):
    """
    Run multiple simulations given a simulation ID and a base directory.
    Each simulation uses different particle geometry (sphere, rectangular prism, irregular).

    Parameters:
        sim_id (int): The unique identifier for the simulation.
        base_plan_dir (str): Directory where simulation outputs will be stored.
        shape_settings (list): List of tuples with shape type and specific parameters.
    """

    logging.basicConfig(level=logging.INFO)
    original_directory = os.getcwd()                                    # Save the current directory to revert back to it later
    
    for settings in geometry_settings:
        geometry, params = settings['type'], settings['params']
        d = params.pop('d', None)  # Properly extract 'd' to avoid duplication
        simulation_directory = os.path.join(base_dir, f"sim_{geometry.lower()}")
        os.makedirs(simulation_directory, exist_ok=True)

        logging.info(f"Starting simulation for {geometry} in {simulation_directory}")

        shape_file_name = "shape.dat" 
        shape_file_content = simulation_env.generate_shape_dat(geometry, d, **params)
        input_generation.save_dust_cloud_info_to_file(shape_file_name, shape_file_content, simulation_directory)

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
        input_generation.save_param_info_to_file(params, par_file_content, simulation_directory)   
        execution.run_ddscat(simulation_directory)                          # Execute ddscat program with provided inputs
        target_file_name = "output"                                         # Convert target.out file to vtk file to display in paraview
        execution.convert_to_vtk(simulation_directory, target_file_name)

        logging.info(f"Simulation {geometry_settings} data generated and processed in {simulation_directory}")


    os.chdir(original_directory)
    logging.info("Simulation completed for all configurations.")

def main(base_dir):
    """
    Execute multiple simulations based on predefined geometries and log the outcomes.
    
    Parameters:
        base_dir (str): Base directory where simulation data will be stored.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    base_dir = os.path.abspath(base_dir)

    geometry_settings = [
        {'type': 'sphere', 'params': {'radius': 0.1, 'd': 0.01}},
        {'type': 'rect_prism', 'params': {'radius': 0.1, 'd': 0.01,'length': 0.02}},
        {'type': 'rect_prism', 'params': {'radius': 0.1, 'd': 0.01}}
    ]
    run_simulation(base_dir, geometry_settings)

    file_paths, labels = visualization.find_data_files(base_dir)
    data_frames = [visualization.extract_data(fp) for fp in file_paths]

    visualization.plot_data(data_frames, labels)
    visualization.plot_polar_data(data_frames, labels)  

    # Generate orbital positions for dust particles
    orbit_height_min = 1  # in km
    orbit_height_max = 10  # in km
    moon_radius = 1737.4  # in km, mean radius of the Moon
    density = 0.000005  # particles per cubic meter
    positions_all = []

    positions = simulation_env.gen_orbital_positions(orbit_height_min, orbit_height_max, moon_radius, density)  # Mock function to generate positions
    positions_all.append((positions, geometry_settings))
    visualization.plot_particle_positions(positions_all)




# Testing script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a DDSCAT simulation.")
    parser.add_argument('--base_dir', type=str, default=os.getcwd(),
                        help="Base directory for simulation output")
    args = parser.parse_args()
    main(args.base_dir)

# Testing script
#if __name__ == "__main__":
#    parser = argparse.ArgumentParser(description=
#                                     "Run a DDSCAT simulation.")
#    parser.add_argument('--sim_id', type=int, default=1, 
#                        help="Identifier for the simulation")
#    parser.add_argument('--base_dir', type=str, default=os.getcwd(), 
#                        help="Base directory for simulation output")
#    parser.add_argument('--num_simulations', type=int, default=1, 
#                        help="Number of simulations to run")
#    args = parser.parse_args()
#    setup_logging()
#
#     main(args.sim_id, args.base_dir, args.num_simulations)