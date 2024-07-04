import simulation_env as simulation_env
import input_generation as input_generation
import visualization as visualization
import execution as execution
import numpy as np
import os
import logging
import argparse
from scipy.stats import norm, uniform
import random

def setup_logging():
    """Configure the logging for the application."""
    logging.basicConfig(level=logging.INFO, 
                        filename='simulation.log', 
                        filemode='a', 
                        format=
                        '%(asctime)s - %(levelname)s - %(message)s')


def sample_parameters(num_samples, random_seed=None):
    """
    Generate sampled parameters for Monte Carlo simulations.
    
    Parameters:
        num_samples (int): Number of samples to generate.
    
    Returns:
        List[Dict[str, Any]]: List of parameter dictionaries.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)

    shapes = ["SPHERE", "RCTGLPRSM"]
    samples = []
    for _ in range(num_samples):
        shape = random.choice(shapes)                                   # Randomly select a shape
        size = norm.rvs(loc=0.3, scale=0.1)                             # Generate a single particle size based on a normal distribution
        size = np.clip(size, 0.01, 0.7)                                 # Clip the value to be within the specified range
        size = round(size, 2)                                           # Round to two decimal places
        wavelength = uniform.rvs(loc=0.380, scale=0.370)                # Generate a single wavelength within the visible spectrum from 0.380 to 0.750
        wavelength = round(wavelength, 4)                               # Round to four decimal places
        #print({aeff})                                                   # Debug print
        
        sample = {
            "shape": shape,
            "wavelength": wavelength,
            "size": size,
        }
        print(f"Generated sample: {sample}")
        samples.append(sample)
    
    return samples

def run_simulation(base_dir, samples):
    """
    Run multiple simulations given a list of samples and a base directory.
    Each simulation uses different particle geometry (sphere, rectangular prism, irregular).

    Parameters:
        base_dir (str): Directory where simulation outputs will be stored.
        samples (List[Dict[str, Any]]): List of sample parameter dictionaries.
    """
    logging.basicConfig(level=logging.INFO)
    original_directory = os.getcwd()                                                    # Save the current directory to revert back to it later

    for i, sample in enumerate(samples):
        shape = sample['shape']
        size = sample ['size']
        aeff = size/2                                                                   #aeff is half the particle size                    
        aeff = round(aeff, 2)   
        simulation_directory = os.path.join(base_dir, f"sim_{i+1}")
        os.makedirs(simulation_directory, exist_ok=True)
        logging.info(f"Starting simulation for {shape} in {simulation_directory}")

        shape_file_name = "shape.dat"
        shape_params = {
                'radius': aeff,  # Use the sampled particle size
                'd': 0.01  # Assuming a constant dipole spacing
            }
        if shape == "SPHERE":
            shape_file_content = simulation_env.generate_shape_dat('sphere', **shape_params)
        elif shape == "RCTGLPRSM":
            shape_file_content = simulation_env.generate_shape_dat('rect_prism', **shape_params)
        else:
            logging.error(f"Unsupported shape type: {shape}")
            continue

        input_generation.save_shape_info_to_file(shape_file_name, shape_file_content, simulation_directory)

        params = {
            "par_file_name": "ddscat.par",
            "shape": "FROM_FILE",
            "material_file": "astrosil",
            "vacuum_wavelengths": (sample['wavelength'], sample['wavelength']),
            "wavelengths_count": 1,
            "aeff_range": (aeff, aeff),
            "beta_params": (0, 0, 1),
            "theta_params": (0, 0, 1),
            "phi_params": (0, 0, 1),
            "plane1": (0, 0, 360, 5),                                       # phi, theta_min, theta_max, dtheta
            "plane2": (90, 0, 360, 5)                                       # phi, theta_min, theta_max, dtheta
            }

        par_file_content = input_generation.generate_par(params)
        input_generation.save_param_info_to_file(params, par_file_content, simulation_directory)
        execution.run_ddscat(simulation_directory)
        target_file_name = "output"
        execution.convert_to_vtk(simulation_directory, target_file_name)

        logging.info(f"Simulation data generated and processed in {simulation_directory}")

    os.chdir(original_directory)
    logging.info("Simulation completed for all samples.")

def main(base_dir):
    """
    Execute multiple simulations based on predefined geometries and log the outcomes.
    
    Parameters:
        base_dir (str): Base directory where simulation data will be stored.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    base_dir = os.path.abspath(base_dir)

    num_samples = 5  # Number of samples to generate
    samples = sample_parameters(num_samples, random_seed=7)

    run_simulation(base_dir, samples)

    file_paths, labels = visualization.find_data_files(base_dir)
    data_frames = [visualization.extract_data(fp) for fp in file_paths]

    visualization.plot_data(data_frames, labels)
    visualization.plot_polar_data(data_frames, labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a DDSCAT simulation.")
    parser.add_argument('--base_dir', type=str, default=os.getcwd(),
                        help="Base directory for simulation output")
    args = parser.parse_args()
    main(args.base_dir)

'''
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
        d = params.pop('d', None)                                       # Properly extract 'd' to avoid duplication
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
'''


# Testing script
#if __name__ == "__main__":
#    parser = argparse.ArgumentParser(description="Run a DDSCAT simulation.")
#    parser.add_argument('--base_dir', type=str, default=os.getcwd(),
#                        help="Base directory for simulation output")
#    args = parser.parse_args()
#    main(args.base_dir)

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
