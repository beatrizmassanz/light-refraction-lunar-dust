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
import pandas as pd
import json

def setup_logging():
    """Configure the logging for the application."""
    logging.basicConfig(level=logging.INFO, 
                        filename='simulation.log', 
                        filemode='a', 
                        format=
                        '%(asctime)s - %(levelname)s - %(message)s')

def sample_parameters(num_samples, random_seed=None, only_spheres=False):
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

    shapes = ["SPHERE"] if only_spheres else ["SPHERE", "RCTGLPRSM"]
    #shapes = ["SPHERE", "RCTGLPRSM"]
    samples = []
    for _ in range(num_samples):
        shape = random.choice(shapes)  # Randomly select a shape
        wavelength = uniform.rvs(loc=0.380, scale=0.370)  # Generate a single wavelength within the visible spectrum from 0.380 to 0.750
        wavelength = round(wavelength, 4)  # Round to four decimal places

        if shape == "SPHERE":
            radius = norm.rvs(loc=0.15, scale=0.05)  # Generate radius based on a normal distribution
            radius = np.clip(radius, 0.005, 0.35)  # Clip the value to be within the specified range
            radius = round(radius, 4)  # Round to four decimal places
            volume = (4/3) * np.pi * radius**3  # Calculate volume of the sphere
            volume = round(volume, 4)  # Round to four decimal places
            size_param = (2 * np.pi * radius) / wavelength  # Calculate size parameter
            size_param = round(size_param, 4)  # Round to four decimal places
            sample = {
                "shape": shape,
                "radius": radius,
                "wavelength": wavelength,
                "size_param": size_param,
                "volume": volume
            }
        
        elif shape == "RCTGLPRSM":
            x_length = norm.rvs(loc=0.24, scale=0.11)  # Generate x_length based on a normal distribution
            x_length = np.clip(x_length, 0.01, 0.56)  # Clip the value to be within the specified range
            x_length = round(x_length, 4)  # Round to four decimal places

            y_length = norm.rvs(loc=0.24, scale=0.11)  # Generate y_length based on a normal distribution
            y_length = np.clip(y_length, 0.01, 0.56)  # Clip the value to be within the specified range
            y_length = round(y_length, 4)  # Round to four decimal places

            z_length = norm.rvs(loc=0.24, scale=0.11)  # Generate z_length based on a normal distribution
            z_length = np.clip(z_length, 0.01, 0.56)  # Clip the value to be within the specified range
            z_length = round(z_length, 4)  # Round to four decimal places

            volume = x_length * y_length * z_length  # Calculate volume of the rectangular prism
            volume = round(volume, 4)  # Round to four decimal places
            radius = ((volume * 3) / (4 * np.pi))**(1/3)  # Calculate radius equivalent for the volume
            radius = round(radius, 4)  # Round to four decimal places
            size_param = (2 * np.pi * radius) / wavelength  # Calculate size parameter
            size_param = round(size_param, 4)  # Round to four decimal places

            sample = {
                "shape": shape,
                "x_length": x_length,
                "y_length": y_length,
                "z_length": z_length,
                "wavelength": wavelength,
                "volume": volume,
                "radius": radius,
                "size_param": size_param
            }

        print(f"Generated sample: {sample}")
        samples.append(sample)

    with open("Generated_samples.json", "w") as f:
        json.dump(samples, f, indent=4)
    
    return samples

def save_sample_parameters(simulation_directory, sample):
    """Save the sample parameters to a file in the simulation directory."""
    sample_file_path = os.path.join(simulation_directory, 'sample_parameters.json')
    with open(sample_file_path, 'w') as f:
        json.dump(sample, f, indent=4)

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
    results =[]
    for i, sample in enumerate(samples):
        shape = sample['shape']
        aeff = sample ['radius']                                                                   #aeff is half the particle size                    
        aeff = round(aeff, 2)   
        simulation_directory = os.path.join(base_dir, f"sim_{i+1}")
        os.makedirs(simulation_directory, exist_ok=True)
        logging.info(f"Starting simulation for {shape} in {simulation_directory}")
        save_sample_parameters(simulation_directory, sample)  # Save sample parameters

        shape_file_name = "shape.dat"
        d = 0.01

        if shape == "SPHERE":
            shape_file_content = simulation_env.generate_shape_dat('sphere', d, sample)
        elif shape == "RCTGLPRSM":
            try:
                shape_file_content = simulation_env.generate_shape_dat('rect_prism', d, sample)
            except ValueError as e:
                logging.error(f"Error generating shape for {shape} in {simulation_directory}: {e}")
                continue
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
        visualization.process_single_result(simulation_directory,results,sample)
        
    os.chdir(original_directory)
    logging.info("Simulation completed for all samples.")
    return pd.concat(results, ignore_index=True)


def main(base_dir, skip_simulation=False):
    """
    Execute multiple simulations based on predefined geometries and log the outcomes.
    
    Parameters:
        base_dir (str): Base directory where simulation data will be stored.
        skip_simulation (bool): If True, skip running simulations and only perform data analysis.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    base_dir = os.path.abspath(base_dir)

    if skip_simulation:
        results_df = visualization.process_results(base_dir)
    else:
        num_samples = 5  # Number of samples to generate
        samples = sample_parameters(num_samples, random_seed=299)
        results_df = run_simulation(base_dir, samples)

    visualization.analyze_results(results_df)
    file_paths, labels = visualization.find_data_files(base_dir)
    data_frames = [visualization.extract_data(fp) for fp in file_paths]
    visualization.plot_data(data_frames, labels)
    visualization.plot_polar_data(data_frames, labels)
    
    # Save the results to a CSV file
    results_df.to_csv(os.path.join(base_dir, 'simulation_results.csv'), index=False)
    print(f"Results saved to {os.path.join(base_dir, 'simulation_results.csv')}")
    return print(results_df)


'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a DDSCAT simulation.")
    parser.add_argument('--base_dir', type=str, default=os.getcwd(),
                        help="Base directory for simulation output")
    args = parser.parse_args()
    main(args.base_dir)
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a DDSCAT simulation.")
    parser.add_argument('--base_dir', type=str, default=os.getcwd(),
                        help="Base directory for simulation output")
    parser.add_argument('--skip_simulation', action='store_true',
                        help="Skip running simulations and only perform data analysis")
    args = parser.parse_args()
    main(args.base_dir, args.skip_simulation)

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
