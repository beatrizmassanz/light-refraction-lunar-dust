import proc_output_ddscat as proc_output_ddscat
import prep_input_ddscat as prep_input_ddscat
import exec_ddscat as exec_ddscat
import os
import pandas as pd
import logging

def process_existing_results(base_dir):
    """
    Process the existing results using proc_output_ddscat's function.

    Parameters:
        base_dir (str): Base directory where simulation data is stored.
    
    Returns:
        pd.DataFrame: A DataFrame containing the aggregated results.
    """
    return proc_output_ddscat.process_existing_results(base_dir)

def run_simulations(base_dir, samples):
    """
    Run multiple simulations given a list of samples and a base directory.
    Each simulation uses different particle geometry (sphere, rectangular 
    prism, irregular (when included)).

    Parameters:
        base_dir (str): Directory where simulation outputs will be stored.
        samples (List[Dict[str, Any]]): List of sample parameter 
            dictionaries.
    """
    logging.basicConfig(level=logging.INFO)
    original_directory = os.getcwd()                                        # Save the current directory to revert back to it later
    results =[]

    for i, sample in enumerate(samples):                                    # Loop to run a ddscat simulation for every sample
        shape = sample['shape']
        aeff = sample ['radius']                                                               
        aeff = round(aeff, 2)                                               # Round aeff to two decimal places
        simulation_directory = os.path.join(base_dir, f"sim_{i+1}")
        os.makedirs(simulation_directory, exist_ok=True)
        logging.info(
            f"Starting simulation for {shape} in {simulation_directory}")
        
        prep_input_ddscat.save_sample_parameters(simulation_directory, 
                                                 sample)                    # Save sample parameters

        shape_file_name = "shape.dat"
        d = 0.01

        if shape == "SPHERE":                                               # Generate shape file content based on sample's shape
            shape_file_content = prep_input_ddscat.generate_shape_dat(
                'sphere', d, sample)
        elif shape == "RCTGLPRSM":
            try:
                shape_file_content = prep_input_ddscat.generate_shape_dat(
                    'rect_prism', d, sample)
            except ValueError as e:
                logging.error(f"Error generating shape for \
                              {shape} in {simulation_directory}: {e}")
                continue
        else:
            logging.error(f"Unsupported shape type: {shape}")
            continue

        prep_input_ddscat.save_shape_info_to_file(shape_file_name,           # Save the generated shape information to file
                                                 shape_file_content, 
                                                 simulation_directory)

        params = {                                                          # Define parameters for the ddscat simulation
            "par_file_name": "ddscat.par",
            "shape": "FROM_FILE",
            "material_file": (sample['mat_file']),
            "vacuum_wavelengths": (sample['wavelength'],
                                   sample['wavelength']),
            "wavelengths_count": 1,
            "aeff_range": (aeff, aeff),
            "beta_params": (0, 0, 1),
            "theta_params": (0, 0, 1),
            "phi_params": (0, 0, 1),
            "plane1": (0, 0, 360, 5),                                       # phi, theta_min, theta_max, dtheta
            "plane2": (90, 0, 360, 5)                                       # phi, theta_min, theta_max, dtheta
            }

        par_file_content = prep_input_ddscat.generate_par(params)                  # Generate and save the parameter information file
        prep_input_ddscat.save_param_info_to_file(params, 
                                                 par_file_content, 
                                                 simulation_directory)
        
        exec_ddscat.run_ddscat(simulation_directory)                          # Run the ddscat simulation
        target_file_name = "output"
        exec_ddscat.convert_to_vtk(simulation_directory,                      # Convert the output to VTK format
                                 target_file_name)

        logging.info(f"Simulation data generated and processed \
                     in {simulation_directory}")
        proc_output_ddscat.process_ddscat_result(simulation_directory,           # Process the simulation results
                                            results,
                                            sample)
        
    os.chdir(original_directory)                                            # Revert back to the original directory
    logging.info("Simulation completed for all samples.")

    return pd.concat(results, ignore_index=True)                            # Combine all results into a DataFrame and return


def extract_simulation_data(base_dir):
    """
    Extract data from the simulation results and return DataFrames and labels.

    Parameters:
        base_dir (str): Base directory where simulation data is stored.

    Returns:
        tuple: (List of DataFrames, List of labels)
    """
    file_paths, labels = proc_output_ddscat.find_data_files(base_dir)
    data_frames = [proc_output_ddscat.extract_data(fp) for fp in file_paths]
    return data_frames, labels
