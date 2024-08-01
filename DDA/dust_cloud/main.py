import input as input
import ddscat_shape_generation as ddscat_shape_gen
import ddscat_input_generation as ddscat_input_gen
import visualization as visualization
import execution as execution
import os
import logging
import argparse
import pandas as pd


def setup_logging():
    """Configure the logging for the application."""
    logging.basicConfig(level=logging.INFO, 
                        filename='simulation.log', 
                        filemode='a', 
                        format=
                        '%(asctime)s - %(levelname)s - %(message)s')

def run_simulation(base_dir, samples):
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
        input.save_sample_parameters(simulation_directory, sample)          # Save sample parameters

        shape_file_name = "shape.dat"
        d = 0.01

        if shape == "SPHERE":                                               # Generate shape file content based on sample's shape
            shape_file_content = ddscat_shape_gen.generate_shape_dat(
                'sphere', d, sample)
        elif shape == "RCTGLPRSM":
            try:
                shape_file_content = ddscat_shape_gen.generate_shape_dat(
                    'rect_prism', d, sample)
            except ValueError as e:
                logging.error(f"Error generating shape for \
                              {shape} in {simulation_directory}: {e}")
                continue
        else:
            logging.error(f"Unsupported shape type: {shape}")
            continue

        ddscat_input_gen.save_shape_info_to_file(shape_file_name,           # Save the generated shape information to file
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

        par_file_content = ddscat_input_gen.generate_par(params)            # Generate and save the parameter information file
        ddscat_input_gen.save_param_info_to_file(params, 
                                                 par_file_content, 
                                                 simulation_directory)
        
        execution.run_ddscat(simulation_directory)                          # Run the ddscat simulation
        target_file_name = "output"
        execution.convert_to_vtk(simulation_directory,                      # Convert the output to VTK format
                                 target_file_name)

        logging.info(f"Simulation data generated and processed \
                     in {simulation_directory}")
        visualization.process_ddscat_result(simulation_directory,           # Process the simulation results
                                            results,
                                            sample)
        
    os.chdir(original_directory)                                            # Revert back to the original directory
    logging.info("Simulation completed for all samples.")

    return pd.concat(results, ignore_index=True)                            # Combine all results into a DataFrame and return


def main(base_dir, skip_simulation=False, only_spheres=False):
    """
    Execute multiple simulations based on predefined geometries 
    and log the outcomes.
    
    Parameters:
        base_dir (str): Base directory where simulation data 
            will be stored.
        skip_simulation (bool): If True, skip running simulations and only 
            perform data analysis.
        only_spheres (bool): If True, only generate spherical samples and 
            perform Mie calculations.
            
    Returns:
        Optional[pd.DataFrame]: DataFrame containing the combined results 
            of all simulations if simulations are run, None if simulations 
            are skipped and only analysis is performed.
    """
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(message)s')
    base_dir = os.path.abspath(base_dir)                                    # Convert base_dir to an absolute path

    if skip_simulation:                                                     # If skipping simulation, process the existing results
        results_df = visualization.process_results(base_dir)

        if only_spheres:                                                    # If only analyzing spherical samples, load/ filter them
            samples = visualization.load_samples("Generated_samples.json")
            samples = [
                sample for sample in samples 
                if sample['shape'].lower() == 'sphere'
                ]
    else:                                                                   # Generate sample parameters for simulations
        num_samples = 2                                                     # SET NUMBER OF SAMPLES TO GENERATE
        samples = input.sample_parameters(num_samples, 
                                          random_seed=297, 
                                          only_spheres=only_spheres)
        results_df = run_simulation(base_dir, samples)
    
    mie_df = None
    if only_spheres:                                                        # Perform Mie calculations if only spherical samples
        mie_df = execution.mie_calculation(samples)

    results = results_df.to_dict('records')                                 # Convert results DataFrame to a list of dictionaries

    if mie_df is not None:                                                  # Process Mie results if available
        results = visualization.process_mie_result(mie_df, 
                                                   results, 
                                                   samples)

    results_df = pd.DataFrame(results)                                      # Convert the processed results back to a DataFrame
    
    visualization.analyze_results(results_df)                               # Analyze and visualize the results
    file_paths, labels = visualization.find_data_files(base_dir)
    data_frames = [
        visualization.extract_data(fp) for fp in file_paths
        ]
    visualization.plot_data(data_frames, labels)
    visualization.plot_polar_data(data_frames, labels)

    if mie_df is not None:                                                  # Plot comparison Mie and DDSCAT if Mie Data available
        visualization.plot_mie_ddscat_comparison(results_df, mie_df)

    results_df.to_csv(os.path.join(base_dir, 'simulation_results.csv'),     # Save the results to a CSV file
                      index=False)
    print(f"Results saved to {os.path.join(base_dir, 
                                           'simulation_results.csv')}")
    return print(results_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run DDSCAT simulations.'
        )
    parser.add_argument(
        'base_dir', type=str, nargs='?', default=os.getcwd(), 
        help='Base directory where simulation data will be stored.'
        )
    parser.add_argument(
        '--skip_simulation', action='store_true', 
        help='Skip running simulations and only perform data analysis.'
        )
    parser.add_argument(
        '--only_spheres', action='store_true', 
        help='Only generate spherical samples and do Mie calculations.'
        )

    args = parser.parse_args()                                                 # Parse the command-line arguments    
    main(args.base_dir,                                                        # Call the main function with parsed arguments
         skip_simulation=args.skip_simulation, 
         only_spheres=args.only_spheres)

