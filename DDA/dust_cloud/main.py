import gen_input as gen_input
import run_ddscat as run_ddscat
import run_mie as run_mie
import proc_results as proc_results
import visualization as visualization
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
        results_df = run_ddscat.process_existing_results(base_dir)
        if only_spheres:                                                    # If only analyzing spherical samples, load/ filter them
            samples = gen_input.load_samples("Generated_samples.json")
            samples = [
                sample for sample in samples 
                if sample['shape'].lower() == 'sphere'
                ]
    else:                                                                   # Generate sample parameters for simulations
        num_samples = 1000                                                  # SET NUMBER OF SAMPLES TO GENERATE
        samples = gen_input.sample_parameters(num_samples, 
                                          random_seed=292, 
                                          only_spheres=only_spheres)
        results_df = run_ddscat.run_simulations(base_dir, samples)
    
    mie_df = None

    if only_spheres:                                                        # Perform Mie calculations if only spherical samples
        mie_df = run_mie.mie_calculation(samples)

    results = results_df.to_dict('records')                                 # Convert results DataFrame to a list of dictionaries

    if mie_df is not None:                                                  # Process Mie results if available
        results = run_mie.process_mie_result(mie_df, 
                                                   results, 
                                                   samples)

    proc_results.save_results_to_csv(results, base_dir)                     # Save results to CSV file

    
    data_frames, labels = run_ddscat.extract_simulation_data (base_dir)

    #visualization.plot_ddscat_correlation_results(results_df)               # Analyze and visualize the results
    #visualization.plot_data(data_frames, labels)
    #visualization.plot_polar_data(data_frames, labels)
    visualization.plot_shape_counts (results_df)
    visualization.plot_size_param_distribution (results_df)
    visualization.plot_wavelength_distribution (results_df)
    visualization.plot_radius_distribution (results_df)
    visualization.plot_qsca_vs_size (results_df)
    visualization.plot_qsca_vs_wavelength (results_df)
    visualization.plot_s11_vs_size_forward_scattering (results_df)
    visualization.plot_s11_vs_wavelength_forward_scattering (results_df)
    visualization.plot_qsca_by_size (results_df)
    visualization.plot_qbk_by_size (results_df)
    visualization.plot_qpol_by_size (results_df)
    visualization.plot_s11_forward_scattering (results_df)
    visualization.plot_s11_back_scattering (results_df)
    #visualization.plot_pol_vs_theta (results_df)
    visualization.plot_average_pol_vs_theta (results_df)
    visualization.plot_average_s11_vs_theta (results_df)

    if mie_df is not None:                                                  # Plot comparison Mie and DDSCAT if Mie Data available
        visualization.plot_mie_ddscat_comparison(results_df, mie_df)

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

