import proc_output_ddscat as proc_output_ddscat
import os
import numpy as np
from scipy.stats import norm, uniform
import subprocess
import pandas as pd
import numpy as np
import logging
import json


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
        save_sample_parameters(simulation_directory, sample)          # Save sample parameters

        shape_file_name = "shape.dat"
        d = 0.01

        if shape == "SPHERE":                                               # Generate shape file content based on sample's shape
            shape_file_content = generate_shape_dat(
                'sphere', d, sample)
        elif shape == "RCTGLPRSM":
            try:
                shape_file_content = generate_shape_dat(
                    'rect_prism', d, sample)
            except ValueError as e:
                logging.error(f"Error generating shape for \
                              {shape} in {simulation_directory}: {e}")
                continue
        else:
            logging.error(f"Unsupported shape type: {shape}")
            continue

        save_shape_info_to_file(shape_file_name,           # Save the generated shape information to file
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

        par_file_content = generate_par(params)                  # Generate and save the parameter information file
        save_param_info_to_file(params, 
                                                 par_file_content, 
                                                 simulation_directory)
        
        run_ddscat(simulation_directory)                          # Run the ddscat simulation
        target_file_name = "output"
        convert_to_vtk(simulation_directory,                      # Convert the output to VTK format
                                 target_file_name)

        logging.info(f"Simulation data generated and processed \
                     in {simulation_directory}")
        proc_output_ddscat.process_ddscat_result(simulation_directory,           # Process the simulation results
                                            results,
                                            sample)
        
    os.chdir(original_directory)                                            # Revert back to the original directory
    logging.info("Simulation completed for all samples.")

    return pd.concat(results, ignore_index=True)                            # Combine all results into a DataFrame and return

def save_sample_parameters(simulation_directory, sample):
    """
    Save the sample parameters to a JSON file in the specified simulation 
    directory.

    Parameters:
        simulation_directory (str): The directory where the sample 
        parameters will be saved.
        sample (dict): The sample parameters to be saved.
    """
    sample_file_path = os.path.join(simulation_directory, 
                                    'sample_parameters.json')
    with open(sample_file_path, 'w') as f:
        json.dump(sample, f, indent=4)

def gen_pos_in_sphere(d, sample):
    """
    Generate positions on a grid within a sphere of given radius.
    
    Parameters:
        radius (float): The radius of the sphere.
        d (float): Spacing between dipoles.
    
    Returns:
        np.ndarray: Array of positions within the sphere.    
    """
    radius = sample['radius']
    num_dipoles_per_dimension = int(2 * radius / d)
    x = np.linspace(-radius, radius, 
                    num_dipoles_per_dimension)
    y = np.linspace(-radius, radius, 
                    num_dipoles_per_dimension)
    z = np.linspace(-radius, radius, 
                    num_dipoles_per_dimension)
    grid = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)

    mask = np.sum(np.square(grid), axis=1) <= radius**2                     # Filter out points outside the sphere   
    unique_positions = np.unique(grid[mask], axis=0)                        # Ensure they are unique
    return unique_positions

def gen_pos_in_rect_prism(d, sample):
    """
    Generate grid positions within a rectangular prism defined by its 
    dimensions.
    
    Parameters:
        length (float): Length of the rectangular prism.
        width (float): Width of the rectangular prism.
        height (float): Height of the rectangular prism.
        d (float): Spacing between dipoles.
    
    Returns:
        np.ndarray: Array of positions in a grid.
    """
    radius = sample['radius']
    x_length = sample ['x_length']                                          # Generate random length and width values
    y_length = sample ['y_length']      
    z_length = sample ['z_length']  
    
    max_dimension = 2*max(x_length, y_length, z_length)

    num_dipoles_x = int(np.floor(x_length / d))                             # Calculate the number of dipoles that fit
    num_dipoles_y = int(np.floor(y_length / d))
    num_dipoles_z = int(np.floor(z_length / d))
    
    grid_x = np.linspace(0, x_length - d, num_dipoles_x)                    # Generate grid positions ensuring they fit in dimensions
    grid_y = np.linspace(0, y_length - d, num_dipoles_y)
    grid_z = np.linspace(0, z_length - d, num_dipoles_z)
    
    grid = np.meshgrid(grid_x, grid_y, grid_z, indexing='ij')
    positions = np.column_stack([g.flatten() for g in grid])
    unique_positions = np.unique(positions, axis=0)

    if len(unique_positions) != len(positions):                             # Verify that there are no duplicates
        duplicate_positions = len(positions) - len(unique_positions)
        #print(f"Duplicate positions detected: {duplicate_positions}")      # Optional debugging pring
        #print(f"Unique: {len(unique_positions)}, Total: {len(positions)}")
        raise ValueError("Duplicate positions detected in the grid.")

    print(                                                                  # Print dimensions for debugging
        f"Rectangular Prism Dimensions - Length: {x_length}, "
        "Width: {y_length}, Height: {z_length}, "
        "Max Dimension: {max_dimension}"
        )
    return unique_positions, max_dimension
    

def generate_shape_dat(geometry, d, sample):
    """
    Generates a formatted shape file content for a dust cloud with unique,
    non-duplicated positions within a specified geometry using a given 
    dipole spacing.

    Parameters:
        geometry (str): The type of geometry to generate. 
            Supported types are:
            'sphere': Generates positions within a spherical geometry.
            'rect_prism': Generates positions within a rectangular 
                prism geometry.
        d (float): The spacing between dipoles.
        sample (dict): Parameters needed for the geometry. Should include:
            'radius' (float) for sphere.
            'x_length', 'y_length', 'z_length' (floats) for rectangular 
                prism.

    Returns:
        list: Formatted lines representing the shape file content, including
        metadata and dipole positions.
    
    Raises:
        ValueError: If an unsupported geometry type is provided.
    """ 
    if geometry == 'sphere':                                                # Generate positions based on the specified geometry    
        unique_positions = gen_pos_in_sphere(d, sample)
        max_dimension = 2 * sample['radius']
    elif geometry == 'rect_prism':
        unique_positions, max_dimension = gen_pos_in_rect_prism(d, sample)
    else:
        raise ValueError("Unsupported geometry type provided")

    num_dipoles = len(unique_positions)                                     # Calculate the number of dipoles
    grid_center = (                                                         # Calculate the grid center to get the lattice offset
        ((np.max(unique_positions, axis=0) 
         - np.min(unique_positions, axis=0)) / 2 
         + np.min(unique_positions, axis=0))
    )

    shape_file_content = [                                                  # Initialize the shape file content with metadata lines
        f">GEOMETRY   {geometry}; AX,AY,AZ= {max_dimension:.4f} {max_dimension:.4f} {max_dimension:.4f}",
        f"     {num_dipoles} = NAT ",
        "  1.000000  0.000000  0.000000 = A_1 vector",
        "  0.000000  1.000000  0.000000 = A_2 vector",
        "  1.000000  1.000000  1.000000 = lattice spacings (d_x,d_y,d_z)/d",
        f" {-grid_center[0]:.5f} {-grid_center[1]:.5f} {-grid_center[2]:.5f} = lattice offset x0(1-3) = (x_TF, y_TF, z_TF)/d for dipole 0 0 0",
        "     JA  IX  IY  IZ ICOMP(x,y,z)"
    ]

    dipole_positions = set()                                                # Track positions to avoid duplicates                       
    min_pos = np.min(unique_positions, axis=0)

    for I, pos in enumerate(unique_positions, start=1):                     # Add each unique dipole position to shape file content
        ix, iy, iz = ((pos - min_pos) / d + 1).astype(int)                  # Calculate indices of dipole position in the grid
        icomp = "1 1 1"                                                     # Default component setting

        dipole_position = (ix, iy, iz)                                      # Check for duplicate positions
        if dipole_position in dipole_positions:
            print(f"Duplicate detected in shape file generation: \
                    {dipole_position}")
            continue

        dipole_positions.add(dipole_position)                               # Add the position to the set to track it
        shape_file_content.append(f"     {I}  {ix}  {iy}  {iz} {icomp}")    # Append the formatted dipole position to file content

    return shape_file_content

def save_shape_info_to_file(shape_file_name, shape_file_content, 
                                 simulation_directory):
    """
    Saves the generated shape file content for a dust cloud to a 
    specified file. 

    Parameters:
        shape_file_name (str): The name of the file to create.
        shape_file_content (list): The content of the shape file as 
            a list of strings.
        simulation_directory (str): The directory where the shape 
            file will be saved.

    Writes the shape file content to a file in 
    the specified directory.
    """
    full_file_path = os.path.join(simulation_directory,                 # Create full path by combining directory and file name
                                  shape_file_name)
    with open(full_file_path, "w") as file:                             # Open the file in write mode and write the content
        file.write("\n".join(shape_file_content))
                                                                        # Print first few lines of the shape file for debugging
    print("\n".join(shape_file_content[:10]))                           # Adjust the number of lines to print as needed
    
def generate_par(params):
    """
    Creates a .par file content for DDSCAT based on a dictionary of 
    parameters.
    
    Parameters:
        params (dict): A dictionary containing all necessary parameters 
            to generate the file.

    Returns:
        list of str: Lines of the .par file as a list of strings.
    """
    par_file_content = [
        ' ========= Parameter file for v7.3 ===================',
        '**** Preliminaries ****',
        '\'NOTORQ\' = CMDTRQ*6 (NOTORQ, DOTORQ) -- either do or skip torque calculations',
        '\'PBCGS2\' = CMDSOL*6 (PBCGS2, PBCGST, GPBICG, PETRKP, QMRCCG) -- CCG method',
        '\'GPFAFT\' = CMDFFT*6 (GPFAFT, FFTMKL) -- FFT method',
        '\'GKDLDR\' = CALPHA*6 (GKDLDR, LATTDR, FLTRCD) -- DDA method',
        '\'NOTBIN\' = CBINFLAG (NOTBIN, ORIBIN, ALLBIN) -- specify binary output',
        '\'**** Initial Memory Allocation ****\'',
        '100 100 100 = dimensioning allowance for target generation',
        '\'**** Target Geometry and Composition ****\'',
        f'\'{params["shape"]} = CSHAPE*9 shape directive\'',
        'no SHPAR parameters needed',
        '1 = NCOMP = number of dielectric materials',
        f'\'../diel/{params["material_file"]}\' = file with refractive index 1',
        '\'**** Additional Nearfield calculation? ****\'',
        '0 = NRFLD (=0 to skip nearfield calc., =1 to calculate nearfield E)',
        '0.0 0.0 0.0 0.0 0.0 0.0 (fract. extens. of calc. vol. in -x,+x,-y,+y,-z,+z)',
        '\'**** Error Tolerance ****\'',
        '1.00e-5 = TOL = MAX ALLOWED (NORM OF |G>=AC|E>-ACA|X>)/(NORM OF AC|E>)',
        '\'**** maximum number of iterations allowed ****\'',
        '300 = MXITER',
        '\'**** Interaction cutoff parameter for PBC calculations ****\'',
        '1.00e-2 = GAMMA (1e-2 is normal, 3e-3 for greater accuracy)',
        '\'**** Angular resolution for calculation of <cos>, etc. ****\'',
        '0.5 = ETASCA (number of angles is proportional to [(3+x)/ETASCA]^2 )',
        '\'**** Vacuum wavelengths (micron) ****\'',
        (f"{params['vacuum_wavelengths'][0]:.4f} {params['vacuum_wavelengths'][1]:.4f} {params['wavelengths_count']} "
            "\'LIN\' = wavelengths (first, last, how many, how=LIN, INV, LOG)"),
        '\'**** Refractive index of ambient medium\'',
        '1.000 = NAMBIENT',
        '\'**** Effective Radii (micron) ****\'',
        (f"{params["aeff_range"][0]} {params["aeff_range"][1]} 1 \'LIN\' = "
            "aeff (first,last,how many,how=LIN,INV,LOG)"),
        '\'**** Define Incident Polarizations ****\'',
        '(0,0) (1.,0.) (0.,0.) = Polarization state e01 (k along x axis)',
        '2 = IORTH  (=1 to do only pol. state e01; =2 to also do orth. pol. state)',
        '\'**** Specify which output files to write ****\'',
        '1 = IWRKSC (=0 to suppress, =1 to write ".sca" file for each target orient.',
        '\'**** Prescribe target Rotations ****\'',
        (f"{params['beta_params'][0]:.1f}    {params['beta_params'][1]:.1f}   {params['beta_params'][2]}  = "
            "BETAMI, BETAMX, NBETA  (beta=rotation around a1)"),
        (f"{params['theta_params'][0]:.1f}    {params['theta_params'][1]:.1f}   {params['theta_params'][2]} = "
            "THETMI, THETMX, NTHETA (theta=angle between a1 and k)"),
        (f"{params['phi_params'][0]:.1f}    {params['phi_params'][1]:.1f}   {params['phi_params'][2]}  = "
            " PHIMIN, PHIMAX, NPHI (phi=rotation angle of a1 around k)"),
        '\'**** Specify first IWAV, IRAD, IORI (normally 0 0 0) ****\'',
        '0 0 0 = first IWAV, first IRAD, first IORI (0 0 0 to begin fresh)',
        '\'**** Select Elements of S_ij Matrix to Print ****\'',
        '6 = NSMELTS = number of elements of S_ij to print (not more than 9)',
        '11 12 21 22 31 41 = indices ij of elements to print',
        '\'**** Specify Scattered Directions ****\'',
        '\'LFRAME\' = CMDFRM (LFRAME, TFRAME for Lab Frame or Target Frame)',
        '2 = NPLANES = number of scattering planes',
        (f"{params["plane1"][0]:.1f} {params["plane1"][1]:.1f} {params["plane1"][2]:.1f} {params["plane1"][3]} = "
        "phi, thetan_min, thetan_max, dtheta (in deg) for plane 1"),
        (f"{params["plane2"][0]:.1f} {params["plane2"][1]:.1f} {params["plane2"][2]:.1f} {params["plane2"][3]} = "
        "phi, thetan_min, thetan_max, dtheta (in deg) for plane 2")
    ]
    return par_file_content

def save_param_info_to_file(params, par_file_content, simulation_directory):
    """
    Saves the parameter file content to two files in the 
    specified directory.

    Parameters:
        params (dict): Dictionary containing 'par_file_name' as 
            a key with the name of the primary .par file.
        par_file_content (list of str): The content of the parameter 
            file to be saved.
        simulation_directory (str): The directory where the parameter 
            files will be saved.

    This function saves the parameter content to a .par file and a 
    .par.sav file in the given directory.
    """
    par_full_file_path = os.path.join(simulation_directory,             # Create full path by combining the directory and file name        
                                      params["par_file_name"])
    sav_full_file_path = os.path.join(simulation_directory, 
                                      "ddscat.par.sav")
    with open(par_full_file_path, "w") as file:                         # Write the parameter content to the .par file
        file.write("\n".join(par_file_content))
    with open(sav_full_file_path, "w") as file:                         # Write the parameter content to the .sav file
        file.write("\n".join(par_file_content))

def run_ddscat(par_file_path):
    """
    Executes the DDSCAT simulation from the given directory.

    Parameters:
    par_file_path (str): Path to the directory containing the 
    'ddscat.par' configuration file.
    """
    original_directory = os.getcwd()                                   # Save the current directory to revert back to it later
    os.chdir(par_file_path)                                            # Change to the directory of DDSCAT configuration file
    ddscat_path = os.path.join('..','..', 'src', 'ddscat')             # Relative path to vtrconvert, two directories up in 'src'
    command = f"./{ddscat_path}"                                       # Construct the DDSCAT command

    try:                                                               # Execute the command
        print("Starting DDSCAT simulation...")
        subprocess.run(command, check=True)
        print("DDSCAT simulation completed successfully.")
    except subprocess.CalledProcessError as e:
        print("DDSCAT simulation failed:", e)
    finally:
        os.chdir(original_directory)                                   # Change back to the original directory

def convert_to_vtk(simulation_directory, target_file_name):
    """
    Converts a DDSCAT shape file to a VTK file using vtrconvert.

    Parameters:
    simulation_directory (str): Path to the directory containing the 
        input file 'target.out'.
    target_file_name (str): Base name for the output VTK file.
    """
    original_directory = os.getcwd()                                   # Save the current directory to revert back to it later
    os.chdir(simulation_directory)                                     # Change to the directory containing the input file
    vtrconvert_path = os.path.join('..', '..', 'src', 'vtrconvert')    # Relative path to vtrconvert, two directories up in 'src'
    command = f"{vtrconvert_path} target.out {target_file_name}"       # Construct the vtrconvert command

    try:                                                               # Execute the command
        output = subprocess.check_output(command, shell=True, 
                                         stderr=subprocess.STDOUT)
        print("Conversion successful:", output.decode())
    except subprocess.CalledProcessError as e:
        print("Error during conversion:", e.output.decode())
    finally:
        os.chdir(original_directory)                                   # Change back to the original directory