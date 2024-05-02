import os
import numpy as np


def generate_dust_cloud_data(num_dipoles, positions, orientations):
    #  Round positions to avoid duplication at the grid level assuming the lattice is integer-indexed
    rounded_positions = np.round(positions).astype(int)
    # Create a unique set of positions
    unique_positions, indices = np.unique(rounded_positions, return_index=True, axis=0)
    unique_orientations = orientations[indices]
    num_dipoles = len(unique_positions)  # Update the number of dipoles based on unique entries

    # Calculate the approximate bounding dimensions of the cloud
    max_dimension = np.max(np.abs(unique_positions)) * 2  # The factor of 2 ensures full coverage

    # Prepare the file content
    shape_file_content = [
        f">DUSTCLOUD   dust cloud; AX,AY,AZ= {max_dimension:.4f} {max_dimension:.4f} {max_dimension:.4f}",
        f"     {num_dipoles} = NAT ",
        "  1.000000  0.000000  0.000000 = A_1 vector",
        "  0.000000  1.000000  0.000000 = A_2 vector",
        "  1.000000  1.000000  1.000000 = lattice spacings (d_x,d_y,d_z)/d",
        f" {-max_dimension/2:.5f} {-max_dimension/2:.5f} {-max_dimension/2:.5f} = lattice offset x0(1-3) = (x_TF,y_TF,z_TF)/d for dipole 0 0 0",
        "     JA  IX  IY  IZ ICOMP(x,y,z)"
    ]

    # Add each dipole's data to the content
    for i, (pos, orient) in enumerate(zip(unique_positions, unique_orientations), start=1):
        ix, iy, iz = pos.astype(int)  # Convert positions to integer for indices
        icomp = "1 1 1"  # Default component setting
        shape_file_content.append(f"     {i}  {ix+1}  {iy+1}  {iz+1} {icomp}")
    
    return shape_file_content


def save_dust_cloud_info_to_file(shape_file_name, shape_file_content, shape_file_path):
    # Create the full path by combining the directory and file name
    full_file_path = os.path.join(shape_file_path, shape_file_name)
    # Open the file in write mode and write the content
    with open(full_file_path, "w") as file:
        file.write("\n".join(shape_file_content))


def generate_par(params):
    """
    Creates a .par file for DDSCAT based on a dictionary of parameters.
    
    Args:
        params (dict): Dictionary containing all the necessary parameters to generate the file.
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
        f"{params['vacuum_wavelengths'][0]:.4f} {params['vacuum_wavelengths'][1]:.4f} {params['wavelengths_count']} \'LIN\' = wavelengths (first, last, how many, how=LIN, INV, LOG)",
        '\'**** Refractive index of ambient medium\'',
        '1.000 = NAMBIENT',
        '\'**** Effective Radii (micron) ****\'',
        f'{params["aeff_range"][0]} {params["aeff_range"][1]} 1 \'LIN\' = aeff (first,last,how many,how=LIN,INV,LOG)',
        '\'**** Define Incident Polarizations ****\'',
        '(0,0) (1.,0.) (0.,0.) = Polarization state e01 (k along x axis)',
        '2 = IORTH  (=1 to do only pol. state e01; =2 to also do orth. pol. state)',
        '\'**** Specify which output files to write ****\'',
        '1 = IWRKSC (=0 to suppress, =1 to write ".sca" file for each target orient.',
        '\'**** Prescribe target Rotations ****\'',
        f'{params['beta_params'][0]:.1f}    {params['beta_params'][1]:.1f}   {params['beta_params'][2]}  = BETAMI, BETAMX, NBETA  (beta=rotation around a1)',
        f'{params['theta_params'][0]:.1f}    {params['theta_params'][1]:.1f}   {params['theta_params'][2]}  = THETMI, THETMX, NTHETA (theta=angle between a1 and k)',
        f'{params['phi_params'][0]:.1f}    {params['phi_params'][1]:.1f}   {params['phi_params'][2]}  = PHIMIN, PHIMAX, NPHI (phi=rotation angle of a1 around k)',
        '\'**** Specify first IWAV, IRAD, IORI (normally 0 0 0) ****\'',
        '0 0 0 = first IWAV, first IRAD, first IORI (0 0 0 to begin fresh)',
        '\'**** Select Elements of S_ij Matrix to Print ****\'',
        '6 = NSMELTS = number of elements of S_ij to print (not more than 9)',
        '11 12 21 22 31 41 = indices ij of elements to print',
        '\'**** Specify Scattered Directions ****\'',
        '\'LFRAME\' = CMDFRM (LFRAME, TFRAME for Lab Frame or Target Frame)',
        '2 = NPLANES = number of scattering planes',
        f'{params["plane1"][0]:.1f} {params["plane1"][1]:.1f} {params["plane1"][2]:.1f} {params["plane1"][3]} = phi, thetan_min, thetan_max, dtheta (in deg) for plane 1',
        f'{params["plane2"][0]:.1f} {params["plane2"][1]:.1f} {params["plane2"][2]:.1f} {params["plane2"][3]} = phi, thetan_min, thetan_max, dtheta (in deg) for plane 2'
    ]
    return par_file_content

def save_param_info_to_file(params, par_file_content, par_file_path):
    # Create the full path by combining the directory and file name
    par_full_file_path = os.path.join(par_file_path, params["par_file_name"])
    sav_full_file_path = os.path.join(par_file_path, "ddscat.par.sav")
    # Open the file in write mode and write the content
    with open(par_full_file_path, "w") as file:
        file.write("\n".join(par_file_content))
    with open(sav_full_file_path, "w") as file:
        file.write("\n".join(par_file_content))
