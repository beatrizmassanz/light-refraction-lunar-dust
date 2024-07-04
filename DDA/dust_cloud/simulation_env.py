import numpy as np
import os
from scipy.stats import norm, uniform
import random

def generate_positions_in_sphere(shape_params):
    """
    Generate positions on a grid within a sphere of given radius.
    
    Parameters:
        radius (float): The radius of the sphere.
        d (float): Spacing between dipoles.
    
    Returns:
        np.ndarray: Array of positions within the sphere.
    """
    radius = shape_params['radius']
    d = shape_params['d']
    num_dipoles_per_dimension = int(2 * radius / d)
    x = np.linspace(-radius, radius, 
                    num_dipoles_per_dimension)
    y = np.linspace(-radius, radius, 
                    num_dipoles_per_dimension)
    z = np.linspace(-radius, radius, 
                    num_dipoles_per_dimension)
    grid = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)

    mask = np.sum(np.square(grid), axis=1) <= radius**2             # Filter out points outside the sphere   
    unique_positions = np.unique(grid[mask], axis=0)                   # Ensure they are unique
    return unique_positions

def generate_positions_in_rect_prism(shape_params):
    """
    Generate grid positions within a rectangular prism defined by its dimensions.
    
    Parameters:
        length (float): Length of the rectangular prism.
        width (float): Width of the rectangular prism.
        height (float): Height of the rectangular prism.
        d (float): Spacing between dipoles.
    
    Returns:
        np.ndarray: Array of positions in a grid.
    """
    radius = shape_params['radius']
    d = shape_params['d']
    sphere_volume = (4/3) * np.pi * radius**3                   # Calculate the volume of the sphere
    length = round(uniform.rvs(loc=0.01, scale=0.09), 2)        # Generate random length and width values with two decimal places
    width = round(uniform.rvs(loc=0.01, scale=0.09), 2)
    height = round(sphere_volume / (length * width),2)                 # Calculate the height to maintain the same volume
    print(f"Rectangular Prism Dimensions - Length: {length}, Width: {width}, Height: {height}")
    # Ensure height is within reasonable limits
    if height > 0.1:
        height = 0.1
    max_dimension = max(length, width, height)
    grid_x = np.arange(0, length, d)                                   # Generate grid positions
    grid_y = np.arange(0, width, d)
    grid_z = np.arange(0, height, d)
    grid = np.meshgrid(grid_x, grid_y, grid_z, indexing='ij')
    positions = np.column_stack([g.flatten() for g in grid])
    unique_positions = np.unique(positions, axis=0)

    # Print dimensions for debugging
    print(f"Rectangular Prism Dimensions - Length: {length}, Width: {width}, Height: {height}, Max Dimension: {max_dimension}")
    return unique_positions, max_dimension
    

def generate_shape_dat(geometry, **shape_params):
    """
    Generates a formatted shape file content for a dust cloud with unique, non-duplicated positions
    within a specified geometry using given dipole spacing.

    Parameters:
        geometry (str): Type of geometry ('sphere', 'rect_prism').
        params (dict): Parameters needed for the geometry, e.g., radius for sphere or dimensions for prism.
        d (float): Spacing between dipoles.

    Returns:
        list: Formatted lines of a shape file representing the dust cloud.
    """
    if geometry == 'sphere':
        unique_positions = generate_positions_in_sphere(shape_params)
        max_dimension = 2 * shape_params['radius']
        grid_center = (np.max(unique_positions, axis=0) - np.min(unique_positions, axis=0)) / 2 + np.min(unique_positions, axis=0)
    elif geometry == 'rect_prism':
        unique_positions, max_dimension = generate_positions_in_rect_prism(shape_params)
        grid_center = (np.max(unique_positions, axis=0) - np.min(unique_positions, axis=0)) / 2 + np.min(unique_positions, axis=0)
    else:
        raise ValueError("Unsupported geometry type provided")

    num_dipoles = len(unique_positions)

    shape_file_content = [
        f">GEOMETRY   {geometry}; AX,AY,AZ= {max_dimension:.4f} {max_dimension:.4f} {max_dimension:.4f}",
        f"     {num_dipoles} = NAT ",
        "  1.000000  0.000000  0.000000 = A_1 vector",
        "  0.000000  1.000000  0.000000 = A_2 vector",
        "  1.000000  1.000000  1.000000 = lattice spacings (d_x,d_y,d_z)/d",
        f" {-grid_center[0]:.5f} {-grid_center[1]:.5f} {-grid_center[2]:.5f} = lattice offset x0(1-3) = (x_TF, y_TF, z_TF)/d for dipole 0 0 0",
        "     JA  IX  IY  IZ ICOMP(x,y,z)"
    ]

    for I, pos in enumerate(unique_positions, start=1):
        ix, iy, iz = ((pos - np.min(unique_positions, axis=0)) / shape_params['d'] + 1).astype(int)
        icomp = "1 1 1"  # Default component setting
        shape_file_content.append(f"     {I}  {ix}  {iy}  {iz} {icomp}")

    return shape_file_content

'''
def gen_orbital_positions(orbit_height_min, orbit_height_max,
                          moon_radius, density):
    """
    Generate positions for n particles in an orbit around the Moon.

    Parameters:
        orbit_height_min (float): Minimum orbit height in km.
        orbit_height_max (float): Maximum orbit height in km.
        moon_radius (float): Radius of the Moon in km.
        density (float): Density of particles per cubic meter.

    Returns:
        np.array: Positions of particles in orbital space.
    """
    volume_min = 4/3 * np.pi * (moon_radius + orbit_height_min)**3     # Total vol available in the shell between min and max orbit
    volume_max = 4/3 * np.pi * (moon_radius + orbit_height_max)**3
    total_volume = volume_max - volume_min
    expected_particles = int(total_volume * density)                   # Convert volume from km^3 to m^3 (1 km^3 = 1e9 m^3)
                                                                       # Number of particles based on density and available volume
    radii = np.random.uniform(moon_radius + orbit_height_min,          # Generate random spherical coordinates within the volume
                              moon_radius + orbit_height_max, 
                              expected_particles)
    theta = np.random.uniform(0, np.pi, expected_particles)
    phi = np.random.uniform(0, 2 * np.pi, expected_particles)
    x = radii * np.sin(theta) * np.cos(phi)                            # Convert spherical to Cartesian coordinates
    y = radii * np.sin(theta) * np.sin(phi)
    z = radii * np.cos(theta)
    return np.vstack((x, y, z)).T


def generate_positions_in_sphere(par_radius, d):
    """
    Generate positions on a grid within a sphere of given radius.
    
    Parameters:
        radius (float): The radius of the sphere.
        dipole_spacing (float): Spacing between dipoles.
    
    Returns:
        np.ndarray: Array of positions within the sphere.
    """
    num_dipoles_per_dimension = int(2 * par_radius / d)
    x = np.linspace(-par_radius, par_radius, 
                    num_dipoles_per_dimension)
    y = np.linspace(-par_radius, par_radius, 
                    num_dipoles_per_dimension)
    z = np.linspace(-par_radius, par_radius, 
                    num_dipoles_per_dimension)
    grid = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)

    mask = np.sum(np.square(grid), axis=1) <= par_radius**2            # Filter out points outside the sphere   
    unique_positions = np.unique(grid[mask], axis=0)                   # Ensure they are unique
    return unique_positions


def generate_positions_in_rect_prism(par_radius, d, dim1=None, 
                                     dim2=None):
    """
    Generate grid positions within a rectangular prism defined by a 
    spherical volume equivalent and optional dimensions.
    
    Parameters:
        par_radius (float): The radius of a sphere with the 
            equivalent volume of the prism.
        d (float): Spacing between dipoles.
        dim1 (float, optional): First dimension of the prism.
        dim2 (float, optional): Second dimension of the prism.
    
    Returns:
        np.ndarray: Array of positions in a grid.
    """
    sphere_volume = (4/3) * np.pi * par_radius**3                      # Calculate the volume of the sphere
    if dim1 is not None and dim2 is not None:                          # If two dimensions are given, calculate the third
        height = sphere_volume / (dim1 * dim2)
    elif dim1 is not None:                                             # Assume a cube if only one dimension is given
        height = dim1
        dim2 = dim1
    elif dim2 is not None:                                             # Assume a cube if only one dimension is given
        height = dim2
        dim1 = dim2
    else:                                                              # If no dimensions are given, assume a cube
        side_length = (sphere_volume) ** (1/3)  
        dim1, dim2, height = side_length, side_length, side_length

    length, width, height = dim1, dim2, height                         # Ensure dimensions are correctly set

    grid_x = np.arange(0, length, d)                                   # Generate grid positions
    grid_y = np.arange(0, width, d)
    grid_z = np.arange(0, height, d)
    grid = np.meshgrid(grid_x, grid_y, grid_z, indexing='ij')
    return np.column_stack([g.flatten() for g in grid])


def generate_shape_dat(geometry, d, **params):
    """
    Generates a formatted shape file content for a dust cloud with 
    unique, non-duplicated positions within a specified geometry 
    using given dipole spacing.

    Parameters:
        geometry (str): Type of geometry ('sphere', 'rect_prism').
        params (dict): Parameters needed for the geometry, e.g., 
            radius for sphere or dimensions for prism.
        d (float): Spacing between dipoles.

    Returns:
        list: Formatted lines of a shape file representing 
            the dust cloud.
    """
    if geometry == 'sphere':
        par_radius = params['radius']
        positions = generate_positions_in_sphere(par_radius, d)
        grid_center = (np.max(positions, axis=0) - np.min(positions, axis=0)) / 2 + np.min(positions, axis=0)
    elif geometry == 'rect_prism':
        # Correctly handling the extraction of dimensions
        par_radius = params.get('radius')
        dim1 = params.get('length')
        dim2 = params.get('width')
        positions = generate_positions_in_rect_prism(par_radius, d, dim1, dim2)
        grid_center = (np.max(positions, axis=0) - np.min(positions, axis=0)) / 2 + np.min(positions, axis=0)
    else:
        raise ValueError("Unsupported geometry type provided")
    
    num_dipoles = len(positions)

    shape_file_content = [
        f">GEOMETRY   {geometry}; AX,AY,AZ= {2*params['radius']:.4f} {2*params['radius']:.4f} {2*params['radius']:.4f}",
        f"     {num_dipoles} = NAT ",
        "  1.000000  0.000000  0.000000 = A_1 vector",
        "  0.000000  1.000000  0.000000 = A_2 vector",
        "  1.000000  1.000000  1.000000 = lattice spacings (d_x,d_y,d_z)/d",
        f" {-grid_center[0]:.5f} {-grid_center[1]:.5f} {-grid_center[2]:.5f} = lattice offset x0(1-3) = (x_TF, y_TF, z_TF)/d for dipole 0 0 0",
        "     JA  IX  IY  IZ ICOMP(x,y,z)"
    ]

    # Formatting grid indices, ensuring they start from 1
    for i, pos in enumerate(positions, start=1):
        ix, iy, iz = ((pos - np.min(positions, axis=0)) / d + 1).astype(int)
        icomp = "1 1 1"  # Default component setting
        shape_file_content.append(f"     {i}  {ix}  {iy}  {iz} {icomp}")

    return shape_file_content
'''
'''
if __name__ == "__main__":
    orbit_height_min = 1                                               # Minimum orbit height in km
    orbit_height_max = 10                                              # Maximum orbit height in km
    moon_radius = 1737.4                                               # Average radius of the Moon in km
    density = 0.005                                                    # Density of particles per cubic km
    positions = gen_orbital_positions(orbit_height_min,                # Generate the orbital position
                                      orbit_height_max, 
                                      moon_radius, density)
    print("Generated Positions:\n", positions)                         # Print the results
    print("Total Particles:", positions.shape[0])

    par_radius = 0.1                                                   # Equivalent spherical radius
    d = 0.01
    dim1 = 0.1                                                         # Optional
    dim2 = 0.2                                                         # Optional

    sphere_positions = generate_positions_in_sphere (radius, d)
    print("Sphere Generated positions count:", sphere_positions.shape[0])
    print("Sphere Sample positions:\n", sphere_positions[:5])          # Show first 5 positionse_lines:

    rect_positions = generate_positions_in_rect_prism(radius, d, dim1, dim2)
    print("Rect Generated positions count:", rect_positions.shape[0])
    print("Rect Sample positions:\n", rect_positions[:5])              # Show first 5 positionse_lines:

    params_sphere = {'radius': 0.1}
    params_prism = {'radius': 0.1, 'length': 0.2, 'width': 0.1}
    geometry = 'sphere'
    shape_file_lines = generate_shape_dat(geometry, params_sphere, d)
    for line in shape_file_lines:
        print(line)
'''