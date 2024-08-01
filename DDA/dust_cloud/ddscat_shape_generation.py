import numpy as np
from scipy.stats import norm, uniform

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