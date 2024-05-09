import numpy as np

def random_cloud_gen(num_dipoles, cube_size):
    """
    Generate randomly distributed dipoles within a cubic volume.

    Parameters:
    num_dipoles (int): Number of dipoles to generate.
    cube_size (float): Side length of the cube within which dipoles are generated.

    Returns:
    tuple: A tuple containing two numpy arrays; positions and orientations of the dipoles.
    """
    positions = np.random.uniform(low=-cube_size/2, high=cube_size/2, size=(num_dipoles, 3))
    orientations = np.random.uniform(low=-1.0, high=1.0, size=(num_dipoles, 3))
    orientations /= np.linalg.norm(orientations, axis=1)[:, np.newaxis]  # Normalize to unit vectors
    return positions, orientations
