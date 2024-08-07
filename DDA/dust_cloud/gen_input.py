from scipy.stats import norm, uniform
import random
import json
import numpy as np


def sample_parameters(num_samples, random_seed=None, only_spheres=False):
    """
    Generate sampled parameters for simulations.
    
    Parameters:
        num_samples (int): Number of samples to generate.
        random_seed (Optional[int]): Seed for random number generators.
        only_spheres (bool): If True, only generate spherical samples.

    Returns:
        List[Dict[str, Any]]: List of parameter dictionaries.
    """
    if random_seed is not None:                                             # Set random seed if not given                    
        np.random.seed(random_seed)
        random.seed(random_seed)

    shapes = ["SPHERE"] if only_spheres else ["SPHERE", "RCTGLPRSM"]        # Shape selection based on flag
    samples = []
    for _ in range(num_samples):
        shape = random.choice(shapes)                                       # Randomly select a shape
        wavelength = uniform.rvs(loc=0.380, scale=0.370)                    # SET WAVELENGTH RANGE AND DISTRIBUTION
        # wavelength = 0.5600                                               # To set a fixed value comment line before and set here
        wavelength = round(wavelength, 4)                                   
        mat_file = "astrosil"                                               # SELECT MATERIAL FILE ("astrosil" or "custom")

        if shape == "SPHERE":                                               # Generate spherical particle sample
            radius = norm.rvs(loc=0.15, scale=0.05)                         # SELECT PARTICLE SIZE RANGE AND DISTRIBUTION
            radius = np.clip(radius, 0.005, 0.35)
            # radius = 0.1                                                  # To set a fixed value comment two lines before                    
            radius = round(radius, 4)                                       
            volume = (4/3) * np.pi * radius**3                              # Calculate volume of the sphere
            volume = round(volume, 4)                                       
            size_param = (2 * np.pi * radius) / wavelength                  # Calculate size parameter
            size_param = round(size_param, 4)                               
            sample = {
                "shape": shape,
                "radius": radius,
                "wavelength": wavelength,
                "size_param": size_param,
                "volume": volume,
                "mat_file": mat_file
            }
        
        elif shape == "RCTGLPRSM":                                          # Generate prism particle sample
            x_length = norm.rvs(loc=0.24, scale=0.11)                       # SELECT x-length RANGE AND DISTRIBUTION
            x_length = np.clip(x_length, 0.01, 0.56)
            # x_length =                                                    # To set a fixed value comment two lines before
            x_length = round(x_length, 4)                                   

            y_length = norm.rvs(loc=0.24, scale=0.11)                       # SELECT y-length RANGE AND DISTRIBUTION
            y_length = np.clip(y_length, 0.01, 0.56)
            # y_length =                                                    # To set a fixed value comment two lines before
            y_length = round(y_length, 4)                                   

            z_length = norm.rvs(loc=0.24, scale=0.11)                       # SELECT y-length RANGE AND DISTRIBUTION
            z_length = np.clip(z_length, 0.01, 0.56)
            # z_length =                                                    # To set a fixed value comment two lines before       
            z_length = round(z_length, 4)                                   

            volume = x_length * y_length * z_length                         # Calculate volume of the rectangular prism
            volume = round(volume, 4)                                       
            radius = ((volume * 3) / (4 * np.pi))**(1/3)                    # Calculate radius equivalent for the volume
            radius = round(radius, 4)                                       
            size_param = (2 * np.pi * radius) / wavelength                  # Calculate size parameter
            size_param = round(size_param, 4)                               

            sample = {
                "shape": shape,
                "x_length": x_length,
                "y_length": y_length,
                "z_length": z_length,
                "wavelength": wavelength,
                "volume": volume,
                "radius": radius,
                "size_param": size_param,
                "mat_file": mat_file
            }

        print(f"Generated sample: {sample}")
        samples.append(sample)

    with open("Generated_samples.json", "w") as f:                          # Generate .json file with all generated samples
        json.dump(samples, f, indent=4)
    
    return samples
