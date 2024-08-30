from scipy.stats import norm, uniform
import random
import json
import numpy as np

# USE THIS FUNCTION FOR VARYING SIZE PARAM AND SETTING CONSTANT OR VARYING WAVELENGTH

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
    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)

    shapes = ["SPHERE"] if only_spheres else ["SPHERE", "RCTGLPRSM"]
    samples = []
    
    prism_size_params = []

    for _ in range(num_samples):
        shape = random.choice(shapes)
        wavelength = uniform.rvs(loc=0.380, scale=0.370)                    # SET WAVELENGTH RANGE AND DISTRIBUTION
        wavelength = round(wavelength, 4)
        mat_file = "custom"

        if shape == "SPHERE":
            if prism_size_params:
                # Draw size parameters from the distribution of prisms
                size_param = random.choice(prism_size_params)
                radius = (size_param * wavelength) / (2 * np.pi)
            else:
                # Fallback in case there are no prisms yet
                radius = norm.rvs(loc=0.15, scale=0.05)
                radius = np.clip(radius, 0.02, 0.35)
                radius = round(radius, 4)
                size_param = (2 * np.pi * radius) / wavelength
            volume = (4/3) * np.pi * radius**3
            volume = round(volume, 4)
            sample = {
                "shape": shape,
                "radius": round(radius, 4),
                "wavelength": wavelength,
                "size_param": round(size_param, 4),
                "volume": volume,
                "mat_file": mat_file
            }

        elif shape == "RCTGLPRSM":
            x_length = norm.rvs(loc=0.35, scale=0.12)
            x_length = np.clip(x_length, 0.05, 0.70)
            x_length = round(x_length, 4)
            
            y_length = norm.rvs(loc=0.35, scale=0.12)
            y_length = np.clip(y_length, 0.05, 0.70)
            y_length = round(y_length, 4)

            z_length = norm.rvs(loc=0.35, scale=0.12)
            z_length = np.clip(z_length, 0.05, 0.70)
            z_length = round(z_length, 4)

            volume = x_length * y_length * z_length
            volume = round(volume, 4)
            radius = ((volume * 3) / (4 * np.pi))**(1/3)  # Equivalent radius
            radius = round(radius, 4)
            size_param = (2 * np.pi * radius) / wavelength
            size_param = round(size_param, 4)
            
            # Store prism size parameters to adjust sphere size parameters
            prism_size_params.append(size_param)

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

    with open("Generated_samples.json", "w") as f:
        json.dump(samples, f, indent=4)
    
    return samples

def load_samples(file_path):
    """Load samples from a file."""
    with open(file_path, "r") as f:
        samples = json.load(f)
    return samples

'''
#USE THIS FUNCTION FOR CONSTANT RADIUS AND VARYING WAVELENGTH

def sample_parameters(num_samples, random_seed=None, only_spheres=False):
    """
    Generate sampled parameters for simulations with a fixed equivalent radius and varying wavelengths.
    
    Parameters:
        num_samples (int): Number of samples to generate.
        random_seed (Optional[int]): Seed for random number generators.
        only_spheres (bool): If True, only generate spherical samples.

    Returns:
        List[Dict[str, Any]]: List of parameter dictionaries.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)

    shapes = ["SPHERE"] if only_spheres else ["SPHERE", "RCTGLPRSM"]
    samples = []
    
    fixed_radius = 0.15  # Set the fixed equivalent radius for all particles
    
    for _ in range(num_samples):
        shape = random.choice(shapes)
        wavelength = uniform.rvs(loc=0.380, scale=0.370)  # Varying wavelength
        wavelength = round(wavelength, 4)
        mat_file = "astrosil"

        if shape == "SPHERE":
            size_param = (2 * np.pi * fixed_radius) / wavelength
            volume = (4/3) * np.pi * fixed_radius**3
            sample = {
                "shape": shape,
                "radius": fixed_radius,
                "wavelength": wavelength,
                "size_param": round(size_param, 4),
                "volume": round(volume, 4),
                "mat_file": mat_file
            }

        elif shape == "RCTGLPRSM":
            # Calculate the volume corresponding to the fixed radius
            volume = (4/3) * np.pi * fixed_radius**3
            
            # Distribute this volume across the x, y, z dimensions
            # For simplicity, assume that all dimensions are normally distributed around a mean value
            mean_length = (volume)**(1/3)
            x_length = norm.rvs(loc=mean_length, scale=mean_length*0.3)
            y_length = norm.rvs(loc=mean_length, scale=mean_length*0.3)
            z_length = volume / (x_length * y_length)

            # Clip the dimensions to reasonable values
            x_length = np.clip(x_length, 0.05, 0.7)
            y_length = np.clip(y_length, 0.05, 0.7)
            z_length = np.clip(z_length, 0.05, 0.7)

            # Recalculate the volume to match the adjusted dimensions
            volume = x_length * y_length * z_length
            
            # The equivalent radius is fixed at 0.15, but for consistency:
            radius = ((volume * 3) / (4 * np.pi))**(1/3)  # Equivalent radius
            size_param = (2 * np.pi * fixed_radius) / wavelength
            
            sample = {
                "shape": shape,
                "x_length": round(x_length, 4),
                "y_length": round(y_length, 4),
                "z_length": round(z_length, 4),
                "wavelength": wavelength,
                "volume": round(volume, 4),
                "radius": fixed_radius,
                "size_param": round(size_param, 4),
                "mat_file": mat_file
            }

        print(f"Generated sample: {sample}")
        samples.append(sample)

    with open("Generated_samples.json", "w") as f:
        json.dump(samples, f, indent=4)
    
    return samples
'''
'''
# USE THIS FUNCTION FOR SIMULATIONS WITH CONSTANT SIZE PARAMETER

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
    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)

    shapes = ["SPHERE"] if only_spheres else ["SPHERE", "RCTGLPRSM"]
    samples = []

    for _ in range(num_samples):
        shape = random.choice(shapes)
        # Randomly sample either radius or wavelength
        if random.choice([True, False]):
            # Fix the radius, calculate the wavelength
            radius = norm.rvs(loc=0.15, scale=0.05)
            radius = np.clip(radius, 0.02, 0.35)
            radius = round(radius, 4)
            wavelength = (2 * np.pi * radius) / 1  # Size parameter is set to 1
        else:
            # Fix the wavelength, calculate the radius
            wavelength = uniform.rvs(loc=0.380, scale=0.370)  # Set wavelength range
            wavelength = round(wavelength, 4)
            radius = wavelength / (2 * np.pi)  # Size parameter is set to 1

        mat_file = "astrosil"
        radius = round(radius, 4)
        wavelength = round(wavelength, 4)
        size_param = 1  # Fixed size parameter
        volume = (4/3) * np.pi * radius**3
        volume = round(volume, 4)

        sample = {
            "shape": shape,
            "radius": radius,
            "wavelength": wavelength,
            "size_param": size_param,
            "volume": volume,
            "mat_file": mat_file
        }

        if shape == "RCTGLPRSM":
            # For rectangular prisms, generate lengths that match the equivalent radius
            x_length = norm.rvs(loc=volume ** (1/3), scale=0.12)
            x_length = np.clip(x_length, 0.05, 0.70)
            x_length = round(x_length, 4)
            
            y_length = norm.rvs(loc=volume ** (1/3), scale=0.12)
            y_length = np.clip(y_length, 0.05, 0.70)
            y_length = round(y_length, 4)

            z_length = norm.rvs(loc=volume ** (1/3), scale=0.12)
            z_length = np.clip(z_length, 0.05, 0.70)
            z_length = round(z_length, 4)

            actual_volume = x_length * y_length * z_length
            actual_volume = round(actual_volume, 4)

            sample.update({
                "x_length": x_length,
                "y_length": y_length,
                "z_length": z_length,
                "volume": actual_volume
            })

        print(f"Generated sample: {sample}")
        samples.append(sample)

    with open("Generated_samples.json", "w") as f:
        json.dump(samples, f, indent=4)
    
    return samples
'''

'''
# INITIAL SAMPLE CREATION 

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
        # wavelength = uniform.rvs(loc=0.380, scale=0.370)                  # SET WAVELENGTH RANGE AND DISTRIBUTION
        wavelength = 0.5600                                                 # To set a fixed value comment line before and set here
        wavelength = round(wavelength, 4)                                   
        mat_file = "astrosil"                                               # SELECT MATERIAL FILE ("astrosil" or "custom")

        if shape == "SPHERE":                                               # Generate spherical particle sample
            radius = norm.rvs(loc=0.15, scale=0.05)                         # SELECT PARTICLE SIZE RANGE AND DISTRIBUTION
            radius = np.clip(radius, 0.02, 0.35)
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
            x_length = norm.rvs(loc=0.35, scale=0.12)
            x_length = np.clip(x_length, 0.05, 0.70)
            x_length = round(x_length, 4)
            
            y_length = norm.rvs(loc=0.35, scale=0.12)
            y_length = np.clip(y_length, 0.05, 0.70)
            y_length = round(y_length, 4)

            z_length = norm.rvs(loc=0.35, scale=0.12)
            z_length = np.clip(z_length, 0.05, 0.70)
            z_length = round(z_length, 4)

            volume = x_length * y_length * z_length
            volume = round(volume, 4)
            radius = ((volume * 3) / (4 * np.pi))**(1/3)  # Equivalent radius
            radius = round(radius, 4)
            size_param = (2 * np.pi * radius) / wavelength
            size_param = round(size_param, 4)

            Please use this ones after!!!
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
            
            Up to here!!!
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
'''


