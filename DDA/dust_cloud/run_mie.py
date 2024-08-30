import miepython as mie
import pandas as pd
import numpy as np


def mie_calculation(samples):
    """
    Perform Mie calculations for the given samples.
    
    Parameters:
        samples (list): List of sample parameters for 
            Mie calculations.
        
    Returns:
        DataFrame: A DataFrame with Mie calculation results.
    """
    mie_results = []
    angles_degrees = np.arange(0, 360, 5)                              # Generate angles from 0 to 355 degrees in 5 degree steps
    angles_cosine = np.cos(np.radians(angles_degrees))

    for sample in samples:
        if sample["shape"] != "SPHERE":
            continue
        
        m_particle = 1.56 + 0.003j                                     # Assuming a constant complex refractive index for simplicity
        wavelength = sample["wavelength"]
        radius = sample["radius"]
        
        x = 2 * np.pi * radius / wavelength
        qext, qsca, qback, g = mie.mie(m_particle, x)

        s1, s2 = mie.mie_S1_S2(m_particle, 
                               x, 
                               angles_cosine, 
                               norm='qsca')
        s_11 = 0.5 * (np.abs(s1)**2 + np.abs(s2)**2)
        norm_factor_qsca =  x * np.sqrt(np.pi)
        print(f"Size parameter (x): {x}, Qsca: {qsca}, \
              Norm factor Mie: {norm_factor_qsca}")

        for angle, s11_values in zip(angles_degrees, s_11):
            mie_result = {
                "radius": radius,
                "wavelength": wavelength,
                "Qext": qext,
                "Qsca": qsca,
                "Qback": qback,
                "G": g,
                "angle": angle,
                "S_11": s11_values
                }
            mie_results.append(mie_result)
    
    mie_df = pd.DataFrame(mie_results)
    mie_df.rename(columns={'angle': 'theta'}, inplace=True)
    mie_df['size_param'] = 2 * np.pi * mie_df['radius'] / mie_df['wavelength']
    return mie_df


def process_mie_result(mie_results, results, samples):
    """
    Process the Mie results and match them with the DDSCAT results.

    Parameters:
        mie_results (DataFrame): DataFrame containing the Mie 
                                calculation results.
        results (list): List to store the final processed results.
        samples (list): List of sample parameters used in the 
                        simulations.
    """
    for sample in samples:
        if sample["shape"] != "SPHERE":
            continue
        
        mie_sample_results = mie_results[
            (mie_results['radius'] == sample['radius']) & 
            (mie_results['wavelength'] == sample['wavelength'])
            ]
        for _, mie_result in mie_sample_results.iterrows():
            result = {
                "theta": mie_result['theta'],
                "phi": 0,                                               # Assuming phi = 0 for comparison with DDSCAT
                "Pol.": 0,                                              # Assuming unpolarized light
                "S_11": mie_result['S_11'],
                "S_12": None,                                           # Not calculated in this example
                "S_21": None,                                           # Not calculated in this example
                "S_22": None,                                           # Not calculated in this example
                "S_31": None,                                           # Not calculated in this example
                "S_41": None,                                           # Not calculated in this example
                "Qsca": mie_result['Qsca'],
                "Qbk": mie_result['Qback'],
                "Qpol": None,                                           # Not calculated in this example
                "shape": sample['shape'],
                "size_param": sample['size_param'],
                "radius": sample['radius'],
                "wavelength": sample['wavelength']
            }
            results.append(result)

    return results




