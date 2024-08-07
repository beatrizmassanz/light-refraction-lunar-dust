
def process_mie_result(mie_results, results, samples):
    """
    Process the Mie results and match them with the DDSCAT results.

    Parameters:
        mie_results (DataFrame): DataFrame containing the Mie calculation results.
        results (list): List to store the final processed results.
        samples (list): List of sample parameters used in the simulations.
    """
    for sample in samples:
        if sample["shape"] != "SPHERE":
            continue
        
        mie_sample_results = mie_results[(mie_results['radius'] == sample['radius']) & 
                                         (mie_results['wavelength'] == sample['wavelength'])]
        for _, mie_result in mie_sample_results.iterrows():
            result = {
                "theta": mie_result['angle'],
                "phi": 0,  # Assuming phi = 0 for simplicity
                "Pol.": 0,  # Assuming unpolarized light for simplicity
                "S_11": mie_result['S_11'],
                "S_12": None,  # Not calculated in this example
                "S_21": None,  # Not calculated in this example
                "S_22": None,  # Not calculated in this example
                "S_31": None,  # Not calculated in this example
                "S_41": None,  # Not calculated in this example
                "Qsca": mie_result['Qsca'],
                "Qbk": mie_result['Qback'],
                "Qpol": None,  # Not calculated in this example
                "shape": sample['shape'],
                "size_param": sample['size_param'],
                "radius": sample['radius'],
                "wavelength": sample['wavelength']
            }
            results.append(result)

    return results