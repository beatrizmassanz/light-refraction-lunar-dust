# light-refraction-lunar-dust

## Overview

This repository contains a Python-based program designed to model light scattering 
by lunar dust particles. The primary objective of this program is to calculate 
scattering characteristics such as scattering efficiency, polarization, and 
direction-dependent scattering phase functions based on input parameters like 
particle size, shape, and wavelength. The program utilizes DDSCAT for simulating 
non-spherical particles and Mie theory for spherical particles to validate the 
results.

## Project Structure

The project is organized into several modules, each responsible for a specific part 
of the simulation workflow:

- `generate_inputs.py`: Generates particle samples based on user-defined or randomly 
generated parameters such as size, shape, and wavelength.

- `run_ddscat.py`: Manages the preparation, execution, and processing of DDSCAT 
simulations. This module includes submodules for preparing input files, running 
the DDSCAT simulations, and processing the outputs.

- `proc_output_ddscat.py`: Extracts and processes results from DDSCAT simulations, 
normalizing the data and compiling it into a structured format for further analysis.

- `run_mie.py`: Performs Mie calculations for spherical particles, providing a 
validation mechanism for the results obtained from DDSCAT.

- `visualization.py`: Contains functions for visualizing the results, including 
correlation plots, line plots, polar plots, and comparison plots between DDSCAT and 
Mie theory results.

## Installation

### 1. Download and Install DDSCAT

To run DDSCAT simulations, you must first download and install DDSCAT from the 
[official website](https://www.astro.princeton.edu/~draine/DDSCAT.html).

Follow these steps to install DDSCAT:

1. Download the latest version of DDSCAT.
2. Extract the files to a directory on your system.
3. Compile the source code according to the instructions provided in the DDSCAT 
documentation.
4. Ensure the `ddscat` executable is in the `src` directory relative to where you 
will run the Python scripts.

### 2. Install Python and Required Libraries

This program requires Python 3.x and the following Python libraries:

- `numpy`
- `scipy`
- `pandas`
- `matplotlib`
- `seaborn`
- `miepython`

You can install these libraries using `pip`:

```bash
pip install numpy scipy pandas matplotlib seaborn miepython
