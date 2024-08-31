# light-scattering-dust-simulation

## Overview

This repository contains a Python-based program designed to model light scattering by dust particles, with an emphasis on lunar dust. The primary objective of this program is to calculate scattering characteristics such as scattering efficiency, polarization, and direction-dependent scattering phase functions based on input parameters like particle size, shape, and wavelength. The program utilizes DDSCAT for simulating non-spherical particles and Mie theory for spherical particles to validate the results.

## Project Structure

The project is organized into several modules, each responsible for a specific part of the simulation workflow:

- **`gen_input.py`**: Generates particle samples based on user-defined or randomly generated parameters such as size, shape, and wavelength.

- **`run_ddscat.py`**: Manages the preparation, execution, and processing of DDSCAT simulations. This module includes submodules for preparing input files, running DDSCAT simulations, and processing the outputs.

- **`run_mie.py`**: Performs Mie calculations for spherical particles, providing a validation mechanism for the results obtained from DDSCAT.

- **`proc_output.py`**: Extracts and processes results from simulations, compiling it into a structured format for further analysis.

- **`visualization.py`**: Contains functions for visualizing the results, including correlation plots, distribution plots, and comparison plots between DDSCAT and Mie theory results.

## Installation

You have two options for using this project:

### Option 1: Download the Entire Repository (Recommended)

This option allows you to download the entire repository, which includes DDSCAT pre-configured and ready to use.

1. Download the repository from [this link](https://github.com/beatrizmassanz/light-refraction-lunar-dust.git).
2. Extract the contents of the repository.
3. The DDSCAT executable is already included in the `DDA` folder. You can run the Python scripts directly without any additional setup.

### Option 2: Manual Setup

If you prefer to set up DDSCAT yourself, follow these steps:

#### 1. Download and Install DDSCAT

To run DDSCAT simulations, you must first download and install DDSCAT from the [official website](https://ddscat.github.io/).

1. Download the latest version of DDSCAT.
2. Extract the files to a directory on your system.
3. Compile the source code according to the instructions provided in the DDSCAT documentation.
4. Ensure the `ddscat` executable is in the `src` directory where you will run the Python scripts.

#### 2. Download and Set Up the Dust Cloud Module

If you are setting up DDSCAT manually, you only need to download the `dust_cloud` folder from this repository:

1. Download the `dust_cloud` folder from [this link](https://github.com/beatrizmassanz/light-refraction-lunar-dust.git/tree/main/dust_cloud).
2. Copy the `dust_cloud` folder into the `DDA` directory of your DDSCAT installation.

#### 3. Install Python and Required Libraries

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