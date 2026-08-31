Add project README documentation
# High-Performance Algorithms for X-ray Tomography

## Research Internship at ESRF

**Intern:** Rana Alouani
**Institution:** European Synchrotron Radiation Facility (ESRF), Grenoble
**Internship period:** June–August 2026
**Supervisor:** Alessandro Mirone

## Project Overview

During this research internship, I worked on the development and evaluation of computational approaches for correcting diffusion and scattering effects in scintillator-based X-ray tomography.

The central challenge was to reduce the computational cost of diffusion correction while preserving the accuracy of the reconstructed correction.

The diffusion process contains contributions acting at different spatial scales. Long-range diffusion components are computationally expensive because they require corrections over large spatial regions.

The main objective of this project was therefore to investigate a multi-scale strategy:

* separate long-range and short-range diffusion contributions;
* identify the low-frequency and high-frequency components of the correction;
* reduce the computational cost of the long-range component using spatial binning;
* reconstruct the correction through interpolation;
* validate the approximation against progressively more realistic numerical tests.

## Scientific Approach

The workflow investigated during the internship can be summarized as:

Measurement / projection data
→ Diffusion model
→ Multi-scale kernel decomposition
→ Separation of long-range and short-range contributions
→ Spatial binning of the low-frequency component
→ Interpolation back to the original resolution
→ Comparison with reference calculations

The key idea is that slowly varying, low-frequency components can potentially be evaluated on a reduced spatial grid, decreasing the computational cost compared with calculations performed entirely at the original resolution.

## Methods and Numerical Tools

The project involved:

* Python scientific computing;
* NumPy;
* SciPy;
* FFT-based numerical methods;
* convolution and deconvolution;
* Gaussian filtering;
* spatial binning;
* interpolation;
* HDF5 data handling;
* numerical validation using synthetic tests;
* comparison between approximated and reference corrections.

Several tests were performed to investigate the effect of different spatial scales and Gaussian blur parameters.

The work also required careful attention to:

* spatial centering;
* barycentric shifts;
* interpolation accuracy;
* boundary effects;
* hidden spatial offsets;
* numerical errors introduced by binning.

## Technical Workflow

The numerical workflow included:

1. Loading tomography-related data and parameters from HDF5 files.
2. Reproducing convolution and deconvolution tests.
3. Investigating diffusion kernels acting at different spatial scales.
4. Separating slowly varying and rapidly varying components.
5. Applying Gaussian filtering to analyse low-frequency behaviour.
6. Reducing the resolution of the low-frequency component through binning.
7. Interpolating the binned result back to the original spatial grid.
8. Comparing the approximation with higher-resolution reference calculations.
9. Analysing numerical errors and spatial alignment issues.

## Key Skills Developed

### Scientific Computing

* Numerical modelling
* Signal and image processing
* FFT-based methods
* Convolution and deconvolution
* Gaussian filtering
* Multi-scale analysis
* Numerical validation

### Programming

* Python
* NumPy
* SciPy
* Matplotlib
* HDF5 / h5py
* Linux environment
* Git and GitHub

### Research Skills

* Understanding an existing scientific codebase
* Reproducing numerical experiments
* Designing validation tests
* Debugging scientific software
* Analysing computational performance
* Documenting research code
* Communicating technical results

## Main Outcome

The internship explored a computational strategy in which the long-range, low-frequency component of the diffusion correction can be treated separately from shorter-range components.

This approach provides a framework for reducing the computational cost of expensive long-range corrections by exploiting their slower spatial variation.

The work also highlighted the importance of accurate spatial centering and interpolation when reducing and reconstructing spatially resolved numerical data.

## Future Work

Possible future developments include:

* further optimisation of the low-frequency correction algorithm;
* implementation of the short-range/high-frequency component in C++;
* performance benchmarking;
* integration into a larger tomography reconstruction workflow;
* validation using additional experimental datasets.

## Note on Data Availability

The original datasets and parts of the development environment are associated with work performed at ESRF and are therefore not included in this public repository.

This repository is intended to document the scientific and computational methods developed during the internship without distributing restricted data or proprietary material.

## Contact

Rana Alouani
Physics student — Scientific Computing, Instrumentation and Imaging

GitHub: https://github.com/alouaniranou2004-droid 
LinkedIn: https://www.linkedin.com/in/rana-alouani-478787346/ 
