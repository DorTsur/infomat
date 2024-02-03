# Information Matrix (InfoMat) Visualization Tool

## Overview
This repository contains the Python code for visualizing information transfer in sequential systems using a novel matrix representation called the InfoMat. The InfoMat visualization tool is designed to help researchers and data scientists better understand the dynamics of information transfer in various datasets, particularly in the context of probabilistic systems and time-series data.

## Contents
- `main.py`: The main script that demonstrates the use of the InfoMat visualization tool across different data scenarios.
- `utils.py`: Contains utility functions for data generation and estimation of conditional mutual information (CMI), supporting both plugin and Gaussian estimators.

## Simulations
- i.i.d. jointly distributed data (both Gaussian and discrete)
- ARMA Gaussian proess
- Ising channel data under both an oblivious and optimal coding scheme.

### Prerequisites
- Python 3.6+
- NumPy
- Matplotlib

## Visualization of continuous data via a Gaussian approximation:
![InfoMat Example](https://github.com/DorTsur/infomat/blob/main/figs_infomat/gaussian_infomat.png)

## Visualization of Ising coding scheme data via a plug-in estimator:
![InfoMat Example](https://github.com/DorTsur/infomat/blob/main/figs_infomat/coding_scheme_infomat.png)

## Citation:
TBD.
