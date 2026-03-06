# Plateau GRBs

This folder contains the main codes used for the modelling and fitting of plateau GRBs within a structured-jet framework.

Files included:
- `fit_combined.py`: joint fitting script for structured-jet HLE and forward-shock components
- `hle.py`: structured-jet high-latitude emission model
- `fs.py`: forward-shock model
- `params_combined.yaml`: example parameter/configuration file
- `predict_lightcurve_from_fit.py`: utility module for optical/UVOT extrapolation and comparison

These codes were used in the thesis workflow for the modelling and fitting of plateau GRB light curves.

Note:
some workflows, especially those involving full UVOT data products, may require additional external input files from the original project environment.
