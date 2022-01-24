#!/bin/bash

# Load the conda environment
source set_env.sh

echo "--- (1) gen_figures.py ---"
python gen_figures.py

echo "--- (2) sim_power.py ---"
# Coverage experiments for power
python sim_power.py

echo "--- (3) run_hydro.py ---"
# Application on real-world-data
python run_hydro.py
