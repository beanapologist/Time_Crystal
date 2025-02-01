#!/bin/bash

echo "Installing Quantum System Dependencies..."

# Update pip
python -m pip install --upgrade pip

# Core dependencies
pip install numpy
pip install torch
pip install scikit-learn
pip install pandas
pip install tqdm

# Visualization dependencies
pip install matplotlib
pip install seaborn

# Testing dependencies
pip install pytest pytest-cov pytest-mock

# Quantum specific dependencies
pip install qiskit
pip install cirq
pip install pennylane

# Machine Learning dependencies
pip install transformers
pip install tensorboard

# Development dependencies
pip install black
pip install flake8
pip install mypy

echo "Dependencies installation completed!"

# Verify installations
python -c "import numpy; import torch; import sklearn; import matplotlib; import seaborn; print('Core imports successful!')"
python -c "import transformers; import tensorboard; print('ML imports successful!')"
python -c "import qiskit; import cirq; import pennylane; print('Quantum imports successful!')"
python -c "import pytest; print('Testing imports successful!')"

echo "Installation verification completed!"

chmod +x install_dependencies.sh
./install_dependencies.sh