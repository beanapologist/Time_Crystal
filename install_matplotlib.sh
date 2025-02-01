# First, make sure you're not in Python (if you see >>>, type exit())
# Then run these commands:

# 1. Uninstall existing matplotlib if any
pip uninstall matplotlib

# 2. Make sure pip is up to date
python -m pip install --upgrade pip

# 3. Install matplotlib
python -m pip install matplotlib

# 4. Verify installation
python -c "import matplotlib.pyplot as plt; print('Matplotlib installed successfully!')"

# 5. Install python3-tk (required for matplotlib on some systems)
# For Ubuntu/Debian:
sudo apt-get install python3-tk

# 6. Install additional required dependencies
pip install pillow
pip install cycler
pip install kiwisolver

import matplotlib
matplotlib.use('TkAgg')  # Set backend explicitly
import matplotlib.pyplot as plt
print("Matplotlib successfully installed!")