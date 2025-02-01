"""
Project setup script
"""
import os
import subprocess
import sys

def setup_project():
    # Create fresh virtual environment
    subprocess.run([sys.executable, "-m", "venv", ".venv"])

    # Install dependencies
    if os.name == 'nt':  # Windows
        subprocess.run([".venv\\Scripts\\pip", "install", "-r", "requirements.txt"])
        subprocess.run([".venv\\Scripts\\pip", "install", "-e", "."])
    else:  # Unix/Linux/Mac
        subprocess.run([".venv/bin/pip", "install", "-r", "requirements.txt"])
        subprocess.run([".venv/bin/pip", "install", "-e", "."])

if __name__ == "__main__":
    setup_project()