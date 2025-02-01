"""
Comprehensive setup script for Quantum Computing System
Handles all dependencies, imports, and initial configuration
"""

import subprocess
import sys
import os
from typing import Dict, List, Optional

class QuantumSystemSetup:
    def __init__(self):
        self.dependencies = {
            # Core Scientific Computing
            'numpy': 'import numpy as np',
            'scipy': 'import scipy',
            'pandas': 'import pandas as pd',
            
            # Machine Learning & Deep Learning
            'torch': 'import torch',
            'torchvision': 'import torchvision',
            
            # NLP & Text Processing
            'nltk': 'import nltk',
            'wikipedia-api': 'import wikipediaapi',
            'arxiv': 'import arxiv',
            'beautifulsoup4': 'from bs4 import BeautifulSoup',
            
            # Visualization
            'matplotlib': 'import matplotlib.pyplot as plt',
            'plotly': 'import plotly',
            'seaborn': 'import seaborn as sns',
            
            # Progress and Utils
            'tqdm': 'from tqdm import tqdm',
            'requests': 'import requests',
            'networkx': 'import networkx as nx'
        }
        
        self.additional_setup = {
            'nltk_data': ['punkt', 'stopwords', 'averaged_perceptron_tagger'],
            'torch_cuda': 'torch.cuda.is_available()',
        }
    
    def install_package(self, package: str) -> bool:
        """Install a single package"""
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ Installed {package}")
            return True
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {package}")
            return False
    
    def verify_import(self, import_statement: str) -> bool:
        """Verify package import"""
        try:
            exec(import_statement)
            return True
        except ImportError:
            return False
    
    def setup_nltk(self):
        """Setup NLTK data"""
        print("\nSetting up NLTK data...")
        import nltk
        for dataset in self.additional_setup['nltk_data']:
            try:
                nltk.download(dataset)
                print(f"✓ Downloaded NLTK {dataset}")
            except Exception as e:
                print(f"✗ Failed to download NLTK {dataset}: {str(e)}")
    
    def verify_cuda(self):
        """Verify CUDA availability for PyTorch"""
        print("\nChecking CUDA availability...")
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("! CUDA not available, using CPU")
    
    def setup_all(self):
        """Run complete setup"""
        print("Starting Quantum System Setup...")
        print("===============================")
        
        # Install and verify all packages
        for package, import_statement in self.dependencies.items():
            print(f"\nProcessing {package}...")
            
            if not self.verify_import(import_statement):
                self.install_package(package)
                
                if not self.verify_import(import_statement):
                    print(f"✗ Failed to setup {package}")
                    continue
            
            print(f"✓ {package} ready")
        
        # Additional setup steps
        self.setup_nltk()
        self.verify_cuda()
        
        # Verify all imports together
        print("\nVerifying all imports...")
        verification_code = "\n".join(self.dependencies.values())
        
        try:
            exec(verification_code)
            print("✓ All imports verified successfully!")
        except Exception as e:
            print(f"✗ Import verification failed: {str(e)}")
            return False
        
        return True

def main():
    setup = QuantumSystemSetup()
    success = setup.setup_all()
    
    if success:
        print("\n✓ Quantum System Setup completed successfully!")
        print("\nYou can now import and use:")
        for package in setup.dependencies.keys():
            print(f"- {package}")
    else:
        print("\n✗ Setup incomplete. Please check the errors above.")

if __name__ == "__main__":
    main() 