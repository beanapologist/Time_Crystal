"""
Setup script for scikit-learn and related packages
"""

import subprocess
import sys
import pkg_resources
import logging
from typing import List, Dict
from pathlib import Path

class SklearnSetup:
    def __init__(self):
        self.required_packages = {
            'scikit-learn': 'sklearn',
            'numpy': 'numpy',
            'scipy': 'scipy',
            'pandas': 'pandas',
            'matplotlib': 'matplotlib',
            'joblib': 'joblib'
        }
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def check_installation(self, package: str) -> bool:
        """Check if package is installed"""
        try:
            pkg_resources.get_distribution(package)
            return True
        except pkg_resources.DistributionNotFound:
            return False
    
    def install_package(self, package: str) -> bool:
        """Install a package using pip"""
        try:
            subprocess.check_call([
                sys.executable, 
                "-m", 
                "pip", 
                "install", 
                "--upgrade",
                package
            ])
            logging.info(f"✓ Successfully installed {package}")
            return True
        except subprocess.CalledProcessError:
            logging.error(f"✗ Failed to install {package}")
            return False
    
    def verify_import(self, module_name: str) -> bool:
        """Verify package can be imported"""
        try:
            __import__(module_name)
            logging.info(f"✓ Successfully imported {module_name}")
            return True
        except ImportError:
            logging.error(f"✗ Failed to import {module_name}")
            return False
    
    def setup_all(self) -> bool:
        """Setup all required packages"""
        logging.info("Starting scikit-learn setup...")
        success = True
        
        for package, module in self.required_packages.items():
            if not self.check_installation(package):
                logging.info(f"Installing {package}...")
                if not self.install_package(package):
                    success = False
                    continue
            
            if not self.verify_import(module):
                success = False
                continue
        
        if success:
            logging.info("✓ Scikit-learn setup completed successfully")
        else:
            logging.error("✗ Some packages failed to install or import")
        
        return success

def main():
    setup = SklearnSetup()
    success = setup.setup_all()
    
    if success:
        # Verify scikit-learn functionality
        try:
            import sklearn
            from sklearn import datasets, metrics
            from sklearn.model_selection import train_test_split
            
            # Load a sample dataset to verify functionality
            iris = datasets.load_iris()
            X_train, X_test, y_train, y_test = train_test_split(
                iris.data, iris.target, test_size=0.2
            )
            
            logging.info("✓ Scikit-learn functionality verified")
            
        except Exception as e:
            logging.error(f"✗ Scikit-learn verification failed: {str(e)}")
            success = False
    
    return success

if __name__ == "__main__":
    success = main()
    if success:
        print("\nSetup completed successfully! You can now use scikit-learn.")
    else:
        print("\nSetup encountered some issues. Please check the logs above.") 