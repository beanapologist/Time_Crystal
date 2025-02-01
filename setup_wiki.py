"""
Wikipedia dataset setup and verification for Quantum Computing System
"""
import subprocess
import sys
import importlib

def setup_wiki_dependencies():
    print("Setting up Wikipedia dataset dependencies...")
    print("==========================================")
    
    # Required packages for wiki dataset
    wiki_packages = {
        'wikipedia-api': 'wikipediaapi',
        'nltk': 'nltk',
        'numpy': 'np',
        'torch': 'torch',
        'pandas': 'pd'
    }
    
    for package, alias in wiki_packages.items():
        print(f"\nInstalling {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} installed successfully")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {package}")
            continue
        
        # Verify import
        try:
            module = importlib.import_module(alias)
            print(f"✓ {package} import verified")
        except ImportError as e:
            print(f"✗ Failed to import {package}: {str(e)}")
    
    # Download required NLTK data
    print("\nDownloading NLTK data...")
    try:
        import nltk
        nltk.download('punkt')
        nltk.download('stopwords')
        print("✓ NLTK data downloaded successfully")
    except Exception as e:
        print(f"✗ Failed to download NLTK data: {str(e)}")
    
    # Test Wikipedia API
    print("\nTesting Wikipedia API...")
    try:
        import wikipediaapi # type: ignore
        wiki = wikipediaapi.Wikipedia('en')
        page = wiki.page('Quantum computing')
        if page.exists():
            print("✓ Wikipedia API working correctly")
        else:
            print("✗ Wikipedia API test failed")
    except Exception as e:
        print(f"✗ Wikipedia API test failed: {str(e)}")

if __name__ == "__main__":
    setup_wiki_dependencies() 
# Run the script
if __name__ == "__main__":
    setup_wiki_dependencies()