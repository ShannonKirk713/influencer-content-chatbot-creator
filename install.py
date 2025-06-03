
"""
Installation script for Fanvue Content Chatbot Extension
This script handles the installation of required dependencies for the extension.
"""

import subprocess
import sys
import os
import importlib.util

def is_package_installed(package_name):
    """Check if a package is already installed."""
    spec = importlib.util.find_spec(package_name)
    return spec is not None

def install_package(package):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package}: {e}")
        return False

def main():
    """Main installation function."""
    print("🚀 Installing Fanvue Content Chatbot Extension dependencies...")
    
    # Required packages
    required_packages = [
        "llama-cpp-python",
        "opencv-python",
        "transformers",
        "python-dateutil",
        "tqdm"
    ]
    
    # Check and install packages
    failed_packages = []
    
    for package in required_packages:
        package_name = package.split(">=")[0].split("==")[0]
        
        if is_package_installed(package_name):
            print(f"✅ {package_name} is already installed")
        else:
            print(f"📦 Installing {package}...")
            if install_package(package):
                print(f"✅ Successfully installed {package}")
            else:
                failed_packages.append(package)
                print(f"❌ Failed to install {package}")
    
    # Report results
    if failed_packages:
        print(f"\n❌ Installation completed with errors. Failed packages: {', '.join(failed_packages)}")
        print("Please install these packages manually:")
        for package in failed_packages:
            print(f"  pip install {package}")
        return False
    else:
        print("\n✅ All dependencies installed successfully!")
        print("🎉 Fanvue Content Chatbot Extension is ready to use!")
        return True

if __name__ == "__main__":
    main()
