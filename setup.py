#!/usr/bin/env python3
"""
Setup script for Influencer Chatbot
Handles model downloading and environment setup
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def install_requirements():
    """Install required packages."""
    print("📦 Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def install_gpu_support():
    """Install GPU support for ctransformers."""
    print("🚀 Installing GPU support...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ctransformers[cuda]"])
        print("✅ GPU support installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install GPU support: {e}")
        print("💡 You can still use CPU-only mode")
        return False

def create_directories():
    """Create necessary directories."""
    directories = ["models", "exports", "logs", "conversation_logs"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Created directory: {directory}")



def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup Influencer Chatbot")

    parser.add_argument("--gpu", action="store_true", help="Install GPU support")
    parser.add_argument("--skip-requirements", action="store_true", help="Skip requirements installation")
    
    args = parser.parse_args()
    
    print("🚀 Setting up Influencer Chatbot...")
    print("="*50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not args.skip_requirements:
        if not install_requirements():
            print("❌ Setup failed during requirements installation")
            sys.exit(1)
    
    # Install GPU support if requested
    if args.gpu:
        install_gpu_support()
    
    # Create directories
    print("📁 Creating directories...")
    create_directories()
    

    
    print("\n" + "="*50)
    print("✅ Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Run 'python main.py' to start the application")
    print("2. Open http://localhost:7861 in your browser")
    print("3. Load a model in the Model Management tab")
    print("4. Start generating content!")
    


if __name__ == "__main__":
    main()
