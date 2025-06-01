
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
    directories = ["models", "exports", "logs"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Created directory: {directory}")

def download_model(model_name="Luna-AI-Llama2-Uncensored"):
    """Download a specific model."""
    print(f"⬇️ Downloading {model_name} model...")
    
    model_configs = {
        "Luna-AI-Llama2-Uncensored": {
            "repo": "TheBloke/Luna-AI-Llama2-Uncensored-GGUF",
            "file": "luna-ai-llama2-uncensored.Q4_K_M.gguf"
        },
        "WizardLM-13B-Uncensored": {
            "repo": "TheBloke/WizardLM-13B-Uncensored-GGUF",
            "file": "WizardLM-13B-Uncensored.Q4_K_M.gguf"
        }
    }
    
    if model_name not in model_configs:
        print(f"❌ Unknown model: {model_name}")
        return False
    
    config = model_configs[model_name]
    
    try:
        # Import huggingface_hub here to avoid import errors during setup
        from huggingface_hub import hf_hub_download
        
        print(f"📥 Downloading from {config['repo']}...")
        model_path = hf_hub_download(
            repo_id=config["repo"],
            filename=config["file"],
            local_dir="./models",
            local_dir_use_symlinks=False
        )
        print(f"✅ Model downloaded to: {model_path}")
        return True
        
    except ImportError:
        print("❌ huggingface_hub not installed. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub>=0.17.1"])
        return download_model(model_name)  # Retry after installation
        
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        return False

def check_system_requirements():
    """Check system requirements."""
    print("🔍 Checking system requirements...")
    
    # Check available RAM
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
        print(f"💾 Available RAM: {ram_gb:.1f} GB")
        
        if ram_gb < 8:
            print("⚠️ Warning: Less than 8GB RAM detected. Consider using Luna-AI model only.")
        elif ram_gb >= 16:
            print("✅ Sufficient RAM for all models")
        else:
            print("✅ Sufficient RAM for Luna-AI and WizardLM models")
            
    except ImportError:
        print("💡 Install psutil for system monitoring: pip install psutil")
    
    # Check disk space
    try:
        disk_usage = os.statvfs('.')
        free_gb = (disk_usage.f_frsize * disk_usage.f_bavail) / (1024**3)
        print(f"💽 Available disk space: {free_gb:.1f} GB")
        
        if free_gb < 10:
            print("⚠️ Warning: Less than 10GB free space. Models require 5-20GB each.")
        else:
            print("✅ Sufficient disk space")
            
    except AttributeError:
        print("💡 Disk space check not available on this system")
    
    # Check for GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"🎮 GPU detected: {gpu_name} ({gpu_memory:.1f} GB VRAM)")
            print(f"🎮 GPU count: {gpu_count}")
        else:
            print("💻 No GPU detected - will use CPU mode")
    except ImportError:
        print("💡 Install PyTorch to check GPU availability")

def run_tests():
    """Run basic functionality tests."""
    print("🧪 Running basic tests...")
    
    try:
        # Test ctransformers import
        import ctransformers
        print("✅ ctransformers import successful")
        
        # Test gradio import
        import gradio
        print("✅ gradio import successful")
        
        # Test other imports
        import json
        import datetime
        print("✅ Standard library imports successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        return False

def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup Influencer Chatbot")
    parser.add_argument("--gpu", action="store_true", help="Install GPU support")
    parser.add_argument("--download-model", type=str, help="Download specific model")
    parser.add_argument("--skip-download", action="store_true", help="Skip model download")
    parser.add_argument("--test-only", action="store_true", help="Run tests only")
    
    args = parser.parse_args()
    
    print("🚀 Influencer Chatbot Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    if args.test_only:
        success = run_tests()
        sys.exit(0 if success else 1)
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Install GPU support if requested
    if args.gpu:
        install_gpu_support()
    
    # Check system requirements
    check_system_requirements()
    
    # Download model
    if not args.skip_download:
        model_to_download = args.download_model or "Luna-AI-Llama2-Uncensored"
        download_model(model_to_download)
    
    # Run tests
    if run_tests():
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Run: python main.py")
        print("2. Open: http://localhost:7860")
        print("3. Load a model in the 'Model Settings' tab")
        print("4. Start generating content!")
        
        if args.gpu:
            print("\n💡 GPU support installed - set GPU layers in model settings")
        
    else:
        print("\n❌ Setup completed with errors")
        print("Check the error messages above and try again")
        sys.exit(1)

if __name__ == "__main__":
    main()
