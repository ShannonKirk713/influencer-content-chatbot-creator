
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

def download_model(model_name="Luna-AI-Llama2-Uncensored"):
    """Download a specific model."""
    print(f"📥 Downloading model: {model_name}")
    
    # Model configurations
    models = {
        "Luna-AI-Llama2-Uncensored": {
            "repo_id": "TheBloke/Luna-AI-Llama2-Uncensored-GGUF",
            "filename": "luna-ai-llama2-uncensored.Q4_K_M.gguf"
        },
        "WizardLM-13B-Uncensored": {
            "repo_id": "TheBloke/WizardLM-13B-Uncensored-GGUF",
            "filename": "wizardlm-13b-uncensored.Q4_K_M.gguf"
        },
        "Wizard-Vicuna-30B-Uncensored": {
            "repo_id": "TheBloke/Wizard-Vicuna-30B-Uncensored-GGUF",
            "filename": "wizard-vicuna-30b-uncensored.Q4_K_M.gguf"
        }
    }
    
    if model_name not in models:
        print(f"❌ Unknown model: {model_name}")
        print(f"Available models: {', '.join(models.keys())}")
        return False
    
    try:
        # Import here to avoid dependency issues during setup
        from llama_cpp import Llama
        
        model_config = models[model_name]
        print(f"📥 Downloading from {model_config['repo_id']}...")
        
        # This will download and cache the model
        llm = Llama.from_pretrained(
            repo_id=model_config["repo_id"],
            filename=model_config["filename"],
            n_gpu_layers=0,  # Don't load into GPU during download
            verbose=False
        )
        
        print(f"✅ Model {model_name} downloaded successfully!")
        return True
        
    except ImportError:
        print("❌ llama-cpp-python not installed. Installing now...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "llama-cpp-python"])
            print("✅ llama-cpp-python installed. Please run setup again.")
            return False
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install llama-cpp-python: {e}")
            return False
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return False

def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup Influencer Chatbot")
    parser.add_argument("--download-model", help="Download a specific model")
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
    
    # Download model if specified
    if args.download_model:
        if not download_model(args.download_model):
            print("❌ Model download failed")
            sys.exit(1)
    
    print("\n" + "="*50)
    print("✅ Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Run 'python main.py' to start the application")
    print("2. Open http://localhost:7861 in your browser")
    print("3. Load a model in the Model Management tab")
    print("4. Start generating content!")
    
    if not args.download_model:
        print("\n💡 Tip: Run 'python setup.py --download-model Luna-AI-Llama2-Uncensored' to download the recommended model")

if __name__ == "__main__":
    main()
