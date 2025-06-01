# Fanvue Adult Content Creation Chatbot

A sophisticated Gradio-based application for generating adult content prompts using uncensored AI models. This chatbot is specifically designed for content creators on platforms like Fanvue, OnlyFans, and other adult content platforms.

## 🌟 Features

- **Uncensored AI Models**: Uses specialized uncensored language models for adult content generation
- **Multiple Model Support**: Luna-AI-Llama2, WizardLM-13B, and Wizard-Vicuna-30B models
- **Specialized Prompts**: Pre-built prompt templates for various adult content scenarios
- **Interactive Web Interface**: Clean, user-friendly Gradio interface
- **Content Export**: Save and export generated content
- **Conversation History**: Track and review previous interactions
- **Model Management**: Easy model loading and configuration

## 🚀 Quick Start (Windows)

### Prerequisites
- Windows 10/11
- Python 3.8 or higher ([Download from python.org](https://python.org))
- At least 8GB RAM (16GB recommended)
- 10GB+ free disk space

### Installation

1. **Download the project**:
   ```bash
   git clone https://github.com/Valorking6/fanvue-content-chatbot.git
   cd fanvue-content-chatbot
   ```

2. **Run the installer**:
   ```bash
   install.bat
   ```
   This will:
   - Create a Python virtual environment
   - Install all required dependencies
   - Download the default AI model (Luna-AI-Llama2-Uncensored)
   - Set up necessary directories

3. **Start the application**:
   ```bash
   start.bat
   ```

4. **Access the interface**:
   Open your browser and go to: `http://localhost:7860`

### Manual Installation (Alternative)

If the batch files don't work, you can install manually:

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate.bat

# Install dependencies
pip install -r requirements.txt

# Download model
python setup.py --download-model Luna-AI-Llama2-Uncensored

# Start application
python main.py
```

## 🐧 Linux/Mac Installation

### Prerequisites
- Python 3.8+
- pip
- At least 8GB RAM (16GB recommended)

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Valorking6/fanvue-content-chatbot.git
   cd fanvue-content-chatbot
   ```

2. **Create virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run setup script**:
   ```bash
   python setup.py
   ```

5. **Start the application**:
   ```bash
   python main.py
   ```

## 🔧 Troubleshooting

### Common Issues

**"'enable_queue' parameter not found" Error**:
This indicates an outdated Gradio version. The current version is compatible with Gradio 4.0+. To fix:

1. **Clear caches and reinstall**:
   ```bash
   pip cache purge
   pip uninstall gradio
   pip install --no-cache-dir "gradio>=4.0.0"
   ```

2. **For persistent issues, see our comprehensive cache clearing guide**:
   - [Cache Clearing Guide](docs/CACHE_CLEARING_GUIDE.md)

**Model won't load**:
- Check available RAM (8GB+ required)
- Try a smaller model (Luna-AI instead of Wizard-Vicuna)
- Restart the application

**Slow generation**:
- Enable GPU acceleration if available
- Reduce max tokens
- Use a smaller model

**Installation fails**:
- Ensure Python 3.8+ is installed
- Check internet connection for model download
- Try manual installation steps

**Windows batch files don't work**:
- Run Command Prompt as Administrator
- Check Python is in PATH
- Use manual installation method

### Getting Help

1. Check the [Cache Clearing Guide](docs/CACHE_CLEARING_GUIDE.md) for cache-related issues
2. Check the logs in the `logs/` directory
3. Review the console output for error messages
4. Ensure all requirements are met
5. Try the manual installation method

## 🎯 Usage Guide

### Getting Started

1. **Load a Model**: Go to the "Model Settings" tab and select a model to load
2. **Choose Content Type**: Select from various adult content categories
3. **Generate Content**: Enter your prompt and click "Generate"
4. **Refine Results**: Use the conversation history to build upon previous responses
5. **Export Content**: Save your generated content for later use

### Available Models

- **Luna-AI-Llama2-Uncensored** (7B): Fast, efficient, good for most tasks
- **WizardLM-13B-Uncensored** (13B): Balanced performance and quality
- **Wizard-Vicuna-30B-Uncensored** (30B): Highest quality, requires more resources

### Content Categories

- Photo/Video Descriptions
- Roleplay Scenarios
- Interactive Stories
- Custom Prompts
- Social Media Captions
- And more...

## ⚙️ Configuration

### GPU Acceleration

For NVIDIA GPUs, install CUDA support:
```bash
pip install ctransformers[cuda]
```

For Apple Silicon Macs:
```bash
pip install ctransformers[metal]
```

### Model Settings

- **Temperature**: Controls creativity (0.1-1.0)
- **Max Tokens**: Maximum response length
- **Top-p**: Nucleus sampling parameter
- **GPU Layers**: Number of layers to run on GPU

## 📁 Project Structure

```
fanvue-content-chatbot/
├── main.py              # Main application file
├── prompt_utils.py      # Prompt templates and utilities
├── image_analyzer.py    # Image analysis functionality
├── setup.py            # Setup and model download script
├── requirements.txt    # Python dependencies
├── install.bat         # Windows installation script
├── start.bat          # Windows startup script
├── docs/              # Documentation
│   └── CACHE_CLEARING_GUIDE.md  # Comprehensive cache clearing guide
├── models/            # Downloaded AI models
├── exports/           # Exported content
└── logs/             # Application logs
```

## ⚠️ Important Notes

### Content Guidelines
- This tool is designed for adult content creators (18+)
- Use responsibly and in accordance with platform guidelines
- Respect consent and legal boundaries
- Review generated content before publishing

### Privacy & Security
- All processing happens locally on your machine
- No data is sent to external servers (except for model downloads)
- Generated content is stored locally only
- Clear conversation history regularly if needed

### Legal Disclaimer
- Users are responsible for compliance with local laws
- Content must comply with platform terms of service
- This tool is for creative assistance only
- Generated content should be reviewed and edited as needed

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [Gradio](https://gradio.app/) for the web interface
- Uses [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) for model inference
- Models from [TheBloke](https://huggingface.co/TheBloke) on Hugging Face
- Inspired by the adult content creator community

---

**Disclaimer**: This software is intended for adult content creators (18+) and should be used responsibly. Users are responsible for ensuring their content complies with applicable laws and platform guidelines.
