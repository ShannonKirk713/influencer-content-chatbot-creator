# Influencer Content Chatbot Creator

A sophisticated standalone AI-powered chatbot application designed for adult content creators. This software provides powerful content generation capabilities using uncensored language models, specifically designed for platforms like Fanvue, OnlyFans, and other adult content platforms.

## 🌟 Features

- **Multiple Uncensored Models**: Support for Luna-AI-Llama2, WizardLM-13B, and Wizard-Vicuna-30B
- **Specialized Templates**: Pre-built prompt templates for various adult content scenarios
- **Image Analysis**: Upload and analyze images for content inspiration
- **Conversation Management**: Save and manage conversation history
- **Standalone Application**: No need for external dependencies or complex setups
- **Real-time Generation**: Interactive content generation with progress tracking
- **Easy Updates**: Simple one-click update system

## 📋 Requirements

- Windows 10/11 (64-bit)
- Python 3.8+ (automatically installed if not present)
- Internet connection for initial setup and model downloads
- At least 8GB RAM recommended
- 10GB+ free disk space for models

## 🚀 Installation

### Method 1: Git Clone (Recommended)

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Valorking6/fanvue-content-chatbot.git
   cd fanvue-content-chatbot
   ```

2. **Run Installation**:
   - Double-click `install.bat` to automatically set up the environment and dependencies
   - Wait for the installation to complete (this may take several minutes)

3. **Start the Application**:
   - Double-click `run.bat` to launch the chatbot
   - The application will open in your default web browser at `http://localhost:7861`

### Method 2: Download ZIP

1. **Download**: Click the green "Code" button above and select "Download ZIP"
2. **Extract**: Extract the ZIP file to your desired location
3. **Install**: Double-click `install.bat` in the extracted folder
4. **Run**: Double-click `run.bat` to start the application

## 🔄 Updating

To update to the latest version:
- Double-click `update.bat` to automatically pull the latest changes and update dependencies
- Restart the application with `run.bat`

## 🎯 Usage

1. **Launch the Application**: Run `run.bat` and wait for the web interface to open

2. **Load a Model**: 
   - Select a model from the dropdown
   - Click "Load Model" and wait for it to download/load (first time only)

3. **Generate Content**:
   - Choose a prompt template or use custom prompt
   - Adjust temperature and max tokens as needed
   - Click "Generate Content"

4. **Analyze Images**:
   - Upload an image in the Image Analysis section
   - Click "Analyze Image" for content suggestions

5. **Manage Conversations**:
   - View conversation history
   - Save conversations to files
   - Clear history when needed

## 🎨 Prompt Templates

The application includes specialized templates for:

- **Seductive Photo**: Artistic photo shoot descriptions
- **Intimate Story**: Short story scenarios
- **Fantasy Roleplay**: Character and setting development
- **Sensual Description**: Atmospheric scene creation
- **Content Caption**: Engaging social media captions
- **Custom Prompt**: Your own creative prompts

## ⚙️ Configuration

### Model Settings
- **Temperature**: Controls creativity (0.1-2.0)
- **Max Tokens**: Response length (50-1000)
- **Context Length**: Varies by model

### Available Models
- **Luna-AI-Llama2-Uncensored**: Best for creative content
- **WizardLM-13B-V1.2-Uncensored**: Great for detailed scenarios
- **Wizard-Vicuna-30B-Uncensored**: Most powerful for complex content

## 🔧 Technical Details

### Application Structure
```
fanvue-content-chatbot/
├── install.bat                # Installation script
├── run.bat                    # Application launcher
├── update.bat                 # Update script
├── main.py                    # Main application file
├── image_analyzer.py          # Image analysis utilities
├── requirements.txt           # Python dependencies
├── models/                    # Downloaded models directory
├── conversations/             # Saved conversations
└── logs/                      # Application logs
```

### Batch Files
- **install.bat**: Sets up Python environment and installs all dependencies
- **run.bat**: Launches the chatbot application
- **update.bat**: Updates the application using `git pull` and refreshes dependencies

## 🛠️ Development

### Adding New Models
Edit the `model_configs` dictionary in `main.py`:

```python
"Your-Model-Name": {
    "repo_id": "huggingface/repo",
    "filename": "model.gguf",
    "description": "Model description",
    "context_length": 4096,
    "temperature": 0.7
}
```

### Adding New Templates
Edit the `prompt_templates` dictionary:

```python
"Template Name": "Template description and instructions..."
```

## 📝 Logging

The application creates detailed logs in:
- `logs/application.log`: Application logs
- `conversations/`: Saved conversations
- `install_log.txt`: Installation log
- `update_log.txt`: Update log

## 🔒 Privacy & Security

- All processing happens locally on your computer
- No data sent to external servers (except model downloads)
- Conversation history stored locally
- Models cached locally after first download
- No telemetry or data collection

## 🐛 Troubleshooting

### Common Issues

1. **Installation Fails**:
   - Run `install.bat` as Administrator
   - Check internet connection
   - Ensure sufficient disk space

2. **Application Won't Start**:
   - Check if Python is installed correctly
   - Run `install.bat` again
   - Check `logs/application.log` for errors

3. **Model Loading Fails**:
   - Check internet connection for first download
   - Ensure sufficient disk space
   - Try a different model

4. **Generation Errors**:
   - Ensure model is loaded first
   - Check prompt length
   - Verify temperature/token settings

### Getting Help

- Check the application logs in the `logs/` folder
- Review the installation log (`install_log.txt`)
- Ensure all dependencies are installed
- Try running `install.bat` again

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

For support and questions:
- Open an issue on GitHub
- Check the documentation
- Review the troubleshooting section

---

**Note**: This software is designed for adult content creation. Please ensure compliance with your local laws and platform guidelines when using this tool.

**System Requirements**: Windows 10/11, 8GB+ RAM, 10GB+ free space, internet connection for setup.