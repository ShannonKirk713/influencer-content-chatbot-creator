
# Fanvue Content Chatbot - Stable Diffusion WebUI Forge Extension

A sophisticated AI-powered chatbot extension for Stable Diffusion WebUI Forge, specifically designed for adult content creators. This extension integrates seamlessly as a new tab in the Forge interface, providing powerful content generation capabilities using uncensored language models.

## 🌟 Features

- **Multiple Uncensored Models**: Support for Luna-AI-Llama2, WizardLM-13B, and Wizard-Vicuna-30B
- **Specialized Templates**: Pre-built prompt templates for various adult content scenarios
- **Image Analysis**: Upload and analyze images for content inspiration
- **Conversation Management**: Save and manage conversation history
- **Forge Integration**: Seamless integration as a new tab in SD WebUI Forge
- **Real-time Generation**: Interactive content generation with progress tracking

## 📋 Requirements

- Stable Diffusion WebUI Forge
- Python 3.8+
- llama-cpp-python
- gradio
- PIL (Pillow)
- torch (for image analysis)

## 🚀 Installation

1. **Clone to Extensions Directory**:
   ```bash
   cd /path/to/stable-diffusion-webui-forge/extensions
   git clone https://github.com/Valorking6/fanvue-content-chatbot.git
   ```

2. **Switch to Extension Branch**:
   ```bash
   cd fanvue-content-chatbot
   git checkout Extension
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Restart Forge**: Restart your Stable Diffusion WebUI Forge instance

## 🎯 Usage

1. **Access the Extension**: Look for the "Fanvue Chatbot" tab in your Forge interface

2. **Load a Model**: 
   - Select a model from the dropdown
   - Click "Load Model" and wait for it to download/load

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

The extension includes specialized templates for:

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

### Extension Structure
```
fanvue-content-chatbot/
├── scripts/
│   └── fanvue_chatbot.py      # Main extension script
├── __init__.py                # Extension initialization
├── image_analyzer.py          # Image analysis utilities
├── sd_forge_utils.py          # Forge integration utilities
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

### Integration Points
- Extends `scripts.Script` for Forge compatibility
- Uses `gr.Tab()` for new tab creation
- Implements proper event handling and state management
- Follows Forge extension conventions

## 🛠️ Development

### Adding New Models
Edit the `model_configs` dictionary in `fanvue_chatbot.py`:

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

The extension creates detailed logs in:
- `fanvue_chatbot.log`: Application logs
- `conversation_logs/`: Saved conversations

## 🔒 Privacy & Security

- All processing happens locally
- No data sent to external servers (except model downloads)
- Conversation history stored locally
- Models cached locally after first download

## 🐛 Troubleshooting

### Common Issues

1. **Model Loading Fails**:
   - Check internet connection for first download
   - Ensure sufficient disk space
   - Verify llama-cpp-python installation

2. **Extension Not Visible**:
   - Restart Forge completely
   - Check extension is in correct directory
   - Verify no Python errors in console

3. **Generation Errors**:
   - Ensure model is loaded first
   - Check prompt length
   - Verify temperature/token settings

### Getting Help

- Check the Forge console for error messages
- Review the `fanvue_chatbot.log` file
- Ensure all dependencies are installed

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

**Note**: This extension is designed for adult content creation. Please ensure compliance with your local laws and platform guidelines when using this tool.
