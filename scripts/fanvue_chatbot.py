
import gradio as gr
import json
import os
import logging
from datetime import datetime
from typing import List, Tuple, Optional
from PIL import Image
import random

from modules import scripts
from modules.ui_components import InputAccordion

# Import our modules
try:
    # Try relative imports first
    import sys
    import os
    current_dir = os.path.dirname(os.path.dirname(__file__))
    sys.path.insert(0, current_dir)
    
    from image_analyzer import image_analyzer
    from sd_forge_utils import PromptComplexityAnalyzer, SDForgeParams
except ImportError as e:
    # Fallback - create minimal implementations
    logger.warning(f"Could not import modules: {e}. Using fallback implementations.")
    
    def image_analyzer(image):
        """Fallback image analyzer."""
        return "Image analysis functionality requires additional setup. Please check the installation."
    
    class PromptComplexityAnalyzer:
        def __init__(self):
            pass
    
    class SDForgeParams:
        def __init__(self):
            pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fanvue_chatbot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class InfluencerChatbot:
    """Main chatbot class with model management and content generation."""
    
    def __init__(self):
        self.llm = None
        self.current_model = None
        self.conversation_history = []
        self.history_folder = "conversation_logs"
        self.complexity_analyzer = PromptComplexityAnalyzer()
        
        # Create history folder
        os.makedirs(self.history_folder, exist_ok=True)
        
        # Available models configuration - ALL THEBLOKE MODELS
        self.model_configs = {
            # TheBloke uncensored models
            "Luna-AI-Llama2-Uncensored": {
                "repo_id": "TheBloke/Luna-AI-Llama2-Uncensored-GGUF",
                "filename": "luna-ai-llama2-uncensored.Q4_K_M.gguf",
                "description": "Luna AI Llama2 Uncensored - Best for creative adult content",
                "context_length": 4096,
                "temperature": 0.7
            },
            "WizardLM-13B-V1.2-Uncensored": {
                "repo_id": "TheBloke/WizardLM-13B-V1.2-Uncensored-GGUF",
                "filename": "wizardlm-13b-v1.2-uncensored.Q4_K_M.gguf",
                "description": "WizardLM 13B Uncensored - Great for detailed scenarios",
                "context_length": 4096,
                "temperature": 0.8
            },
            "Wizard-Vicuna-30B-Uncensored": {
                "repo_id": "TheBloke/Wizard-Vicuna-30B-Uncensored-GGUF",
                "filename": "wizard-vicuna-30b-uncensored.Q4_K_M.gguf",
                "description": "Wizard Vicuna 30B - Most powerful for complex content",
                "context_length": 2048,
                "temperature": 0.7
            }
        }
        
        # Specialized prompts for adult content creation
        self.prompt_templates = {
            "Seductive Photo": "Create a detailed description for a seductive photo shoot. Include setting, pose, outfit, lighting, and mood. Make it artistic and alluring.",
            "Intimate Story": "Write a short, intimate story scenario. Include character details, setting, and emotional connection. Keep it tasteful but engaging.",
            "Fantasy Roleplay": "Develop a fantasy roleplay scenario. Include character backgrounds, setting details, and interaction dynamics.",
            "Sensual Description": "Create a sensual scene description focusing on atmosphere, emotions, and subtle details that create intimacy.",
            "Content Caption": "Write an engaging caption for adult content that would attract viewers while maintaining class and sophistication.",
            "Custom Prompt": "Enter your own custom prompt for personalized content generation."
        }

    def load_model(self, model_name: str, progress=gr.Progress()) -> str:
        """Load the specified model."""
        try:
            if model_name not in self.model_configs:
                return f"❌ Model {model_name} not found in configurations"
            
            if self.current_model == model_name and self.llm is not None:
                return f"✅ Model {model_name} is already loaded"
            
            progress(0.1, desc="Initializing model...")
            
            # Import here to avoid issues if not available
            try:
                from llama_cpp import Llama
            except ImportError:
                return "❌ llama-cpp-python not installed. Please install it first."
            
            config = self.model_configs[model_name]
            
            progress(0.3, desc="Downloading model if needed...")
            
            # Load model
            self.llm = Llama.from_pretrained(
                repo_id=config["repo_id"],
                filename=config["filename"],
                n_ctx=config["context_length"],
                verbose=False
            )
            
            progress(0.8, desc="Finalizing...")
            
            self.current_model = model_name
            logger.info(f"Successfully loaded model: {model_name}")
            
            progress(1.0, desc="Complete!")
            
            return f"✅ Successfully loaded {model_name}"
            
        except Exception as e:
            error_msg = f"❌ Error loading model {model_name}: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def generate_response(self, prompt: str, temperature: float = 0.7, max_tokens: int = 500) -> str:
        """Generate response using the loaded model."""
        try:
            if self.llm is None:
                return "❌ No model loaded. Please load a model first."
            
            # Generate response
            response = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["Human:", "Assistant:", "\n\n"],
                echo=False
            )
            
            generated_text = response['choices'][0]['text'].strip()
            
            # Add to conversation history
            self.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "prompt": prompt,
                "response": generated_text,
                "model": self.current_model,
                "temperature": temperature,
                "max_tokens": max_tokens
            })
            
            return generated_text
            
        except Exception as e:
            error_msg = f"❌ Error generating response: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def save_conversation(self) -> str:
        """Save conversation history to file."""
        try:
            if not self.conversation_history:
                return "❌ No conversation to save"
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.history_folder}/conversation_{timestamp}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_history, f, indent=2, ensure_ascii=False)
            
            return f"✅ Conversation saved to {filename}"
            
        except Exception as e:
            error_msg = f"❌ Error saving conversation: {str(e)}"
            logger.error(error_msg)
            return error_msg

# Global chatbot instance
chatbot_instance = InfluencerChatbot()

class FanvueChatbotForForge(scripts.Script):
    """Fanvue Content Chatbot extension for Stable Diffusion WebUI Forge."""
    
    def title(self):
        return "Fanvue Content Chatbot"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        """Create the UI for the chatbot extension."""
        
        with gr.Tab("Fanvue Chatbot"):
            gr.Markdown("""
            # 🎭 Fanvue Content Chatbot
            
            Advanced AI assistant for adult content creation using uncensored language models.
            Perfect for generating creative prompts, scenarios, and content ideas.
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    # Model Management Section
                    with gr.Group():
                        gr.Markdown("### 🤖 Model Management")
                        
                        model_dropdown = gr.Dropdown(
                            choices=list(chatbot_instance.model_configs.keys()),
                            label="Select Model",
                            value="Luna-AI-Llama2-Uncensored",
                            info="Choose an uncensored model for content generation"
                        )
                        
                        load_model_btn = gr.Button("🔄 Load Model", variant="primary")
                        model_status = gr.Textbox(
                            label="Model Status",
                            value="No model loaded",
                            interactive=False
                        )
                    
                    # Generation Settings
                    with gr.Group():
                        gr.Markdown("### ⚙️ Generation Settings")
                        
                        temperature = gr.Slider(
                            minimum=0.1,
                            maximum=2.0,
                            value=0.7,
                            step=0.1,
                            label="Temperature",
                            info="Higher = more creative, Lower = more focused"
                        )
                        
                        max_tokens = gr.Slider(
                            minimum=50,
                            maximum=1000,
                            value=500,
                            step=50,
                            label="Max Tokens",
                            info="Maximum length of generated response"
                        )
                
                with gr.Column(scale=2):
                    # Content Generation Section
                    with gr.Group():
                        gr.Markdown("### 📝 Content Generation")
                        
                        prompt_template = gr.Dropdown(
                            choices=list(chatbot_instance.prompt_templates.keys()),
                            label="Prompt Template",
                            value="Custom Prompt",
                            info="Select a template or use custom prompt"
                        )
                        
                        user_input = gr.Textbox(
                            label="Your Prompt",
                            placeholder="Enter your prompt here...",
                            lines=3,
                            info="Describe what kind of content you want to generate"
                        )
                        
                        generate_btn = gr.Button("✨ Generate Content", variant="primary", size="lg")
                        
                        output_text = gr.Textbox(
                            label="Generated Content",
                            lines=10,
                            placeholder="Generated content will appear here...",
                            interactive=False
                        )
            
            with gr.Row():
                with gr.Column():
                    # Image Analysis Section
                    with gr.Group():
                        gr.Markdown("### 🖼️ Image Analysis")
                        
                        image_input = gr.Image(
                            label="Upload Image for Analysis",
                            type="pil",
                            info="Upload an image to get content suggestions"
                        )
                        
                        analyze_btn = gr.Button("🔍 Analyze Image")
                        
                        image_analysis = gr.Textbox(
                            label="Image Analysis",
                            lines=5,
                            placeholder="Image analysis will appear here...",
                            interactive=False
                        )
                
                with gr.Column():
                    # Conversation Management
                    with gr.Group():
                        gr.Markdown("### 💾 Conversation Management")
                        
                        conversation_display = gr.Textbox(
                            label="Conversation History",
                            lines=8,
                            placeholder="Conversation history will appear here...",
                            interactive=False
                        )
                        
                        with gr.Row():
                            save_btn = gr.Button("💾 Save Conversation")
                            clear_btn = gr.Button("🗑️ Clear History")
                        
                        save_status = gr.Textbox(
                            label="Save Status",
                            placeholder="Save status will appear here...",
                            interactive=False
                        )

            # Event handlers
            def on_template_change(template):
                if template in chatbot_instance.prompt_templates:
                    return chatbot_instance.prompt_templates[template]
                return ""

            def on_load_model(model_name, progress=gr.Progress()):
                return chatbot_instance.load_model(model_name, progress)

            def on_generate(prompt, temp, max_tok, progress=gr.Progress()):
                if not prompt.strip():
                    return "❌ Please enter a prompt"
                
                progress(0.1, desc="Generating content...")
                response = chatbot_instance.generate_response(prompt, temp, max_tok)
                progress(1.0, desc="Complete!")
                return response

            def on_analyze_image(image):
                if image is None:
                    return "❌ Please upload an image"
                
                try:
                    analysis = image_analyzer(image)
                    return analysis
                except Exception as e:
                    return f"❌ Error analyzing image: {str(e)}"

            def on_save_conversation():
                return chatbot_instance.save_conversation()

            def on_clear_history():
                chatbot_instance.conversation_history = []
                return "✅ Conversation history cleared", ""

            def update_conversation_display():
                if not chatbot_instance.conversation_history:
                    return "No conversation history"
                
                display_text = ""
                for i, entry in enumerate(chatbot_instance.conversation_history[-5:], 1):
                    display_text += f"--- Entry {i} ---\n"
                    display_text += f"Prompt: {entry['prompt'][:100]}...\n"
                    display_text += f"Response: {entry['response'][:200]}...\n\n"
                
                return display_text

            # Connect event handlers
            prompt_template.change(
                fn=on_template_change,
                inputs=[prompt_template],
                outputs=[user_input]
            )
            
            load_model_btn.click(
                fn=on_load_model,
                inputs=[model_dropdown],
                outputs=[model_status]
            )
            
            generate_btn.click(
                fn=on_generate,
                inputs=[user_input, temperature, max_tokens],
                outputs=[output_text]
            ).then(
                fn=update_conversation_display,
                outputs=[conversation_display]
            )
            
            analyze_btn.click(
                fn=on_analyze_image,
                inputs=[image_input],
                outputs=[image_analysis]
            )
            
            save_btn.click(
                fn=on_save_conversation,
                outputs=[save_status]
            )
            
            clear_btn.click(
                fn=on_clear_history,
                outputs=[save_status, conversation_display]
            )

        return []

    def process(self, p, *args):
        # This extension doesn't modify the image generation process
        pass
