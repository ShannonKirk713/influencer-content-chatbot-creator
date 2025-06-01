#!/usr/bin/env python3
"""
Fanvue Chatbot - Advanced AI Assistant for Adult Content Creation
Enhanced with multiple models, image analysis, and comprehensive content generation
"""

import gradio as gr
import json
import os
import logging
from datetime import datetime
from typing import List, Tuple, Optional
from PIL import Image
import random

# Import our modules
from image_analyzer import image_analyzer
from sd_forge_utils import PromptComplexityAnalyzer, SDForgeParams

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FanvueChatbot:
    """Main chatbot class with model management and content generation."""
    
    def __init__(self):
        self.llm = None
        self.current_model = None
        self.conversation_history = []
        self.history_folder = "conversation_logs"
        self.complexity_analyzer = PromptComplexityAnalyzer()
        
        # Create history folder
        os.makedirs(self.history_folder, exist_ok=True)
        
        # Available models configuration - RESTORED UNCENSORED MODELS
        self.model_configs = {
            # RESTORED: TheBloke uncensored models
            "Luna-AI-Llama2-Uncensored": {
                "repo_id": "TheBloke/Luna-AI-Llama2-Uncensored-GGUF",
                "filename": "luna-ai-llama2-uncensored.Q4_K_M.gguf",
                "template": "USER: {prompt}\nASSISTANT:",
                "description": "Efficient 7B uncensored model, good for most adult content tasks"
            },
            "WizardLM-13B-Uncensored": {
                "repo_id": "TheBloke/WizardLM-13B-Uncensored-GGUF", 
                "filename": "WizardLM-13B-Uncensored.Q4_K_M.gguf",
                "template": "You are a helpful AI assistant.\n\nUSER: {prompt}\nASSISTANT:",
                "description": "Balanced 13B uncensored model, high quality responses"
            },
            "Wizard-Vicuna-30B-Uncensored": {
                "repo_id": "TheBloke/Wizard-Vicuna-30B-Uncensored-GGUF",
                "filename": "Wizard-Vicuna-30B-Uncensored.Q4_K_M.gguf", 
                "template": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {prompt} ASSISTANT:",
                "description": "High-end 30B uncensored model, best quality (requires powerful hardware)"
            },
            "Nous-Hermes-13B-Uncensored": {
                "repo_id": "TheBloke/Nous-Hermes-13b-Chinese-GGUF",
                "filename": "nous-hermes-13b-chinese.Q4_K_M.gguf",
                "template": "### Instruction:\n{prompt}\n\n### Response:",
                "description": "Creative 13B uncensored model, excellent for roleplay and creative content"
            },
            # Modern models for comparison
            "llama-3.2-3b-instruct": {
                "repo_id": "huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF",
                "filename": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
                "template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                "description": "Fast 3B model, good for quick responses"
            },
            "llama-3.1-8b-instruct": {
                "repo_id": "huggingface.co/bartowski/Llama-3.1-8B-Instruct-GGUF",
                "filename": "Llama-3.1-8B-Instruct-Q4_K_M.gguf",
                "template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                "description": "Balanced 8B model, good quality and speed"
            },
            "qwen-2.5-14b-instruct": {
                "repo_id": "huggingface.co/bartowski/Qwen2.5-14B-Instruct-GGUF",
                "filename": "Qwen2.5-14B-Instruct-Q4_K_M.gguf",
                "template": "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
                "description": "High-quality 14B model, excellent for detailed content"
            },
            "llama-3.1-70b-instruct": {
                "repo_id": "huggingface.co/bartowski/Llama-3.1-70B-Instruct-GGUF",
                "filename": "Llama-3.1-70B-Instruct-IQ2_M.gguf",
                "template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                "description": "Premium 70B model, best quality (requires powerful hardware)"
            },
            "goliath-120b": {
                "repo_id": "huggingface.co/bartowski/goliath-120b-GGUF",
                "filename": "goliath-120b-Q2_K.gguf",
                "template": "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
                "description": "Ultra-high-end 120B model, exceptional quality (requires very powerful hardware)"
            },
            "deepseek-v2.5": {
                "repo_id": "huggingface.co/bartowski/DeepSeek-V2.5-GGUF",
                "filename": "DeepSeek-V2.5-Q4_K_M.gguf",
                "template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                "description": "Advanced reasoning model, great for complex prompts"
            },
            "mixtral-8x22b-instruct": {
                "repo_id": "huggingface.co/bartowski/Mixtral-8x22B-Instruct-v0.1-GGUF",
                "filename": "Mixtral-8x22B-Instruct-v0.1-Q2_K.gguf",
                "template": "<s>[INST] {system}\n\n{prompt} [/INST]",
                "description": "High-end 30B model, best quality (requires powerful hardware)"
            }
        }
        
        # Content type templates with multiple examples for randomization
        self.content_templates = {
            "image_prompt": {
                "system": """You are an expert at creating detailed image prompts for adult content generation. Generate prompts in this exact format:

📸 CONCEPT: [Brief concept description]
👥 SUBJECT(S): [Detailed subject description including appearance, age (18+), positioning]
👗 CLOTHING: [Clothing/lingerie/styling details]
🏞️ SETTING: [Location and environment details]
💃 POSE & EXPRESSION: [Specific pose, facial expression, body language]
📷 TECHNICAL: [Camera settings, lighting, composition, style]

Focus on creating visually appealing, tasteful adult content. Be specific and detailed.""",
                "examples": [
                    "Create an image prompt for a sensual bedroom scene",
                    "Generate a prompt for an elegant boudoir photography session",
                    "Create a romantic sunset portrait concept",
                    "Design a luxurious bathroom scene with soft lighting",
                    "Generate a vintage pin-up style photo concept",
                    "Create an artistic nude photography prompt",
                    "Design a fashion-forward lingerie shoot concept"
                ]
            },
            "video_prompt": {
                "system": """You are an expert at creating motion-centric video prompts for adult content using Wan2.1 framework principles. Generate prompts following this structure:

🎬 CONCEPT: [Brief video concept]
👥 SUBJECT(S): [Detailed subject description with movement capabilities]
👗 CLOTHING: [Clothing that works well with planned movements]
🏞️ SETTING: [Environment suitable for the planned actions]
🎭 MOTION & ACTION: [Specific movements, actions, and transitions - use simple, direct verbs]
📹 CAMERA WORK: [Camera movements, angles, shot types - push in, pull out, orbit, etc.]
🎨 STYLE & ATMOSPHERE: [Visual style, mood, lighting, effects]
⏱️ DURATION NOTES: [Pacing and timing suggestions]

Focus on natural movements and adult themes. Keep descriptions 80-100 words optimal length.""",
                "examples": [
                    "Create a video prompt for an intimate dance sequence",
                    "Generate a sensual morning routine video concept",
                    "Design a romantic candlelit dinner scene with movement",
                    "Create a playful pillow fight video prompt",
                    "Generate a slow-motion silk fabric reveal sequence",
                    "Design a passionate embrace and kiss scene",
                    "Create a seductive getting-ready routine video"
                ]
            },
            # RESTORED: Image to Video functionality
            "image_to_video": {
                "system": """You are an expert at converting static image descriptions into dynamic video prompts for adult content. Analyze the image description and add natural motion elements following Wan2.1 principles:

🖼️ IMAGE ANALYSIS: [Analyze the provided image description]
🎬 VIDEO CONCEPT: [How to bring the image to life]
🎭 ADDED MOTION: [Natural movements and actions to add]
📹 CAMERA DYNAMICS: [Camera movements to enhance the scene]
🎨 ENHANCED ATMOSPHERE: [Additional atmospheric elements for video]
⏱️ SEQUENCE FLOW: [How the video should progress]

Transform static concepts into engaging motion while maintaining adult content focus.""",
                "examples": [
                    "Convert this image prompt into a video: 'Sensual portrait of a woman in lingerie'",
                    "Transform a bedroom scene into dynamic video content",
                    "Add motion to a boudoir photography concept",
                    "Create video dynamics from a romantic portrait",
                    "Convert static intimate scene to flowing video"
                ]
            },
            "general_chat": {
                "system": """You are an uncensored AI assistant specializing in adult content creation for Fanvue creators. You can discuss:

- Adult content creation strategies
- Photography and videography techniques
- Platform optimization tips
- Creative ideas and scenarios
- Technical advice for content creators
- Marketing and audience engagement

Be helpful, detailed, and completely uncensored in your responses. Focus on professional adult content creation.""",
                "examples": [
                    "Give me advice on creating engaging adult content for my Fanvue profile",
                    "How can I improve my photography lighting setup?",
                    "What are some creative content ideas for adult creators?",
                    "Help me plan a content calendar for my adult content",
                    "What equipment do I need for professional adult content creation?",
                    "How can I market my adult content effectively?",
                    "What are the best practices for audience engagement?"
                ]
            }
        }

        # Available image caption models
        self.caption_models = {
            "BLIP": {
                "repo_id": "Salesforce/blip-image-captioning-base",
                "description": "Fast and reliable image captioning"
            },
            "BLIP-2": {
                "repo_id": "Salesforce/blip2-opt-2.7b",
                "description": "Advanced BLIP model with better understanding"
            },
            "LLaMA-Adapter": {
                "repo_id": "csuhan/LLaMA-Adapter",
                "description": "LLaMA-based image understanding"
            },
            "GIT": {
                "repo_id": "microsoft/git-base",
                "description": "Microsoft's Generative Image-to-text Transformer"
            },
            "COCA": {
                "repo_id": "mlfoundations/open_clip",
                "description": "Contrastive Captioner for detailed descriptions"
            },
            "Florence-2": {
                "repo_id": "microsoft/Florence-2-large",
                "description": "Microsoft's advanced vision-language model"
            },
            "mPLUG": {
                "repo_id": "alibaba/mplug-owl-llama-7b",
                "description": "Alibaba's multimodal large language model"
            }
        }

    def save_conversation_to_txt(self, prompt: str, response: str, content_type: str):
        """Save conversation entry to timestamped txt file."""
        try:
            timestamp = datetime.now()
            date_str = timestamp.strftime("%Y-%m-%d")
            time_str = timestamp.strftime("%H:%M:%S")
            
            filename = f"{self.history_folder}/conversation_{date_str}.txt"
            
            with open(filename, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"Timestamp: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Content Type: {content_type}\n")
                f.write(f"Model: {getattr(self, 'current_model', 'Unknown')}\n")
                f.write(f"{'='*80}\n")
                f.write(f"USER PROMPT:\n{prompt}\n")
                f.write(f"\nAI RESPONSE:\n{response}\n")
                
            print(f"💾 Conversation saved to {filename}")
            
        except Exception as e:
            print(f"❌ Error saving conversation: {e}")

    def load_model(self, model_name: str, gpu_layers: int = 35, progress=gr.Progress()) -> str:
        """Load the specified model."""
        try:
            print(f"🔄 Loading model: {model_name}")
            progress(0.1, f"Initializing {model_name}...")
            
            # Import llama-cpp-python
            try:
                from llama_cpp import Llama
            except ImportError:
                error_msg = "llama-cpp-python not installed. Please install with: pip install llama-cpp-python"
                print(f"❌ {error_msg}")
                return f"❌ {error_msg}"
            
            if model_name not in self.model_configs:
                error_msg = f"Unknown model: {model_name}"
                print(f"❌ {error_msg}")
                return f"❌ {error_msg}"
            
            model_config = self.model_configs[model_name]
            progress(0.3, f"Loading {model_name} from {model_config['repo_id']}...")
            
            # Load the model
            self.llm = Llama.from_pretrained(
                repo_id=model_config["repo_id"],
                filename=model_config["filename"],
                n_gpu_layers=gpu_layers,
                n_ctx=4096,
                verbose=False
            )
            
            self.current_model = model_name
            progress(1.0, f"✅ {model_name} loaded successfully!")
            
            success_msg = f"✅ Model {model_name} loaded successfully with {gpu_layers} GPU layers!"
            print(success_msg)
            return success_msg
            
        except Exception as e:
            error_msg = f"Error loading model {model_name}: {str(e)}"
            logger.error(error_msg)
            print(f"❌ {error_msg}")
            return f"❌ {error_msg}"

    def download_additional_model(self, repo_id: str, filename: str, progress=gr.Progress()) -> str:
        """Download additional models from Hugging Face."""
        try:
            print(f"📥 Downloading model from {repo_id}")
            progress(0.1, f"Connecting to {repo_id}...")
            
            from llama_cpp import Llama
            
            progress(0.3, "Downloading model files...")
            
            # Download the model (this will cache it locally)
            llm = Llama.from_pretrained(
                repo_id=repo_id,
                filename=filename,
                n_gpu_layers=0,  # Don't load into GPU, just download
                verbose=False
            )
            
            progress(0.8, "Finalizing download...")
            
            # Add to available models if not already present
            model_name = f"custom_{repo_id.split('/')[-1]}"
            if model_name not in self.model_configs:
                self.model_configs[model_name] = {
                    "repo_id": repo_id,
                    "filename": filename,
                    "template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                    "description": f"Custom model from {repo_id}"
                }
            
            progress(1.0, "Download complete!")
            success_msg = f"✅ Model downloaded successfully! Available as: {model_name}"
            print(success_msg)
            return success_msg
            
        except Exception as e:
            error_msg = f"Error downloading model: {str(e)}"
            logger.error(error_msg)
            print(f"❌ {error_msg}")
            return error_msg

    def generate_response(self, prompt: str, content_type: str, temperature: float = 0.7, progress=gr.Progress()) -> str:
        """Generate response using the loaded model."""
        if not self.llm:
            return "❌ No model loaded. Please load a model first."
        
        try:
            print(f"🤖 Generating {content_type} response...")
            progress(0.1, "Preparing prompt...")
            
            # Get template for content type
            template_info = self.content_templates.get(content_type, self.content_templates["general_chat"])
            system_prompt = template_info["system"]
            
            # Format the full prompt based on model type
            model_config = self.model_configs[self.current_model]
            
            # Handle different template formats
            if "{system}" in model_config["template"]:
                # Modern format with separate system and user prompts
                full_prompt = model_config["template"].format(
                    system=system_prompt,
                    prompt=prompt
                )
            else:
                # Legacy format - combine system and user prompt
                full_prompt = model_config["template"].format(
                    prompt=f"{system_prompt}\n\nUser Request: {prompt}"
                )
            
            progress(0.3, "Generating response...")
            
            # Generate response
            response = self.llm(
                full_prompt,
                max_tokens=1024,
                temperature=temperature,
                top_p=0.9,
                stop=["<|eot_id|>", "<|im_end|>", "</s>", "[/INST]", "USER:", "ASSISTANT:", "\n\nUSER:", "\n\nASSISTANT:"],
                echo=False
            )
            
            generated_text = response["choices"][0]["text"].strip()
            
            progress(0.9, "Finalizing response...")
            
            # Save to conversation history
            conversation_entry = {
                "timestamp": datetime.now().isoformat(),
                "prompt": prompt,
                "response": generated_text,
                "content_type": content_type,
                "model": self.current_model,
                "temperature": temperature
            }
            self.conversation_history.append(conversation_entry)
            
            # Save to text file
            self.save_conversation_to_txt(prompt, generated_text, content_type)
            
            progress(1.0, "Response generated!")
            print(f"✅ Response generated successfully ({len(generated_text)} characters)")
            
            return generated_text
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            logger.error(error_msg)
            print(f"❌ {error_msg}")
            return f"❌ {error_msg}"

    def get_example_prompt(self, content_type: str) -> str:
        """Get a random example prompt for content type."""
        template_info = self.content_templates.get(content_type, {})
        examples = template_info.get("examples", ["No examples available"])
        return random.choice(examples)

# Initialize chatbot
chatbot = FanvueChatbot()

def load_model_interface(model_name: str, gpu_layers: int, progress=gr.Progress()) -> str:
    """Interface function for loading models."""
    return chatbot.load_model(model_name, gpu_layers, progress)

def download_model_interface(repo_id: str, filename: str, progress=gr.Progress()) -> str:
    """Interface function for downloading additional models."""
    if not repo_id.strip() or not filename.strip():
        return "❌ Please provide both repository ID and filename"
    return chatbot.download_additional_model(repo_id, filename, progress)

def generate_content(prompt: str, content_type: str, temperature: float, progress=gr.Progress()) -> str:
    """Interface function for generating content."""
    if not prompt.strip():
        return "Please enter a prompt."
    
    return chatbot.generate_response(prompt, content_type, temperature, progress)

def analyze_uploaded_image(image: Image.Image, caption_model: str, progress=gr.Progress()) -> Tuple[str, str]:
    """
    Analyze uploaded image and return caption and detailed description.
    """
    if image is None:
        return "❌ No image uploaded", ""
    
    try:
        print(f"🖼️ Analyzing image with {caption_model} model...")
        progress(0.1, f"Loading {caption_model} model...")
        
        # For now, use the default BLIP model regardless of selection
        # In a full implementation, you would switch models based on caption_model parameter
        result = image_analyzer.analyze_image(image)
        
        if result["success"]:
            caption = result["caption"]
            detailed_description = result["detailed_description"]
            
            progress(1.0, "Analysis complete!")
            print(f"✅ Image analysis completed")
            
            return caption, detailed_description
        else:
            error_msg = f"❌ Image analysis failed: {result['error']}"
            print(error_msg)
            return error_msg, ""
            
    except Exception as e:
        error_msg = f"❌ Error analyzing image: {str(e)}"
        print(error_msg)
        return error_msg, ""

def generate_video_prompt_from_image(image: Image.Image, user_request: str, progress=gr.Progress()) -> str:
    """Generate video prompt from uploaded image."""
    if image is None:
        return "❌ No image uploaded"
    
    try:
        print("🎬 Generating video prompt from image...")
        progress(0.2, "Analyzing image...")
        
        # First analyze the image
        analysis_result = image_analyzer.analyze_image(image)
        
        if not analysis_result["success"]:
            return f"❌ Could not analyze image: {analysis_result['error']}"
        
        progress(0.6, "Creating video prompt...")
        
        # Generate video prompt based on analysis
        video_prompt = image_analyzer.generate_video_prompt_from_image(analysis_result, user_request)
        
        progress(1.0, "Video prompt generated!")
        print("✅ Video prompt generated successfully")
        
        return video_prompt
        
    except Exception as e:
        error_msg = f"❌ Error generating video prompt: {str(e)}"
        print(error_msg)
        return error_msg

def analyze_prompt_complexity(prompt: str) -> Tuple[str, str]:
    """Analyze prompt complexity and suggest SD Forge parameters."""
    if not prompt.strip():
        return "Please enter a prompt to analyze.", ""
    
    try:
        # Analyze complexity
        analysis = chatbot.complexity_analyzer.analyze_prompt_complexity(prompt)
        
        # Get SD Forge recommendations
        sd_params = chatbot.complexity_analyzer.recommend_sd_forge_params(prompt)
        
        # Format analysis results
        analysis_text = f"""🔍 **Prompt Complexity Analysis**

**Overall Complexity:** {analysis['complexity_level']} ({analysis['complexity_score']}/100)

**Metrics:**
- Word Count: {analysis['word_count']}
- Technical Terms: {analysis['technical_score']}
- Detail Indicators: {analysis['detail_score']}
- Complexity Modifiers: {analysis['modifier_score']}

**Technical Categories Found:**
"""
        
        for category, count in analysis['technical_categories'].items():
            if count > 0:
                analysis_text += f"- {category.title()}: {count} terms\n"
        
        # Format SD Forge parameters
        params_text = f"""⚙️ **Recommended Stable Diffusion Forge Parameters**

**Generation Settings:**
- Steps: {sd_params.steps}
- Sampler: {sd_params.sampler}
- Schedule Type: {sd_params.schedule_type}
- CFG Scale: {sd_params.cfg_scale}
- Distilled CFG Scale: {sd_params.distilled_cfg_scale}

**Image Settings:**
- Width: {sd_params.width}
- Height: {sd_params.height}
- Seed: {sd_params.seed} (use -1 for random)

**Explanation:** Based on your prompt's complexity level ({analysis['complexity_level']}), these parameters are optimized for the best quality-to-time ratio."""
        
        return analysis_text, params_text
        
    except Exception as e:
        error_msg = f"❌ Error analyzing prompt: {str(e)}"
        return error_msg, ""

def get_example(content_type: str) -> str:
    """Get example prompt for the selected content type."""
    return chatbot.get_example_prompt(content_type)

def clear_conversation():
    """Clear conversation history."""
    chatbot.conversation_history = []
    print("🗑️ Conversation history cleared.")
    return "Conversation history cleared."

def export_conversation() -> str:
    """Export conversation history as JSON."""
    if not chatbot.conversation_history:
        return "No conversation history to export."
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"fanvue_conversation_{timestamp}.json"
    
    try:
        with open(filename, 'w') as f:
            json.dump(chatbot.conversation_history, f, indent=2)
        
        success_msg = f"✅ Conversation exported to {filename}"
        print(success_msg)
        return success_msg
        
    except Exception as e:
        error_msg = f"❌ Error exporting conversation: {str(e)}"
        print(error_msg)
        return error_msg

def create_interface():
    """Create and configure the Gradio interface."""
    
    with gr.Blocks(
        title="Fanvue Chatbot - Advanced AI Assistant",
        theme=gr.themes.Soft(),
        css="""
        .main-header { text-align: center; margin-bottom: 2rem; }
        .model-info { background: #f0f0f0; padding: 1rem; border-radius: 8px; margin: 1rem 0; }
        .status-box { background: #e8f5e8; padding: 0.5rem; border-radius: 4px; }
        .error-box { background: #ffe8e8; padding: 0.5rem; border-radius: 4px; }
        .warning-box { background: #fff3cd; border: 1px solid #ffeaa7; padding: 1rem; border-radius: 8px; margin: 1rem 0; }
        """
    ) as interface:
        
        gr.HTML("""
        <div class="main-header">
            <h1>🔥 Fanvue Chatbot - Advanced AI Assistant</h1>
            <p>Professional AI assistant for adult content creation with multiple models, image analysis, and advanced features</p>
        </div>
        """)
        
        # Warning notice
        gr.HTML("""
        <div class="warning-box">
            <strong>⚠️ ADULT CONTENT WARNING</strong><br>
            This application is designed for creating adult content (18+). By using this tool, you confirm that:
            <ul>
                <li>You are 18 years of age or older</li>
                <li>You understand this tool generates adult content</li>
                <li>You will use this responsibly and in compliance with applicable laws</li>
                <li>You will respect platform guidelines and terms of service</li>
            </ul>
        </div>
        """)
        
        with gr.Tabs():
            # Model Management Tab
            with gr.Tab("🤖 Model Management"):
                gr.Markdown("## Load and Manage AI Models")
                
                with gr.Row():
                    with gr.Column():
                        model_dropdown = gr.Dropdown(
                            choices=list(chatbot.model_configs.keys()),
                            label="Select Model",
                            value="Luna-AI-Llama2-Uncensored"
                        )
                        
                        gpu_layers_slider = gr.Slider(
                            minimum=0,
                            maximum=80,
                            value=35,
                            step=1,
                            label="GPU Layers",
                        )
                        
                        load_btn = gr.Button("🚀 Load Model", variant="primary", size="lg")
                        
                    with gr.Column():
                        model_status = gr.Textbox(
                            label="Model Status",
                            value="No model loaded",
                            interactive=False,
                            lines=3
                        )
                
                # Model descriptions
                gr.Markdown("### Available Models")
                model_info_html = "<div class='model-info'>"
                for name, config in chatbot.model_configs.items():
                    model_info_html += f"<p><strong>{name}:</strong> {config['description']}</p>"
                model_info_html += "</div>"
                gr.HTML(model_info_html)
                
                # Additional model download section
                gr.Markdown("## Download Additional Models")
                with gr.Row():
                    with gr.Column():
                        repo_id_input = gr.Textbox(
                            label="Hugging Face Repository ID",
                            placeholder="e.g., TheBloke/Luna-AI-Llama2-Uncensored-GGUF",
                        )
                        filename_input = gr.Textbox(
                            label="Model Filename",
                            placeholder="e.g., luna-ai-llama2-uncensored.Q4_K_M.gguf",
                        )
                        download_btn = gr.Button("📥 Download Model", variant="secondary")
                    
                    with gr.Column():
                        download_status = gr.Textbox(
                            label="Download Status",
                            interactive=False,
                            lines=3
                        )
                
                # Wire up model management events
                load_btn.click(
                    fn=load_model_interface,
                    inputs=[model_dropdown, gpu_layers_slider],
                    outputs=model_status,
                    show_progress=True
                )
                
                download_btn.click(
                    fn=download_model_interface,
                    inputs=[repo_id_input, filename_input],
                    outputs=download_status,
                    show_progress=True
                )
            
            # Content Generation Tab
            with gr.Tab("✨ Content Generation"):
                gr.Markdown("## Generate AI Content")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        content_type_dropdown = gr.Dropdown(
                            choices=["image_prompt", "video_prompt", "image_to_video", "general_chat"],
                            value="image_prompt",
                            label="Content Type",
                        )
                        
                        prompt_input = gr.Textbox(
                            label="Your Prompt",
                            placeholder="Enter your request here...",
                            lines=4,
                        )
                        
                        with gr.Row():
                            temperature_slider = gr.Slider(
                                minimum=0.1,
                                maximum=1.5,
                                value=0.7,
                                step=0.1,
                                label="Creativity (Temperature)",
                            )
                            
                            example_btn = gr.Button("💡 Get Example", variant="secondary")
                        
                        generate_btn = gr.Button("🎨 Generate Content", variant="primary", size="lg")
                        
                    with gr.Column(scale=3):
                        output_text = gr.Textbox(
                            label="Generated Content",
                            lines=20,
                            max_lines=30,
                            interactive=False,
                            show_copy_button=True
                        )
                
                # Wire up content generation events
                generate_btn.click(
                    fn=generate_content,
                    inputs=[prompt_input, content_type_dropdown, temperature_slider],
                    outputs=output_text,
                    show_progress=True
                )
                
                example_btn.click(
                    fn=get_example,
                    inputs=content_type_dropdown,
                    outputs=prompt_input
                )
            
            # Image Analysis Tab
            with gr.Tab("🖼️ Image Analysis"):
                gr.Markdown("## Analyze Images and Generate Video Prompts")
                
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(
                            label="Upload Image",
                            type="pil",
                            height=400
                        )
                        
                        caption_model_dropdown = gr.Dropdown(
                            choices=list(chatbot.caption_models.keys()),
                            value="BLIP",
                            label="Caption Model",
                        )
                        
                        analyze_btn = gr.Button("🔍 Analyze Image", variant="primary")
                        
                        gr.Markdown("### Generate Video from Image")
                        video_request_input = gr.Textbox(
                            label="Additional Requirements (Optional)",
                            placeholder="e.g., focus on movement, add romantic elements...",
                            lines=2
                        )
                        
                        video_btn = gr.Button("🎬 Generate Video Prompt", variant="secondary")
                    
                    with gr.Column():
                        image_caption = gr.Textbox(
                            label="Image Caption",
                            lines=3,
                            interactive=False
                        )
                        
                        image_description = gr.Textbox(
                            label="Detailed Description",
                            lines=5,
                            interactive=False
                        )
                        
                        video_output = gr.Textbox(
                            label="Generated Video Prompt",
                            lines=15,
                            interactive=False,
                            show_copy_button=True
                        )
                
                # Wire up image analysis events
                analyze_btn.click(
                    fn=analyze_uploaded_image,
                    inputs=[image_input, caption_model_dropdown],
                    outputs=[image_caption, image_description],
                    show_progress=True
                )
                
                video_btn.click(
                    fn=generate_video_prompt_from_image,
                    inputs=[image_input, video_request_input],
                    outputs=video_output,
                    show_progress=True
                )
            
            # Prompt Analysis Tab
            with gr.Tab("🔍 Prompt Analysis"):
                gr.Markdown("## Analyze Prompt Complexity and Get SD Forge Parameters")
                
                with gr.Row():
                    with gr.Column():
                        analysis_input = gr.Textbox(
                            label="Prompt to Analyze",
                            placeholder="Enter your image generation prompt here...",
                            lines=5
                        )
                        
                        analyze_complexity_btn = gr.Button("🔍 Analyze Complexity", variant="primary")
                    
                    with gr.Column():
                        complexity_output = gr.Textbox(
                            label="Complexity Analysis",
                            lines=10,
                            interactive=False
                        )
                        
                        params_output = gr.Textbox(
                            label="Recommended SD Forge Parameters",
                            lines=10,
                            interactive=False,
                            show_copy_button=True
                        )
                
                # Wire up prompt analysis events
                analyze_complexity_btn.click(
                    fn=analyze_prompt_complexity,
                    inputs=analysis_input,
                    outputs=[complexity_output, params_output]
                )
            
            # History & Export Tab
            with gr.Tab("📚 History & Export"):
                gr.Markdown("## Conversation History and Export")
                
                with gr.Row():
                    with gr.Column():
                        clear_btn = gr.Button("🗑️ Clear History", variant="secondary")
                        export_btn = gr.Button("📤 Export Conversation", variant="primary")
                    
                    with gr.Column():
                        history_status = gr.Textbox(
                            label="Status",
                            interactive=False,
                            lines=3
                        )
                
                gr.Markdown("### Recent Conversations")
                conversation_display = gr.JSON(
                    label="Conversation History",
                    value=lambda: chatbot.conversation_history[-5:] if chatbot.conversation_history else []
                )
                
                # Wire up history events
                clear_btn.click(
                    fn=clear_conversation,
                    outputs=history_status
                )
                
                export_btn.click(
                    fn=export_conversation,
                    outputs=history_status
                )
    
    return interface

if __name__ == "__main__":
    print("🚀 Starting Fanvue Chatbot...")
    
    # Create and launch the interface
    interface = create_interface()
    
    # Launch with custom settings
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False,
        show_error=True,
        quiet=False
    )
