#!/usr/bin/env python3
"""
Influencer Chatbot - Advanced AI Assistant for Adult Content Creation
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
        
        # Available models configuration - SPECIFIC UNCENSORED MODELS
        self.model_configs = {
            # Specific uncensored models requested
            "Orenguteng/Llama-3-8B-Lexi-Uncensored": {
                "repo_id": "Orenguteng/Llama-3-8B-Lexi-Uncensored-GGUF",
                "filename": "Llama-3-8B-Lexi-Uncensored.Q4_K_M.gguf",
                "template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\\\\n\\\\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\\\\n\\\\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\\\n\\\\n",
                "description": "Lexi uncensored 8B model, specialized for adult content"
            },
            "aifeifei798/DarkIdol-Llama-3.1-8B-Instruct-1.2-Uncensored": {
                "repo_id": "aifeifei798/DarkIdol-Llama-3.1-8B-Instruct-1.2-Uncensored-GGUF",
                "filename": "DarkIdol-Llama-3.1-8B-Instruct-1.2-Uncensored.Q4_K_M.gguf",
                "template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\\\\n\\\\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\\\\n\\\\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\\\n\\\\n",
                "description": "DarkIdol uncensored 8B model, enhanced for creative content"
            },
            "DevsDoCode/LLama-3-8b-Uncensored": {
                "repo_id": "DevsDoCode/LLama-3-8b-Uncensored-GGUF",
                "filename": "LLama-3-8b-Uncensored.Q4_K_M.gguf",
                "template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\\\\n\\\\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\\\\n\\\\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\\\n\\\\n",
                "description": "DevsDoCode uncensored 8B model, optimized for unrestricted content"
            },
            # Keep these specific models as requested
            "Luna-AI-Llama2-Uncensored": {
                "repo_id": "TheBloke/Luna-AI-Llama2-Uncensored-GGUF",
                "filename": "luna-ai-llama2-uncensored.Q4_K_M.gguf",
                "template": "USER: {prompt}\\\\nASSISTANT:",
                "description": "Efficient 7B uncensored model, good for most adult content tasks"
            },
            "WizardLM-13B-Uncensored": {
                "repo_id": "TheBloke/WizardLM-13B-Uncensored-GGUF", 
                "filename": "wizardlm-13b-uncensored.Q4_K_M.gguf",
                "template": "You are a helpful AI assistant.\\\\n\\\\nUSER: {prompt}\\\\nASSISTANT:",
                "description": "Balanced 13B uncensored model, high quality responses"
            },
            "Wizard-Vicuna-30B-Uncensored": {
                "repo_id": "TheBloke/Wizard-Vicuna-30B-Uncensored-GGUF",
                "filename": "wizard-vicuna-30b-uncensored.Q4_K_M.gguf", 
                "template": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {prompt} ASSISTANT:",
                "description": "High-end 30B uncensored model, best quality (requires powerful hardware)"
            },
            "Nous-Hermes-13B-Uncensored": {
                "repo_id": "TheBloke/Nous-Hermes-13b-GGUF",
                "filename": "nous-hermes-13b.Q4_K_M.gguf",
                "template": "### Instruction:\\\\n{prompt}\\\\n\\\\n### Response:",
                "description": "Creative 13B uncensored model, excellent for roleplay and creative content"
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
                "system": """You are an expert at creating video prompts using WAN 2.1 format (Subject + Scene + Motion structure). Generate prompts following this exact structure:

🎬 **SUBJECT**: [Detailed description of the main subject - appearance, clothing, positioning]
🏞️ **SCENE**: [Environment, setting, lighting, atmosphere, background elements]
🎭 **MOTION**: [Specific movements, actions, camera work - use simple, direct verbs]

**Technical Recommendations:**
📹 Recommended FPS: 24-30 fps for smooth motion
⏱️ Optimal Duration: 3-5 seconds for best quality
🎞️ Frame Count: 72-150 frames (24fps × 3-5 seconds)

Focus on clear, concise descriptions. Keep each section under 30 words for optimal video generation.""",
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
                "system": """You are an expert at converting static image descriptions into dynamic video prompts using WAN 2.1 format (Subject + Scene + Motion structure). Analyze the image and convert to video format:

🖼️ **IMAGE ANALYSIS**: [Brief analysis of the static image]

**WAN 2.1 VIDEO FORMAT:**
🎬 **SUBJECT**: [Subject from image with added movement capabilities]
🏞️ **SCENE**: [Scene from image enhanced for video]
🎭 **MOTION**: [Natural movements and camera work to bring image to life]

**Technical Recommendations:**
📹 Recommended FPS: 24-30 fps for smooth motion
⏱️ Optimal Duration: 3-5 seconds for best quality
🎞️ Frame Count: 72-150 frames (24fps × 3-5 seconds)

Transform static concepts into engaging motion while maintaining focus.""",
                "examples": [
                    "Convert this image prompt into a video: 'Sensual portrait of a woman in lingerie'",
                    "Transform a bedroom scene into dynamic video content",
                    "Add motion to a boudoir photography concept",
                    "Create video dynamics from a romantic portrait",
                    "Convert static intimate scene to flowing video"
                ]
            },

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

        # Diverse appearance descriptions for random generation
        self.appearance_descriptors = {
            "age_ranges": ["18-22", "23-27", "28-32", "33-37", "38-42", "43-47", "48+"],
            "ethnicities": ["Caucasian", "Asian", "Latina", "African American", "Middle Eastern", "Mixed heritage", "Mediterranean", "Nordic"],
            "hair_colors": ["blonde", "brunette", "black", "red", "auburn", "platinum", "honey blonde", "dark brown"],
            "hair_styles": ["long and wavy", "short and sleek", "curly", "straight", "bob cut", "pixie cut", "braided", "updo"],
            "eye_colors": ["blue", "brown", "green", "hazel", "gray", "amber", "dark brown", "light blue"],
            "body_types": ["petite", "athletic", "curvy", "slim", "tall", "average height", "hourglass figure", "pear-shaped"],
            "skin_tones": ["fair", "olive", "tan", "dark", "porcelain", "golden", "bronze", "caramel"]
        }

    def generate_diverse_appearance(self) -> str:
        """Generate a diverse appearance description."""
        age = random.choice(self.appearance_descriptors["age_ranges"])
        ethnicity = random.choice(self.appearance_descriptors["ethnicities"])
        hair_color = random.choice(self.appearance_descriptors["hair_colors"])
        hair_style = random.choice(self.appearance_descriptors["hair_styles"])
        eye_color = random.choice(self.appearance_descriptors["eye_colors"])
        body_type = random.choice(self.appearance_descriptors["body_types"])
        skin_tone = random.choice(self.appearance_descriptors["skin_tones"])
        
        return f"{age} year old {ethnicity} woman with {hair_color}, {hair_style} hair, {eye_color} eyes, {body_type} build, and {skin_tone} skin"

    def save_conversation_to_txt(self, prompt: str, response: str, content_type: str):
        """Save conversation entry to timestamped txt file."""
        try:
            timestamp = datetime.now()
            date_str = timestamp.strftime("%Y-%m-%d")
            time_str = timestamp.strftime("%H:%M:%S")
            
            filename = f"{self.history_folder}/conversation_{date_str}.txt"
            
            with open(filename, "a", encoding="utf-8") as f:
                f.write(f"\\n{'='*80}\\n")
                f.write(f"Timestamp: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}\\n")
                f.write(f"Content Type: {content_type}\\n")
                f.write(f"Model: {getattr(self, 'current_model', 'Unknown')}\\n")
                f.write(f"{'='*80}\\n")
                f.write(f"USER PROMPT:\\n{prompt}\\n")
                f.write(f"\\nAI RESPONSE:\\n{response}\\n")
                
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



    def generate_response(self, prompt: str, content_type: str, main_influencer: str, temperature: float = 0.7, progress=gr.Progress()) -> str:
        """Generate response using the loaded model."""
        if not self.llm:
            return "❌ No model loaded. Please load a model first."
        
        try:
            print(f"🤖 Generating {content_type} response...")
            progress(0.1, "Preparing prompt...")
            
            # Get template for content type
            template_info = self.content_templates.get(content_type, self.content_templates["image_prompt"])
            system_prompt = template_info["system"]
            
            # Handle Main Influencer description
            final_prompt = prompt
            if main_influencer.strip():
                # If Main Influencer is provided, incorporate it into the prompt
                final_prompt = f"Main Influencer: {main_influencer.strip()}\\\\n\\\\nRequest: {prompt}"
            else:
                # If Main Influencer is empty, generate diverse appearance for image/video prompts
                if content_type in ["image_prompt", "video_prompt", "image_to_video"]:
                    diverse_appearance = self.generate_diverse_appearance()
                    final_prompt = f"Subject appearance: {diverse_appearance}\\\\n\\\\nRequest: {prompt}"
            
            # Format the full prompt based on model type
            model_config = self.model_configs[self.current_model]
            
            # Handle different template formats
            if "{system}" in model_config["template"]:
                # Modern format with separate system and user prompts
                full_prompt = model_config["template"].format(
                    system=system_prompt,
                    prompt=final_prompt
                )
            else:
                # Legacy format - combine system and user prompt
                full_prompt = model_config["template"].format(
                    prompt=f"{system_prompt}\\\\n\\\\nUser Request: {final_prompt}"
                )
            
            progress(0.3, "Generating response...")
            
            # Generate response
            response = self.llm(
                full_prompt,
                max_tokens=1024,
                temperature=temperature,
                top_p=0.9,
                stop=["<|eot_id|>", "<|im_end|>", "</s>", "[/INST]", "USER:", "ASSISTANT:", "\\\\n\\\\nUSER:", "\\\\n\\\\nASSISTANT:"],
                echo=False
            )
            
            generated_text = response["choices"][0]["text"].strip()
            
            progress(0.9, "Finalizing response...")
            
            # Save to conversation history
            conversation_entry = {
                "timestamp": datetime.now().isoformat(),
                "prompt": prompt,
                "main_influencer": main_influencer,
                "response": generated_text,
                "content_type": content_type,
                "model": self.current_model,
                "temperature": temperature
            }
            self.conversation_history.append(conversation_entry)
            
            # Save to text file
            self.save_conversation_to_txt(f"Main Influencer: {main_influencer}\\\\n\\\\n{prompt}", generated_text, content_type)
            
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
chatbot = InfluencerChatbot()

def load_model_interface(model_name: str, gpu_layers: int, progress=gr.Progress()) -> str:
    """Interface function for loading models."""
    return chatbot.load_model(model_name, gpu_layers, progress)


def generate_content(prompt: str, content_type: str, main_influencer: str, temperature: float, progress=gr.Progress()) -> str:
    """Interface function for generating content."""
    if not prompt.strip():
        return "Please enter a prompt."
    
    return chatbot.generate_response(prompt, content_type, main_influencer, temperature, progress)

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
    """Analyze prompt complexity and automatically suggest optimal SD Forge parameters."""
    if not prompt.strip():
        return "Please enter a prompt to analyze.", ""
    
    try:
        # Analyze complexity
        analysis = chatbot.complexity_analyzer.analyze_prompt_complexity(prompt)
        
        # Get automatically recommended SD Forge parameters
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
                analysis_text += f"- {category.title()}: {count} terms\\\\n"
        
        # Format SD Forge parameters in Flux format with automatic selections
        params_text = f"""⚙️ **Automatically Recommended Stable Diffusion Forge Parameters (Flux Format)**

**Parameters String:**
Steps: {sd_params.steps}, Sampler: {sd_params.sampler}, Schedule type: {sd_params.schedule_type}, CFG scale: {sd_params.cfg_scale}, Distilled CFG Scale: {sd_params.distilled_cfg_scale}, Seed: {sd_params.seed}, Size: {sd_params.width}x{sd_params.height}

**Individual Settings:**
- Steps: {sd_params.steps} (automatically optimized for complexity)
- Sampler: {sd_params.sampler} (automatically selected based on prompt analysis)
- Schedule Type: {sd_params.schedule_type} (automatically chosen for optimal quality)
- CFG Scale: {sd_params.cfg_scale} (always 1 for Flux)
- Distilled CFG Scale: {sd_params.distilled_cfg_scale} (automatically tuned for prompt complexity)
- Seed: {sd_params.seed} (use -1 for random)
- Size: {sd_params.width}x{sd_params.height} (automatically determined from prompt content)

**Explanation:** Based on your prompt's complexity level ({analysis['complexity_level']}), these parameters are automatically optimized for Flux model generation. The system analyzes your prompt's technical complexity, detail level, and content to select the most appropriate sampler, scheduler, and distilled CFG scale for optimal results."""
        
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
    filename = f"influencer_conversation_{timestamp}.json"
    
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
        title="Influencer Chatbot - Advanced AI Assistant",
        theme=gr.themes.Soft(),
        css="""
        .main-header { text-align: center; margin-bottom: 2rem; }
        .model-info { background: #f0f0f0; padding: 1rem; border-radius: 8px; margin: 1rem 0; }
        .status-box { background: #e8f5e8; padding: 0.5rem; border-radius: 4px; }
        .error-box { background: #ffe8e8; padding: 0.5rem; border-radius: 4px; }
        .warning-box { 
            background: #fff3cd; 
            border: 1px solid #ffeaa7; 
            padding: 1rem; 
            border-radius: 8px; 
            margin: 1rem 0; 
            color: #856404 !important;
        }
        
        /* Dark theme support */
        .dark .warning-box {
            background: #664d03 !important;
            border: 1px solid #b08900 !important;
            color: #fff3cd !important;
        }
        
        .dark .model-info {
            background: #374151 !important;
            color: #f9fafb !important;
        }
        
        /* Fix dropdown and text visibility in dark theme */
        .dark .gr-dropdown,
        .dark .gr-textbox,
        .dark .gr-button {
            color: #f9fafb !important;
        }
        
        .dark .gr-dropdown .gr-box {
            background: #374151 !important;
            border-color: #6b7280 !important;
        }
        
        /* Available models text visibility */
        .dark h3,
        .dark p,
        .dark li {
            color: #f9fafb !important;
        }
        """
    ) as interface:
        
        gr.HTML("""
        <div class="main-header">
            <h1>🔥 Influencer Chatbot - Advanced AI Assistant</h1>
            <p>Professional AI assistant for adult content creation with multiple models, image analysis, and comprehensive content generation</p>
        </div>
        """)
        
        # Warning about adult content
        gr.HTML("""
        <div class="warning-box">
            <h3>⚠️ Adult Content Warning</h3>
            <p>This application is designed for creating adult content (18+). It contains uncensored AI models and explicit content generation capabilities. By using this application, you confirm that you are 18 years or older and consent to viewing and creating adult content.</p>
        </div>
        """)
        
        with gr.Tabs():
            # Model Management Tab
            with gr.Tab("🤖 Model Management"):
                gr.HTML("""
                <div class="model-info">
                    <h3>Available Uncensored Models</h3>
                    <p>Specialized uncensored models optimized for adult content creation:</p>
                    <ul>
                        <li><strong>Orenguteng/Llama-3-8B-Lexi-Uncensored</strong> - Lexi uncensored 8B model, specialized for adult content</li>
                        <li><strong>aifeifei798/DarkIdol-Llama-3.1-8B-Instruct-1.2-Uncensored</strong> - DarkIdol uncensored 8B model, enhanced for creative content</li>
                        <li><strong>DevsDoCode/LLama-3-8b-Uncensored</strong> - DevsDoCode uncensored 8B model, optimized for unrestricted content</li>
                        <li><strong>Luna-AI-Llama2-Uncensored</strong> - Efficient 7B uncensored model, good for most adult content tasks</li>
                        <li><strong>WizardLM-13B-Uncensored</strong> - Balanced 13B uncensored model, high quality responses</li>
                        <li><strong>Wizard-Vicuna-30B-Uncensored</strong> - High-end 30B uncensored model, best quality (requires powerful hardware)</li>
                        <li><strong>Nous-Hermes-13B-Uncensored</strong> - Creative 13B uncensored model, excellent for roleplay and creative content</li>
                    </ul>
                </div>
                """)
                
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
                            label="GPU Layers"
                        )
                        
                        load_model_btn = gr.Button("🔄 Load Model", variant="primary")
                        model_status = gr.Textbox(label="Model Status", interactive=False)

            
            # Content Generation Tab
            with gr.Tab("✨ Content Generation"):
                with gr.Row():
                    with gr.Column(scale=2):
                        content_type = gr.Dropdown(
                            choices=["image_prompt", "video_prompt", "image_to_video"],
                            value="image_prompt",
                            label="Content Type"
                        )
                        
                        main_influencer = gr.Textbox(
                            label="Main Influencer Description (Optional)",
                            placeholder="e.g., 25-year-old blonde woman with blue eyes, athletic build..."
                        )
                        
                        user_prompt = gr.Textbox(
                            label="Your Prompt",
                            placeholder="Describe what you want to create...",
                            lines=4
                        )
                        
                        with gr.Row():
                            temperature_slider = gr.Slider(
                                minimum=0.1,
                                maximum=1.5,
                                value=0.7,
                                step=0.1,
                                label="Creativity (Temperature)"
                            )
                        
                        with gr.Row():
                            generate_btn = gr.Button("✨ Generate Content", variant="primary", size="lg")
                            example_btn = gr.Button("🎲 Get Example", variant="secondary")
                    
                    with gr.Column(scale=1):
                        gr.HTML("""
                        <div class="model-info">
                            <h3>💡 Content Types</h3>
                            <ul>
                                <li><strong>Image Prompt:</strong> Detailed prompts for image generation</li>
                                <li><strong>Video Prompt:</strong> WAN 2.1 format prompts for video creation (24-30 fps, 3-5 seconds)</li>
                                <li><strong>Image to Video:</strong> Convert image descriptions to video prompts</li>
                            </ul>
                        </div>
                        """)
                
                generated_content = gr.Textbox(
                    label="Generated Content",
                    lines=15,
                    interactive=False
                )
            
            # Image Analysis Tab
            with gr.Tab("🖼️ Image Analysis"):
                with gr.Row():
                    with gr.Column():
                        uploaded_image = gr.Image(
                            label="Upload Image for Analysis",
                            type="pil"
                        )
                        
                        caption_model_dropdown = gr.Dropdown(
                            choices=list(chatbot.caption_models.keys()),
                            value="BLIP",
                            label="Caption Model"
                        )
                        
                        analyze_image_btn = gr.Button("🔍 Analyze Image", variant="primary")
                        
                    with gr.Column():
                        image_caption = gr.Textbox(
                            label="Image Caption",
                            lines=3,
                            interactive=False
                        )
                        
                        image_description = gr.Textbox(
                            label="Detailed Description",
                            lines=8,
                            interactive=False
                        )
                
                # Image to Video Section
                gr.HTML("<hr><h3>🎬 Image to Video Conversion</h3>")
                
                with gr.Row():
                    with gr.Column():
                        video_request = gr.Textbox(
                            label="Video Request",
                            placeholder="Describe how you want to animate this image...",
                            lines=3
                        )
                        
                        generate_video_prompt_btn = gr.Button("🎬 Generate Video Prompt", variant="primary")
                    
                    with gr.Column():
                        video_prompt_output = gr.Textbox(
                            label="Generated Video Prompt",
                            lines=10,
                            interactive=False
                        )
            
            # Prompt Analysis Tab
            with gr.Tab("🔍 Prompt Analysis"):
                gr.HTML("""
                <div class="model-info">
                    <h3>🎯 Automatic SD Forge Parameter Optimization</h3>
                    <p>Analyze your prompts and get automatically optimized Stable Diffusion Forge parameters for Flux models. The system intelligently selects the best sampler, scheduler, and settings based on your prompt's complexity and content.</p>
                </div>
                """)
                
                with gr.Row():
                    with gr.Column():
                        analysis_prompt = gr.Textbox(
                            label="Prompt to Analyze",
                            placeholder="Enter your image generation prompt here...",
                            lines=5
                        )
                        
                        analyze_prompt_btn = gr.Button("🔍 Analyze & Optimize", variant="primary", size="lg")
                
                with gr.Row():
                    with gr.Column():
                        complexity_analysis = gr.Textbox(
                            label="Complexity Analysis",
                            lines=12,
                            interactive=False
                        )
                    
                    with gr.Column():
                        sd_parameters = gr.Textbox(
                            label="Automatically Optimized SD Forge Parameters",
                            lines=12,
                            interactive=False
                        )
            
            # Conversation History Tab
            with gr.Tab("📚 History"):
                with gr.Row():
                    clear_history_btn = gr.Button("🗑️ Clear History", variant="secondary")
                    export_history_btn = gr.Button("💾 Export History", variant="primary")
                
                history_status = gr.Textbox(
                    label="History Status",
                    interactive=False
                )
                
                gr.HTML("""
                <div class="model-info">
                    <h3>📁 Conversation Logs</h3>
                    <p>All conversations are automatically saved to timestamped text files in the 'conversation_logs' folder. You can also export your conversation history as JSON for backup or analysis.</p>
                </div>
                """)
        
        # Event handlers
        load_model_btn.click(
            fn=load_model_interface,
            inputs=[model_dropdown, gpu_layers_slider],
            outputs=[model_status]
        )
        
        generate_btn.click(
            fn=generate_content,
            inputs=[user_prompt, content_type, main_influencer, temperature_slider],
            outputs=[generated_content]
        )
        
        example_btn.click(
            fn=get_example,
            inputs=[content_type],
            outputs=[user_prompt]
        )
        
        analyze_image_btn.click(
            fn=analyze_uploaded_image,
            inputs=[uploaded_image, caption_model_dropdown],
            outputs=[image_caption, image_description]
        )
        
        generate_video_prompt_btn.click(
            fn=generate_video_prompt_from_image,
            inputs=[uploaded_image, video_request],
            outputs=[video_prompt_output]
        )
        
        analyze_prompt_btn.click(
            fn=analyze_prompt_complexity,
            inputs=[analysis_prompt],
            outputs=[complexity_analysis, sd_parameters]
        )
        
        clear_history_btn.click(
            fn=clear_conversation,
            outputs=[history_status]
        )
        
        export_history_btn.click(
            fn=export_conversation,
            outputs=[history_status]
        )
    
    return interface

if __name__ == "__main__":
    print("🚀 Starting Influencer Chatbot...")
    print("📋 Available models:", list(chatbot.model_configs.keys()))
    
    # Create and launch interface
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        show_error=True,
        debug=False
    )
