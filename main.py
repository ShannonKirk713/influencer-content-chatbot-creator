#!/usr/bin/env python3
"""
Influencer Chatbot - Advanced AI Assistant for Adult Content Creation
Enhanced with intelligent prompt analysis, advanced image analysis, auto-open webpage, and dynamic parameter optimization
"""

import gradio as gr
import json
import os
import logging
import webbrowser
from threading import Timer
from datetime import datetime
from typing import List, Tuple, Optional
from PIL import Image
import random

# Import our modules
from image_analyzer import image_analyzer
from sd_forge_utils import PromptComplexityAnalyzer, SDForgeParams

# Import the new intelligent prompt analysis system
from prompt_analyzer import IntelligentPromptAnalyzer, ComplexityLevel, ContentType
from api_wrapper import PromptAnalysisAPI

# Enhanced image analysis modules (CLIP, YOLO, SAM, GPT-4V)
try:
    from vision.analyze import ImageAnalyzer
    from vision.generate_video import VideoGenerator
    from utils.args_compat import get_compatible_args
    ENHANCED_VISION_AVAILABLE = True
except ImportError:
    ENHANCED_VISION_AVAILABLE = False
    print("⚠️ Enhanced vision modules not available. Basic image analysis will be used.")

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

def auto_open_browser():
    """Auto-open browser functionality"""
    try:
        webbrowser.open('http://127.0.0.1:7861')
        print("🌐 Browser opened automatically at http://127.0.0.1:7861")
    except Exception as e:
        print(f"⚠️ Could not auto-open browser: {e}")

class InfluencerChatbot:
    """Main chatbot class with model management and content generation."""
    
    def __init__(self):
        self.llm = None
        self.current_model = None
        self.conversation_history = []
        self.history_folder = "conversation_logs"
        
        # Initialize intelligent prompt analysis system
        self.intelligent_analyzer = IntelligentPromptAnalyzer()
        self.analysis_api = PromptAnalysisAPI()
        
        # Keep legacy analyzer for backward compatibility
        self.complexity_analyzer = PromptComplexityAnalyzer()
        
        # Initialize enhanced vision components if available
        if ENHANCED_VISION_AVAILABLE:
            self.enhanced_image_analyzer = ImageAnalyzer()
            self.video_generator = VideoGenerator()
        
        # Create history folder
        os.makedirs(self.history_folder, exist_ok=True)
        
        # Available models configuration - ALL THEBLOKE MODELS
        self.model_configs = {
            # TheBloke uncensored models
            "Luna-AI-Llama2-Uncensored": {
                "repo_id": "TheBloke/Luna-AI-Llama2-Uncensored-GGUF",
                "filename": "luna-ai-llama2-uncensored.Q4_K_M.gguf",
                "template": "USER: {prompt}\\nASSISTANT:",
                "description": "Efficient 7B uncensored model, good for most adult content tasks"
            },
            "WizardLM-13B-Uncensored": {
                "repo_id": "TheBloke/WizardLM-13B-Uncensored-GGUF", 
                "filename": "wizardlm-13b-uncensored.Q4_K_M.gguf",
                "template": "You are a helpful AI assistant.\\n\\nUSER: {prompt}\\nASSISTANT:",
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
                "template": "### Instruction:\\n{prompt}\\n\\n### Response:",
                "description": "Creative 13B uncensored model, excellent for roleplay and creative content"
            },
            # Modern TheBloke models
            "Llama-3.2-3B-Instruct": {
                "repo_id": "TheBloke/Llama-3.2-3B-Instruct-GGUF",
                "filename": "llama-3.2-3b-instruct-q4_k_m.gguf",
                "template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\\n\\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\\n\\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n",
                "description": "Fast 3B model, good for quick responses"
            },
            "Llama-3.1-8B-Instruct": {
                "repo_id": "TheBloke/Llama-3.1-8B-Instruct-GGUF",
                "filename": "llama-3.1-8b-instruct-q4_k_m.gguf",
                "template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\\n\\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\\n\\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n",
                "description": "Balanced 8B model, good quality and speed"
            },
            "Qwen2.5-14B-Instruct": {
                "repo_id": "TheBloke/Qwen2.5-14B-Instruct-GGUF",
                "filename": "qwen2.5-14b-instruct-q4_k_m.gguf",
                "template": "<|im_start|>system\\n{system}<|im_end|>\\n<|im_start|>user\\n{prompt}<|im_end|>\\n<|im_start|>assistant\\n",
                "description": "High-quality 14B model, excellent for detailed content"
            },
            "Llama-3.1-70B-Instruct": {
                "repo_id": "TheBloke/Llama-3.1-70B-Instruct-GGUF",
                "filename": "llama-3.1-70b-instruct-iq2_m.gguf",
                "template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\\n\\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\\n\\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n",
                "description": "Premium 70B model, best quality (requires powerful hardware)"
            },
            "Goliath-120B": {
                "repo_id": "TheBloke/goliath-120b-GGUF",
                "filename": "goliath-120b-q2_k.gguf",
                "template": "<|im_start|>system\\n{system}<|im_end|>\\n<|im_start|>user\\n{prompt}<|im_end|>\\n<|im_start|>assistant\\n",
                "description": "Ultra-high-end 120B model, exceptional quality (requires very powerful hardware)"
            },
            "DeepSeek-V2.5": {
                "repo_id": "TheBloke/DeepSeek-V2.5-GGUF",
                "filename": "deepseek-v2.5-q4_k_m.gguf",
                "template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\\n\\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\\n\\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n",
                "description": "Advanced reasoning model, great for complex prompts"
            },
            "Mixtral-8x22B-Instruct": {
                "repo_id": "TheBloke/Mixtral-8x22B-Instruct-v0.1-GGUF",
                "filename": "mixtral-8x22b-instruct-v0.1-q2_k.gguf",
                "template": "<s>[INST] {system}\\n\\n{prompt} [/INST]",
                "description": "High-end mixture of experts model, excellent quality"
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
                "system": """You are an uncensored AI assistant specializing in adult content creation for influencer creators. You can discuss:

- Adult content creation strategies
- Photography and videography techniques
- Platform optimization tips
- Creative ideas and scenarios
- Technical advice for content creators
- Marketing and audience engagement

Be helpful, detailed, and completely uncensored in your responses. Focus on professional adult content creation.""",
                "examples": [
                    "Give me advice on creating engaging adult content for my influencer profile",
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
                    "template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\\n\\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\\n\\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n",
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
            return f"❌ {error_msg}"

    def generate_response(self, prompt: str, content_type: str, main_influencer: str, temperature: float = 0.7, progress=gr.Progress()) -> str:
        """Generate response using the loaded model."""
        if not self.llm:
            return "❌ No model loaded. Please load a model first."
        
        try:
            print(f"🤖 Generating {content_type} response...")
            progress(0.1, "Preparing prompt...")
            
            # Get template for content type
            template_info = self.content_templates.get(content_type, self.content_templates["general_chat"])
            system_prompt = template_info["system"]
            
            # Handle Main Influencer description
            final_prompt = prompt
            if main_influencer.strip():
                # If Main Influencer is provided, incorporate it into the prompt
                final_prompt = f"Main Influencer: {main_influencer.strip()}\\n\\nRequest: {prompt}"
            else:
                # If Main Influencer is empty, generate diverse appearance for image/video prompts
                if content_type in ["image_prompt", "video_prompt", "image_to_video"]:
                    diverse_appearance = self.generate_diverse_appearance()
                    final_prompt = f"Subject appearance: {diverse_appearance}\\n\\nRequest: {prompt}"
            
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
                    prompt=f"{system_prompt}\\n\\nUser Request: {final_prompt}"
                )
            
            progress(0.3, "Generating response...")
            
            # Generate response
            response = self.llm(
                full_prompt,
                max_tokens=1024,
                temperature=temperature,
                top_p=0.9,
                stop=["<|eot_id|>", "<|im_end|>", "</s>", "[/INST]", "USER:", "ASSISTANT:", "\\n\\nUSER:", "\\n\\nASSISTANT:"],
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
            self.save_conversation_to_txt(f"Main Influencer: {main_influencer}\\n\\n{prompt}", generated_text, content_type)
            
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

def download_model_interface(repo_id: str, filename: str, progress=gr.Progress()) -> str:
    """Interface function for downloading additional models."""
    if not repo_id.strip() or not filename.strip():
        return "❌ Please provide both repository ID and filename"
    return chatbot.download_additional_model(repo_id, filename, progress)

def generate_content(prompt: str, content_type: str, main_influencer: str, temperature: float, progress=gr.Progress()) -> str:
    """Interface function for generating content."""
    if not prompt.strip():
        return "Please enter a prompt."
    
    return chatbot.generate_response(prompt, content_type, main_influencer, temperature, progress)

def analyze_uploaded_image(image: Image.Image, caption_model: str, progress=gr.Progress()) -> Tuple[str, str]:
    """
    Analyze uploaded image and return caption and detailed description.
    Enhanced with CLIP, YOLO, SAM, and GPT-4V analysis when available.
    """
    if image is None:
        return "❌ No image uploaded", ""
    
    try:
        print(f"🖼️ Analyzing image with {caption_model} model...")
        progress(0.1, f"Loading {caption_model} model...")
        
        # Use enhanced analysis if available
        if ENHANCED_VISION_AVAILABLE:
            # Save image temporarily for enhanced analysis
            temp_path = "/tmp/temp_image.jpg"
            image.save(temp_path)
            
            # Stage 1: Comprehensive analysis with CLIP, YOLO, SAM, GPT-4V
            analysis_result = chatbot.enhanced_image_analyzer.analyze_comprehensive(temp_path)
            
            progress(0.8, "Processing enhanced analysis...")
            
            # Format results
            caption = analysis_result.get('caption', 'No caption available')
            detailed_description = analysis_result.get('detailed_description', 'No detailed description available')
            
            # Add enhanced analysis details
            if 'clip_analysis' in analysis_result:
                detailed_description += f"\n\n🔍 CLIP Analysis: {analysis_result['clip_analysis']}"
            if 'yolo_detections' in analysis_result:
                detailed_description += f"\n\n🎯 YOLO Detections: {analysis_result['yolo_detections']}"
            if 'sam_segments' in analysis_result:
                detailed_description += f"\n\n🎨 SAM Segments: {analysis_result['sam_segments']}"
            if 'gpt4v_analysis' in analysis_result:
                detailed_description += f"\n\n🧠 GPT-4V Analysis: {analysis_result['gpt4v_analysis']}"
            
            # Clean up temp file
            os.remove(temp_path)
            
        else:
            # Fallback to basic analysis
            result = image_analyzer.analyze_image(image)
            
            if result["success"]:
                caption = result["caption"]
                detailed_description = result["detailed_description"]
            else:
                error_msg = f"❌ Image analysis failed: {result['error']}"
                print(error_msg)
                return error_msg, ""
        
        progress(1.0, "Analysis complete!")
        print(f"✅ Image analysis completed")
        
        return caption, detailed_description
        
    except Exception as e:
        error_msg = f"❌ Error analyzing image: {str(e)}"
        print(error_msg)
        return error_msg, ""

def generate_video_prompt_from_image(image: Image.Image, user_request: str, progress=gr.Progress()) -> str:
    """Generate video prompt from uploaded image with enhanced two-stage process."""
    if image is None:
        return "❌ No image uploaded"
    
    try:
        print("🎬 Generating video prompt from image...")
        progress(0.2, "Analyzing image...")
        
        if ENHANCED_VISION_AVAILABLE:
            # Enhanced two-stage process
            temp_path = "/tmp/temp_image_video.jpg"
            image.save(temp_path)
            
            # Stage 1: Comprehensive image analysis
            analysis_result = chatbot.enhanced_image_analyzer.analyze_comprehensive(temp_path)
            
            progress(0.6, "Generating dynamic video prompt...")
            
            # Stage 2: Generate dynamic video prompt with CFG=1
            video_prompt = chatbot.enhanced_image_analyzer.generate_video_prompt(analysis_result)
            
            # Add user request if provided
            if user_request.strip():
                video_prompt += f"\n\n🎯 User Requirements: {user_request}"
            
            # Clean up temp file
            os.remove(temp_path)
            
        else:
            # Fallback to basic analysis
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

def analyze_prompt_complexity_intelligent(prompt: str, sampler: str, scheduler: str, distilled_cfg: float) -> Tuple[str, str]:
    """Analyze prompt complexity using the new intelligent system with automatic CFG=1."""
    if not prompt.strip():
        return "Please enter a prompt to analyze.", ""
    
    try:
        # Use the new intelligent analysis system
        analysis_result = chatbot.analysis_api.analyze(prompt)
        parameters_result = chatbot.analysis_api.get_optimal_parameters(
            prompt, sampler, scheduler, 1.0  # CFG always set to 1
        )
        
        # Override CFG to always be 1
        parameters_result['cfg_scale'] = 1.0
        parameters_result['distilled_cfg_scale'] = distilled_cfg
        
        # Format analysis results
        complexity = analysis_result['complexity']
        content = analysis_result['content']
        recommendations = analysis_result['recommendations']
        
        analysis_text = f"""🔍 **Intelligent Prompt Analysis (Enhanced)**

**Overall Complexity:** {complexity['level'].title()} ({complexity['score']}/100)

**Content Analysis:**
- Content Type: {content['type'].title()}
- Word Count: {complexity['word_count']}
- Technical Score: {complexity['technical_score']}
- Detail Score: {complexity['detail_score']}
- Modifier Score: {complexity['modifier_score']}

**Content Elements Found:**
- Style Indicators: {', '.join(content['style_indicators']) if content['style_indicators'] else 'None detected'}
- Quality Terms: {', '.join(content['quality_indicators']) if content['quality_indicators'] else 'None detected'}
- Composition Elements: {', '.join(content['composition_elements']) if content['composition_elements'] else 'None detected'}
- Lighting Terms: {', '.join(content['lighting_terms']) if content['lighting_terms'] else 'None detected'}

**Technical Categories:**"""
        
        for category, count in analysis_result['technical_categories'].items():
            if count > 0:
                analysis_text += f"\n- {category.title()}: {count} terms"
        
        # Format parameters with automatic CFG=1
        params_text = f"""⚙️ **Optimized Generation Parameters (Enhanced)**

**Flux Model Parameters:**
Steps: {parameters_result['steps']}, Sampler: {parameters_result['sampler']}, Schedule type: {parameters_result['scheduler']}, CFG scale: {parameters_result['cfg_scale']}, Distilled CFG Scale: {parameters_result['distilled_cfg_scale']}, Seed: {parameters_result['seed']}, Size: {parameters_result['width']}x{parameters_result['height']}

**Individual Settings:**
- Steps: {parameters_result['steps']} (dynamically calculated)
- Sampler: {parameters_result['sampler']} (user selected)
- Schedule Type: {parameters_result['scheduler']} (user selected)
- CFG Scale: {parameters_result['cfg_scale']} (automatically set to 1.0)
- Distilled CFG Scale: {parameters_result['distilled_cfg_scale']} (user selected)
- Seed: {parameters_result['seed']} (use -1 for random)
- Size: {parameters_result['width']}x{parameters_result['height']} (optimized for content type)
- Guidance Scale: {parameters_result['guidance_scale']} (for advanced models)
- Negative Prompt Strength: {parameters_result['negative_prompt_strength']}

**Enhanced Analysis:** This system automatically sets CFG=1 for maximum creative freedom and uses advanced image analysis with CLIP, YOLO, SAM, and GPT-4V when available. Parameters are dynamically optimized based on prompt complexity ({complexity['level']}) and content type ({content['type']})."""
        
        return analysis_text, params_text
        
    except Exception as e:
        error_msg = f"❌ Error in intelligent analysis: {str(e)}"
        logger.error(error_msg)
        return error_msg, ""

def analyze_prompt_complexity_legacy(prompt: str, sampler: str, scheduler: str, distilled_cfg: float) -> Tuple[str, str]:
    """Legacy prompt analysis for backward compatibility."""
    if not prompt.strip():
        return "Please enter a prompt to analyze.", ""
    
    try:
        # Analyze complexity
        analysis = chatbot.complexity_analyzer.analyze_prompt_complexity(prompt)
        
        # Get SD Forge recommendations with custom parameters
        sd_params = chatbot.complexity_analyzer.recommend_sd_forge_params(prompt)
        
        # Override with user selections and force CFG=1
        sd_params.sampler = sampler
        sd_params.schedule_type = scheduler
        sd_params.cfg_scale = 1.0  # Always set to 1
        sd_params.distilled_cfg_scale = distilled_cfg
        
        # Format analysis results
        analysis_text = f"""🔍 **Legacy Prompt Complexity Analysis**

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
        
        # Format SD Forge parameters in Flux format with user selections
        params_text = f"""⚙️ **Legacy SD Forge Parameters (Flux Format)**

**Parameters String:**
Steps: {sd_params.steps}, Sampler: {sd_params.sampler}, Schedule type: {sd_params.schedule_type}, CFG scale: {sd_params.cfg_scale}, Distilled CFG Scale: {sd_params.distilled_cfg_scale}, Seed: {sd_params.seed}, Size: {sd_params.width}x{sd_params.height}

**Individual Settings:**
- Steps: {sd_params.steps} (static: 55)
- Sampler: {sd_params.sampler} (user selected)
- Schedule Type: {sd_params.schedule_type} (user selected)
- CFG Scale: {sd_params.cfg_scale} (automatically set to 1.0)
- Distilled CFG Scale: {sd_params.distilled_cfg_scale} (user selected)
- Seed: {sd_params.seed} (use -1 for random)
- Size: {sd_params.width}x{sd_params.height}

**Note:** This is the legacy analysis system with static parameters and automatic CFG=1. Use the Intelligent Analysis for dynamic optimization with enhanced vision models."""
        
        return analysis_text, params_text
        
    except Exception as e:
        error_msg = f"❌ Error analyzing prompt: {str(e)}"
        return error_msg, ""

def get_compatible_args() -> str:
    """Get compatible arguments documentation."""
    if ENHANCED_VISION_AVAILABLE:
        try:
            args_list = get_compatible_args()
            return f"✅ Enhanced vision modules loaded. Compatible arguments:\n\n{args_list}"
        except Exception as e:
            return f"⚠️ Error getting compatible args: {str(e)}"
    else:
        return "⚠️ Enhanced vision modules not available. Basic functionality only."

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

def get_gpu_status():
    """Get GPU detection and status information."""
    try:
        import subprocess
        import platform
        
        gpu_info = []
        
        # Try to get NVIDIA GPU info
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,temperature.gpu', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for i, line in enumerate(lines):
                    if line.strip():
                        parts = line.split(', ')
                        if len(parts) >= 4:
                            name, total_mem, used_mem, temp = parts[:4]
                            gpu_info.append(f"🎮 GPU {i}: {name}")
                            gpu_info.append(f"   Memory: {used_mem}MB / {total_mem}MB used")
                            gpu_info.append(f"   Temperature: {temp}°C")
                        else:
                            gpu_info.append(f"🎮 GPU {i}: {line}")
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            pass
        
        # If no NVIDIA GPUs found, check for other GPUs
        if not gpu_info:
            try:
                # Try lspci for Linux
                if platform.system() == "Linux":
                    result = subprocess.run(['lspci'], capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        lines = result.stdout.lower()
                        if 'vga' in lines or 'display' in lines or '3d' in lines:
                            gpu_info.append("🎮 Non-NVIDIA GPU detected (limited CUDA support)")
                            for line in result.stdout.split('\n'):
                                if any(term in line.lower() for term in ['vga', 'display', '3d controller']):
                                    gpu_info.append(f"   {line.strip()}")
            except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
                pass
        
        # System info
        system_info = [
            f"💻 System: {platform.system()} {platform.release()}",
            f"🏗️ Architecture: {platform.machine()}",
            f"🐍 Python: {platform.python_version()}"
        ]
        
        # Enhanced vision status
        vision_status = "✅ Enhanced vision modules loaded" if ENHANCED_VISION_AVAILABLE else "⚠️ Enhanced vision modules not available"
        system_info.append(f"🔍 Vision: {vision_status}")
        
        if gpu_info:
            return "\\n".join(system_info + [""] + gpu_info)
        else:
            return "\\n".join(system_info + ["", "⚠️ No CUDA-compatible GPUs detected", "   CPU-only mode will be used (slower performance)"])
            
    except Exception as e:
        return f"❌ Error detecting GPU: {str(e)}"

def create_interface():
    """Create and configure the Gradio interface."""
    
    with gr.Blocks(
        title="Influencer Chatbot - Enhanced AI Assistant",
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
        .enhancement-box {
            background: #d4edda;
            border: 1px solid #c3e6cb;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            color: #155724 !important;
        }
        .new-feature-box {
            background: #e7f3ff;
            border: 1px solid #b3d9ff;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            color: #004085 !important;
        }
        
        /* Dark theme support */
        .dark .warning-box {
            background: #664d03 !important;
            border: 1px solid #b08900 !important;
            color: #fff3cd !important;
        }
        
        .dark .enhancement-box {
            background: #155724 !important;
            border: 1px solid #28a745 !important;
            color: #d4edda !important;
        }
        
        .dark .new-feature-box {
            background: #1e3a8a !important;
            border: 1px solid #3b82f6 !important;
            color: #dbeafe !important;
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
            <h1>🔥 Influencer Chatbot - Enhanced AI Assistant</h1>
            <p>Professional AI assistant for adult content creation with intelligent prompt analysis, advanced image analysis (CLIP/YOLO/SAM/GPT-4V), auto-open webpage, and comprehensive content generation</p>
        </div>
        """)
        
        # Enhancement status
        enhancement_status = "✅ Enhanced vision modules loaded (CLIP, YOLO, SAM, GPT-4V)" if ENHANCED_VISION_AVAILABLE else "⚠️ Enhanced vision modules not available - using basic analysis"
        
        gr.HTML(f"""
        <div class="new-feature-box">
            <h3>🚀 NEW ENHANCEMENTS INTEGRATED</h3>
            <ul>
                <li><strong>Auto-Open Webpage:</strong> Browser automatically opens on startup</li>
                <li><strong>Automatic Prompt Analysis:</strong> CFG scale automatically set to 1.0 for maximum creative freedom</li>
                <li><strong>Advanced Image Analysis:</strong> {enhancement_status}</li>
                <li><strong>Two-Stage Image-to-Video:</strong> Enhanced process with comprehensive analysis</li>
                <li><strong>ARGS Documentation:</strong> Complete parameter compatibility guide available</li>
            </ul>
        </div>
        """)
        
        with gr.Tabs() as tabs:
            # Main Content Generation Tab
            with gr.Tab("💬 Content Generation", id="main"):
                with gr.Row():
                    with gr.Column(scale=2):
                        # Main Influencer input box
                        main_influencer_input = gr.Textbox(
                            label="👑 Main Influencer Description (Optional)",
                            placeholder="Describe the main influencer/subject for this content (e.g., '25-year-old blonde fitness model with blue eyes and athletic build'). Leave empty for diverse random generation.",
                            lines=2,
                            info="This will be used as the primary subject for all generated content. If left empty, diverse appearances will be randomly generated."
                        )
                        
                        prompt_input = gr.Textbox(
                            label="✍️ Your Prompt",
                            placeholder="Enter your content request here...",
                            lines=4
                        )
                        
                        with gr.Row():
                            content_type = gr.Dropdown(
                                choices=["image_prompt", "video_prompt", "image_to_video", "general_chat"],
                                value="image_prompt",
                                label="📝 Content Type"
                            )
                            
                            temperature = gr.Slider(
                                minimum=0.1,
                                maximum=1.0,
                                value=0.7,
                                step=0.1,
                                label="🌡️ Creativity (Temperature)"
                            )
                        
                        with gr.Row():
                            generate_btn = gr.Button("🚀 Generate Content", variant="primary", size="lg")
                            example_btn = gr.Button("💡 Get Example", variant="secondary")
                    
                    with gr.Column(scale=3):
                        output_text = gr.Textbox(
                            label="🎯 Generated Content",
                            lines=20,
                            max_lines=30,
                            show_copy_button=True
                        )
                        
                        with gr.Row():
                            clear_btn = gr.Button("🗑️ Clear History")
                            export_btn = gr.Button("📤 Export Conversation")
                        
                        status_output = gr.Textbox(label="📊 Status", lines=2)
            
            # Enhanced Image Analysis Tab
            with gr.Tab("🖼️ Enhanced Image Analysis", id="image"):
                gr.HTML(f"""
                <div class="enhancement-box">
                    <h3>🚀 Enhanced Image Analysis with CLIP, YOLO, SAM, and GPT-4V</h3>
                    <p><strong>Status:</strong> {enhancement_status}</p>
                    <p>This enhanced system provides comprehensive image analysis using multiple AI models:</p>
                    <ul>
                        <li><strong>CLIP:</strong> Image-text understanding and semantic analysis</li>
                        <li><strong>YOLO:</strong> Object detection and localization</li>
                        <li><strong>SAM:</strong> Segment Anything Model for precise segmentation</li>
                        <li><strong>GPT-4V:</strong> Advanced visual reasoning and description</li>
                    </ul>
                </div>
                """)
                
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(
                            label="📸 Upload Image",
                            type="pil",
                            height=400
                        )
                        
                        caption_model = gr.Dropdown(
                            choices=list(chatbot.caption_models.keys()),
                            value="BLIP",
                            label="🤖 Caption Model",
                            info="Select image captioning model (enhanced analysis uses all models)"
                        )
                        
                        analyze_btn = gr.Button("🔍 Analyze Image (Enhanced)", variant="primary")
                    
                    with gr.Column():
                        image_caption = gr.Textbox(
                            label="📝 Image Caption",
                            lines=3,
                            show_copy_button=True
                        )
                        
                        image_description = gr.Textbox(
                            label="📋 Enhanced Analysis Results",
                            lines=12,
                            show_copy_button=True
                        )
                
                gr.HTML("<hr>")
                
                with gr.Row():
                    with gr.Column():
                        gr.HTML("<h3>🎬 Enhanced Two-Stage Image-to-Video Generation</h3>")
                        video_request = gr.Textbox(
                            label="🎯 Video Requirements (Optional)",
                            placeholder="Specify any particular movements, actions, or video style you want...",
                            lines=2
                        )
                        
                        video_btn = gr.Button("🎬 Generate Video Prompt (Enhanced)", variant="secondary")
                    
                    with gr.Column():
                        video_prompt_output = gr.Textbox(
                            label="🎥 Enhanced Video Prompt (CFG=1)",
                            lines=15,
                            show_copy_button=True
                        )
            
            # Enhanced Prompt Analysis Tab
            with gr.Tab("⚙️ Enhanced Prompt Analysis", id="analysis"):
                gr.HTML("""
                <div class="enhancement-box">
                    <h3>🚀 Enhanced Intelligent Prompt Analysis System</h3>
                    <p>This enhanced system automatically optimizes parameters and sets CFG=1 for maximum creative freedom:</p>
                    <ul>
                        <li><strong>Automatic CFG=1:</strong> CFG scale automatically set to 1.0 (no manual selection needed)</li>
                        <li><strong>Dynamic Steps:</strong> Automatically calculates optimal steps (20-80) based on prompt complexity</li>
                        <li><strong>Smart Sampler Selection:</strong> Chooses best sampler based on content type and complexity</li>
                        <li><strong>Content-Aware:</strong> Recognizes portraits, landscapes, artistic styles, etc.</li>
                        <li><strong>Technical Analysis:</strong> Detects camera, lighting, composition terms</li>
                        <li><strong>Enhanced Vision Integration:</strong> Works with CLIP, YOLO, SAM, and GPT-4V analysis</li>
                    </ul>
                </div>
                """)
                
                with gr.Row():
                    with gr.Column():
                        analysis_prompt = gr.Textbox(
                            label="📝 Prompt to Analyze",
                            placeholder="Enter your image generation prompt here...",
                            lines=4
                        )
                        
                        with gr.Row():
                            sampler_choice = gr.Dropdown(
                                choices=["Euler a", "Euler", "DPM++ 2M", "DPM++ 2M SDE", "DPM++ 2M Karras", "DPM++ 2M SDE Karras", "DDIM", "PLMS"],
                                value="DPM++ 2M",
                                label="🎛️ Sampler"
                            )
                            
                            scheduler_choice = gr.Dropdown(
                                choices=["Automatic", "Karras", "Exponential", "Polyexponential", "SGM Uniform"],
                                value="Karras",
                                label="📅 Scheduler"
                            )
                        
                        distilled_cfg_slider = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=10,
                            step=1,
                            label="🎚️ Distilled CFG Scale"
                        )
                        
                        gr.HTML("""
                        <div class="new-feature-box">
                            <p><strong>Note:</strong> CFG Scale is automatically set to 1.0 for maximum creative freedom. This cannot be changed.</p>
                        </div>
                        """)
                        
                        with gr.Row():
                            analyze_intelligent_btn = gr.Button("🧠 Enhanced Intelligent Analysis", variant="primary")
                            analyze_legacy_btn = gr.Button("📊 Legacy Analysis", variant="secondary")
                    
                    with gr.Column():
                        complexity_output = gr.Textbox(
                            label="📊 Enhanced Analysis Results",
                            lines=15,
                            show_copy_button=True
                        )
                        
                        parameters_output = gr.Textbox(
                            label="⚙️ Optimized Parameters (CFG=1)",
                            lines=15,
                            show_copy_button=True
                        )
            
            # ARGS Documentation Tab
            with gr.Tab("📚 ARGS Documentation", id="args"):
                gr.HTML("""
                <div class="new-feature-box">
                    <h3>📚 Compatible Arguments Documentation</h3>
                    <p>Complete documentation of all compatible arguments and parameters for the enhanced system.</p>
                </div>
                """)
                
                args_btn = gr.Button("📋 Load Compatible Arguments", variant="primary")
                args_output = gr.Textbox(
                    label="📋 Compatible Arguments",
                    lines=25,
                    show_copy_button=True
                )
            
            # Model Settings Tab
            with gr.Tab("🤖 Model Settings", id="models"):
                with gr.Row():
                    with gr.Column():
                        gr.HTML("""
                        <div class="model-info">
                            <h3>🤖 Available Models</h3>
                            <p>Select and load AI models for content generation. Models are downloaded automatically from Hugging Face.</p>
                        </div>
                        """)
                        
                        model_choice = gr.Dropdown(
                            choices=list(chatbot.model_configs.keys()),
                            value="Luna-AI-Llama2-Uncensored",
                            label="🎯 Select Model",
                            info="Choose the AI model for content generation"
                        )
                        
                        gpu_layers = gr.Slider(
                            minimum=0,
                            maximum=50,
                            value=35,
                            step=1,
                            label="🎮 GPU Layers",
                            info="Number of layers to run on GPU (0 = CPU only, higher = more GPU usage)"
                        )
                        
                        load_btn = gr.Button("⬇️ Load Model", variant="primary", size="lg")
                        
                        model_status = gr.Textbox(label="📊 Model Status", lines=3)
                    
                    with gr.Column():
                        gr.HTML("""
                        <div class="model-info">
                            <h3>📥 Download Additional Models</h3>
                            <p>Download custom models from Hugging Face repositories.</p>
                        </div>
                        """)
                        
                        repo_id = gr.Textbox(
                            label="🏪 Repository ID",
                            placeholder="e.g., TheBloke/Llama-2-7B-Chat-GGUF",
                            info="Hugging Face repository ID"
                        )
                        
                        filename = gr.Textbox(
                            label="📄 Model Filename",
                            placeholder="e.g., llama-2-7b-chat.Q4_K_M.gguf",
                            info="Specific model file to download"
                        )
                        
                        download_btn = gr.Button("📥 Download Model", variant="secondary")
                        
                        download_status = gr.Textbox(label="📊 Download Status", lines=3)
                
                # Model descriptions
                gr.HTML("""
                <div class="model-info">
                    <h3>📋 Model Descriptions</h3>
                    <ul>
                        <li><strong>Luna-AI-Llama2-Uncensored (7B):</strong> Fast and efficient, good for most adult content tasks</li>
                        <li><strong>WizardLM-13B-Uncensored (13B):</strong> Balanced performance and quality</li>
                        <li><strong>Wizard-Vicuna-30B-Uncensored (30B):</strong> Highest quality, requires powerful hardware</li>
                        <li><strong>Nous-Hermes-13B-Uncensored (13B):</strong> Excellent for roleplay and creative content</li>
                        <li><strong>Llama-3.1-8B-Instruct (8B):</strong> Modern model with good balance of speed and quality</li>
                        <li><strong>Qwen2.5-14B-Instruct (14B):</strong> High-quality model for detailed content</li>
                        <li><strong>Llama-3.1-70B-Instruct (70B):</strong> Premium model, best quality (requires very powerful hardware)</li>
                    </ul>
                    <p><strong>💡 Recommendation:</strong> Start with Luna-AI for testing, upgrade to WizardLM-13B for better quality, or Qwen2.5-14B for the best balance.</p>
                </div>
                """)
        
        # Event handlers
        generate_btn.click(
            fn=generate_content,
            inputs=[prompt_input, content_type, main_influencer_input, temperature],
            outputs=[output_text]
        )
        
        example_btn.click(
            fn=get_example,
            inputs=[content_type],
            outputs=[prompt_input]
        )
        
        load_btn.click(
            fn=load_model_interface,
            inputs=[model_choice, gpu_layers],
            outputs=[model_status]
        )
        
        download_btn.click(
            fn=download_model_interface,
            inputs=[repo_id, filename],
            outputs=[download_status]
        )
        
        analyze_btn.click(
            fn=analyze_uploaded_image,
            inputs=[image_input, caption_model],
            outputs=[image_caption, image_description]
        )
        
        video_btn.click(
            fn=generate_video_prompt_from_image,
            inputs=[image_input, video_request],
            outputs=[video_prompt_output]
        )
        
        analyze_intelligent_btn.click(
            fn=analyze_prompt_complexity_intelligent,
            inputs=[analysis_prompt, sampler_choice, scheduler_choice, distilled_cfg_slider],
            outputs=[complexity_output, parameters_output]
        )
        
        analyze_legacy_btn.click(
            fn=analyze_prompt_complexity_legacy,
            inputs=[analysis_prompt, sampler_choice, scheduler_choice, distilled_cfg_slider],
            outputs=[complexity_output, parameters_output]
        )
        
        args_btn.click(
            fn=get_compatible_args,
            outputs=[args_output]
        )
        
        clear_btn.click(
            fn=clear_conversation,
            outputs=[status_output]
        )
        
        export_btn.click(
            fn=export_conversation,
            outputs=[status_output]
        )
    
    return interface

if __name__ == "__main__":
    print("🚀 Starting Enhanced Influencer Chatbot with Advanced Image Analysis...")
    print("📍 Access the interface at: http://localhost:7861")
    
    # Print GPU detection results
    print("\n" + "="*60)
    print("🎮 ENHANCED GPU DETECTION RESULTS")
    print("="*60)
    print(get_gpu_status())
    print("="*60)
    
    # Auto-open browser functionality
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        Timer(1.5, auto_open_browser).start()
    
    # Create and launch the interface
    interface = create_interface()
    interface.launch(
        server_name="127.0.0.1",
        server_port=7861,
        share=False,
        show_error=True
        # Removed deprecated enable_queue parameter
    )