#!/usr/bin/env python3
"""
Influencer Chatbot - Advanced AI Assistant for Adult Content Creation
Expanded with comprehensive LLM support, refresh functionality, and streamlined text-only interface
"""

import gradio as gr
import json
import os
import logging
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
import random
import glob
from pathlib import Path

# Import our modules
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
    """Main chatbot class with expanded model management and content generation."""
    
    def __init__(self):
        self.llm = None
        self.current_model = None
        self.conversation_history = []
        self.history_folder = "conversation_logs"
        self.complexity_analyzer = PromptComplexityAnalyzer()
        
        # Create history folder
        os.makedirs(self.history_folder, exist_ok=True)
        
        # Expanded model configurations - ALL COMPATIBLE HUGGING FACE MODELS
        self.model_configs = {
            # === LLAMA FAMILY ===
            "Llama-3.2-3B-Instruct": {
                "repo_id": "meta-llama/Llama-3.2-3B-Instruct",
                "filename": "*.gguf",
                "template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                "description": "Fast 3B Llama model, good for quick responses"
            },
            "Llama-3.1-8B-Instruct": {
                "repo_id": "meta-llama/Llama-3.1-8B-Instruct",
                "filename": "*.gguf",
                "template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                "description": "Balanced 8B Llama model, excellent quality and speed"
            },
            "Llama-3.1-70B-Instruct": {
                "repo_id": "meta-llama/Llama-3.1-70B-Instruct",
                "filename": "*.gguf",
                "template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                "description": "Premium 70B Llama model, exceptional quality (requires powerful hardware)"
            },
            "CodeLlama-34B-Instruct": {
                "repo_id": "codellama/CodeLlama-34b-Instruct-hf",
                "filename": "*.gguf",
                "template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                "description": "Code-specialized Llama model, great for technical content"
            },
            
            # === QWEN FAMILY ===
            "Qwen2.5-7B-Instruct": {
                "repo_id": "Qwen/Qwen2.5-7B-Instruct",
                "filename": "*.gguf",
                "template": "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
                "description": "High-quality 7B Qwen model, excellent for detailed content"
            },
            "Qwen2.5-14B-Instruct": {
                "repo_id": "Qwen/Qwen2.5-14B-Instruct",
                "filename": "*.gguf",
                "template": "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
                "description": "Premium 14B Qwen model, exceptional reasoning capabilities"
            },
            "Qwen2.5-32B-Instruct": {
                "repo_id": "Qwen/Qwen2.5-32B-Instruct",
                "filename": "*.gguf",
                "template": "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
                "description": "High-end 32B Qwen model, superior quality (requires powerful hardware)"
            },
            "Qwen2.5-Coder-7B-Instruct": {
                "repo_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
                "filename": "*.gguf",
                "template": "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
                "description": "Code-specialized Qwen model, excellent for technical tasks"
            },
            
            # === MISTRAL FAMILY ===
            "Mistral-7B-Instruct-v0.3": {
                "repo_id": "mistralai/Mistral-7B-Instruct-v0.3",
                "filename": "*.gguf",
                "template": "<s>[INST] {system}\n\n{prompt} [/INST]",
                "description": "Efficient 7B Mistral model, balanced performance"
            },
            "Mixtral-8x7B-Instruct-v0.1": {
                "repo_id": "mistralai/Mixtral-8x7B-Instruct-v0.1",
                "filename": "*.gguf",
                "template": "<s>[INST] {system}\n\n{prompt} [/INST]",
                "description": "Mixture of experts model, excellent quality and efficiency"
            },
            "Mixtral-8x22B-Instruct-v0.1": {
                "repo_id": "mistralai/Mixtral-8x22B-Instruct-v0.1",
                "filename": "*.gguf",
                "template": "<s>[INST] {system}\n\n{prompt} [/INST]",
                "description": "High-end mixture of experts, premium quality (requires powerful hardware)"
            },
            
            # === PHI FAMILY ===
            "Phi-3-Mini-4K-Instruct": {
                "repo_id": "microsoft/Phi-3-mini-4k-instruct",
                "filename": "*.gguf",
                "template": "<|user|>\n{system}\n\n{prompt}<|end|>\n<|assistant|>\n",
                "description": "Compact 3.8B Phi model, efficient and capable"
            },
            "Phi-3-Medium-4K-Instruct": {
                "repo_id": "microsoft/Phi-3-medium-4k-instruct",
                "filename": "*.gguf",
                "template": "<|user|>\n{system}\n\n{prompt}<|end|>\n<|assistant|>\n",
                "description": "Balanced 14B Phi model, high quality responses"
            },
            
            # === GEMMA FAMILY ===
            "Gemma-2-9B-IT": {
                "repo_id": "google/gemma-2-9b-it",
                "filename": "*.gguf",
                "template": "<start_of_turn>user\n{system}\n\n{prompt}<end_of_turn>\n<start_of_turn>model\n",
                "description": "Google's 9B Gemma model, excellent reasoning"
            },
            "Gemma-2-27B-IT": {
                "repo_id": "google/gemma-2-27b-it",
                "filename": "*.gguf",
                "template": "<start_of_turn>user\n{system}\n\n{prompt}<end_of_turn>\n<start_of_turn>model\n",
                "description": "High-end 27B Gemma model, superior quality (requires powerful hardware)"
            },
            
            # === DEEPSEEK FAMILY ===
            "DeepSeek-V2.5": {
                "repo_id": "deepseek-ai/DeepSeek-V2.5",
                "filename": "*.gguf",
                "template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                "description": "Advanced reasoning model, excellent for complex prompts"
            },
            "DeepSeek-Coder-V2-Lite-Instruct": {
                "repo_id": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
                "filename": "*.gguf",
                "template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                "description": "Code-specialized DeepSeek model, excellent for technical content"
            },
            
            # === UNCENSORED MODELS ===
            "Luna-AI-Llama2-Uncensored": {
                "repo_id": "TheBloke/Luna-AI-Llama2-Uncensored-GGUF",
                "filename": "luna-ai-llama2-uncensored.Q4_K_M.gguf",
                "template": "USER: {prompt}\nASSISTANT:",
                "description": "Efficient 7B uncensored model, good for adult content tasks"
            },
            "WizardLM-13B-Uncensored": {
                "repo_id": "TheBloke/WizardLM-13B-Uncensored-GGUF", 
                "filename": "wizardlm-13b-uncensored.Q4_K_M.gguf",
                "template": "You are a helpful AI assistant.\n\nUSER: {prompt}\nASSISTANT:",
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
                "template": "### Instruction:\n{prompt}\n\n### Response:",
                "description": "Creative 13B uncensored model, excellent for roleplay and creative content"
            },
            
            # === SPECIALIZED MODELS ===
            "Dolphin-2.9-Llama3-8B": {
                "repo_id": "cognitivecomputations/dolphin-2.9-llama3-8b",
                "filename": "*.gguf",
                "template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                "description": "Uncensored Dolphin model, excellent for creative tasks"
            },
            "OpenHermes-2.5-Mistral-7B": {
                "repo_id": "teknium/OpenHermes-2.5-Mistral-7B",
                "filename": "*.gguf",
                "template": "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
                "description": "High-quality instruction-tuned model, versatile and capable"
            },
            "Starling-LM-7B-Alpha": {
                "repo_id": "berkeley-nest/Starling-LM-7B-alpha",
                "filename": "*.gguf",
                "template": "GPT4 Correct User: {system}\n\n{prompt}<|end_of_turn|>GPT4 Correct Assistant:",
                "description": "High-performance 7B model trained with RLAIF"
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

    def discover_local_models(self) -> List[str]:
        """Discover locally cached models from Hugging Face cache."""
        try:
            from huggingface_hub import scan_cache_dir
            
            print("🔍 Scanning for locally cached models...")
            cache_info = scan_cache_dir()
            
            discovered_models = []
            for repo in cache_info.repos:
                repo_id = repo.repo_id
                # Check if it's a text generation model
                if any(keyword in repo_id.lower() for keyword in ['llama', 'mistral', 'qwen', 'phi', 'gemma', 'deepseek', 'dolphin', 'hermes', 'wizard', 'vicuna']):
                    discovered_models.append(repo_id)
            
            print(f"📦 Found {len(discovered_models)} cached models")
            return discovered_models
            
        except Exception as e:
            print(f"⚠️ Could not scan cache: {e}")
            return []

    def refresh_model_list(self) -> Tuple[List[str], str]:
        """Refresh the model list by discovering new models."""
        try:
            print("🔄 Refreshing model list...")
            
            # Discover local models
            local_models = self.discover_local_models()
            
            # Add discovered models to our config if not already present
            new_models_added = 0
            for model_id in local_models:
                model_name = model_id.split('/')[-1]  # Use just the model name part
                if model_name not in self.model_configs:
                    # Add with generic configuration
                    self.model_configs[model_name] = {
                        "repo_id": model_id,
                        "filename": "*.gguf",
                        "template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                        "description": f"Discovered model from {model_id}"
                    }
                    new_models_added += 1
            
            updated_choices = list(self.model_configs.keys())
            status_msg = f"✅ Model list refreshed! Found {len(local_models)} cached models, added {new_models_added} new models."
            
            print(status_msg)
            return updated_choices, status_msg
            
        except Exception as e:
            error_msg = f"❌ Error refreshing model list: {str(e)}"
            print(error_msg)
            return list(self.model_configs.keys()), error_msg

    def load_model(self, model_name: str, gpu_layers: int = 35, progress=gr.Progress()) -> str:
        """Load the specified model with enhanced compatibility."""
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
            
            # Enhanced model loading with better file detection
            try:
                # Try to load with specific filename first
                if "*" not in model_config["filename"]:
                    self.llm = Llama.from_pretrained(
                        repo_id=model_config["repo_id"],
                        filename=model_config["filename"],
                        n_gpu_layers=gpu_layers,
                        n_ctx=4096,
                        verbose=False
                    )
                else:
                    # For wildcard filenames, let llama-cpp-python auto-detect
                    self.llm = Llama.from_pretrained(
                        repo_id=model_config["repo_id"],
                        n_gpu_layers=gpu_layers,
                        n_ctx=4096,
                        verbose=False
                    )
            except Exception as load_error:
                # Fallback: try different common GGUF filenames
                common_patterns = [
                    "*Q4_K_M*.gguf",
                    "*q4_k_m*.gguf", 
                    "*Q4_0*.gguf",
                    "*q4_0*.gguf",
                    "*.gguf"
                ]
                
                loaded = False
                for pattern in common_patterns:
                    try:
                        print(f"🔄 Trying pattern: {pattern}")
                        self.llm = Llama.from_pretrained(
                            repo_id=model_config["repo_id"],
                            filename=pattern,
                            n_gpu_layers=gpu_layers,
                            n_ctx=4096,
                            verbose=False
                        )
                        loaded = True
                        break
                    except:
                        continue
                
                if not loaded:
                    raise load_error
            
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
        """Download additional models from Hugging Face with enhanced support."""
        try:
            print(f"📥 Downloading model from {repo_id}")
            progress(0.1, f"Connecting to {repo_id}...")
            
            from llama_cpp import Llama
            
            progress(0.3, "Downloading model files...")
            
            # Enhanced download with better filename handling
            try:
                if filename and filename.strip() and "*" not in filename:
                    # Specific filename provided
                    llm = Llama.from_pretrained(
                        repo_id=repo_id,
                        filename=filename,
                        n_gpu_layers=0,  # Don't load into GPU, just download
                        verbose=False
                    )
                else:
                    # Auto-detect best file
                    llm = Llama.from_pretrained(
                        repo_id=repo_id,
                        n_gpu_layers=0,  # Don't load into GPU, just download
                        verbose=False
                    )
            except Exception as download_error:
                # Try common GGUF patterns
                common_patterns = ["*Q4_K_M*.gguf", "*q4_k_m*.gguf", "*.gguf"]
                downloaded = False
                
                for pattern in common_patterns:
                    try:
                        print(f"🔄 Trying pattern: {pattern}")
                        llm = Llama.from_pretrained(
                            repo_id=repo_id,
                            filename=pattern,
                            n_gpu_layers=0,
                            verbose=False
                        )
                        downloaded = True
                        break
                    except:
                        continue
                
                if not downloaded:
                    raise download_error
            
            progress(0.8, "Finalizing download...")
            
            # Add to available models if not already present
            model_name = f"custom_{repo_id.split('/')[-1]}"
            if model_name not in self.model_configs:
                self.model_configs[model_name] = {
                    "repo_id": repo_id,
                    "filename": filename if filename and "*" not in filename else "*.gguf",
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
                final_prompt = f"Main Influencer: {main_influencer.strip()}\n\nRequest: {prompt}"
            else:
                # If Main Influencer is empty, generate diverse appearance for image/video prompts
                if content_type in ["image_prompt", "video_prompt"]:
                    diverse_appearance = self.generate_diverse_appearance()
                    final_prompt = f"Subject appearance: {diverse_appearance}\n\nRequest: {prompt}"
            
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
                    prompt=f"{system_prompt}\n\nUser Request: {final_prompt}"
                )
            
            progress(0.3, "Generating response...")
            
            # Generate response with enhanced parameters
            response = self.llm(
                full_prompt,
                max_tokens=1024,
                temperature=temperature,
                top_p=0.9,
                stop=["<|eot_id|>", "<|im_end|>", "</s>", "[/INST]", "USER:", "ASSISTANT:", "\n\nUSER:", "\n\nASSISTANT:", "<|end_of_turn|>", "<end_of_turn>"],
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
            self.save_conversation_to_txt(f"Main Influencer: {main_influencer}\n\n{prompt}", generated_text, content_type)
            
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
    if not repo_id.strip():
        return "❌ Please provide repository ID"
    return chatbot.download_additional_model(repo_id, filename, progress)

def refresh_models_interface() -> Tuple[gr.Dropdown, str]:
    """Interface function for refreshing model list."""
    updated_choices, status_msg = chatbot.refresh_model_list()
    return gr.Dropdown(choices=updated_choices), status_msg

def generate_content(prompt: str, content_type: str, main_influencer: str, temperature: float, progress=gr.Progress()) -> str:
    """Interface function for generating content."""
    if not prompt.strip():
        return "Please enter a prompt."
    
    return chatbot.generate_response(prompt, content_type, main_influencer, temperature, progress)

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
                analysis_text += f"- {category.title()}: {count} terms\n"
        
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
        .workflow-instructions {
            background: #e3f2fd;
            border: 1px solid #2196f3;
            padding: 1.5rem;
            border-radius: 8px;
            margin: 1rem 0;
            color: #0d47a1 !important;
        }
        
        /* Dark theme support */
        .dark .warning-box {
            background: #664d03 !important;
            border: 1px solid #b08900 !important;
            color: #fff3cd !important;
        }
        
        .dark .workflow-instructions {
            background: #1a237e !important;
            border: 1px solid #3f51b5 !important;
            color: #e3f2fd !important;
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
            <p>Professional AI assistant for adult content creation with expanded model support and enhanced functionality</p>
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
                    <h3>🚀 Expanded Model Support</h3>
                    <p>Now supporting ALL compatible LLMs from Hugging Face, including:</p>
                    <ul>
                        <li><strong>Llama Family:</strong> 3.2-3B, 3.1-8B, 3.1-70B, CodeLlama-34B</li>
                        <li><strong>Qwen Family:</strong> 2.5-7B, 2.5-14B, 2.5-32B, Coder-7B</li>
                        <li><strong>Mistral Family:</strong> 7B-Instruct, Mixtral-8x7B, Mixtral-8x22B</li>
                        <li><strong>Phi Family:</strong> 3-Mini-4K, 3-Medium-4K</li>
                        <li><strong>Gemma Family:</strong> 2-9B-IT, 2-27B-IT</li>
                        <li><strong>DeepSeek Family:</strong> V2.5, Coder-V2-Lite</li>
                        <li><strong>Uncensored Models:</strong> Luna-AI, WizardLM, Wizard-Vicuna, Nous-Hermes</li>
                        <li><strong>Specialized Models:</strong> Dolphin, OpenHermes, Starling-LM</li>
                    </ul>
                    <p><strong>✨ New Features:</strong> Refresh button to discover newly downloaded models, enhanced compatibility with GGUF and safetensors formats, automatic model detection.</p>
                </div>
                """)
                
                with gr.Row():
                    with gr.Column():
                        model_dropdown = gr.Dropdown(
                            choices=list(chatbot.model_configs.keys()),
                            label="Select Model",
                            value="Llama-3.1-8B-Instruct",
                            info="Choose the AI model to load"
                        )
                        
                        with gr.Row():
                            refresh_models_btn = gr.Button("🔄 Refresh Models", variant="secondary")
                            load_model_btn = gr.Button("🚀 Load Model", variant="primary")
                        
                        gpu_layers_slider = gr.Slider(
                            minimum=0,
                            maximum=80,
                            value=35,
                            step=1,
                            label="GPU Layers",
                            info="Number of layers to offload to GPU (0 = CPU only, higher = more GPU usage)"
                        )
                        
                        model_status = gr.Textbox(label="Model Status", interactive=False)
                
                # Custom model download section
                gr.HTML("<h3>📥 Download Additional Models</h3>")
                with gr.Row():
                    with gr.Column():
                        custom_repo_id = gr.Textbox(
                            label="Repository ID",
                            placeholder="e.g., microsoft/Phi-3-mini-4k-instruct",
                            info="Hugging Face repository ID"
                        )
                        custom_filename = gr.Textbox(
                            label="Model Filename (Optional)",
                            placeholder="e.g., model.gguf (leave empty for auto-detection)",
                            info="Specific GGUF filename or leave empty for auto-detection"
                        )
                        download_model_btn = gr.Button("📥 Download Model")
                        download_status = gr.Textbox(label="Download Status", interactive=False)
            
            # Content Generation Tab
            with gr.Tab("✨ Content Generation"):
                with gr.Row():
                    with gr.Column(scale=2):
                        content_type = gr.Dropdown(
                            choices=["image_prompt", "video_prompt", "general_chat"],
                            value="image_prompt",
                            label="Content Type",
                            info="Select the type of content to generate"
                        )
                        
                        main_influencer = gr.Textbox(
                            label="Main Influencer Description (Optional)",
                            placeholder="e.g., 25-year-old blonde woman with blue eyes, athletic build...",
                            info="Describe the main subject. Leave empty for diverse random generation.",
                            lines=2
                        )
                        
                        prompt_input = gr.Textbox(
                            label="Your Request",
                            placeholder="Describe what you want to create...",
                            lines=4
                        )
                        
                        temperature_slider = gr.Slider(
                            minimum=0.1,
                            maximum=2.0,
                            value=0.7,
                            step=0.1,
                            label="Creativity (Temperature)",
                            info="Lower = more focused, Higher = more creative"
                        )
                        
                        with gr.Row():
                            generate_btn = gr.Button("🚀 Generate Content", variant="primary")
                            example_btn = gr.Button("💡 Get Example")
                    
                    with gr.Column(scale=3):
                        output_text = gr.Textbox(
                            label="Generated Content",
                            lines=20,
                            interactive=False
                        )
            
            # Prompt Analysis Tab
            with gr.Tab("🔍 Prompt Analysis"):
                gr.HTML("""
                <div class="model-info">
                    <h3>Intelligent Prompt Analysis & SD Forge Parameter Optimization</h3>
                    <p>Analyze your prompts for complexity and get automatically optimized Stable Diffusion Forge parameters for Flux models.</p>
                </div>
                """)
                
                with gr.Row():
                    with gr.Column():
                        analysis_prompt_input = gr.Textbox(
                            label="Prompt to Analyze",
                            placeholder="Enter your image generation prompt here...",
                            lines=5
                        )
                        
                        analyze_prompt_btn = gr.Button("🔍 Analyze Prompt & Get Parameters", variant="primary")
                    
                    with gr.Column():
                        complexity_analysis = gr.Textbox(
                            label="Complexity Analysis",
                            lines=12,
                            interactive=False
                        )
                        
                        sd_parameters = gr.Textbox(
                            label="Recommended SD Forge Parameters",
                            lines=12,
                            interactive=False
                        )
            
            # Conversation Management Tab
            with gr.Tab("💾 Conversation Management"):
                gr.HTML("""
                <div class="model-info">
                    <h3>Conversation History Management</h3>
                    <p>Manage your conversation history, export data, and clear sessions.</p>
                </div>
                """)
                
                with gr.Row():
                    with gr.Column():
                        clear_btn = gr.Button("🗑️ Clear History", variant="secondary")
                        export_btn = gr.Button("📤 Export Conversation", variant="primary")
                    
                    with gr.Column():
                        management_status = gr.Textbox(
                            label="Status",
                            interactive=False
                        )
        
        # Event handlers
        refresh_models_btn.click(
            fn=refresh_models_interface,
            outputs=[model_dropdown, model_status]
        )
        
        load_model_btn.click(
            fn=load_model_interface,
            inputs=[model_dropdown, gpu_layers_slider],
            outputs=[model_status]
        )
        
        download_model_btn.click(
            fn=download_model_interface,
            inputs=[custom_repo_id, custom_filename],
            outputs=[download_status]
        )
        
        generate_btn.click(
            fn=generate_content,
            inputs=[prompt_input, content_type, main_influencer, temperature_slider],
            outputs=[output_text]
        )
        
        example_btn.click(
            fn=get_example,
            inputs=[content_type],
            outputs=[prompt_input]
        )
        
        analyze_prompt_btn.click(
            fn=analyze_prompt_complexity,
            inputs=[analysis_prompt_input],
            outputs=[complexity_analysis, sd_parameters]
        )
        
        clear_btn.click(
            fn=clear_conversation,
            outputs=[management_status]
        )
        
        export_btn.click(
            fn=export_conversation,
            outputs=[management_status]
        )
    
    return interface

if __name__ == "__main__":
    print("🚀 Starting Enhanced Influencer Chatbot...")
    
    # Create and launch the interface
    interface = create_interface()
    
    # Launch with specific settings and enable queueing for progress tracking
    interface.queue()  # Enable queueing for progress tracking in Gradio 5.x
    interface.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        debug=False,
        show_error=True,
        quiet=False
    )
