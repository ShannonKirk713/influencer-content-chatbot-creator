#!/usr/bin/env python3
"""
Fanvue Adult Content Creation Chatbot
A Gradio-based application for generating adult content prompts using uncensored AI models.
Enhanced with image upload and analysis functionality, progress tracking, and conversation history saving.
"""

import gradio as gr
import os
import json
import time
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
from PIL import Image
from tqdm import tqdm

# Import image analyzer
from image_analyzer import image_analyzer

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

try:
    from ctransformers import AutoModelForCausalLM
    CTRANSFORMERS_AVAILABLE = True
except ImportError:
    CTRANSFORMERS_AVAILABLE = False
    logger.warning("ctransformers not available. Install with: pip install ctransformers")

class FanvueChatbot:
    """Main chatbot class for Fanvue adult content generation."""
    
    def __init__(self):
        self.model = None
        self.model_loaded = False
        self.conversation_history = []
        self.history_folder = "conversation_logs"
        
        # Create history folder if it doesn't exist
        os.makedirs(self.history_folder, exist_ok=True)
        
        # Model configurations
        self.model_configs = {
            "Luna-AI-Llama2-Uncensored": {
                "repo": "TheBloke/Luna-AI-Llama2-Uncensored-GGUF",
                "file": "luna-ai-llama2-uncensored.Q4_K_M.gguf",
                "template": "USER: {prompt}\nASSISTANT:",
                "description": "Efficient 7B model, good for most tasks"
            },
            "WizardLM-13B-Uncensored": {
                "repo": "TheBloke/WizardLM-13B-Uncensored-GGUF", 
                "file": "WizardLM-13B-Uncensored.Q4_K_M.gguf",
                "template": "You are a helpful AI assistant.\n\nUSER: {prompt}\nASSISTANT:",
                "description": "Balanced 13B model, high quality responses"
            },
            "Wizard-Vicuna-30B-Uncensored": {
                "repo": "TheBloke/Wizard-Vicuna-30B-Uncensored-GGUF",
                "file": "Wizard-Vicuna-30B-Uncensored.Q4_K_M.gguf", 
                "template": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {prompt} ASSISTANT:",
                "description": "High-end 30B model, best quality (requires powerful hardware)"
            }
        }
        
        # Content type templates
        self.content_templates = {
            "image_prompt": {
                "system": """You are an expert at creating detailed image prompts for adult content generation. Generate prompts in this exact format:

📸 CONCEPT: [Brief concept description]
👥 SUBJECT(S): [Detailed subject description including appearance, age (18+), positioning]
👗 CLOTHING: [Detailed clothing/styling description]
🏞️ SETTING: [Location and environment details]
💃 POSE & EXPRESSION: [Specific pose instructions and facial expressions]
📷 TECHNICAL: [Camera angle, lighting, style, quality specifications]

Focus on adult themes while maintaining artistic quality. Be explicit but tasteful.""",
                "example": "Create an image prompt for a sensual bedroom scene"
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
                "example": "Create a video prompt for an intimate dance sequence"
            },
            "image_to_video": {
                "system": """You are an expert at converting static image descriptions into dynamic video prompts for adult content. Analyze the image description and add natural motion elements following Wan2.1 principles:

🖼️ IMAGE ANALYSIS: [Analyze the provided image description]
🎬 VIDEO CONCEPT: [How to bring the image to life]
🎭 ADDED MOTION: [Natural movements and actions to add]
📹 CAMERA DYNAMICS: [Camera movements to enhance the scene]
🎨 ENHANCED ATMOSPHERE: [Additional atmospheric elements for video]
⏱️ SEQUENCE FLOW: [How the video should progress]

Transform static concepts into engaging motion while maintaining adult content focus.""",
                "example": "Convert this image prompt into a video: 'Sensual portrait of a woman in lingerie'"
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
                "example": "Give me advice on creating engaging adult content for my Fanvue profile"
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
                f.write(f"{'='*80}\n\n")
            
            print(f"💾 Conversation saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")
            print(f"❌ Error saving conversation: {e}")

    def load_model(self, model_name: str, gpu_layers: int = 0, progress=gr.Progress()) -> str:
        """Load the specified model with progress tracking."""
        if not CTRANSFORMERS_AVAILABLE:
            return "❌ ctransformers library not available. Please install it first."
        
        try:
            if model_name not in self.model_configs:
                return f"❌ Unknown model: {model_name}"
            
            config = self.model_configs[model_name]
            
            print(f"🔄 Loading model: {model_name}")
            logger.info(f"Loading model: {model_name}")
            
            progress(0.1, "Initializing model loading...")
            
            # Load model with ctransformers
            progress(0.3, "Downloading model files...")
            print("📥 Downloading model files (this may take a while)...")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                config["repo"],
                model_file=config["file"],
                model_type="llama",
                gpu_layers=gpu_layers,
                context_length=4096,
                max_new_tokens=1024,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.1
            )
            
            progress(0.8, "Finalizing model setup...")
            
            self.model_loaded = True
            self.current_model = model_name
            
            progress(1.0, "Model loaded successfully!")
            print(f"✅ Successfully loaded {model_name}")
            
            return f"✅ Successfully loaded {model_name}"
            
        except Exception as e:
            error_msg = f"Error loading model: {str(e)}"
            logger.error(error_msg)
            print(f"❌ {error_msg}")
            return f"❌ {error_msg}"

    def generate_response(self, prompt: str, content_type: str, temperature: float = 0.7, progress=gr.Progress()) -> str:
        """Generate response based on content type with progress tracking."""
        if not self.model_loaded:
            return "❌ No model loaded. Please load a model first."
        
        try:
            print(f"🤖 Generating {content_type} response...")
            progress(0.1, "Preparing prompt...")
            
            # Get template for content type
            template_info = self.content_templates.get(content_type, self.content_templates["general_chat"])
            system_prompt = template_info["system"]
            
            # Format the full prompt
            model_config = self.model_configs[self.current_model]
            full_prompt = model_config["template"].format(
                prompt=f"{system_prompt}\n\nUser Request: {prompt}"
            )
            
            progress(0.3, "Generating response...")
            print("⚡ AI is thinking...")
            
            # Generate response
            response = self.model(
                full_prompt,
                max_new_tokens=1024,
                temperature=temperature,
                top_p=0.95,
                stop=["USER:", "ASSISTANT:", "\n\nUSER:", "\n\nASSISTANT:"]
            )
            
            progress(0.8, "Processing response...")
            
            # Clean up response
            response = response.strip()
            if response.startswith("ASSISTANT:"):
                response = response[10:].strip()
            
            progress(0.9, "Saving conversation...")
            
            # Add to conversation history
            conversation_entry = {
                "timestamp": datetime.now().isoformat(),
                "prompt": prompt,
                "response": response,
                "content_type": content_type,
                "model": self.current_model
            }
            self.conversation_history.append(conversation_entry)
            
            # Save to txt file
            self.save_conversation_to_txt(prompt, response, content_type)
            
            progress(1.0, "Response generated!")
            print("✅ Response generated and saved!")
            
            return response
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            logger.error(error_msg)
            print(f"❌ {error_msg}")
            return f"❌ {error_msg}"

    def get_example_prompt(self, content_type: str) -> str:
        """Get example prompt for content type."""
        return self.content_templates.get(content_type, {}).get("example", "")

# Initialize chatbot
chatbot = FanvueChatbot()

def load_model_interface(model_name: str, gpu_layers: int, progress=gr.Progress()) -> str:
    """Interface function for loading models."""
    return chatbot.load_model(model_name, gpu_layers, progress)

def generate_content(prompt: str, content_type: str, temperature: float, progress=gr.Progress()) -> str:
    """Interface function for generating content."""
    if not prompt.strip():
        return "Please enter a prompt."
    
    return chatbot.generate_response(prompt, content_type, temperature, progress)

def analyze_uploaded_image(image: Image.Image, progress=gr.Progress()) -> Tuple[str, str]:
    """
    Analyze uploaded image and return caption and detailed description.
    
    Returns:
        Tuple of (caption, detailed_description)
    """
    if image is None:
        return "No image uploaded", ""
    
    try:
        progress(0.1, "Starting image analysis...")
        print("🖼️ Analyzing uploaded image...")
        
        # Analyze the image
        analysis = image_analyzer.analyze_image(image)
        
        if analysis["success"]:
            caption = analysis["caption"]
            detailed = analysis["detailed_description"]
            progress(1.0, "Image analysis complete!")
            print("✅ Image analysis completed successfully!")
            return caption, detailed
        else:
            error_msg = f"❌ Error analyzing image: {analysis['error']}"
            print(error_msg)
            return error_msg, ""
            
    except Exception as e:
        error_msg = f"❌ Error processing image: {str(e)}"
        logger.error(error_msg)
        print(error_msg)
        return error_msg, ""

def generate_video_from_image(image: Image.Image, user_request: str = "", progress=gr.Progress()) -> str:
    """
    Generate video prompt from uploaded image.
    
    Args:
        image: Uploaded PIL Image
        user_request: Additional user requirements
        
    Returns:
        Generated video prompt
    """
    if image is None:
        return "❌ Please upload an image first."
    
    try:
        progress(0.1, "Analyzing image...")
        print("🎬 Generating video prompt from image...")
        
        # Analyze the image
        analysis = image_analyzer.analyze_image(image)
        
        if not analysis["success"]:
            return f"❌ Could not analyze image: {analysis['error']}"
        
        progress(0.6, "Generating video prompt...")
        
        # Generate video prompt from analysis
        video_prompt = image_analyzer.generate_video_prompt_from_image(analysis, user_request)
        
        progress(0.9, "Saving conversation...")
        
        # Save the video prompt generation to history
        chatbot.save_conversation_to_txt(
            f"Image to Video Generation - User Request: {user_request}",
            video_prompt,
            "image_to_video"
        )
        
        progress(1.0, "Video prompt generated!")
        print("✅ Video prompt generated and saved!")
        
        return video_prompt
        
    except Exception as e:
        error_msg = f"❌ Error generating video prompt: {str(e)}"
        logger.error(error_msg)
        print(error_msg)
        return error_msg

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
        print(f"💾 Conversation exported to {filename}")
        return f"Conversation exported to {filename}"
    except Exception as e:
        error_msg = f"Error exporting conversation: {str(e)}"
        print(f"❌ {error_msg}")
        return error_msg

def get_conversation_stats() -> str:
    """Get statistics about conversation history."""
    if not chatbot.conversation_history:
        return "No conversation history available."
    
    total_conversations = len(chatbot.conversation_history)
    content_types = {}
    
    for entry in chatbot.conversation_history:
        content_type = entry.get("content_type", "unknown")
        content_types[content_type] = content_types.get(content_type, 0) + 1
    
    stats = f"📊 Conversation Statistics:\n"
    stats += f"Total conversations: {total_conversations}\n"
    stats += f"Content type breakdown:\n"
    for content_type, count in content_types.items():
        stats += f"  - {content_type}: {count}\n"
    
    # Check txt files in history folder
    txt_files = [f for f in os.listdir(chatbot.history_folder) if f.endswith('.txt')]
    stats += f"\nTXT files saved: {len(txt_files)}\n"
    stats += f"History folder: {chatbot.history_folder}"
    
    return stats

# Create Gradio interface
def create_interface():
    """Create the main Gradio interface."""
    
    with gr.Blocks(
        title="Fanvue Adult Content Creation Chatbot",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .warning-box {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 5px;
            padding: 15px;
            margin: 10px 0;
        }
        .model-info {
            background-color: #e3f2fd;
            border-radius: 5px;
            padding: 10px;
            margin: 5px 0;
        }
        .image-analysis-box {
            background-color: #f0f8ff;
            border: 1px solid #b0d4f1;
            border-radius: 5px;
            padding: 10px;
            margin: 5px 0;
        }
        .success-box {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            border-radius: 5px;
            padding: 10px;
            margin: 5px 0;
        }
        """
    ) as interface:
        
        # Header
        gr.Markdown("""
        # 🔞 Fanvue Adult Content Creation Chatbot
        
        **Professional AI Assistant for Adult Content Creators**
        
        This application uses uncensored AI models to help create high-quality adult content prompts for Fanvue creators.
        Now enhanced with **image upload and analysis** functionality, **progress tracking**, and **automatic conversation history saving**!
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
        
        # Success notice for new features
        gr.HTML("""
        <div class="success-box">
            <strong>🆕 NEW FEATURES:</strong><br>
            ✅ Progress bars for all operations<br>
            ✅ Automatic conversation history saving to TXT files<br>
            ✅ Enhanced image analysis with better scene descriptions<br>
            ✅ Improved error handling and console output<br>
            ✅ Changed port to 7861 for better compatibility
        </div>
        """)
        
        with gr.Tabs():
            # Main Chat Tab
            with gr.Tab("💬 Content Generation"):
                with gr.Row():
                    with gr.Column(scale=2):
                        content_type = gr.Dropdown(
                            choices=["image_prompt", "video_prompt", "image_to_video", "general_chat"],
                            value="image_prompt",
                            label="Content Type",
                            info="Select the type of content you want to generate"
                        )
                        
                        prompt_input = gr.Textbox(
                            label="Your Prompt",
                            placeholder="Describe what you want to create...",
                            lines=4
                        )
                        
                        with gr.Row():
                            example_btn = gr.Button("📝 Get Example", size="sm")
                            generate_btn = gr.Button("🚀 Generate Content", variant="primary")
                            clear_btn = gr.Button("🗑️ Clear", size="sm")
                        
                        temperature = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.7,
                            step=0.1,
                            label="Creativity (Temperature)",
                            info="Higher values = more creative, lower = more focused"
                        )
                    
                    with gr.Column(scale=3):
                        output = gr.Textbox(
                            label="Generated Content",
                            lines=20,
                            max_lines=30
                        )
            
            # Image to Video Tab (Enhanced)
            with gr.Tab("🖼️➡️🎬 Image to Video"):
                gr.Markdown("""
                ## Upload Image and Generate Video Prompts
                Upload an image and automatically generate video prompts based on AI analysis of the image content.
                **Now with enhanced progress tracking and better scene descriptions!**
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        # Image upload
                        image_input = gr.Image(
                            label="Upload Image",
                            type="pil",
                            height=400
                        )
                        
                        # Additional user request
                        user_request = gr.Textbox(
                            label="Additional Requirements (Optional)",
                            placeholder="e.g., 'focus on sensual movements', 'add romantic lighting'...",
                            lines=3
                        )
                        
                        with gr.Row():
                            analyze_btn = gr.Button("🔍 Analyze Image", variant="secondary")
                            generate_video_btn = gr.Button("🎬 Generate Video Prompt", variant="primary")
                    
                    with gr.Column(scale=2):
                        # Image analysis results
                        gr.HTML('<div class="image-analysis-box"><strong>📊 Image Analysis Results</strong></div>')
                        
                        image_caption = gr.Textbox(
                            label="Image Caption",
                            lines=2,
                            interactive=False
                        )
                        
                        image_description = gr.Textbox(
                            label="Detailed Description",
                            lines=4,
                            interactive=False
                        )
                        
                        # Generated video prompt
                        video_prompt_output = gr.Textbox(
                            label="Generated Video Prompt",
                            lines=15,
                            max_lines=25
                        )
            
            # Model Management Tab
            with gr.Tab("🤖 Model Settings"):
                gr.Markdown("## Model Configuration")
                
                with gr.Row():
                    with gr.Column():
                        model_choice = gr.Dropdown(
                            choices=list(chatbot.model_configs.keys()),
                            value="Luna-AI-Llama2-Uncensored",
                            label="Select Model",
                            info="Choose the AI model to use"
                        )
                        
                        gpu_layers = gr.Slider(
                            minimum=0,
                            maximum=50,
                            value=0,
                            step=1,
                            label="GPU Layers",
                            info="Number of layers to offload to GPU (0 = CPU only)"
                        )
                        
                        load_btn = gr.Button("📥 Load Model", variant="primary")
                        
                        model_status = gr.Textbox(
                            label="Model Status",
                            value="No model loaded",
                            interactive=False
                        )
                    
                    with gr.Column():
                        gr.Markdown("### Model Information")
                        for name, config in chatbot.model_configs.items():
                            gr.HTML(f"""
                            <div class="model-info">
                                <strong>{name}</strong><br>
                                {config['description']}<br>
                                <small>Repository: {config['repo']}</small>
                            </div>
                            """)
                        
                        gr.Markdown("### Image Analysis")
                        gr.HTML("""
                        <div class="model-info">
                            <strong>BLIP Image Captioning</strong><br>
                            Automatically loaded for image analysis<br>
                            Enhanced with better scene descriptions<br>
                            <small>Model: Salesforce/blip-image-captioning-base</small>
                        </div>
                        """)
            
            # History Tab (Enhanced)
            with gr.Tab("📚 Conversation History"):
                gr.Markdown("## Conversation History Management")
                gr.Markdown("**All conversations are automatically saved to timestamped TXT files!**")
                
                with gr.Row():
                    with gr.Column():
                        stats_btn = gr.Button("📊 Show Statistics")
                        export_btn = gr.Button("💾 Export JSON History")
                        clear_history_btn = gr.Button("🗑️ Clear Memory History")
                    
                    with gr.Column():
                        gr.Markdown("### Automatic Saving")
                        gr.HTML("""
                        <div class="success-box">
                            <strong>✅ Auto-Save Features:</strong><br>
                            • All conversations saved to TXT files<br>
                            • Organized by date in conversation_logs/ folder<br>
                            • Includes timestamps and metadata<br>
                            • No manual action required!
                        </div>
                        """)
                
                history_status = gr.Textbox(
                    label="History Status",
                    interactive=False,
                    lines=10
                )
                
                gr.Markdown("""
                ### Usage Tips:
                - **Image Prompts**: Generate detailed descriptions for static adult content
                - **Video Prompts**: Create motion-focused prompts for video content  
                - **Image to Video**: Upload images and convert them into dynamic video ideas
                - **General Chat**: Get advice and discuss adult content creation strategies
                - **Auto-Save**: All conversations automatically saved with timestamps
                - **Progress Tracking**: Visual progress bars for all operations
                """)
            
            # Help Tab (Updated)
            with gr.Tab("❓ Help & Examples"):
                gr.Markdown("""
                ## How to Use This Enhanced Chatbot
                
                ### 🆕 What's New in This Version:
                - **Progress Bars**: Visual feedback for all operations
                - **Auto-Save**: Conversations automatically saved to TXT files
                - **Enhanced Image Analysis**: Better scene descriptions with BLIP
                - **Console Output**: Real-time feedback in terminal
                - **Port Change**: Now runs on port 7861
                - **Better Error Handling**: More informative error messages
                
                ### 1. Load a Model
                - Go to "Model Settings" tab
                - Choose a model (Luna-AI recommended for most users)
                - Set GPU layers if you have a compatible GPU
                - Click "Load Model" and watch the progress bar
                
                ### 2. Generate Content
                - Select your content type
                - Enter your prompt or click "Get Example"
                - Adjust creativity slider if needed
                - Click "Generate Content" and watch progress
                - **Conversations are automatically saved!**
                
                ### 3. Enhanced Image to Video Feature
                - Go to "Image to Video" tab
                - Upload an image (JPG, PNG, etc.)
                - Click "Analyze Image" with progress tracking
                - Add optional requirements
                - Click "Generate Video Prompt" for motion-based content
                - **Analysis and prompts are automatically saved!**
                
                ### 4. Monitor Progress and History
                - Watch progress bars for real-time feedback
                - Check console output for detailed status
                - View "Conversation History" for statistics
                - Find saved TXT files in conversation_logs/ folder
                
                ### Content Types Explained:
                
                **📸 Image Prompts**
                - Creates detailed prompts for static adult images
                - Includes concept, subjects, clothing, setting, pose, and technical details
                - Perfect for photo shoots and digital art
                
                **🎬 Video Prompts**
                - Generates motion-focused prompts for adult videos
                - Based on Wan2.1 framework principles
                - Includes camera work, movements, and pacing
                
                **🖼️➡️🎬 Image to Video (Enhanced)**
                - Upload any image for AI analysis with progress tracking
                - Enhanced BLIP model for better scene descriptions
                - Converts static concepts into dynamic video prompts
                - Adds natural motion and camera dynamics
                - Great for expanding existing content ideas
                
                **💬 General Chat**
                - Open conversation about adult content creation
                - Get advice, tips, and creative ideas
                - Discuss platform strategies and techniques
                
                ### Example Prompts:
                
                **Image Prompt Example:**
                "Create a sensual boudoir photo concept with soft lighting"
                
                **Video Prompt Example:**
                "Design a slow-motion intimate dance sequence"
                
                **Image to Video Example:**
                Upload a photo → AI analyzes it → Generates video prompt automatically
                
                ### Technical Requirements:
                - **RAM**: 8GB+ (16GB+ recommended)
                - **GPU**: Optional but recommended for faster generation
                - **Storage**: 5-20GB for model files
                - **Image Analysis**: Automatic BLIP model download (~1GB)
                - **Port**: Application runs on port 7861
                
                ### Enhanced Dependencies:
                - **transformers**: For BLIP image analysis
                - **torch**: Deep learning backend
                - **Pillow**: Image processing
                - **tqdm**: Progress tracking
                - **ctransformers**: Model loading
                
                ### File Organization:
                - **conversation_logs/**: Auto-saved TXT files by date
                - **app.log**: Application logs and errors
                - **fanvue_conversation_*.json**: Exported JSON history
                
                ### Troubleshooting:
                - **Progress bars not showing**: Check console output for errors
                - **Image analysis failing**: Ensure transformers and torch are installed
                - **Model loading issues**: Check internet connection and disk space
                - **Port conflicts**: Application now uses port 7861 instead of 7860
                
                ### Safety and Legal Notes:
                - This tool is for creating legal adult content (18+)
                - Always comply with platform terms of service
                - Respect local laws and regulations
                - Use responsibly and ethically
                - All conversations are logged for quality assurance
                """)
        
        # Event handlers with progress tracking
        example_btn.click(
            fn=get_example,
            inputs=[content_type],
            outputs=[prompt_input]
        )
        
        generate_btn.click(
            fn=generate_content,
            inputs=[prompt_input, content_type, temperature],
            outputs=[output]
        )
        
        clear_btn.click(
            fn=lambda: "",
            outputs=[output]
        )
        
        # Image analysis event handlers with progress
        analyze_btn.click(
            fn=analyze_uploaded_image,
            inputs=[image_input],
            outputs=[image_caption, image_description]
        )
        
        generate_video_btn.click(
            fn=generate_video_from_image,
            inputs=[image_input, user_request],
            outputs=[video_prompt_output]
        )
        
        # Model management event handlers with progress
        load_btn.click(
            fn=load_model_interface,
            inputs=[model_choice, gpu_layers],
            outputs=[model_status]
        )
        
        # History event handlers
        stats_btn.click(
            fn=get_conversation_stats,
            outputs=[history_status]
        )
        
        export_btn.click(
            fn=export_conversation,
            outputs=[history_status]
        )
        
        clear_history_btn.click(
            fn=clear_conversation,
            outputs=[history_status]
        )
    
    return interface

if __name__ == "__main__":
    print("🚀 Starting Fanvue Chatbot...")
    print("🔧 Enhanced with progress tracking and auto-save features")
    print("🌐 Application will be available at http://localhost:7861")
    
    # Create and launch the interface
    interface = create_interface()
    
    # Launch with updated port and settings
    interface.launch(
        server_name="0.0.0.0",
        server_port=7861,  # Changed from 7860 to 7861
        share=False,  # Set to True if you want a public link
        debug=False,
        show_error=True,
        quiet=False
    )