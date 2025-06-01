#!/usr/bin/env python3
"""
Fanvue Adult Content Creation Chatbot
A Gradio-based application for generating adult content prompts using uncensored AI models.
"""

import gradio as gr
import os
import json
import time
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
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

    def load_model(self, model_name: str, gpu_layers: int = 0) -> str:
        """Load the specified model."""
        if not CTRANSFORMERS_AVAILABLE:
            return "❌ ctransformers library not available. Please install it first."
        
        try:
            if model_name not in self.model_configs:
                return f"❌ Unknown model: {model_name}"
            
            config = self.model_configs[model_name]
            
            logger.info(f"Loading model: {model_name}")
            
            # Load model with ctransformers
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
            
            self.model_loaded = True
            self.current_model = model_name
            
            return f"✅ Successfully loaded {model_name}"
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return f"❌ Error loading model: {str(e)}"

    def generate_response(self, prompt: str, content_type: str, temperature: float = 0.7) -> str:
        """Generate response based on content type."""
        if not self.model_loaded:
            return "❌ No model loaded. Please load a model first."
        
        try:
            # Get template for content type
            template_info = self.content_templates.get(content_type, self.content_templates["general_chat"])
            system_prompt = template_info["system"]
            
            # Format the full prompt
            model_config = self.model_configs[self.current_model]
            full_prompt = model_config["template"].format(
                prompt=f"{system_prompt}\n\nUser Request: {prompt}"
            )
            
            # Generate response
            response = self.model(
                full_prompt,
                max_new_tokens=1024,
                temperature=temperature,
                top_p=0.95,
                stop=["USER:", "ASSISTANT:", "\n\nUSER:", "\n\nASSISTANT:"]
            )
            
            # Clean up response
            response = response.strip()
            if response.startswith("ASSISTANT:"):
                response = response[10:].strip()
            
            # Add to conversation history
            self.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "prompt": prompt,
                "response": response,
                "content_type": content_type,
                "model": self.current_model
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"❌ Error generating response: {str(e)}"

    def get_example_prompt(self, content_type: str) -> str:
        """Get example prompt for content type."""
        return self.content_templates.get(content_type, {}).get("example", "")

# Initialize chatbot
chatbot = FanvueChatbot()

def load_model_interface(model_name: str, gpu_layers: int) -> str:
    """Interface function for loading models."""
    return chatbot.load_model(model_name, gpu_layers)

def generate_content(prompt: str, content_type: str, temperature: float) -> str:
    """Interface function for generating content."""
    if not prompt.strip():
        return "Please enter a prompt."
    
    return chatbot.generate_response(prompt, content_type, temperature)

def get_example(content_type: str) -> str:
    """Get example prompt for the selected content type."""
    return chatbot.get_example_prompt(content_type)

def clear_conversation():
    """Clear conversation history."""
    chatbot.conversation_history = []
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
        return f"Conversation exported to {filename}"
    except Exception as e:
        return f"Error exporting conversation: {str(e)}"

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
        """
    ) as interface:
        
        # Header
        gr.Markdown("""
        # 🔞 Fanvue Adult Content Creation Chatbot
        
        **Professional AI Assistant for Adult Content Creators**
        
        This application uses uncensored AI models to help create high-quality adult content prompts for Fanvue creators.
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
            
            # History Tab
            with gr.Tab("📚 Conversation History"):
                with gr.Row():
                    export_btn = gr.Button("💾 Export History")
                    clear_history_btn = gr.Button("🗑️ Clear History")
                
                history_status = gr.Textbox(
                    label="History Status",
                    interactive=False
                )
                
                gr.Markdown("""
                ### Usage Tips:
                - **Image Prompts**: Generate detailed descriptions for static adult content
                - **Video Prompts**: Create motion-focused prompts for video content
                - **Image to Video**: Convert static concepts into dynamic video ideas
                - **General Chat**: Get advice and discuss adult content creation strategies
                """)
            
            # Help Tab
            with gr.Tab("❓ Help & Examples"):
                gr.Markdown("""
                ## How to Use This Chatbot
                
                ### 1. Load a Model
                - Go to "Model Settings" tab
                - Choose a model (Luna-AI recommended for most users)
                - Set GPU layers if you have a compatible GPU
                - Click "Load Model"
                
                ### 2. Generate Content
                - Select your content type
                - Enter your prompt or click "Get Example"
                - Adjust creativity slider if needed
                - Click "Generate Content"
                
                ### Content Types Explained:
                
                **📸 Image Prompts**
                - Creates detailed prompts for static adult images
                - Includes concept, subjects, clothing, setting, pose, and technical details
                - Perfect for photo shoots and digital art
                
                **🎬 Video Prompts**
                - Generates motion-focused prompts for adult videos
                - Based on Wan2.1 framework principles
                - Includes camera work, movements, and pacing
                
                **🖼️➡️🎬 Image to Video**
                - Converts static image concepts into dynamic video prompts
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
                "Turn this lingerie photo concept into a video: [describe image]"
                
                ### Technical Requirements:
                - **RAM**: 8GB+ (16GB+ recommended)
                - **GPU**: Optional but recommended for faster generation
                - **Storage**: 5-20GB for model files
                
                ### Safety and Legal Notes:
                - This tool is for creating legal adult content (18+)
                - Always comply with platform terms of service
                - Respect local laws and regulations
                - Use responsibly and ethically
                """)
        
        # Event handlers
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
        
        load_btn.click(
            fn=load_model_interface,
            inputs=[model_choice, gpu_layers],
            outputs=[model_status]
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
    # Create and launch the interface
    interface = create_interface()
    
    # Launch with appropriate settings
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True if you want a public link
        debug=False,
        show_error=True,
        quiet=False
    )