#!/usr/bin/env python3
"""
Image Analysis Module for Fanvue Chatbot
Uses BLIP model for image captioning and analysis
Enhanced with better error handling, progress tracking, and detailed descriptions
"""

import logging
from PIL import Image
from typing import Optional, Dict, Any
import torch
from tqdm import tqdm
import sys

logger = logging.getLogger(__name__)

class ImageAnalyzer:
    """Image analysis class using BLIP model for captioning."""
    
    def __init__(self):
        self.processor = None
        self.model = None
        self.model_loaded = False
        
    def load_model(self, progress_callback=None) -> bool:
        """Load BLIP model for image captioning with progress tracking."""
        try:
            print("🔄 Loading BLIP image captioning model...")
            if progress_callback:
                progress_callback(0.1, "Importing transformers...")
            
            from transformers import BlipProcessor, BlipForConditionalGeneration
            
            if progress_callback:
                progress_callback(0.3, "Loading processor...")
            
            logger.info("Loading BLIP image captioning model...")
            
            # Load the model and processor with progress tracking
            print("📥 Downloading BLIP processor...")
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            
            if progress_callback:
                progress_callback(0.6, "Loading model...")
            
            print("📥 Downloading BLIP model...")
            self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            
            if progress_callback:
                progress_callback(0.8, "Setting up device...")
            
            # Move to GPU if available
            if torch.cuda.is_available():
                print("🚀 Moving model to GPU...")
                self.model = self.model.cuda()
                logger.info("BLIP model loaded on GPU")
            else:
                print("💻 Using CPU for inference...")
                logger.info("BLIP model loaded on CPU")
            
            if progress_callback:
                progress_callback(1.0, "Model ready!")
            
            self.model_loaded = True
            print("✅ BLIP model loaded successfully!")
            return True
            
        except ImportError as e:
            error_msg = f"Required libraries not installed: {e}. Please install with: pip install transformers torch"
            logger.error(error_msg)
            print(f"❌ {error_msg}")
            return False
        except Exception as e:
            error_msg = f"Error loading BLIP model: {e}"
            logger.error(error_msg)
            print(f"❌ {error_msg}")
            return False
    
    def analyze_image(self, image: Image.Image, prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze an image and generate caption and detailed description.
        
        Args:
            image: PIL Image object
            prompt: Optional prompt to guide captioning
            
        Returns:
            Dictionary containing analysis results
        """
        print("🔍 Starting image analysis...")
        
        if not self.model_loaded:
            print("📥 Model not loaded, loading now...")
            if not self.load_model():
                return {
                    "success": False,
                    "error": "Failed to load image analysis model",
                    "caption": "",
                    "detailed_description": ""
                }
        
        try:
            print("📸 Generating basic caption...")
            
            # Generate basic caption
            inputs = self.processor(image, return_tensors="pt")
            if torch.cuda.is_available() and hasattr(self.model, 'cuda'):
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=50, 
                    num_beams=5,
                    do_sample=True,
                    temperature=0.7
                )
            
            caption = self.processor.decode(outputs[0], skip_special_tokens=True)
            print(f"✅ Basic caption: {caption}")
            
            print("📝 Generating detailed description...")
            
            # Generate detailed description with specific prompt for better scene analysis
            detailed_prompt = "Describe this image in detail including: the setting, people or subjects, their clothing and appearance, poses and expressions, lighting and atmosphere, and any notable objects or elements in the scene"
            detailed_inputs = self.processor(image, text=detailed_prompt, return_tensors="pt")
            if torch.cuda.is_available() and hasattr(self.model, 'cuda'):
                detailed_inputs = {k: v.cuda() for k, v in detailed_inputs.items()}
            
            with torch.no_grad():
                detailed_outputs = self.model.generate(
                    **detailed_inputs, 
                    max_new_tokens=150, 
                    num_beams=5,
                    do_sample=True,
                    temperature=0.8,
                    repetition_penalty=1.2
                )
            
            detailed_description = self.processor.decode(detailed_outputs[0], skip_special_tokens=True)
            
            # Clean up the detailed description to remove the prompt
            if detailed_prompt in detailed_description:
                detailed_description = detailed_description.replace(detailed_prompt, "").strip()
            
            print(f"✅ Detailed description generated ({len(detailed_description)} characters)")
            
            return {
                "success": True,
                "caption": caption,
                "detailed_description": detailed_description,
                "error": ""
            }
            
        except Exception as e:
            error_msg = f"Error analyzing image: {str(e)}"
            logger.error(error_msg)
            print(f"❌ {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "caption": "",
                "detailed_description": ""
            }
    
    def generate_video_prompt_from_image(self, image_analysis: Dict[str, Any], user_request: str = "") -> str:
        """
        Generate a video prompt based on image analysis.
        
        Args:
            image_analysis: Results from analyze_image()
            user_request: Additional user requirements
            
        Returns:
            Formatted video prompt
        """
        print("🎬 Generating video prompt from image analysis...")
        
        if not image_analysis["success"]:
            return f"❌ Could not analyze image: {image_analysis['error']}"
        
        caption = image_analysis["caption"]
        detailed_desc = image_analysis["detailed_description"]
        
        # Create a more sophisticated video prompt based on the image
        video_prompt = f"""🖼️ IMAGE ANALYSIS: {caption}

🎬 VIDEO CONCEPT: Transform this static scene into a dynamic, engaging video sequence
- Base scene: {detailed_desc}
- User request: {user_request if user_request else 'Create engaging motion and atmosphere from this image'}

👥 SUBJECT(S): Based on the analyzed image, enhance with natural movements
- Breathing and subtle body language
- Natural facial expressions and micro-movements
- Realistic interaction with environment
- Age-appropriate adult content (18+)

👗 CLOTHING: Maintain the styling from the image with added dynamics
- Natural fabric movement and flow
- Realistic response to lighting and movement
- Enhanced textures and materials

🏞️ SETTING: Expand the environment for video
- Enhanced atmospheric elements
- Dynamic lighting variations
- Environmental interactions (breeze, ambient sounds)
- Depth and dimensionality

🎭 MOTION & ACTION: Natural movements that bring the scene to life
- Gentle, flowing movements
- Realistic physics and timing
- Sensual but tasteful motion
- Interactive elements with the environment

📹 CAMERA WORK: Professional cinematography
- Slow push-in for intimacy (0-3 seconds)
- Gentle camera movements and subtle shifts
- Focus pulls to highlight key elements
- Smooth transitions between angles
- Professional adult content aesthetic

🎨 STYLE & ATMOSPHERE: Enhanced visual appeal
- Cinematic lighting with soft shadows
- Atmospheric effects (steam, particles, soft focus)
- Enhanced mood and sensuality
- High-quality production values
- Warm, inviting color palette

⏱️ DURATION NOTES: Optimal pacing for engagement
- Opening: Establish scene and mood (2-3 seconds)
- Development: Build movement and interaction (6-8 seconds)
- Peak: Highlight moment or climax (2-3 seconds)
- Resolution: Gentle conclusion (2-3 seconds)

Total suggested duration: 12-17 seconds for maximum impact and engagement."""

        print("✅ Video prompt generated successfully!")
        return video_prompt

# Global instance
image_analyzer = ImageAnalyzer()