#!/usr/bin/env python3
"""
Image Analysis Module for Fanvue Chatbot
Uses BLIP model for image captioning and analysis
"""

import logging
from PIL import Image
from typing import Optional, Dict, Any
import torch

logger = logging.getLogger(__name__)

class ImageAnalyzer:
    """Image analysis class using BLIP model for captioning."""
    
    def __init__(self):
        self.processor = None
        self.model = None
        self.model_loaded = False
        
    def load_model(self) -> bool:
        """Load BLIP model for image captioning."""
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            
            logger.info("Loading BLIP image captioning model...")
            
            # Load the model and processor
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                logger.info("BLIP model loaded on GPU")
            else:
                logger.info("BLIP model loaded on CPU")
            
            self.model_loaded = True
            return True
            
        except ImportError as e:
            logger.error(f"Required libraries not installed: {e}")
            return False
        except Exception as e:
            logger.error(f"Error loading BLIP model: {e}")
            return False
    
    def analyze_image(self, image: Image.Image, prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze an image and generate caption and description.
        
        Args:
            image: PIL Image object
            prompt: Optional prompt to guide captioning
            
        Returns:
            Dictionary containing analysis results
        """
        if not self.model_loaded:
            if not self.load_model():
                return {
                    "success": False,
                    "error": "Failed to load image analysis model",
                    "caption": "",
                    "detailed_description": ""
                }
        
        try:
            # Generate basic caption
            inputs = self.processor(image, return_tensors="pt")
            if torch.cuda.is_available() and hasattr(self.model, 'cuda'):
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=50, num_beams=5)
            
            caption = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Generate detailed description with prompt
            detailed_prompt = "Describe this image in detail, including the setting, subjects, clothing, pose, and atmosphere:"
            detailed_inputs = self.processor(image, text=detailed_prompt, return_tensors="pt")
            if torch.cuda.is_available() and hasattr(self.model, 'cuda'):
                detailed_inputs = {k: v.cuda() for k, v in detailed_inputs.items()}
            
            with torch.no_grad():
                detailed_outputs = self.model.generate(**detailed_inputs, max_new_tokens=100, num_beams=5)
            
            detailed_description = self.processor.decode(detailed_outputs[0], skip_special_tokens=True)
            
            return {
                "success": True,
                "caption": caption,
                "detailed_description": detailed_description,
                "error": ""
            }
            
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return {
                "success": False,
                "error": f"Error analyzing image: {str(e)}",
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
        if not image_analysis["success"]:
            return f"❌ Could not analyze image: {image_analysis['error']}"
        
        caption = image_analysis["caption"]
        detailed_desc = image_analysis["detailed_description"]
        
        # Create a structured video prompt based on the image
        video_prompt = f"""🖼️ IMAGE ANALYSIS: {caption}

🎬 VIDEO CONCEPT: Transform this static scene into a dynamic video sequence
- Base scene: {detailed_desc}
- User request: {user_request if user_request else 'Create engaging motion from this image'}

🎭 ADDED MOTION: 
- Subtle movements and natural gestures
- Breathing and micro-expressions
- Gentle clothing movement from air/breeze
- Natural lighting variations

📹 CAMERA DYNAMICS:
- Slow push-in to create intimacy
- Gentle camera movements to add life
- Focus pulls to highlight key elements
- Smooth transitions between angles

🎨 ENHANCED ATMOSPHERE:
- Ambient lighting effects
- Atmospheric elements (steam, breeze, etc.)
- Enhanced mood and sensuality
- Professional adult content aesthetic

⏱️ SEQUENCE FLOW:
- Opening: Establish the scene (2-3 seconds)
- Development: Add movement and interaction (5-8 seconds)
- Climax: Peak moment of the sequence (2-3 seconds)
- Resolution: Gentle conclusion (2-3 seconds)

Total suggested duration: 10-15 seconds for optimal impact."""

        return video_prompt

# Global instance
image_analyzer = ImageAnalyzer()