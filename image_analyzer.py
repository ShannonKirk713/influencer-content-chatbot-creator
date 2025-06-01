#!/usr/bin/env python3
"""
Image Analysis Module for Fanvue Chatbot
Uses BLIP model for image captioning and analysis
Enhanced with better error handling, progress tracking, and detailed descriptions
FIXED: Improved video prompt generation to actually analyze uploaded images
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
            
            # If the description is too similar to the prompt, generate a more natural description
            if len(detailed_description) < 20 or detailed_description.lower().startswith("describe"):
                # Try a different approach for detailed description
                natural_prompt = "a detailed description of"
                natural_inputs = self.processor(image, text=natural_prompt, return_tensors="pt")
                if torch.cuda.is_available() and hasattr(self.model, 'cuda'):
                    natural_inputs = {k: v.cuda() for k, v in natural_inputs.items()}
                
                with torch.no_grad():
                    natural_outputs = self.model.generate(
                        **natural_inputs, 
                        max_new_tokens=100, 
                        num_beams=3,
                        do_sample=True,
                        temperature=0.9
                    )
                
                detailed_description = self.processor.decode(natural_outputs[0], skip_special_tokens=True)
                # Remove the prompt prefix if present
                if natural_prompt in detailed_description:
                    detailed_description = detailed_description.replace(natural_prompt, "").strip()
            
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
        FIXED: Generate a video prompt based on actual image analysis.
        
        Args:
            image_analysis: Results from analyze_image()
            user_request: Additional user requirements
            
        Returns:
            Formatted video prompt based on actual image content
        """
        print("🎬 Generating video prompt from image analysis...")
        
        if not image_analysis["success"]:
            return f"❌ Could not analyze image: {image_analysis['error']}"
        
        caption = image_analysis["caption"]
        detailed_desc = image_analysis["detailed_description"]
        
        # FIXED: Create a sophisticated video prompt that actually uses the image analysis
        # Extract key elements from the image analysis
        subjects = self._extract_subjects(caption, detailed_desc)
        setting = self._extract_setting(caption, detailed_desc)
        clothing = self._extract_clothing(caption, detailed_desc)
        mood = self._extract_mood(caption, detailed_desc)
        
        # Create a comprehensive video prompt based on actual image content
        video_prompt = f"""🖼️ **IMAGE ANALYSIS RESULTS:**
- Caption: {caption}
- Detailed Description: {detailed_desc}

🎬 **VIDEO CONCEPT:** Transform this static scene into dynamic adult content
Based on the analyzed image, create a sensual video sequence that brings this scene to life while maintaining the original aesthetic and mood.

👥 **SUBJECT(S):** {subjects}
- Add natural breathing and subtle body movements
- Enhance facial expressions with gentle transitions
- Include realistic micro-movements and gestures
- Maintain the original positioning while adding fluid motion
- Age-appropriate adult content (18+)

👗 **CLOTHING & STYLING:** {clothing}
- Natural fabric movement and realistic physics
- Subtle adjustments and repositioning
- Enhanced textures responding to movement and lighting
- Maintain the original style while adding dynamic elements

🏞️ **SETTING & ENVIRONMENT:** {setting}
- Enhance atmospheric elements from the original scene
- Add environmental interactions (lighting changes, ambient effects)
- Create depth and dimensionality beyond the static image
- Maintain the original mood and aesthetic

🎭 **MOTION & ACTION:** Natural movements that enhance the scene
- Gentle, flowing movements that complement the original pose
- Realistic physics and natural timing
- Sensual but tasteful motion appropriate for adult content
- Interactive elements that build on the existing composition
- {f"User-requested elements: {user_request}" if user_request else "Focus on natural, sensual movements"}

📹 **CAMERA WORK:** Professional cinematography
- Slow push-in to create intimacy (0-3 seconds)
- Gentle camera movements that enhance the original composition
- Focus pulls to highlight key elements identified in the image
- Smooth transitions between angles while respecting the original framing
- Professional adult content aesthetic matching the image style

🎨 **STYLE & ATMOSPHERE:** Enhanced visual appeal
- Maintain and enhance the lighting style from the original image
- {mood}
- Atmospheric effects that complement the existing scene
- Enhanced mood and sensuality building on the original
- High-quality production values matching the image aesthetic
- Color palette consistent with the analyzed image

⏱️ **DURATION NOTES:** Optimal pacing for engagement
- Opening: Establish the enhanced scene (2-3 seconds)
- Development: Build movement and interaction (6-8 seconds)
- Peak: Highlight the most compelling moment (2-3 seconds)
- Resolution: Gentle conclusion maintaining the mood (2-3 seconds)

**Total suggested duration:** 12-17 seconds for maximum impact and engagement

**TECHNICAL NOTES:**
- Maintain aspect ratio and composition style from the original image
- Ensure smooth transitions that feel natural and not jarring
- Focus on quality over quantity of movement
- Preserve the artistic integrity of the original image while adding motion"""

        print("✅ Video prompt generated successfully based on actual image analysis!")
        return video_prompt
    
    def _extract_subjects(self, caption: str, description: str) -> str:
        """Extract subject information from image analysis."""
        text = f"{caption} {description}".lower()
        
        # Look for people/subjects
        if "woman" in text or "girl" in text or "female" in text:
            if "man" in text or "male" in text:
                return "Multiple subjects including woman and man as described in the image"
            else:
                return "Woman as the primary subject, maintaining her appearance and positioning from the image"
        elif "man" in text or "male" in text:
            return "Man as the primary subject, maintaining his appearance and positioning from the image"
        elif "person" in text or "people" in text:
            return "Person(s) as described in the analyzed image, maintaining their original characteristics"
        else:
            return "Subject(s) as identified in the image analysis, maintaining original appearance and positioning"
    
    def _extract_setting(self, caption: str, description: str) -> str:
        """Extract setting information from image analysis."""
        text = f"{caption} {description}".lower()
        
        # Look for location indicators
        locations = ["bedroom", "bathroom", "kitchen", "living room", "outdoor", "beach", "garden", "studio", "office", "hotel"]
        for location in locations:
            if location in text:
                return f"The {location} setting as shown in the image, enhanced with dynamic elements"
        
        # Look for general environment clues
        if "indoor" in text or "inside" in text:
            return "Indoor setting as shown in the image, enhanced with atmospheric elements"
        elif "outdoor" in text or "outside" in text:
            return "Outdoor setting as shown in the image, enhanced with natural environmental effects"
        else:
            return "The setting as depicted in the analyzed image, enhanced with appropriate atmospheric elements"
    
    def _extract_clothing(self, caption: str, description: str) -> str:
        """Extract clothing information from image analysis."""
        text = f"{caption} {description}".lower()
        
        # Look for clothing items
        clothing_items = ["dress", "lingerie", "underwear", "bra", "panties", "shirt", "blouse", "skirt", "pants", "jeans", "bikini", "swimsuit", "robe", "nightgown"]
        found_items = [item for item in clothing_items if item in text]
        
        if found_items:
            return f"The {', '.join(found_items)} as shown in the image, with natural fabric movement and realistic physics"
        elif "naked" in text or "nude" in text or "topless" in text:
            return "Minimal or no clothing as shown in the image, focusing on natural skin tones and body positioning"
        else:
            return "Clothing and styling as depicted in the analyzed image, enhanced with realistic movement"
    
    def _extract_mood(self, caption: str, description: str) -> str:
        """Extract mood and atmosphere from image analysis."""
        text = f"{caption} {description}".lower()
        
        # Look for mood indicators
        if "romantic" in text or "intimate" in text:
            return "Romantic and intimate atmosphere as captured in the original image"
        elif "sensual" in text or "seductive" in text:
            return "Sensual and seductive mood matching the original image"
        elif "playful" in text or "fun" in text:
            return "Playful and engaging atmosphere as shown in the image"
        elif "elegant" in text or "sophisticated" in text:
            return "Elegant and sophisticated mood consistent with the image"
        elif "soft" in text or "gentle" in text:
            return "Soft and gentle atmosphere as depicted in the original"
        else:
            return "Atmospheric mood consistent with the analyzed image, enhanced for video"

# Global instance
image_analyzer = ImageAnalyzer()
