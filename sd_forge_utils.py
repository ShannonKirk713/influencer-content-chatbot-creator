"""
Stable Diffusion Forge parameter recommendation utilities.
Analyzes prompt complexity and suggests appropriate generation parameters.
"""

import re
from typing import Dict, Tuple, List
from dataclasses import dataclass

@dataclass
class SDForgeParams:
    """Stable Diffusion Forge parameters."""
    steps: int
    sampler: str
    schedule_type: str
    cfg_scale: float
    distilled_cfg_scale: int
    seed: int
    width: int
    height: int

class PromptComplexityAnalyzer:
    """Analyzes prompt complexity to recommend SD Forge parameters."""
    
    def __init__(self):
        # Technical terms that indicate complexity
        self.technical_terms = {
            'lighting': ['lighting', 'illumination', 'shadows', 'highlights', 'backlit', 'rim light', 
                        'soft light', 'hard light', 'dramatic lighting', 'natural light', 'studio lighting',
                        'golden hour', 'blue hour', 'chiaroscuro', 'volumetric', 'ambient'],
            'camera': ['bokeh', 'depth of field', 'macro', 'wide angle', 'telephoto', 'fisheye',
                      'tilt-shift', 'long exposure', 'motion blur', 'sharp focus', 'shallow focus',
                      'aperture', 'f-stop', 'ISO', 'shutter speed', 'focal length'],
            'artistic': ['impressionist', 'renaissance', 'baroque', 'art nouveau', 'surreal',
                        'abstract', 'minimalist', 'maximalist', 'photorealistic', 'hyperrealistic',
                        'stylized', 'painterly', 'sketch', 'watercolor', 'oil painting'],
            'composition': ['rule of thirds', 'symmetry', 'asymmetry', 'leading lines', 'framing',
                           'perspective', 'foreground', 'background', 'negative space', 'balance',
                           'contrast', 'harmony', 'rhythm', 'emphasis', 'proportion'],
            'quality': ['4K', '8K', 'ultra high resolution', 'masterpiece', 'award winning',
                       'professional', 'studio quality', 'commercial', 'editorial', 'fine art',
                       'museum quality', 'gallery worthy', 'trending', 'viral'],
            'effects': ['lens flare', 'chromatic aberration', 'vignette', 'grain', 'noise',
                       'film grain', 'digital noise', 'glitch', 'distortion', 'reflection',
                       'refraction', 'caustics', 'subsurface scattering', 'global illumination']
        }
        
        # Detail indicators
        self.detail_indicators = [
            'detailed', 'intricate', 'elaborate', 'complex', 'ornate', 'decorative',
            'textured', 'patterned', 'layered', 'multi-layered', 'rich', 'luxurious',
            'fine', 'delicate', 'subtle', 'nuanced', 'sophisticated', 'refined'
        ]
        
        # Complexity modifiers
        self.complexity_modifiers = [
            'extremely', 'highly', 'very', 'ultra', 'super', 'incredibly', 'exceptionally',
            'remarkably', 'extraordinarily', 'intensely', 'dramatically', 'strikingly'
        ]

    def analyze_prompt_complexity(self, prompt: str) -> Dict[str, any]:
        """
        Analyze prompt complexity and return detailed metrics.
        
        Args:
            prompt: The text prompt to analyze
            
        Returns:
            Dictionary containing complexity metrics and scores
        """
        prompt_lower = prompt.lower()
        
        # Basic metrics
        word_count = len(prompt.split())
        char_count = len(prompt)
        sentence_count = len([s for s in re.split(r'[.!?]+', prompt) if s.strip()])
        
        # Technical term analysis
        technical_score = 0
        technical_categories = {}
        
        for category, terms in self.technical_terms.items():
            category_count = sum(1 for term in terms if term in prompt_lower)
            technical_categories[category] = category_count
            technical_score += category_count
        
        # Detail level analysis
        detail_score = sum(1 for indicator in self.detail_indicators if indicator in prompt_lower)
        
        # Complexity modifier analysis
        modifier_score = sum(1 for modifier in self.complexity_modifiers if modifier in prompt_lower)
        
        # Punctuation complexity (commas, semicolons indicate detailed descriptions)
        punctuation_score = prompt.count(',') + prompt.count(';') * 2 + prompt.count(':') * 1.5
        
        # Calculate overall complexity score (0-100)
        complexity_score = min(100, (
            (word_count / 100) * 25 +  # Word count contribution (25%)
            (technical_score / 10) * 30 +  # Technical terms (30%)
            (detail_score / 5) * 20 +  # Detail indicators (20%)
            (modifier_score / 3) * 10 +  # Complexity modifiers (10%)
            (punctuation_score / 10) * 15  # Punctuation complexity (15%)
        ))
        
        return {
            'complexity_score': round(complexity_score, 1),
            'word_count': word_count,
            'char_count': char_count,
            'sentence_count': sentence_count,
            'technical_score': technical_score,
            'technical_categories': technical_categories,
            'detail_score': detail_score,
            'modifier_score': modifier_score,
            'punctuation_score': punctuation_score,
            'complexity_level': self._get_complexity_level(complexity_score)
        }

    def _get_complexity_level(self, score: float) -> str:
        """Convert complexity score to descriptive level."""
        if score < 20:
            return "Simple"
        elif score < 40:
            return "Moderate"
        elif score < 60:
            return "Detailed"
        elif score < 80:
            return "Complex"
        else:
            return "Highly Complex"

    def recommend_sd_forge_params(self, prompt: str) -> SDForgeParams:
        """
        Recommend Stable Diffusion Forge parameters based on prompt complexity.
        
        Args:
            prompt: The text prompt to analyze
            
        Returns:
            SDForgeParams object with recommended settings
        """
        analysis = self.analyze_prompt_complexity(prompt)
        complexity_score = analysis['complexity_score']
        
        # Steps: 15-55 based on complexity
        if complexity_score < 20:
            steps = 15  # Simple prompts need fewer steps
        elif complexity_score < 40:
            steps = 25  # Moderate complexity
        elif complexity_score < 60:
            steps = 35  # Detailed prompts
        elif complexity_score < 80:
            steps = 45  # Complex prompts
        else:
            steps = 55  # Highly complex prompts
        
        # Distilled CFG Scale: 7-16 based on complexity
        if complexity_score < 20:
            distilled_cfg = 7   # Simple prompts
        elif complexity_score < 40:
            distilled_cfg = 10  # Moderate complexity
        elif complexity_score < 60:
            distilled_cfg = 13  # Detailed prompts
        elif complexity_score < 80:
            distilled_cfg = 15  # Complex prompts
        else:
            distilled_cfg = 16  # Highly complex prompts
        
        # CFG Scale: 6.0-12.0 based on complexity
        if complexity_score < 20:
            cfg_scale = 6.5
        elif complexity_score < 40:
            cfg_scale = 7.5
        elif complexity_score < 60:
            cfg_scale = 8.5
        elif complexity_score < 80:
            cfg_scale = 10.0
        else:
            cfg_scale = 11.5
        
        # Sampler selection based on complexity
        if complexity_score < 30:
            sampler = "Euler a"  # Fast for simple prompts
        elif complexity_score < 60:
            sampler = "DPM++ 2M"  # Balanced quality/speed
        else:
            sampler = "DPM++ 2M SDE"  # High quality for complex prompts
        
        # Schedule type based on complexity
        if complexity_score < 40:
            schedule_type = "Automatic"
        elif complexity_score < 70:
            schedule_type = "Karras"
        else:
            schedule_type = "Exponential"
        
        # Image dimensions based on content analysis
        # Look for aspect ratio hints in the prompt
        prompt_lower = prompt.lower()
        
        if any(term in prompt_lower for term in ['portrait', 'headshot', 'face', 'bust']):
            width, height = 768, 1024  # Portrait orientation
        elif any(term in prompt_lower for term in ['landscape', 'wide', 'panoramic', 'horizon']):
            width, height = 1024, 768  # Landscape orientation
        elif any(term in prompt_lower for term in ['square', 'centered', 'symmetrical']):
            width, height = 1024, 1024  # Square format
        else:
            width, height = 1024, 1024  # Default square
        
        # Seed (-1 for random)
        seed = -1
        
        return SDForgeParams(
            steps=steps,
            sampler=sampler,
            schedule_type=schedule_type,
            cfg_scale=cfg_scale,
            distilled_cfg_scale=distilled_cfg,
            seed=seed,
            width=width,
            height=height
        )

    def get_parameter_explanation(self, params: SDForgeParams, complexity_score: float) -> str:
        """
        Get a human-readable explanation of the recommended parameters.
        
        Args:
            params: The recommended parameters
            complexity_score: The complexity score of the prompt
            
        Returns:
            String explanation of the parameter choices
        """
        explanation = f"""Parameter Recommendations Explanation:

**Steps ({params.steps}):** Based on your prompt's complexity score of {complexity_score:.1f}, this step count balances quality and generation time. """
        
        if complexity_score < 30:
            explanation += "Simple prompts don't need many steps to achieve good results."
        elif complexity_score < 60:
            explanation += "Moderate complexity requires more steps for proper detail rendering."
        else:
            explanation += "Complex prompts benefit from higher step counts for fine detail resolution."
        
        explanation += f"""

**Sampler ({params.sampler}):** """
        
        if params.sampler == "Euler a":
            explanation += "Fast and efficient sampler, perfect for simple prompts and quick iterations."
        elif params.sampler == "DPM++ 2M":
            explanation += "Balanced sampler offering good quality with reasonable speed."
        else:
            explanation += "High-quality sampler that excels with complex, detailed prompts."
        
        explanation += f"""

**CFG Scale ({params.cfg_scale}):** Controls how closely the AI follows your prompt. """
        
        if params.cfg_scale < 7.5:
            explanation += "Lower values for simple prompts to avoid over-processing."
        elif params.cfg_scale < 9.0:
            explanation += "Moderate values for balanced prompt adherence."
        else:
            explanation += "Higher values to ensure complex prompts are fully realized."
        
        explanation += f"""

**Image Size ({params.width}x{params.height}):** """
        
        if params.width > params.height:
            explanation += "Landscape orientation detected from prompt content."
        elif params.height > params.width:
            explanation += "Portrait orientation suggested by prompt content."
        else:
            explanation += "Square format for balanced composition."
        
        return explanation

    def suggest_prompt_improvements(self, prompt: str) -> list[str]:
        """
        Suggest improvements to enhance prompt effectiveness.
        
        Args:
            prompt: The prompt to analyze
            
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        analysis = self.analyze_prompt_complexity(prompt)
        prompt_lower = prompt.lower()
        
        # Check for missing technical elements
        if analysis['technical_categories']['lighting'] == 0:
            suggestions.append("Add lighting details (e.g., 'soft natural lighting', 'dramatic shadows', 'golden hour')")
        
        if analysis['technical_categories']['camera'] == 0:
            suggestions.append("Include camera/lens details (e.g., 'shallow depth of field', 'wide angle', 'macro lens')")
        
        if analysis['technical_categories']['quality'] == 0:
            suggestions.append("Add quality modifiers (e.g., 'high resolution', 'professional', 'masterpiece')")
        
        # Check prompt length
        if analysis['word_count'] < 10:
            suggestions.append("Consider adding more descriptive details - very short prompts may lack specificity")
        elif analysis['word_count'] > 100:
            suggestions.append("Consider condensing the prompt - very long prompts can be less effective")
        
        # Check for style specification
        if not any(style in prompt_lower for style in ['style', 'artistic', 'photographic', 'painted', 'drawn']):
            suggestions.append("Specify an artistic style (e.g., 'photorealistic', 'oil painting style', 'digital art')")
        
        # Check for composition details
        if analysis['technical_categories']['composition'] == 0:
            suggestions.append("Add composition details (e.g., 'rule of thirds', 'centered composition', 'dynamic angle')")
        
        return suggestions

# Example usage and testing functions
def test_complexity_analyzer():
    """Test the complexity analyzer with sample prompts."""
    analyzer = PromptComplexityAnalyzer()
    
    test_prompts = [
        "A woman in red dress",
        "Professional portrait of an elegant woman in a flowing red evening gown, soft studio lighting, shallow depth of field, high resolution",
        "Hyperrealistic oil painting style portrait of a sophisticated woman with intricate jewelry, dramatic chiaroscuro lighting, renaissance composition with golden ratio, museum quality, trending on artstation, 8K resolution, award winning photography"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- Test Prompt {i} ---")
        print(f"Prompt: {prompt}")
        
        analysis = analyzer.analyze_prompt_complexity(prompt)
        params = analyzer.recommend_sd_forge_params(prompt)
        
        print(f"Complexity: {analysis['complexity_level']} ({analysis['complexity_score']}/100)")
        print(f"Recommended Steps: {params.steps}")
        print(f"Recommended Sampler: {params.sampler}")
        print(f"Recommended CFG: {params.cfg_scale}")

if __name__ == "__main__":
    test_complexity_analyzer()
