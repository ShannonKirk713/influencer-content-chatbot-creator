#!/usr/bin/env python3
"""
Intelligent Prompt Analysis System
Replaces static parameter selection with dynamic, research-based analysis
"""

import re
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)

class ComplexityLevel(Enum):
    """Prompt complexity levels"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"

class ContentType(Enum):
    """Content type categories"""
    PORTRAIT = "portrait"
    LANDSCAPE = "landscape"
    ARTISTIC = "artistic"
    PHOTOREALISTIC = "photorealistic"
    FANTASY = "fantasy"
    ABSTRACT = "abstract"
    TECHNICAL = "technical"

@dataclass
class PromptAnalysis:
    """Comprehensive prompt analysis results"""
    complexity_level: ComplexityLevel
    complexity_score: int
    word_count: int
    technical_score: int
    detail_score: int
    modifier_score: int
    content_type: ContentType
    technical_categories: Dict[str, int]
    style_indicators: List[str]
    quality_indicators: List[str]
    composition_elements: List[str]
    lighting_terms: List[str]
    recommended_steps: int
    recommended_cfg: float
    recommended_sampler: str
    recommended_scheduler: str

@dataclass
class GenerationParameters:
    """Optimized generation parameters"""
    steps: int
    cfg_scale: float
    distilled_cfg_scale: float
    sampler: str
    scheduler: str
    seed: int
    width: int
    height: int
    guidance_scale: float
    negative_prompt_strength: float

class IntelligentPromptAnalyzer:
    """
    Advanced prompt analysis system that dynamically determines optimal parameters
    based on prompt content, complexity, and research-backed optimization strategies.
    """
    
    def __init__(self):
        self.technical_terms = {
            'camera': ['dslr', 'canon', 'nikon', 'sony', 'leica', 'hasselblad', 'fujifilm', 'pentax', 
                      'camera', 'lens', 'focal length', 'aperture', 'f/', 'mm', 'macro', 'telephoto', 
                      'wide angle', 'fisheye', 'prime', 'zoom'],
            
            'lighting': ['soft light', 'hard light', 'natural light', 'studio lighting', 'golden hour',
                        'blue hour', 'backlighting', 'rim lighting', 'key light', 'fill light',
                        'ambient light', 'dramatic lighting', 'moody lighting', 'cinematic lighting',
                        'volumetric', 'god rays', 'chiaroscuro', 'rembrandt lighting'],
            
            'composition': ['rule of thirds', 'leading lines', 'symmetry', 'asymmetry', 'depth of field',
                           'bokeh', 'foreground', 'background', 'midground', 'perspective', 'angle',
                           'framing', 'crop', 'close-up', 'medium shot', 'wide shot', 'bird\'s eye',
                           'worm\'s eye', 'dutch angle'],
            
            'style': ['photorealistic', 'hyperrealistic', 'surreal', 'abstract', 'minimalist',
                     'maximalist', 'vintage', 'retro', 'modern', 'contemporary', 'classical',
                     'baroque', 'renaissance', 'impressionist', 'expressionist', 'cubist',
                     'art nouveau', 'art deco', 'pop art', 'street art'],
            
            'quality': ['4k', '8k', 'uhd', 'hd', 'high resolution', 'sharp', 'crisp', 'detailed',
                       'intricate', 'fine details', 'texture', 'grain', 'smooth', 'polished',
                       'professional', 'award winning', 'masterpiece', 'trending', 'viral'],
            
            'post_processing': ['hdr', 'tone mapping', 'color grading', 'lut', 'film grain',
                               'vignette', 'bloom', 'lens flare', 'chromatic aberration',
                               'noise reduction', 'sharpening', 'contrast', 'saturation',
                               'vibrance', 'clarity', 'structure']
        }
        
        self.complexity_modifiers = [
            'extremely', 'highly', 'very', 'ultra', 'super', 'hyper', 'incredibly',
            'exceptionally', 'remarkably', 'extraordinarily', 'intensely', 'dramatically',
            'stunningly', 'breathtakingly', 'magnificently', 'spectacularly'
        ]
        
        self.detail_indicators = [
            'detailed', 'intricate', 'complex', 'elaborate', 'ornate', 'rich',
            'layered', 'textured', 'nuanced', 'sophisticated', 'refined',
            'precise', 'meticulous', 'careful', 'thorough', 'comprehensive'
        ]
        
        # Research-based parameter mappings
        self.sampler_profiles = {
            'simple': ['Euler a', 'DPM++ 2M'],
            'moderate': ['DPM++ 2M Karras', 'DPM++ 2M SDE'],
            'complex': ['DPM++ 2M SDE Karras', 'DDIM'],
            'very_complex': ['DDIM', 'PLMS']
        }
        
        self.scheduler_profiles = {
            'simple': ['Automatic', 'Karras'],
            'moderate': ['Karras', 'Exponential'],
            'complex': ['Exponential', 'Polyexponential'],
            'very_complex': ['Polyexponential', 'SGM Uniform']
        }

    def analyze_prompt_comprehensive(self, prompt: str) -> PromptAnalysis:
        """
        Perform comprehensive prompt analysis with detailed categorization
        """
        if not prompt or not prompt.strip():
            return self._create_default_analysis()
        
        prompt_lower = prompt.lower()
        words = prompt.split()
        word_count = len(words)
        
        # Analyze technical categories
        technical_categories = {}
        total_technical_score = 0
        
        for category, terms in self.technical_terms.items():
            count = sum(1 for term in terms if term.lower() in prompt_lower)
            technical_categories[category] = count
            total_technical_score += count
        
        # Analyze complexity modifiers
        modifier_score = sum(1 for modifier in self.complexity_modifiers 
                           if modifier in prompt_lower)
        
        # Analyze detail indicators
        detail_score = sum(1 for indicator in self.detail_indicators 
                         if indicator in prompt_lower)
        
        # Calculate overall complexity score
        complexity_score = min(100, (
            (word_count * 0.5) +
            (total_technical_score * 3) +
            (modifier_score * 2) +
            (detail_score * 2.5)
        ))
        
        # Determine complexity level
        if complexity_score < 20:
            complexity_level = ComplexityLevel.SIMPLE
        elif complexity_score < 40:
            complexity_level = ComplexityLevel.MODERATE
        elif complexity_score < 70:
            complexity_level = ComplexityLevel.COMPLEX
        else:
            complexity_level = ComplexityLevel.VERY_COMPLEX
        
        # Determine content type
        content_type = self._determine_content_type(prompt_lower)
        
        # Extract style and quality indicators
        style_indicators = self._extract_style_indicators(prompt_lower)
        quality_indicators = self._extract_quality_indicators(prompt_lower)
        composition_elements = self._extract_composition_elements(prompt_lower)
        lighting_terms = self._extract_lighting_terms(prompt_lower)
        
        # Calculate recommended parameters
        recommended_steps = self._calculate_optimal_steps(complexity_level, technical_categories)
        recommended_cfg = self._calculate_optimal_cfg(complexity_level, content_type)
        recommended_sampler = self._select_optimal_sampler(complexity_level, content_type)
        recommended_scheduler = self._select_optimal_scheduler(complexity_level, technical_categories)
        
        return PromptAnalysis(
            complexity_level=complexity_level,
            complexity_score=int(complexity_score),
            word_count=word_count,
            technical_score=total_technical_score,
            detail_score=detail_score,
            modifier_score=modifier_score,
            content_type=content_type,
            technical_categories=technical_categories,
            style_indicators=style_indicators,
            quality_indicators=quality_indicators,
            composition_elements=composition_elements,
            lighting_terms=lighting_terms,
            recommended_steps=recommended_steps,
            recommended_cfg=recommended_cfg,
            recommended_sampler=recommended_sampler,
            recommended_scheduler=recommended_scheduler
        )

    def generate_optimal_parameters(self, prompt: str, 
                                  user_sampler: Optional[str] = None,
                                  user_scheduler: Optional[str] = None,
                                  user_cfg: Optional[float] = None) -> GenerationParameters:
        """
        Generate optimal parameters based on prompt analysis with user overrides
        """
        analysis = self.analyze_prompt_comprehensive(prompt)
        
        # Base parameters from analysis
        steps = analysis.recommended_steps
        cfg_scale = user_cfg if user_cfg is not None else analysis.recommended_cfg
        sampler = user_sampler if user_sampler else analysis.recommended_sampler
        scheduler = user_scheduler if user_scheduler else analysis.recommended_scheduler
        
        # Calculate distilled CFG based on complexity
        distilled_cfg = self._calculate_distilled_cfg(analysis.complexity_level, cfg_scale)
        
        # Determine optimal resolution based on content type
        width, height = self._calculate_optimal_resolution(analysis.content_type)
        
        # Calculate guidance scale for advanced models
        guidance_scale = self._calculate_guidance_scale(analysis.complexity_level)
        
        # Calculate negative prompt strength
        negative_prompt_strength = self._calculate_negative_prompt_strength(analysis.complexity_level)
        
        return GenerationParameters(
            steps=steps,
            cfg_scale=cfg_scale,
            distilled_cfg_scale=distilled_cfg,
            sampler=sampler,
            scheduler=scheduler,
            seed=-1,  # Random seed by default
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            negative_prompt_strength=negative_prompt_strength
        )

    def _create_default_analysis(self) -> PromptAnalysis:
        """Create default analysis for empty prompts"""
        return PromptAnalysis(
            complexity_level=ComplexityLevel.SIMPLE,
            complexity_score=0,
            word_count=0,
            technical_score=0,
            detail_score=0,
            modifier_score=0,
            content_type=ContentType.PORTRAIT,
            technical_categories={category: 0 for category in self.technical_terms.keys()},
            style_indicators=[],
            quality_indicators=[],
            composition_elements=[],
            lighting_terms=[],
            recommended_steps=20,
            recommended_cfg=7.0,
            recommended_sampler="Euler a",
            recommended_scheduler="Automatic"
        )

    def _determine_content_type(self, prompt_lower: str) -> ContentType:
        """Determine content type based on prompt analysis"""
        type_indicators = {
            ContentType.PORTRAIT: ['portrait', 'face', 'headshot', 'person', 'character', 'model'],
            ContentType.LANDSCAPE: ['landscape', 'scenery', 'nature', 'outdoor', 'mountain', 'forest', 'beach'],
            ContentType.ARTISTIC: ['artistic', 'painting', 'drawing', 'sketch', 'illustration', 'art'],
            ContentType.PHOTOREALISTIC: ['photorealistic', 'realistic', 'photo', 'photograph'],
            ContentType.FANTASY: ['fantasy', 'magical', 'mythical', 'dragon', 'fairy', 'wizard'],
            ContentType.ABSTRACT: ['abstract', 'surreal', 'conceptual', 'experimental'],
            ContentType.TECHNICAL: ['technical', 'diagram', 'blueprint', 'schematic', 'engineering']
        }
        
        scores = {}
        for content_type, indicators in type_indicators.items():
            score = sum(1 for indicator in indicators if indicator in prompt_lower)
            scores[content_type] = score
        
        # Return the type with highest score, default to portrait
        return max(scores.keys(), key=lambda k: scores[k]) if max(scores.values()) > 0 else ContentType.PORTRAIT

    def _extract_style_indicators(self, prompt_lower: str) -> List[str]:
        """Extract style indicators from prompt"""
        found_styles = []
        for style in self.technical_terms['style']:
            if style.lower() in prompt_lower:
                found_styles.append(style)
        return found_styles

    def _extract_quality_indicators(self, prompt_lower: str) -> List[str]:
        """Extract quality indicators from prompt"""
        found_quality = []
        for quality in self.technical_terms['quality']:
            if quality.lower() in prompt_lower:
                found_quality.append(quality)
        return found_quality

    def _extract_composition_elements(self, prompt_lower: str) -> List[str]:
        """Extract composition elements from prompt"""
        found_composition = []
        for element in self.technical_terms['composition']:
            if element.lower() in prompt_lower:
                found_composition.append(element)
        return found_composition

    def _extract_lighting_terms(self, prompt_lower: str) -> List[str]:
        """Extract lighting terms from prompt"""
        found_lighting = []
        for term in self.technical_terms['lighting']:
            if term.lower() in prompt_lower:
                found_lighting.append(term)
        return found_lighting

    def _calculate_optimal_steps(self, complexity_level: ComplexityLevel, 
                               technical_categories: Dict[str, int]) -> int:
        """Calculate optimal steps based on complexity and technical content"""
        base_steps = {
            ComplexityLevel.SIMPLE: 20,
            ComplexityLevel.MODERATE: 30,
            ComplexityLevel.COMPLEX: 40,
            ComplexityLevel.VERY_COMPLEX: 50
        }
        
        steps = base_steps[complexity_level]
        
        # Adjust based on technical complexity
        technical_bonus = sum(technical_categories.values()) * 2
        steps += min(technical_bonus, 20)  # Cap the bonus
        
        return min(steps, 80)  # Maximum reasonable steps

    def _calculate_optimal_cfg(self, complexity_level: ComplexityLevel, 
                             content_type: ContentType) -> float:
        """Calculate optimal CFG scale"""
        base_cfg = {
            ComplexityLevel.SIMPLE: 7.0,
            ComplexityLevel.MODERATE: 8.0,
            ComplexityLevel.COMPLEX: 9.0,
            ComplexityLevel.VERY_COMPLEX: 10.0
        }
        
        cfg = base_cfg[complexity_level]
        
        # Adjust based on content type
        if content_type in [ContentType.ARTISTIC, ContentType.FANTASY]:
            cfg += 1.0
        elif content_type == ContentType.PHOTOREALISTIC:
            cfg -= 0.5
        
        return max(1.0, min(cfg, 20.0))

    def _select_optimal_sampler(self, complexity_level: ComplexityLevel, 
                              content_type: ContentType) -> str:
        """Select optimal sampler based on analysis"""
        samplers = self.sampler_profiles[complexity_level.value]
        
        # Prefer specific samplers for certain content types
        if content_type == ContentType.PHOTOREALISTIC:
            return "DPM++ 2M Karras"
        elif content_type in [ContentType.ARTISTIC, ContentType.FANTASY]:
            return "DPM++ 2M SDE Karras"
        
        return samplers[0]  # Default to first option

    def _select_optimal_scheduler(self, complexity_level: ComplexityLevel, 
                                technical_categories: Dict[str, int]) -> str:
        """Select optimal scheduler based on analysis"""
        schedulers = self.scheduler_profiles[complexity_level.value]
        
        # Prefer specific schedulers for technical content
        if technical_categories.get('post_processing', 0) > 2:
            return "Polyexponential"
        elif technical_categories.get('lighting', 0) > 3:
            return "Exponential"
        
        return schedulers[0]  # Default to first option

    def _calculate_distilled_cfg(self, complexity_level: ComplexityLevel, 
                               base_cfg: float) -> float:
        """Calculate distilled CFG scale for Flux models"""
        # Distilled CFG is typically higher than regular CFG for Flux
        multiplier = {
            ComplexityLevel.SIMPLE: 1.2,
            ComplexityLevel.MODERATE: 1.4,
            ComplexityLevel.COMPLEX: 1.6,
            ComplexityLevel.VERY_COMPLEX: 1.8
        }
        
        distilled_cfg = base_cfg * multiplier[complexity_level]
        return max(1.0, min(distilled_cfg, 20.0))

    def _calculate_optimal_resolution(self, content_type: ContentType) -> Tuple[int, int]:
        """Calculate optimal resolution based on content type"""
        resolutions = {
            ContentType.PORTRAIT: (768, 1024),
            ContentType.LANDSCAPE: (1024, 768),
            ContentType.ARTISTIC: (1024, 1024),
            ContentType.PHOTOREALISTIC: (1024, 1024),
            ContentType.FANTASY: (1024, 1024),
            ContentType.ABSTRACT: (1024, 1024),
            ContentType.TECHNICAL: (1024, 768)
        }
        
        return resolutions.get(content_type, (1024, 1024))

    def _calculate_guidance_scale(self, complexity_level: ComplexityLevel) -> float:
        """Calculate guidance scale for advanced models"""
        guidance_scales = {
            ComplexityLevel.SIMPLE: 7.5,
            ComplexityLevel.MODERATE: 8.0,
            ComplexityLevel.COMPLEX: 8.5,
            ComplexityLevel.VERY_COMPLEX: 9.0
        }
        
        return guidance_scales[complexity_level]

    def _calculate_negative_prompt_strength(self, complexity_level: ComplexityLevel) -> float:
        """Calculate negative prompt strength"""
        strengths = {
            ComplexityLevel.SIMPLE: 0.7,
            ComplexityLevel.MODERATE: 0.8,
            ComplexityLevel.COMPLEX: 0.9,
            ComplexityLevel.VERY_COMPLEX: 1.0
        }
        
        return strengths[complexity_level]

    def export_analysis_report(self, prompt: str, analysis: PromptAnalysis, 
                             parameters: GenerationParameters) -> Dict[str, Any]:
        """Export comprehensive analysis report"""
        return {
            "prompt": prompt,
            "analysis": asdict(analysis),
            "parameters": asdict(parameters),
            "timestamp": "2025-06-01T00:00:00Z",
            "analyzer_version": "1.0.0"
        }

# Convenience function for backward compatibility
def analyze_prompt_complexity(prompt: str) -> Dict[str, Any]:
    """
    Backward compatibility function for existing code
    """
    analyzer = IntelligentPromptAnalyzer()
    analysis = analyzer.analyze_prompt_comprehensive(prompt)
    
    return {
        'complexity_level': analysis.complexity_level.value,
        'complexity_score': analysis.complexity_score,
        'word_count': analysis.word_count,
        'technical_score': analysis.technical_score,
        'detail_score': analysis.detail_score,
        'modifier_score': analysis.modifier_score,
        'technical_categories': analysis.technical_categories
    }

def recommend_sd_forge_params(prompt: str, sampler: str = None, 
                            scheduler: str = None, cfg: float = None) -> Dict[str, Any]:
    """
    Backward compatibility function for existing code
    """
    analyzer = IntelligentPromptAnalyzer()
    parameters = analyzer.generate_optimal_parameters(prompt, sampler, scheduler, cfg)
    
    return {
        'steps': parameters.steps,
        'sampler': parameters.sampler,
        'schedule_type': parameters.scheduler,
        'cfg_scale': parameters.cfg_scale,
        'distilled_cfg_scale': parameters.distilled_cfg_scale,
        'seed': parameters.seed,
        'width': parameters.width,
        'height': parameters.height
    }