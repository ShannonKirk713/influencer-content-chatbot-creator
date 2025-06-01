#!/usr/bin/env python3
"""
API Wrapper for Intelligent Prompt Analysis System
Provides easy integration with existing applications
"""

import json
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import asdict

from prompt_analyzer import (
    IntelligentPromptAnalyzer, 
    PromptAnalysis, 
    GenerationParameters,
    ComplexityLevel,
    ContentType
)

# Configure logging
logger = logging.getLogger(__name__)

class PromptAnalysisAPI:
    """
    High-level API wrapper for the intelligent prompt analysis system.
    Provides simplified interfaces for common use cases.
    """
    
    def __init__(self):
        self.analyzer = IntelligentPromptAnalyzer()
        self._cache = {}  # Simple caching for repeated prompts
    
    def analyze(self, prompt: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Analyze a prompt and return comprehensive analysis results.
        
        Args:
            prompt: The prompt to analyze
            use_cache: Whether to use cached results for identical prompts
            
        Returns:
            Dictionary containing analysis results
        """
        if not prompt or not prompt.strip():
            return self._empty_analysis_result()
        
        # Check cache
        if use_cache and prompt in self._cache:
            logger.debug(f"Using cached analysis for prompt: {prompt[:50]}...")
            return self._cache[prompt]
        
        try:
            analysis = self.analyzer.analyze_prompt_comprehensive(prompt)
            result = self._format_analysis_result(analysis)
            
            # Cache result
            if use_cache:
                self._cache[prompt] = result
            
            logger.info(f"Analyzed prompt with complexity: {analysis.complexity_level.value}")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing prompt: {e}")
            return self._error_analysis_result(str(e))
    
    def get_optimal_parameters(self, prompt: str, 
                             sampler: Optional[str] = None,
                             scheduler: Optional[str] = None,
                             cfg_scale: Optional[float] = None,
                             use_cache: bool = True) -> Dict[str, Any]:
        """
        Get optimal generation parameters for a prompt.
        
        Args:
            prompt: The prompt to analyze
            sampler: Override sampler selection
            scheduler: Override scheduler selection
            cfg_scale: Override CFG scale
            use_cache: Whether to use cached results
            
        Returns:
            Dictionary containing optimal parameters
        """
        if not prompt or not prompt.strip():
            return self._default_parameters()
        
        cache_key = f"{prompt}_{sampler}_{scheduler}_{cfg_scale}"
        
        # Check cache
        if use_cache and cache_key in self._cache:
            logger.debug(f"Using cached parameters for prompt: {prompt[:50]}...")
            return self._cache[cache_key]
        
        try:
            parameters = self.analyzer.generate_optimal_parameters(
                prompt, sampler, scheduler, cfg_scale
            )
            result = self._format_parameters_result(parameters)
            
            # Cache result
            if use_cache:
                self._cache[cache_key] = result
            
            logger.info(f"Generated parameters: {parameters.steps} steps, {parameters.sampler} sampler")
            return result
            
        except Exception as e:
            logger.error(f"Error generating parameters: {e}")
            return self._default_parameters()
    
    def quick_analyze(self, prompt: str) -> Dict[str, Union[str, int, float]]:
        """
        Quick analysis returning only essential information.
        
        Args:
            prompt: The prompt to analyze
            
        Returns:
            Dictionary with essential analysis data
        """
        try:
            analysis = self.analyzer.analyze_prompt_comprehensive(prompt)
            return {
                'complexity': analysis.complexity_level.value,
                'score': analysis.complexity_score,
                'recommended_steps': analysis.recommended_steps,
                'recommended_cfg': analysis.recommended_cfg,
                'recommended_sampler': analysis.recommended_sampler,
                'content_type': analysis.content_type.value
            }
        except Exception as e:
            logger.error(f"Error in quick analysis: {e}")
            return {
                'complexity': 'simple',
                'score': 0,
                'recommended_steps': 20,
                'recommended_cfg': 7.0,
                'recommended_sampler': 'Euler a',
                'content_type': 'portrait'
            }
    
    def batch_analyze(self, prompts: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze multiple prompts in batch.
        
        Args:
            prompts: List of prompts to analyze
            
        Returns:
            List of analysis results
        """
        results = []
        for i, prompt in enumerate(prompts):
            try:
                result = self.analyze(prompt, use_cache=True)
                result['batch_index'] = i
                results.append(result)
            except Exception as e:
                logger.error(f"Error analyzing prompt {i}: {e}")
                error_result = self._error_analysis_result(str(e))
                error_result['batch_index'] = i
                results.append(error_result)
        
        logger.info(f"Batch analyzed {len(prompts)} prompts")
        return results
    
    def compare_prompts(self, prompt1: str, prompt2: str) -> Dict[str, Any]:
        """
        Compare two prompts and their analysis results.
        
        Args:
            prompt1: First prompt
            prompt2: Second prompt
            
        Returns:
            Dictionary containing comparison results
        """
        try:
            analysis1 = self.analyzer.analyze_prompt_comprehensive(prompt1)
            analysis2 = self.analyzer.analyze_prompt_comprehensive(prompt2)
            
            return {
                'prompt1': {
                    'text': prompt1,
                    'analysis': self._format_analysis_result(analysis1)
                },
                'prompt2': {
                    'text': prompt2,
                    'analysis': self._format_analysis_result(analysis2)
                },
                'comparison': {
                    'complexity_difference': analysis2.complexity_score - analysis1.complexity_score,
                    'steps_difference': analysis2.recommended_steps - analysis1.recommended_steps,
                    'more_complex': prompt2 if analysis2.complexity_score > analysis1.complexity_score else prompt1,
                    'same_content_type': analysis1.content_type == analysis2.content_type,
                    'same_complexity_level': analysis1.complexity_level == analysis2.complexity_level
                }
            }
        except Exception as e:
            logger.error(f"Error comparing prompts: {e}")
            return {'error': str(e)}
    
    def get_suggestions(self, prompt: str) -> Dict[str, List[str]]:
        """
        Get suggestions for improving a prompt.
        
        Args:
            prompt: The prompt to analyze
            
        Returns:
            Dictionary containing improvement suggestions
        """
        try:
            analysis = self.analyzer.analyze_prompt_comprehensive(prompt)
            suggestions = {
                'technical_improvements': [],
                'style_suggestions': [],
                'quality_enhancements': [],
                'composition_tips': []
            }
            
            # Technical improvements
            if analysis.technical_score < 3:
                suggestions['technical_improvements'].extend([
                    "Add camera specifications (e.g., 'shot with Canon EOS R5')",
                    "Include lens details (e.g., '85mm lens', 'macro photography')",
                    "Specify lighting conditions (e.g., 'soft natural light', 'golden hour')"
                ])
            
            # Style suggestions
            if not analysis.style_indicators:
                suggestions['style_suggestions'].extend([
                    "Add artistic style (e.g., 'photorealistic', 'impressionist', 'modern')",
                    "Include mood descriptors (e.g., 'dramatic', 'serene', 'vibrant')",
                    "Specify art movement (e.g., 'renaissance style', 'art nouveau')"
                ])
            
            # Quality enhancements
            if not analysis.quality_indicators:
                suggestions['quality_enhancements'].extend([
                    "Add quality terms (e.g., '4K', 'high resolution', 'award winning')",
                    "Include detail descriptors (e.g., 'intricate details', 'sharp focus')",
                    "Specify professional terms (e.g., 'professional photography', 'masterpiece')"
                ])
            
            # Composition tips
            if not analysis.composition_elements:
                suggestions['composition_tips'].extend([
                    "Add composition rules (e.g., 'rule of thirds', 'leading lines')",
                    "Include depth information (e.g., 'shallow depth of field', 'bokeh')",
                    "Specify camera angles (e.g., 'low angle', 'bird\'s eye view')"
                ])
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error generating suggestions: {e}")
            return {'error': str(e)}
    
    def export_report(self, prompt: str, format: str = 'json') -> Union[str, Dict[str, Any]]:
        """
        Export comprehensive analysis report.
        
        Args:
            prompt: The prompt to analyze
            format: Export format ('json', 'dict')
            
        Returns:
            Report in specified format
        """
        try:
            analysis = self.analyzer.analyze_prompt_comprehensive(prompt)
            parameters = self.analyzer.generate_optimal_parameters(prompt)
            
            report = self.analyzer.export_analysis_report(prompt, analysis, parameters)
            
            if format.lower() == 'json':
                return json.dumps(report, indent=2, default=str)
            else:
                return report
                
        except Exception as e:
            logger.error(f"Error exporting report: {e}")
            error_report = {'error': str(e), 'prompt': prompt}
            
            if format.lower() == 'json':
                return json.dumps(error_report, indent=2)
            else:
                return error_report
    
    def clear_cache(self):
        """Clear the analysis cache."""
        self._cache.clear()
        logger.info("Analysis cache cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            'cached_items': len(self._cache),
            'cache_size_bytes': sum(len(str(v)) for v in self._cache.values())
        }
    
    def _format_analysis_result(self, analysis: PromptAnalysis) -> Dict[str, Any]:
        """Format analysis result for API response."""
        return {
            'complexity': {
                'level': analysis.complexity_level.value,
                'score': analysis.complexity_score,
                'word_count': analysis.word_count,
                'technical_score': analysis.technical_score,
                'detail_score': analysis.detail_score,
                'modifier_score': analysis.modifier_score
            },
            'content': {
                'type': analysis.content_type.value,
                'style_indicators': analysis.style_indicators,
                'quality_indicators': analysis.quality_indicators,
                'composition_elements': analysis.composition_elements,
                'lighting_terms': analysis.lighting_terms
            },
            'technical_categories': analysis.technical_categories,
            'recommendations': {
                'steps': analysis.recommended_steps,
                'cfg_scale': analysis.recommended_cfg,
                'sampler': analysis.recommended_sampler,
                'scheduler': analysis.recommended_scheduler
            }
        }
    
    def _format_parameters_result(self, parameters: GenerationParameters) -> Dict[str, Any]:
        """Format parameters result for API response."""
        return asdict(parameters)
    
    def _empty_analysis_result(self) -> Dict[str, Any]:
        """Return empty analysis result."""
        return {
            'complexity': {
                'level': 'simple',
                'score': 0,
                'word_count': 0,
                'technical_score': 0,
                'detail_score': 0,
                'modifier_score': 0
            },
            'content': {
                'type': 'portrait',
                'style_indicators': [],
                'quality_indicators': [],
                'composition_elements': [],
                'lighting_terms': []
            },
            'technical_categories': {},
            'recommendations': {
                'steps': 20,
                'cfg_scale': 7.0,
                'sampler': 'Euler a',
                'scheduler': 'Automatic'
            }
        }
    
    def _error_analysis_result(self, error_message: str) -> Dict[str, Any]:
        """Return error analysis result."""
        result = self._empty_analysis_result()
        result['error'] = error_message
        return result
    
    def _default_parameters(self) -> Dict[str, Any]:
        """Return default parameters."""
        return {
            'steps': 20,
            'cfg_scale': 7.0,
            'distilled_cfg_scale': 8.4,
            'sampler': 'Euler a',
            'scheduler': 'Automatic',
            'seed': -1,
            'width': 1024,
            'height': 1024,
            'guidance_scale': 7.5,
            'negative_prompt_strength': 0.7
        }

# Global API instance for convenience
api = PromptAnalysisAPI()

# Convenience functions
def analyze_prompt(prompt: str) -> Dict[str, Any]:
    """Convenience function for prompt analysis."""
    return api.analyze(prompt)

def get_optimal_params(prompt: str, **kwargs) -> Dict[str, Any]:
    """Convenience function for getting optimal parameters."""
    return api.get_optimal_parameters(prompt, **kwargs)

def quick_analysis(prompt: str) -> Dict[str, Union[str, int, float]]:
    """Convenience function for quick analysis."""
    return api.quick_analyze(prompt)