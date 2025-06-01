#!/usr/bin/env python3
"""
Prompt utilities for Fanvue Chatbot
Utility functions for processing and formatting prompts
"""

import re
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime

class PromptFormatter:
    """Utility class for formatting and processing prompts."""
    
    def __init__(self):
        self.emoji_patterns = {
            'concept': '📸',
            'subjects': '👥', 
            'clothing': '👗',
            'setting': '🏞️',
            'pose': '💃',
            'technical': '📷',
            'video_concept': '🎬',
            'motion': '🎭',
            'camera': '📹',
            'style': '🎨',
            'duration': '⏱️',
            'image_analysis': '🖼️'
        }
    
    def validate_image_prompt(self, prompt: str) -> Tuple[bool, List[str]]:
        """Validate if an image prompt contains all required sections."""
        required_sections = ['📸 CONCEPT:', '👥 SUBJECT(S):', '👗 CLOTHING:', 
                           '🏞️ SETTING:', '💃 POSE & EXPRESSION:', '📷 TECHNICAL:']
        
        missing_sections = []
        for section in required_sections:
            if section not in prompt:
                missing_sections.append(section)
        
        is_valid = len(missing_sections) == 0
        return is_valid, missing_sections
    
    def validate_video_prompt(self, prompt: str) -> Tuple[bool, List[str]]:
        """Validate if a video prompt contains all required sections."""
        required_sections = ['🎬 CONCEPT:', '👥 SUBJECT(S):', '👗 CLOTHING:',
                           '🏞️ SETTING:', '🎭 MOTION & ACTION:', '📹 CAMERA WORK:',
                           '🎨 STYLE & ATMOSPHERE:', '⏱️ DURATION NOTES:']
        
        missing_sections = []
        for section in required_sections:
            if section not in prompt:
                missing_sections.append(section)
        
        is_valid = len(missing_sections) == 0
        return is_valid, missing_sections
    
    def extract_sections(self, prompt: str) -> Dict[str, str]:
        """Extract sections from a formatted prompt."""
        sections = {}
        
        # Define patterns for different emoji sections
        patterns = {
            'concept': r'📸 CONCEPT:(.*?)(?=👥|$)',
            'video_concept': r'🎬 CONCEPT:(.*?)(?=👥|$)',
            'subjects': r'👥 SUBJECT\(S\):(.*?)(?=👗|🏞️|$)',
            'clothing': r'👗 CLOTHING:(.*?)(?=🏞️|💃|$)',
            'setting': r'🏞️ SETTING:(.*?)(?=💃|🎭|📹|$)',
            'pose': r'💃 POSE & EXPRESSION:(.*?)(?=📷|$)',
            'motion': r'🎭 MOTION & ACTION:(.*?)(?=📹|$)',
            'camera': r'📹 CAMERA WORK:(.*?)(?=🎨|$)',
            'technical': r'📷 TECHNICAL:(.*?)$',
            'style': r'🎨 STYLE & ATMOSPHERE:(.*?)(?=⏱️|$)',
            'duration': r'⏱️ DURATION NOTES:(.*?)$',
            'image_analysis': r'🖼️ IMAGE ANALYSIS:(.*?)(?=🎬|$)'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, prompt, re.DOTALL | re.IGNORECASE)
            if match:
                sections[key] = match.group(1).strip()
        
        return sections
    
    def format_prompt_for_export(self, prompt: str, metadata: Dict) -> Dict:
        """Format a prompt for export with metadata."""
        sections = self.extract_sections(prompt)
        
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata,
            'raw_prompt': prompt,
            'sections': sections,
            'word_count': len(prompt.split()),
            'character_count': len(prompt)
        }
        
        return export_data
    
    def clean_response(self, response: str) -> str:
        """Clean up AI model response."""
        # Remove common AI response prefixes
        prefixes_to_remove = [
            "ASSISTANT:", "Assistant:", "AI:", "Response:", 
            "Here's", "Here is", "I'll create", "I'll generate"
        ]
        
        cleaned = response.strip()
        
        for prefix in prefixes_to_remove:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
                break
        
        # Remove excessive newlines
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        
        return cleaned
    
    def enhance_prompt_with_keywords(self, prompt: str, keywords: List[str]) -> str:
        """Enhance a prompt by incorporating additional keywords."""
        if not keywords:
            return prompt
        
        keyword_string = ", ".join(keywords)
        enhanced_prompt = f"{prompt}\n\nAdditional elements to consider: {keyword_string}"
        
        return enhanced_prompt
    
    def generate_variations(self, base_prompt: str, variation_count: int = 3) -> List[str]:
        """Generate variations of a base prompt."""
        variations = []
        
        # Define variation templates
        variation_templates = [
            "Create a more artistic version of: {prompt}",
            "Design a more intimate variation of: {prompt}", 
            "Generate a more dramatic interpretation of: {prompt}",
            "Develop a softer, more romantic version of: {prompt}",
            "Create a more explicit variation of: {prompt}"
        ]
        
        for i in range(min(variation_count, len(variation_templates))):
            variation = variation_templates[i].format(prompt=base_prompt)
            variations.append(variation)
        
        return variations

class ContentAnalyzer:
    """Analyze and categorize generated content."""
    
    def __init__(self):
        self.content_categories = {
            'artistic': ['artistic', 'aesthetic', 'beautiful', 'elegant', 'graceful'],
            'intimate': ['intimate', 'sensual', 'romantic', 'tender', 'passionate'],
            'explicit': ['explicit', 'erotic', 'sexual', 'provocative', 'seductive'],
            'fashion': ['fashion', 'style', 'clothing', 'lingerie', 'outfit'],
            'technical': ['lighting', 'camera', 'photography', 'technical', 'professional']
        }
    
    def analyze_content_type(self, content: str) -> Dict[str, float]:
        """Analyze content and return category scores."""
        content_lower = content.lower()
        scores = {}
        
        for category, keywords in self.content_categories.items():
            score = 0
            for keyword in keywords:
                score += content_lower.count(keyword)
            
            # Normalize score by content length
            word_count = len(content.split())
            normalized_score = score / max(word_count, 1) * 100
            scores[category] = normalized_score
        
        return scores
    
    def get_dominant_category(self, content: str) -> str:
        """Get the dominant category for content."""
        scores = self.analyze_content_type(content)
        if not scores:
            return 'general'
        
        return max(scores, key=scores.get)
    
    def calculate_readability_score(self, content: str) -> float:
        """Calculate a simple readability score."""
        sentences = content.split('.')
        words = content.split()
        
        if len(sentences) == 0 or len(words) == 0:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        
        # Simple readability score (lower is better)
        score = max(0, 100 - (avg_sentence_length * 2))
        return score

class PromptLibrary:
    """Manage a library of prompts and templates."""
    
    def __init__(self, library_file: str = "prompt_library.json"):
        self.library_file = library_file
        self.prompts = self.load_library()
    
    def load_library(self) -> Dict:
        """Load prompt library from file."""
        try:
            with open(self.library_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {'image_prompts': [], 'video_prompts': [], 'templates': {}}
    
    def save_library(self):
        """Save prompt library to file."""
        with open(self.library_file, 'w') as f:
            json.dump(self.prompts, f, indent=2)
    
    def add_prompt(self, prompt: str, category: str, tags: List[str] = None):
        """Add a prompt to the library."""
        if tags is None:
            tags = []
        
        prompt_entry = {
            'content': prompt,
            'tags': tags,
            'created_at': datetime.now().isoformat(),
            'usage_count': 0
        }
        
        if category not in self.prompts:
            self.prompts[category] = []
        
        self.prompts[category].append(prompt_entry)
        self.save_library()
    
    def search_prompts(self, query: str, category: str = None) -> List[Dict]:
        """Search prompts by query."""
        results = []
        query_lower = query.lower()
        
        categories_to_search = [category] if category else self.prompts.keys()
        
        for cat in categories_to_search:
            if cat in self.prompts:
                for prompt in self.prompts[cat]:
                    if (query_lower in prompt['content'].lower() or 
                        any(query_lower in tag.lower() for tag in prompt['tags'])):
                        results.append({**prompt, 'category': cat})
        
        return results
    
    def get_popular_prompts(self, category: str = None, limit: int = 10) -> List[Dict]:
        """Get most popular prompts by usage count."""
        all_prompts = []
        
        categories_to_check = [category] if category else self.prompts.keys()
        
        for cat in categories_to_check:
            if cat in self.prompts:
                for prompt in self.prompts[cat]:
                    all_prompts.append({**prompt, 'category': cat})
        
        # Sort by usage count
        sorted_prompts = sorted(all_prompts, key=lambda x: x['usage_count'], reverse=True)
        
        return sorted_prompts[:limit]
    
    def increment_usage(self, prompt_content: str):
        """Increment usage count for a prompt."""
        for category in self.prompts:
            for prompt in self.prompts[category]:
                if prompt['content'] == prompt_content:
                    prompt['usage_count'] += 1
                    self.save_library()
                    return

# Utility functions
def create_prompt_formatter() -> PromptFormatter:
    """Create a new prompt formatter instance."""
    return PromptFormatter()

def create_content_analyzer() -> ContentAnalyzer:
    """Create a new content analyzer instance."""
    return ContentAnalyzer()

def create_prompt_library(library_file: str = "prompt_library.json") -> PromptLibrary:
    """Create a new prompt library instance."""
    return PromptLibrary(library_file)

# Example usage and testing
if __name__ == "__main__":
    # Test the utilities
    formatter = create_prompt_formatter()
    analyzer = create_content_analyzer()
    
    # Test prompt validation
    sample_image_prompt = """
    📸 CONCEPT: Elegant boudoir photography
    👥 SUBJECT(S): Beautiful woman, 25 years old
    👗 CLOTHING: Black lace lingerie
    🏞️ SETTING: Luxurious bedroom
    💃 POSE & EXPRESSION: Reclining pose
    📷 TECHNICAL: Soft lighting, 85mm lens
    """
    
    is_valid, missing = formatter.validate_image_prompt(sample_image_prompt)
    print(f"Image prompt valid: {is_valid}")
    if missing:
        print(f"Missing sections: {missing}")
    
    # Test section extraction
    sections = formatter.extract_sections(sample_image_prompt)
    print(f"Extracted sections: {list(sections.keys())}")
    
    # Test content analysis
    scores = analyzer.analyze_content_type(sample_image_prompt)
    print(f"Content scores: {scores}")
    
    dominant = analyzer.get_dominant_category(sample_image_prompt)
    print(f"Dominant category: {dominant}")
