
"""
Utility functions for prompt processing and enhancement.
"""

import re
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime

class PromptProcessor:
    """Utility class for processing and enhancing prompts."""
    
    def __init__(self):
        self.image_prompt_structure = {
            "concept": "📸 CONCEPT:",
            "subjects": "👥 SUBJECT(S):",
            "clothing": "👗 CLOTHING:",
            "setting": "🏞️ SETTING:",
            "pose": "💃 POSE & EXPRESSION:",
            "technical": "📷 TECHNICAL:"
        }
        
        self.video_prompt_structure = {
            "concept": "🎬 CONCEPT:",
            "subjects": "👥 SUBJECT(S):",
            "clothing": "👗 CLOTHING:",
            "setting": "🏞️ SETTING:",
            "motion": "🎭 MOTION & ACTION:",
            "camera": "📹 CAMERA WORK:",
            "style": "🎨 STYLE & ATMOSPHERE:",
            "duration": "⏱️ DURATION NOTES:"
        }
        
        self.image_to_video_structure = {
            "analysis": "🖼️ IMAGE ANALYSIS:",
            "concept": "🎬 VIDEO CONCEPT:",
            "motion": "🎭 ADDED MOTION:",
            "camera": "📹 CAMERA DYNAMICS:",
            "atmosphere": "🎨 ENHANCED ATMOSPHERE:",
            "sequence": "⏱️ SEQUENCE FLOW:"
        }

    def validate_prompt_structure(self, prompt: str, prompt_type: str) -> Dict[str, bool]:
        """Validate if a generated prompt follows the expected structure."""
        if prompt_type == "image_prompt":
            structure = self.image_prompt_structure
        elif prompt_type == "video_prompt":
            structure = self.video_prompt_structure
        elif prompt_type == "image_to_video":
            structure = self.image_to_video_structure
        else:
            return {"valid": True, "missing_sections": []}
        
        missing_sections = []
        for section, emoji_header in structure.items():
            if emoji_header not in prompt:
                missing_sections.append(section)
        
        return {
            "valid": len(missing_sections) == 0,
            "missing_sections": missing_sections,
            "total_sections": len(structure),
            "found_sections": len(structure) - len(missing_sections)
        }

    def extract_prompt_sections(self, prompt: str, prompt_type: str) -> Dict[str, str]:
        """Extract individual sections from a structured prompt."""
        if prompt_type == "image_prompt":
            structure = self.image_prompt_structure
        elif prompt_type == "video_prompt":
            structure = self.video_prompt_structure
        elif prompt_type == "image_to_video":
            structure = self.image_to_video_structure
        else:
            return {"raw_content": prompt}
        
        sections = {}
        lines = prompt.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this line starts a new section
            section_found = False
            for section_name, emoji_header in structure.items():
                if line.startswith(emoji_header):
                    # Save previous section if exists
                    if current_section and current_content:
                        sections[current_section] = ' '.join(current_content).strip()
                    
                    # Start new section
                    current_section = section_name
                    current_content = [line[len(emoji_header):].strip()]
                    section_found = True
                    break
            
            # If not a section header, add to current content
            if not section_found and current_section:
                current_content.append(line)
        
        # Save the last section
        if current_section and current_content:
            sections[current_section] = ' '.join(current_content).strip()
        
        return sections

    def enhance_prompt_with_keywords(self, prompt: str, keywords: List[str]) -> str:
        """Enhance a prompt by incorporating additional keywords."""
        if not keywords:
            return prompt
        
        keyword_string = ", ".join(keywords)
        enhanced_prompt = f"{prompt}\n\nAdditional elements to incorporate: {keyword_string}"
        return enhanced_prompt

    def generate_style_variations(self, base_prompt: str) -> List[str]:
        """Generate style variations of a base prompt."""
        styles = [
            "film noir style with dramatic shadows",
            "soft romantic style with warm lighting",
            "high fashion editorial style",
            "vintage pin-up aesthetic",
            "modern minimalist approach",
            "artistic black and white photography",
            "golden hour natural lighting",
            "cyberpunk futuristic style"
        ]
        
        variations = []
        for style in styles:
            variation = f"{base_prompt}\n\nStyle direction: {style}"
            variations.append(variation)
        
        return variations

    def clean_generated_text(self, text: str) -> str:
        """Clean up generated text by removing artifacts and formatting issues."""
        # Remove common AI artifacts
        text = re.sub(r'\[.*?\]', '', text)  # Remove bracketed content
        text = re.sub(r'\(.*?\)', '', text)  # Remove parenthetical content
        text = re.sub(r'<.*?>', '', text)    # Remove HTML-like tags
        
        # Fix spacing issues
        text = re.sub(r'\s+', ' ', text)     # Multiple spaces to single
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Multiple newlines to double
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text

    def format_for_export(self, prompt: str, metadata: Dict) -> Dict:
        """Format a prompt for export with metadata."""
        return {
            "timestamp": datetime.now().isoformat(),
            "prompt_content": prompt,
            "metadata": metadata,
            "word_count": len(prompt.split()),
            "character_count": len(prompt)
        }

class ContentAnalyzer:
    """Analyze and categorize adult content prompts."""
    
    def __init__(self):
        self.content_categories = {
            "artistic": ["artistic", "art", "painting", "sculpture", "renaissance", "classical"],
            "boudoir": ["boudoir", "intimate", "bedroom", "lingerie", "sensual"],
            "fashion": ["fashion", "editorial", "runway", "designer", "haute couture"],
            "lifestyle": ["lifestyle", "casual", "everyday", "natural", "authentic"],
            "glamour": ["glamour", "luxury", "elegant", "sophisticated", "high-end"],
            "fetish": ["latex", "leather", "bondage", "fetish", "kink", "alternative"],
            "vintage": ["vintage", "retro", "pin-up", "classic", "1950s", "1960s"],
            "outdoor": ["outdoor", "nature", "beach", "garden", "natural light"]
        }
        
        self.technical_elements = {
            "lighting": ["lighting", "light", "shadow", "illumination", "glow"],
            "camera": ["camera", "lens", "angle", "shot", "perspective", "framing"],
            "composition": ["composition", "rule of thirds", "symmetry", "balance"],
            "color": ["color", "palette", "hue", "saturation", "contrast", "tone"],
            "texture": ["texture", "fabric", "material", "surface", "pattern"],
            "mood": ["mood", "atmosphere", "feeling", "emotion", "vibe", "energy"]
        }

    def analyze_content_category(self, prompt: str) -> Dict[str, float]:
        """Analyze what category of content a prompt represents."""
        prompt_lower = prompt.lower()
        category_scores = {}
        
        for category, keywords in self.content_categories.items():
            score = sum(1 for keyword in keywords if keyword in prompt_lower)
            category_scores[category] = score / len(keywords)  # Normalize by keyword count
        
        return category_scores

    def analyze_technical_elements(self, prompt: str) -> Dict[str, bool]:
        """Analyze what technical elements are present in a prompt."""
        prompt_lower = prompt.lower()
        elements_present = {}
        
        for element, keywords in self.technical_elements.items():
            elements_present[element] = any(keyword in prompt_lower for keyword in keywords)
        
        return elements_present

    def suggest_improvements(self, prompt: str) -> List[str]:
        """Suggest improvements for a prompt based on analysis."""
        suggestions = []
        technical_analysis = self.analyze_technical_elements(prompt)
        
        if not technical_analysis.get("lighting"):
            suggestions.append("Consider adding specific lighting details (soft, dramatic, natural, etc.)")
        
        if not technical_analysis.get("camera"):
            suggestions.append("Add camera angle or shot type specifications")
        
        if not technical_analysis.get("mood"):
            suggestions.append("Include mood or atmosphere descriptions")
        
        if len(prompt.split()) < 50:
            suggestions.append("Consider adding more descriptive details for richer results")
        
        if len(prompt.split()) > 150:
            suggestions.append("Consider condensing for more focused results")
        
        return suggestions

class PromptLibrary:
    """Manage a library of saved prompts and templates."""
    
    def __init__(self, library_file: str = "prompt_library.json"):
        self.library_file = library_file
        self.library = self.load_library()

    def load_library(self) -> Dict:
        """Load prompt library from file."""
        try:
            with open(self.library_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                "templates": {},
                "saved_prompts": {},
                "favorites": [],
                "tags": {}
            }

    def save_library(self):
        """Save prompt library to file."""
        with open(self.library_file, 'w') as f:
            json.dump(self.library, f, indent=2)

    def add_template(self, name: str, template: str, category: str, tags: List[str] = None):
        """Add a new template to the library."""
        self.library["templates"][name] = {
            "content": template,
            "category": category,
            "tags": tags or [],
            "created": datetime.now().isoformat(),
            "usage_count": 0
        }
        self.save_library()

    def save_prompt(self, name: str, prompt: str, metadata: Dict = None):
        """Save a generated prompt to the library."""
        self.library["saved_prompts"][name] = {
            "content": prompt,
            "metadata": metadata or {},
            "saved": datetime.now().isoformat()
        }
        self.save_library()

    def search_templates(self, query: str, category: str = None) -> List[Dict]:
        """Search templates by query and optionally category."""
        results = []
        query_lower = query.lower()
        
        for name, template in self.library["templates"].items():
            if category and template["category"] != category:
                continue
            
            if (query_lower in name.lower() or 
                query_lower in template["content"].lower() or
                any(query_lower in tag.lower() for tag in template["tags"])):
                results.append({
                    "name": name,
                    "template": template,
                    "relevance_score": self._calculate_relevance(query_lower, name, template)
                })
        
        # Sort by relevance
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results

    def _calculate_relevance(self, query: str, name: str, template: Dict) -> float:
        """Calculate relevance score for search results."""
        score = 0
        
        # Name match (highest weight)
        if query in name.lower():
            score += 10
        
        # Content match
        content_matches = template["content"].lower().count(query)
        score += content_matches * 2
        
        # Tag matches
        tag_matches = sum(1 for tag in template["tags"] if query in tag.lower())
        score += tag_matches * 3
        
        # Usage count (popularity)
        score += template.get("usage_count", 0) * 0.1
        
        return score

    def get_popular_templates(self, limit: int = 10) -> List[Dict]:
        """Get most popular templates by usage count."""
        templates = [
            {"name": name, "template": template}
            for name, template in self.library["templates"].items()
        ]
        
        templates.sort(key=lambda x: x["template"].get("usage_count", 0), reverse=True)
        return templates[:limit]

    def add_to_favorites(self, prompt_name: str):
        """Add a prompt to favorites."""
        if prompt_name not in self.library["favorites"]:
            self.library["favorites"].append(prompt_name)
            self.save_library()

    def remove_from_favorites(self, prompt_name: str):
        """Remove a prompt from favorites."""
        if prompt_name in self.library["favorites"]:
            self.library["favorites"].remove(prompt_name)
            self.save_library()

    def get_statistics(self) -> Dict:
        """Get library statistics."""
        return {
            "total_templates": len(self.library["templates"]),
            "total_saved_prompts": len(self.library["saved_prompts"]),
            "total_favorites": len(self.library["favorites"]),
            "categories": list(set(t["category"] for t in self.library["templates"].values())),
            "most_used_template": max(
                self.library["templates"].items(),
                key=lambda x: x[1].get("usage_count", 0),
                default=(None, {"usage_count": 0})
            )[0]
        }

# Example usage and default templates
DEFAULT_TEMPLATES = {
    "boudoir_basic": {
        "content": "Create an elegant boudoir photo concept with {subject_description} in {setting} featuring {lighting_style} lighting",
        "category": "boudoir",
        "tags": ["boudoir", "intimate", "elegant", "template"]
    },
    "fashion_editorial": {
        "content": "Design a high-fashion editorial concept featuring {clothing_style} in {location} with {mood} atmosphere",
        "category": "fashion", 
        "tags": ["fashion", "editorial", "high-end", "template"]
    },
    "artistic_nude": {
        "content": "Create an artistic nude concept inspired by {art_style} with {lighting_description} and {composition_notes}",
        "category": "artistic",
        "tags": ["artistic", "nude", "fine art", "template"]
    }
}

def initialize_default_library():
    """Initialize library with default templates."""
    library = PromptLibrary()
    
    for name, template_data in DEFAULT_TEMPLATES.items():
        if name not in library.library["templates"]:
            library.add_template(
                name=name,
                template=template_data["content"],
                category=template_data["category"],
                tags=template_data["tags"]
            )
    
    return library
