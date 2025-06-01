#!/usr/bin/env python3
"""
Test suite for the Intelligent Prompt Analysis System
"""

import unittest
import json
from typing import Dict, Any

from prompt_analyzer import (
    IntelligentPromptAnalyzer, 
    ComplexityLevel, 
    ContentType,
    analyze_prompt_complexity,
    recommend_sd_forge_params
)
from api_wrapper import PromptAnalysisAPI, analyze_prompt, get_optimal_params

class TestIntelligentPromptAnalyzer(unittest.TestCase):
    """Test cases for the core prompt analyzer"""
    
    def setUp(self):
        self.analyzer = IntelligentPromptAnalyzer()
    
    def test_empty_prompt(self):
        """Test handling of empty prompts"""
        analysis = self.analyzer.analyze_prompt_comprehensive("")
        self.assertEqual(analysis.complexity_level, ComplexityLevel.SIMPLE)
        self.assertEqual(analysis.complexity_score, 0)
        self.assertEqual(analysis.word_count, 0)
    
    def test_simple_prompt(self):
        """Test analysis of simple prompts"""
        prompt = "a beautiful woman"
        analysis = self.analyzer.analyze_prompt_comprehensive(prompt)
        
        self.assertEqual(analysis.complexity_level, ComplexityLevel.SIMPLE)
        self.assertEqual(analysis.word_count, 3)
        self.assertGreater(analysis.complexity_score, 0)
        self.assertEqual(analysis.content_type, ContentType.PORTRAIT)
    
    def test_complex_prompt(self):
        """Test analysis of complex prompts"""
        prompt = """extremely detailed photorealistic portrait of a stunning 25-year-old woman 
                   with intricate jewelry, shot with Canon EOS R5, 85mm lens, shallow depth of field,
                   dramatic lighting, golden hour, award winning photography, 4K resolution"""
        
        analysis = self.analyzer.analyze_prompt_comprehensive(prompt)
        
        self.assertIn(analysis.complexity_level, [ComplexityLevel.COMPLEX, ComplexityLevel.VERY_COMPLEX])
        self.assertGreater(analysis.technical_score, 5)
        self.assertGreater(analysis.detail_score, 2)
        self.assertGreater(len(analysis.quality_indicators), 0)
        self.assertGreater(len(analysis.lighting_terms), 0)
    
    def test_technical_categories(self):
        """Test technical category detection"""
        prompt = "DSLR camera shot with 50mm lens, soft lighting, rule of thirds composition"
        analysis = self.analyzer.analyze_prompt_comprehensive(prompt)
        
        self.assertGreater(analysis.technical_categories['camera'], 0)
        self.assertGreater(analysis.technical_categories['lighting'], 0)
        self.assertGreater(analysis.technical_categories['composition'], 0)
    
    def test_content_type_detection(self):
        """Test content type detection"""
        test_cases = [
            ("portrait of a person", ContentType.PORTRAIT),
            ("beautiful landscape with mountains", ContentType.LANDSCAPE),
            ("abstract art painting", ContentType.ARTISTIC),
            ("photorealistic image", ContentType.PHOTOREALISTIC),
            ("fantasy dragon scene", ContentType.FANTASY)
        ]
        
        for prompt, expected_type in test_cases:
            analysis = self.analyzer.analyze_prompt_comprehensive(prompt)
            self.assertEqual(analysis.content_type, expected_type, 
                           f"Failed for prompt: {prompt}")
    
    def test_parameter_generation(self):
        """Test parameter generation"""
        prompt = "detailed portrait with dramatic lighting"
        params = self.analyzer.generate_optimal_parameters(prompt)
        
        self.assertIsInstance(params.steps, int)
        self.assertGreater(params.steps, 0)
        self.assertLessEqual(params.steps, 80)
        
        self.assertIsInstance(params.cfg_scale, float)
        self.assertGreaterEqual(params.cfg_scale, 1.0)
        self.assertLessEqual(params.cfg_scale, 20.0)
        
        self.assertIsInstance(params.sampler, str)
        self.assertIsInstance(params.scheduler, str)
    
    def test_user_overrides(self):
        """Test user parameter overrides"""
        prompt = "simple portrait"
        custom_sampler = "DDIM"
        custom_scheduler = "Exponential"
        custom_cfg = 12.0
        
        params = self.analyzer.generate_optimal_parameters(
            prompt, custom_sampler, custom_scheduler, custom_cfg
        )
        
        self.assertEqual(params.sampler, custom_sampler)
        self.assertEqual(params.scheduler, custom_scheduler)
        self.assertEqual(params.cfg_scale, custom_cfg)
    
    def test_backward_compatibility(self):
        """Test backward compatibility functions"""
        prompt = "beautiful portrait with soft lighting"
        
        # Test analyze_prompt_complexity
        analysis = analyze_prompt_complexity(prompt)
        self.assertIn('complexity_level', analysis)
        self.assertIn('complexity_score', analysis)
        self.assertIn('technical_categories', analysis)
        
        # Test recommend_sd_forge_params
        params = recommend_sd_forge_params(prompt)
        self.assertIn('steps', params)
        self.assertIn('sampler', params)
        self.assertIn('cfg_scale', params)

class TestPromptAnalysisAPI(unittest.TestCase):
    """Test cases for the API wrapper"""
    
    def setUp(self):
        self.api = PromptAnalysisAPI()
    
    def test_api_analyze(self):
        """Test API analyze function"""
        prompt = "beautiful landscape photography"
        result = self.api.analyze(prompt)
        
        self.assertIn('complexity', result)
        self.assertIn('content', result)
        self.assertIn('recommendations', result)
        
        # Check structure
        self.assertIn('level', result['complexity'])
        self.assertIn('score', result['complexity'])
        self.assertIn('type', result['content'])
    
    def test_api_parameters(self):
        """Test API parameter generation"""
        prompt = "detailed portrait"
        result = self.api.get_optimal_parameters(prompt)
        
        self.assertIn('steps', result)
        self.assertIn('cfg_scale', result)
        self.assertIn('sampler', result)
        self.assertIn('scheduler', result)
    
    def test_quick_analyze(self):
        """Test quick analysis function"""
        prompt = "artistic portrait"
        result = self.api.quick_analyze(prompt)
        
        expected_keys = ['complexity', 'score', 'recommended_steps', 
                        'recommended_cfg', 'recommended_sampler', 'content_type']
        
        for key in expected_keys:
            self.assertIn(key, result)
    
    def test_batch_analyze(self):
        """Test batch analysis"""
        prompts = [
            "simple portrait",
            "complex detailed landscape with dramatic lighting",
            "abstract art"
        ]
        
        results = self.api.batch_analyze(prompts)
        
        self.assertEqual(len(results), len(prompts))
        for i, result in enumerate(results):
            self.assertEqual(result['batch_index'], i)
            self.assertIn('complexity', result)
    
    def test_compare_prompts(self):
        """Test prompt comparison"""
        prompt1 = "simple portrait"
        prompt2 = "extremely detailed photorealistic portrait with dramatic lighting"
        
        comparison = self.api.compare_prompts(prompt1, prompt2)
        
        self.assertIn('prompt1', comparison)
        self.assertIn('prompt2', comparison)
        self.assertIn('comparison', comparison)
        
        # The second prompt should be more complex
        self.assertGreater(comparison['comparison']['complexity_difference'], 0)
    
    def test_suggestions(self):
        """Test improvement suggestions"""
        prompt = "woman"  # Very simple prompt
        suggestions = self.api.get_suggestions(prompt)
        
        expected_categories = ['technical_improvements', 'style_suggestions', 
                             'quality_enhancements', 'composition_tips']
        
        for category in expected_categories:
            self.assertIn(category, suggestions)
            self.assertIsInstance(suggestions[category], list)
    
    def test_export_report(self):
        """Test report export"""
        prompt = "detailed portrait"
        
        # Test JSON export
        json_report = self.api.export_report(prompt, 'json')
        self.assertIsInstance(json_report, str)
        
        # Verify it's valid JSON
        parsed = json.loads(json_report)
        self.assertIn('prompt', parsed)
        self.assertIn('analysis', parsed)
        self.assertIn('parameters', parsed)
        
        # Test dict export
        dict_report = self.api.export_report(prompt, 'dict')
        self.assertIsInstance(dict_report, dict)
    
    def test_caching(self):
        """Test caching functionality"""
        prompt = "test prompt for caching"
        
        # Clear cache first
        self.api.clear_cache()
        stats = self.api.get_cache_stats()
        self.assertEqual(stats['cached_items'], 0)
        
        # Analyze with caching
        result1 = self.api.analyze(prompt, use_cache=True)
        stats = self.api.get_cache_stats()
        self.assertEqual(stats['cached_items'], 1)
        
        # Analyze again (should use cache)
        result2 = self.api.analyze(prompt, use_cache=True)
        self.assertEqual(result1, result2)
    
    def test_convenience_functions(self):
        """Test convenience functions"""
        prompt = "beautiful portrait"
        
        # Test analyze_prompt
        analysis = analyze_prompt(prompt)
        self.assertIn('complexity', analysis)
        
        # Test get_optimal_params
        params = get_optimal_params(prompt)
        self.assertIn('steps', params)

class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def setUp(self):
        self.analyzer = IntelligentPromptAnalyzer()
        self.api = PromptAnalysisAPI()
    
    def test_very_long_prompt(self):
        """Test handling of very long prompts"""
        # Create a very long prompt
        long_prompt = " ".join(["detailed"] * 100)
        
        analysis = self.analyzer.analyze_prompt_comprehensive(long_prompt)
        self.assertIsNotNone(analysis)
        self.assertEqual(analysis.word_count, 100)
    
    def test_special_characters(self):
        """Test handling of special characters"""
        prompt = "portrait with émotions and café lighting, 50% opacity"
        
        analysis = self.analyzer.analyze_prompt_comprehensive(prompt)
        self.assertIsNotNone(analysis)
        self.assertGreater(analysis.word_count, 0)
    
    def test_numeric_prompts(self):
        """Test handling of numeric content"""
        prompt = "4K 8K resolution 1080p HD quality 50mm f/1.4"
        
        analysis = self.analyzer.analyze_prompt_comprehensive(prompt)
        self.assertGreater(analysis.technical_score, 0)
        self.assertGreater(len(analysis.quality_indicators), 0)
    
    def test_mixed_case_prompts(self):
        """Test handling of mixed case"""
        prompt = "DRAMATIC Lighting with Soft SHADOWS"
        
        analysis = self.analyzer.analyze_prompt_comprehensive(prompt)
        self.assertGreater(analysis.technical_categories['lighting'], 0)

def run_performance_tests():
    """Run performance tests"""
    import time
    
    analyzer = IntelligentPromptAnalyzer()
    api = PromptAnalysisAPI()
    
    test_prompts = [
        "simple portrait",
        "detailed landscape with mountains and rivers",
        "extremely complex photorealistic portrait with intricate details, dramatic lighting, shot with professional camera",
        "abstract art with vibrant colors and geometric shapes",
        "fantasy scene with magical elements and mystical atmosphere"
    ]
    
    print("Running performance tests...")
    
    # Test analyzer performance
    start_time = time.time()
    for prompt in test_prompts * 10:  # 50 total analyses
        analyzer.analyze_prompt_comprehensive(prompt)
    analyzer_time = time.time() - start_time
    
    # Test API performance
    start_time = time.time()
    for prompt in test_prompts * 10:  # 50 total analyses
        api.analyze(prompt)
    api_time = time.time() - start_time
    
    print(f"Analyzer: {analyzer_time:.3f}s for 50 analyses ({analyzer_time/50*1000:.1f}ms per analysis)")
    print(f"API: {api_time:.3f}s for 50 analyses ({api_time/50*1000:.1f}ms per analysis)")
    
    # Test batch performance
    start_time = time.time()
    api.batch_analyze(test_prompts * 10)
    batch_time = time.time() - start_time
    
    print(f"Batch API: {batch_time:.3f}s for 50 analyses ({batch_time/50*1000:.1f}ms per analysis)")

if __name__ == '__main__':
    # Run unit tests
    print("Running unit tests...")
    unittest.main(verbosity=2, exit=False)
    
    # Run performance tests
    print("\n" + "="*50)
    run_performance_tests()