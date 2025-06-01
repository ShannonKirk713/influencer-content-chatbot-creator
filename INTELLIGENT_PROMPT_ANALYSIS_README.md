# Intelligent Prompt Analysis System

## Overview

This system replaces static parameter selection (like fixed 55 steps) with dynamic, research-based analysis that adapts to your prompt's complexity and content type.

## Key Features

### 🧠 Dynamic Parameter Optimization
- **Adaptive Steps**: Automatically calculates optimal steps (20-80) based on prompt complexity
- **Smart Sampler Selection**: Chooses best sampler based on content type and complexity
- **Dynamic CFG**: Optimizes CFG scale for your specific prompt
- **Content-Aware Resolution**: Adjusts resolution based on content type (portrait vs landscape)

### 📊 Comprehensive Analysis
- **Complexity Scoring**: Multi-factor analysis including word count, technical terms, detail indicators
- **Content Type Detection**: Automatically identifies portraits, landscapes, artistic styles, etc.
- **Technical Category Recognition**: Detects camera, lighting, composition, style, and quality terms
- **Style and Quality Assessment**: Identifies artistic styles and quality indicators

### ⚙️ Research-Based Optimization
- **Sampler Profiles**: Different samplers optimized for different complexity levels
- **Scheduler Selection**: Intelligent scheduler choice based on technical content
- **CFG Scaling**: Dynamic CFG calculation with Flux-optimized distilled CFG
- **Resolution Optimization**: Content-type-aware resolution selection

## Usage

### Basic Analysis
```python
from prompt_analyzer import IntelligentPromptAnalyzer

analyzer = IntelligentPromptAnalyzer()
analysis = analyzer.analyze_prompt_comprehensive("your prompt here")
parameters = analyzer.generate_optimal_parameters("your prompt here")
```

### API Wrapper
```python
from api_wrapper import PromptAnalysisAPI

api = PromptAnalysisAPI()
result = api.analyze("your prompt here")
params = api.get_optimal_parameters("your prompt here")
```

### Integration with Existing Code
The system provides backward compatibility functions:
```python
from prompt_analyzer import analyze_prompt_complexity, recommend_sd_forge_params

# Legacy compatibility
analysis = analyze_prompt_complexity("your prompt")
params = recommend_sd_forge_params("your prompt")
```

## System Architecture

### Core Components

1. **IntelligentPromptAnalyzer**: Main analysis engine
2. **PromptAnalysisAPI**: High-level API wrapper
3. **ComplexityLevel**: Enum for complexity categorization
4. **ContentType**: Enum for content type classification
5. **PromptAnalysis**: Comprehensive analysis results
6. **GenerationParameters**: Optimized parameter set

### Analysis Pipeline

1. **Text Processing**: Tokenization and preprocessing
2. **Technical Analysis**: Detection of technical terms across categories
3. **Complexity Scoring**: Multi-factor complexity calculation
4. **Content Classification**: Automatic content type detection
5. **Parameter Optimization**: Research-based parameter selection
6. **Result Formatting**: Structured output for easy integration

## Technical Categories

The system analyzes prompts across multiple technical categories:

- **Camera**: DSLR, lens specifications, camera models
- **Lighting**: Natural light, studio lighting, dramatic lighting
- **Composition**: Rule of thirds, depth of field, framing
- **Style**: Photorealistic, artistic styles, art movements
- **Quality**: Resolution terms, professional descriptors
- **Post-processing**: HDR, color grading, effects

## Complexity Levels

- **Simple** (0-19): Basic prompts with minimal technical content
- **Moderate** (20-39): Prompts with some technical terms and detail
- **Complex** (40-69): Detailed prompts with multiple technical elements
- **Very Complex** (70+): Highly detailed prompts with extensive technical content

## Parameter Optimization

### Steps Calculation
- Base steps determined by complexity level
- Technical bonus added based on technical term count
- Capped at reasonable maximum (80 steps)

### Sampler Selection
- Simple prompts: Euler a, DPM++ 2M
- Moderate prompts: DPM++ 2M Karras, DPM++ 2M SDE
- Complex prompts: DPM++ 2M SDE Karras, DDIM
- Very complex prompts: DDIM, PLMS

### CFG Optimization
- Base CFG determined by complexity level
- Adjusted for content type (artistic vs photorealistic)
- Distilled CFG calculated with appropriate multipliers

### Resolution Selection
- Portrait: 768x1024
- Landscape: 1024x768
- Square formats: 1024x1024
- Content-type optimized

## Integration Guide

### Replacing Static Parameters

**Before (Static):**
```python
steps = 55  # Always the same
sampler = "DPM++ 2M"  # Fixed choice
cfg = 7.0  # Static value
```

**After (Dynamic):**
```python
analyzer = IntelligentPromptAnalyzer()
params = analyzer.generate_optimal_parameters(prompt)
steps = params.steps  # 20-80 based on complexity
sampler = params.sampler  # Optimized choice
cfg = params.cfg_scale  # Dynamic calculation
```

### Gradio Interface Integration

The system integrates seamlessly with Gradio interfaces:
```python
def analyze_prompt_intelligent(prompt, user_sampler, user_scheduler, user_cfg):
    api = PromptAnalysisAPI()
    analysis = api.analyze(prompt)
    params = api.get_optimal_parameters(prompt, user_sampler, user_scheduler, user_cfg)
    return format_results(analysis, params)
```

## Performance

- **Analysis Speed**: ~1-5ms per prompt
- **Memory Usage**: Minimal (< 10MB)
- **Caching**: Built-in caching for repeated prompts
- **Batch Processing**: Efficient batch analysis support

## Testing

Run the comprehensive test suite:
```bash
python test_prompt_analyzer.py
```

Tests include:
- Unit tests for all components
- Edge case handling
- Performance benchmarks
- Integration tests
- Backward compatibility verification

## Migration from Legacy System

1. **Install**: No additional dependencies required
2. **Import**: Add new imports to existing code
3. **Replace**: Replace static parameter calls with dynamic analysis
4. **Test**: Verify functionality with existing prompts
5. **Optimize**: Fine-tune based on results

## Benefits

### For Users
- **Better Results**: Optimized parameters for each prompt
- **Consistency**: Reliable parameter selection
- **Efficiency**: No manual parameter tuning needed
- **Quality**: Research-based optimization

### For Developers
- **Maintainability**: Centralized parameter logic
- **Extensibility**: Easy to add new analysis features
- **Compatibility**: Backward compatible with existing code
- **Documentation**: Comprehensive analysis reporting

## Future Enhancements

- **Model-Specific Optimization**: Parameters optimized for specific models
- **User Learning**: Adaptation based on user preferences
- **Advanced Content Detection**: More sophisticated content analysis
- **Performance Optimization**: Further speed improvements
- **Extended Technical Categories**: Additional technical term categories

## Support

For issues, questions, or contributions:
1. Check the test suite for examples
2. Review the API documentation
3. Examine the integration examples
4. Test with the provided test cases

The system is designed to be robust, efficient, and easy to integrate while providing significant improvements over static parameter selection.