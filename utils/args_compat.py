"""
Compatible arguments utility for the enhanced chatbot system.
"""

def get_compatible_args():
    """
    Return a comprehensive list of all compatible arguments and parameters.
    
    Returns:
        str: Formatted string containing all compatible arguments
    """
    
    args_documentation = """
# Compatible Arguments Documentation

This document provides a comprehensive list of all compatible arguments and parameters for the Enhanced Image-to-Video Chatbot system.

## Image Analysis

Arguments for advanced image analysis using CLIP, YOLO, SAM, and GPT-4V

### `clip_model`
- **Type**: string
- **Default**: ViT-B/32
- **Options**: ViT-B/32, ViT-B/16, ViT-L/14, RN50, RN101
- **Description**: CLIP model variant for image-text understanding

### `yolo_model`
- **Type**: string
- **Default**: yolov8n.pt
- **Options**: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
- **Description**: YOLOv8 model size for object detection

### `sam_model`
- **Type**: string
- **Default**: vit_h
- **Options**: vit_h, vit_l, vit_b
- **Description**: Segment Anything Model variant

### `confidence_threshold`
- **Type**: float
- **Default**: 0.25
- **Range**: 0.1 - 1.0
- **Description**: Confidence threshold for object detection

### `max_detections`
- **Type**: integer
- **Default**: 100
- **Range**: 1 - 1000
- **Description**: Maximum number of objects to detect

## Video Generation

Arguments for video generation with automatic prompt analysis

### `cfg_scale`
- **Type**: float
- **Default**: 1.0
- **Fixed**: True (cannot be changed)
- **Description**: CFG scale fixed at 1.0 for maximum creative freedom

### `num_frames`
- **Type**: integer
- **Default**: 16
- **Options**: 8, 16, 24, 32
- **Description**: Number of frames in generated video

### `fps`
- **Type**: integer
- **Default**: 8
- **Options**: 4, 8, 12, 16, 24
- **Description**: Frames per second for output video

### `resolution`
- **Type**: string
- **Default**: 512x512
- **Options**: 256x256, 512x512, 768x768, 1024x1024
- **Description**: Output video resolution

### `motion_strength`
- **Type**: float
- **Default**: 0.8
- **Range**: 0.1 - 1.0
- **Description**: Strength of motion in generated video

## Prompt Analysis

Automatic prompt analysis parameters (no manual input required)

### `sampler`
- **Type**: string
- **Default**: DPMSolverMultistep
- **Auto-selected**: True
- **Description**: Automatically selected optimal sampler

### `scheduler`
- **Type**: string
- **Default**: karras
- **Auto-selected**: True
- **Description**: Automatically selected optimal scheduler

### `distilled_cfg`
- **Type**: boolean
- **Default**: True
- **Auto-enabled**: True
- **Description**: Distilled CFG automatically enabled for better quality

### `prompt_enhancement`
- **Type**: boolean
- **Default**: True
- **Description**: Automatic prompt enhancement using GPT-4V analysis

### `style_detection`
- **Type**: boolean
- **Default**: True
- **Description**: Automatic style and mood detection from image

## System Settings

System-wide settings and configurations

### `auto_open_browser`
- **Type**: boolean
- **Default**: True
- **Description**: Automatically open web browser on startup

### `max_file_size`
- **Type**: string
- **Default**: 16MB
- **Description**: Maximum upload file size

### `supported_formats`
- **Type**: array
- **Default**: ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp']
- **Description**: Supported image formats

### `cache_models`
- **Type**: boolean
- **Default**: True
- **Description**: Cache loaded models for faster inference

### `gpu_acceleration`
- **Type**: boolean
- **Default**: True
- **Description**: Use GPU acceleration when available

## Advanced Features

Advanced features and experimental options

### `two_stage_analysis`
- **Type**: boolean
- **Default**: True
- **Description**: Split image analysis into two parts: detection + interpretation

### `multi_model_ensemble`
- **Type**: boolean
- **Default**: True
- **Description**: Use ensemble of CLIP, YOLO, SAM, and GPT-4V for analysis

### `dynamic_prompt_generation`
- **Type**: boolean
- **Default**: True
- **Description**: Generate dynamic video prompts automatically

### `scene_understanding`
- **Type**: boolean
- **Default**: True
- **Description**: Advanced scene understanding and context analysis

### `motion_prediction`
- **Type**: boolean
- **Default**: True
- **Description**: Predict natural motion patterns from static images
"""
    
    return args_documentation

def get_model_configs():
    """
    Return configuration for all supported models.
    
    Returns:
        dict: Dictionary containing model configurations
    """
    
    return {
        "clip_models": {
            "ViT-B/32": {"size": "small", "speed": "fast", "accuracy": "good"},
            "ViT-B/16": {"size": "medium", "speed": "medium", "accuracy": "better"},
            "ViT-L/14": {"size": "large", "speed": "slow", "accuracy": "best"},
            "RN50": {"size": "medium", "speed": "fast", "accuracy": "good"},
            "RN101": {"size": "large", "speed": "medium", "accuracy": "better"}
        },
        "yolo_models": {
            "yolov8n.pt": {"size": "nano", "speed": "fastest", "accuracy": "good"},
            "yolov8s.pt": {"size": "small", "speed": "fast", "accuracy": "better"},
            "yolov8m.pt": {"size": "medium", "speed": "medium", "accuracy": "good"},
            "yolov8l.pt": {"size": "large", "speed": "slow", "accuracy": "better"},
            "yolov8x.pt": {"size": "extra_large", "speed": "slowest", "accuracy": "best"}
        },
        "sam_models": {
            "vit_h": {"size": "huge", "speed": "slow", "accuracy": "best"},
            "vit_l": {"size": "large", "speed": "medium", "accuracy": "better"},
            "vit_b": {"size": "base", "speed": "fast", "accuracy": "good"}
        }
    }

def validate_args(args_dict):
    """
    Validate provided arguments against compatible arguments.
    
    Args:
        args_dict (dict): Dictionary of arguments to validate
        
    Returns:
        tuple: (is_valid, error_messages)
    """
    
    valid_ranges = {
        "confidence_threshold": (0.1, 1.0),
        "max_detections": (1, 1000),
        "motion_strength": (0.1, 1.0),
        "num_frames": [8, 16, 24, 32],
        "fps": [4, 8, 12, 16, 24],
        "resolution": ["256x256", "512x512", "768x768", "1024x1024"]
    }
    
    errors = []
    
    for arg_name, arg_value in args_dict.items():
        if arg_name in valid_ranges:
            valid_range = valid_ranges[arg_name]
            
            if isinstance(valid_range, tuple):
                # Range validation
                if not (valid_range[0] <= arg_value <= valid_range[1]):
                    errors.append(f"{arg_name} must be between {valid_range[0]} and {valid_range[1]}")
            elif isinstance(valid_range, list):
                # Choice validation
                if arg_value not in valid_range:
                    errors.append(f"{arg_name} must be one of {valid_range}")
    
    return len(errors) == 0, errors