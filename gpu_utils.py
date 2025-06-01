#!/usr/bin/env python3
"""
NVIDIA RTX GPU Detection and Optimization Utility
Comprehensive support for RTX 10, 20, 30, 40, and 50 series GPUs
with VRAM-based recommendations for optimal GPU layer allocation.
"""

import torch
import subprocess
import re
import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RTXSeries(Enum):
    """RTX GPU Series enumeration"""
    RTX_10 = "RTX 10 Series"
    RTX_20 = "RTX 20 Series" 
    RTX_30 = "RTX 30 Series"
    RTX_40 = "RTX 40 Series"
    RTX_50 = "RTX 50 Series"
    UNKNOWN = "Unknown Series"

@dataclass
class GPUSpecs:
    """GPU specifications dataclass"""
    model: str
    series: RTXSeries
    vram_gb: float
    memory_type: str
    memory_bus: int
    bandwidth_gbps: float
    cuda_cores: int
    tensor_cores: Optional[int] = None
    rt_cores: Optional[int] = None

@dataclass
class LayerRecommendation:
    """GPU layer allocation recommendation"""
    recommended_layers: int
    max_layers: int
    optimal_batch_size: int
    memory_usage_percent: float
    performance_tier: str
    notes: str

class RTXGPUDetector:
    """Comprehensive NVIDIA RTX GPU detection and optimization system"""
    
    # Comprehensive RTX GPU database with VRAM specifications
    RTX_GPU_DATABASE = {
        # RTX 10 Series (Pascal Architecture)
        "GTX 1080 Ti": GPUSpecs("GTX 1080 Ti", RTXSeries.RTX_10, 11.0, "GDDR5X", 352, 484, 3584),
        "GTX 1080": GPUSpecs("GTX 1080", RTXSeries.RTX_10, 8.0, "GDDR5X", 256, 320, 2560),
        "GTX 1070 Ti": GPUSpecs("GTX 1070 Ti", RTXSeries.RTX_10, 8.0, "GDDR5", 256, 256, 2432),
        "GTX 1070": GPUSpecs("GTX 1070", RTXSeries.RTX_10, 8.0, "GDDR5", 256, 256, 1920),
        "GTX 1060": GPUSpecs("GTX 1060", RTXSeries.RTX_10, 6.0, "GDDR5", 192, 192, 1280),
        
        # RTX 20 Series (Turing Architecture)
        "RTX 2080 Ti": GPUSpecs("RTX 2080 Ti", RTXSeries.RTX_20, 11.0, "GDDR6", 352, 616, 4352, 544, 68),
        "RTX 2080 SUPER": GPUSpecs("RTX 2080 SUPER", RTXSeries.RTX_20, 8.0, "GDDR6", 256, 496, 3072, 384, 48),
        "RTX 2080": GPUSpecs("RTX 2080", RTXSeries.RTX_20, 8.0, "GDDR6", 256, 448, 2944, 368, 46),
        "RTX 2070 SUPER": GPUSpecs("RTX 2070 SUPER", RTXSeries.RTX_20, 8.0, "GDDR6", 256, 448, 2560, 320, 40),
        "RTX 2070": GPUSpecs("RTX 2070", RTXSeries.RTX_20, 8.0, "GDDR6", 256, 448, 2304, 288, 36),
        "RTX 2060 SUPER": GPUSpecs("RTX 2060 SUPER", RTXSeries.RTX_20, 8.0, "GDDR6", 256, 448, 2176, 272, 34),
        "RTX 2060": GPUSpecs("RTX 2060", RTXSeries.RTX_20, 6.0, "GDDR6", 192, 336, 1920, 240, 30),
        
        # RTX 30 Series (Ampere Architecture)
        "RTX 3090 Ti": GPUSpecs("RTX 3090 Ti", RTXSeries.RTX_30, 24.0, "GDDR6X", 384, 1008, 10752, 336, 84),
        "RTX 3090": GPUSpecs("RTX 3090", RTXSeries.RTX_30, 24.0, "GDDR6X", 384, 936, 10496, 328, 82),
        "RTX 3080 Ti": GPUSpecs("RTX 3080 Ti", RTXSeries.RTX_30, 12.0, "GDDR6X", 384, 912, 10240, 320, 80),
        "RTX 3080": GPUSpecs("RTX 3080", RTXSeries.RTX_30, 10.0, "GDDR6X", 320, 760, 8704, 272, 68),
        "RTX 3070 Ti": GPUSpecs("RTX 3070 Ti", RTXSeries.RTX_30, 8.0, "GDDR6X", 256, 608, 6144, 192, 48),
        "RTX 3070": GPUSpecs("RTX 3070", RTXSeries.RTX_30, 8.0, "GDDR6", 256, 448, 5888, 184, 46),
        "RTX 3060 Ti": GPUSpecs("RTX 3060 Ti", RTXSeries.RTX_30, 8.0, "GDDR6", 256, 448, 4864, 152, 38),
        "RTX 3060": GPUSpecs("RTX 3060", RTXSeries.RTX_30, 12.0, "GDDR6", 192, 360, 3584, 112, 28),
        
        # RTX 40 Series (Ada Lovelace Architecture)
        "RTX 4090": GPUSpecs("RTX 4090", RTXSeries.RTX_40, 24.0, "GDDR6X", 384, 1008, 16384, 512, 128),
        "RTX 4080 SUPER": GPUSpecs("RTX 4080 SUPER", RTXSeries.RTX_40, 16.0, "GDDR6X", 256, 736, 10240, 320, 80),
        "RTX 4080": GPUSpecs("RTX 4080", RTXSeries.RTX_40, 16.0, "GDDR6X", 256, 716, 9728, 304, 76),
        "RTX 4070 Ti SUPER": GPUSpecs("RTX 4070 Ti SUPER", RTXSeries.RTX_40, 16.0, "GDDR6X", 256, 672, 8448, 264, 66),
        "RTX 4070 Ti": GPUSpecs("RTX 4070 Ti", RTXSeries.RTX_40, 12.0, "GDDR6X", 192, 504, 7680, 240, 60),
        "RTX 4070 SUPER": GPUSpecs("RTX 4070 SUPER", RTXSeries.RTX_40, 12.0, "GDDR6X", 192, 504, 7168, 224, 56),
        "RTX 4070": GPUSpecs("RTX 4070", RTXSeries.RTX_40, 12.0, "GDDR6X", 192, 504, 5888, 184, 46),
        "RTX 4060 Ti": GPUSpecs("RTX 4060 Ti", RTXSeries.RTX_40, 16.0, "GDDR6", 128, 288, 4352, 136, 34),
        "RTX 4060": GPUSpecs("RTX 4060", RTXSeries.RTX_40, 8.0, "GDDR6", 128, 272, 3072, 96, 24),
        
        # RTX 50 Series (Blackwell Architecture) - Based on leaks and speculation
        "RTX 5090": GPUSpecs("RTX 5090", RTXSeries.RTX_50, 32.0, "GDDR7", 512, 1500, 21504, 672, 168),
        "RTX 5080": GPUSpecs("RTX 5080", RTXSeries.RTX_50, 16.0, "GDDR7", 256, 960, 10752, 336, 84),
        "RTX 5070 Ti": GPUSpecs("RTX 5070 Ti", RTXSeries.RTX_50, 16.0, "GDDR7", 256, 896, 8960, 280, 70),
        "RTX 5070": GPUSpecs("RTX 5070", RTXSeries.RTX_50, 12.0, "GDDR7", 192, 672, 6912, 216, 54),
        "RTX 5060 Ti": GPUSpecs("RTX 5060 Ti", RTXSeries.RTX_50, 16.0, "GDDR7", 128, 448, 4608, 144, 36),
        "RTX 5060": GPUSpecs("RTX 5060", RTXSeries.RTX_50, 8.0, "GDDR7", 128, 384, 3584, 112, 28),
    }
    
    # Performance tier mapping based on VRAM and compute capability
    PERFORMANCE_TIERS = {
        "FLAGSHIP": ["RTX 4090", "RTX 3090 Ti", "RTX 3090", "RTX 5090"],
        "HIGH_END": ["RTX 4080", "RTX 4080 SUPER", "RTX 3080 Ti", "RTX 3080", "RTX 2080 Ti", "RTX 5080"],
        "UPPER_MID": ["RTX 4070 Ti", "RTX 4070 Ti SUPER", "RTX 4070", "RTX 4070 SUPER", "RTX 3070 Ti", "RTX 3070", "RTX 5070 Ti", "RTX 5070"],
        "MID_RANGE": ["RTX 4060 Ti", "RTX 3060 Ti", "RTX 2070", "RTX 2070 SUPER", "RTX 5060 Ti"],
        "ENTRY_LEVEL": ["RTX 4060", "RTX 3060", "RTX 2060", "RTX 2060 SUPER", "RTX 5060"],
        "LEGACY_HIGH": ["GTX 1080 Ti", "GTX 1080", "GTX 1070 Ti", "GTX 1070"]
    }

    def __init__(self):
        """Initialize the RTX GPU detector"""
        self.detected_gpus = []
        self.primary_gpu = None
        self._detect_gpus()

    def _detect_gpus(self) -> None:
        """Detect all available NVIDIA RTX GPUs"""
        try:
            # Method 1: PyTorch CUDA detection
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
                    
                    # Normalize GPU name for database lookup
                    normalized_name = self._normalize_gpu_name(gpu_name)
                    gpu_specs = self._get_gpu_specs(normalized_name, gpu_memory)
                    
                    if gpu_specs:
                        self.detected_gpus.append({
                            'index': i,
                            'name': gpu_name,
                            'normalized_name': normalized_name,
                            'specs': gpu_specs,
                            'available_memory': gpu_memory,
                            'method': 'pytorch'
                        })
            
            # Method 2: nvidia-smi detection (fallback/verification)
            self._detect_via_nvidia_smi()
            
            # Set primary GPU
            if self.detected_gpus:
                self.primary_gpu = self.detected_gpus[0]
                logger.info(f"Primary GPU detected: {self.primary_gpu['name']}")
            else:
                logger.warning("No NVIDIA RTX GPUs detected")
                
        except Exception as e:
            logger.error(f"Error detecting GPUs: {str(e)}")

    def _detect_via_nvidia_smi(self) -> None:
        """Detect GPUs using nvidia-smi command"""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for i, line in enumerate(lines):
                    if line.strip():
                        parts = line.split(', ')
                        if len(parts) >= 2:
                            gpu_name = parts[0].strip()
                            memory_mb = float(parts[1].strip())
                            memory_gb = memory_mb / 1024
                            
                            # Check if this GPU is already detected
                            already_detected = any(gpu['name'] == gpu_name for gpu in self.detected_gpus)
                            
                            if not already_detected:
                                normalized_name = self._normalize_gpu_name(gpu_name)
                                gpu_specs = self._get_gpu_specs(normalized_name, memory_gb)
                                
                                if gpu_specs:
                                    self.detected_gpus.append({
                                        'index': i,
                                        'name': gpu_name,
                                        'normalized_name': normalized_name,
                                        'specs': gpu_specs,
                                        'available_memory': memory_gb,
                                        'method': 'nvidia-smi'
                                    })
                                    
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning(f"nvidia-smi detection failed: {str(e)}")

    def _normalize_gpu_name(self, gpu_name: str) -> str:
        """Normalize GPU name for database lookup"""
        # Remove common prefixes and suffixes
        name = gpu_name.upper()
        name = re.sub(r'NVIDIA\s+GEFORCE\s+', '', name)
        name = re.sub(r'GEFORCE\s+', '', name)
        name = re.sub(r'\s+FOUNDERS?\s+EDITION', '', name)
        name = re.sub(r'\s+FE\b', '', name)
        name = re.sub(r'\s+OC\b', '', name)
        name = re.sub(r'\s+GAMING\b', '', name)
        name = re.sub(r'\s+\d+GB\b', '', name)
        
        # Handle specific model variations
        if 'RTX' in name or 'GTX' in name:
            # Extract the core model name
            patterns = [
                r'(RTX\s+\d{4}\s+TI\s+SUPER)',
                r'(RTX\s+\d{4}\s+SUPER)',
                r'(RTX\s+\d{4}\s+TI)',
                r'(RTX\s+\d{4})',
                r'(GTX\s+\d{4}\s+TI)',
                r'(GTX\s+\d{4})'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, name)
                if match:
                    return match.group(1).strip()
        
        return name.strip()

    def _get_gpu_specs(self, normalized_name: str, detected_memory: float) -> Optional[GPUSpecs]:
        """Get GPU specifications from database"""
        # Direct lookup
        if normalized_name in self.RTX_GPU_DATABASE:
            return self.RTX_GPU_DATABASE[normalized_name]
        
        # Fuzzy matching for variations
        for db_name, specs in self.RTX_GPU_DATABASE.items():
            if self._fuzzy_match(normalized_name, db_name):
                return specs
        
        # Memory-based inference for unknown models
        return self._infer_specs_from_memory(normalized_name, detected_memory)

    def _fuzzy_match(self, name1: str, name2: str) -> bool:
        """Fuzzy matching for GPU names"""
        name1_clean = re.sub(r'\s+', '', name1.upper())
        name2_clean = re.sub(r'\s+', '', name2.upper())
        
        # Check if core model numbers match
        model1 = re.search(r'(\d{4})', name1_clean)
        model2 = re.search(r'(\d{4})', name2_clean)
        
        if model1 and model2:
            return model1.group(1) == model2.group(1) and \
                   ('TI' in name1_clean) == ('TI' in name2_clean) and \
                   ('SUPER' in name1_clean) == ('SUPER' in name2_clean)
        
        return False

    def _infer_specs_from_memory(self, name: str, memory_gb: float) -> Optional[GPUSpecs]:
        """Infer GPU specs based on memory size for unknown models"""
        # Determine series from name
        series = RTXSeries.UNKNOWN
        if 'RTX 50' in name or '50' in name:
            series = RTXSeries.RTX_50
        elif 'RTX 40' in name or '40' in name:
            series = RTXSeries.RTX_40
        elif 'RTX 30' in name or '30' in name:
            series = RTXSeries.RTX_30
        elif 'RTX 20' in name or '20' in name:
            series = RTXSeries.RTX_20
        elif 'GTX 10' in name or '10' in name:
            series = RTXSeries.RTX_10
        
        # Estimate specs based on memory
        if memory_gb >= 20:
            return GPUSpecs(name, series, memory_gb, "GDDR6X", 384, 900, 10000, 300, 75)
        elif memory_gb >= 15:
            return GPUSpecs(name, series, memory_gb, "GDDR6X", 256, 700, 8000, 250, 60)
        elif memory_gb >= 10:
            return GPUSpecs(name, series, memory_gb, "GDDR6", 256, 500, 6000, 200, 50)
        elif memory_gb >= 6:
            return GPUSpecs(name, series, memory_gb, "GDDR6", 192, 400, 4000, 150, 35)
        else:
            return GPUSpecs(name, series, memory_gb, "GDDR6", 128, 300, 2000, 100, 25)

    def get_layer_recommendations(self, model_size_gb: float = 7.0, context_length: int = 4096) -> Dict[str, LayerRecommendation]:
        """Get layer allocation recommendations for all detected GPUs"""
        recommendations = {}
        
        for gpu in self.detected_gpus:
            recommendation = self._calculate_layer_recommendation(
                gpu['specs'], gpu['available_memory'], model_size_gb, context_length
            )
            recommendations[gpu['name']] = recommendation
            
        return recommendations

    def _calculate_layer_recommendation(self, specs: GPUSpecs, available_memory: float, 
                                      model_size_gb: float, context_length: int) -> LayerRecommendation:
        """Calculate optimal layer allocation for a specific GPU"""
        # Get performance tier
        performance_tier = self._get_performance_tier(specs.model)
        
        # Calculate memory overhead (OS, CUDA, etc.)
        memory_overhead = min(2.0, available_memory * 0.15)  # 15% overhead, max 2GB
        usable_memory = available_memory - memory_overhead
        
        # Calculate context memory usage (approximate)
        context_memory = (context_length * 4 * 2) / (1024**3)  # 4 bytes per token, 2 for KV cache
        
        # Available memory for model layers
        layer_memory = usable_memory - context_memory
        
        # Estimate layers based on model size and available memory
        if layer_memory <= 0:
            recommended_layers = 0
            max_layers = 0
            optimal_batch_size = 1
            memory_usage = 100.0
            notes = "Insufficient VRAM for this model"
        else:
            # Estimate layer count (assuming uniform layer distribution)
            estimated_total_layers = self._estimate_total_layers(model_size_gb)
            memory_per_layer = model_size_gb / estimated_total_layers
            
            max_layers = min(estimated_total_layers, int(layer_memory / memory_per_layer))
            
            # Recommended layers (conservative estimate)
            recommended_layers = int(max_layers * 0.8)  # 80% of max for safety
            
            # Optimal batch size based on remaining memory
            remaining_memory = layer_memory - (recommended_layers * memory_per_layer)
            optimal_batch_size = max(1, int(remaining_memory / context_memory)) if context_memory > 0 else 4
            
            # Memory usage percentage
            used_memory = (recommended_layers * memory_per_layer) + context_memory + memory_overhead
            memory_usage = (used_memory / available_memory) * 100
            
            # Generate notes based on performance tier and specs
            notes = self._generate_optimization_notes(specs, performance_tier, memory_usage)
        
        return LayerRecommendation(
            recommended_layers=recommended_layers,
            max_layers=max_layers,
            optimal_batch_size=optimal_batch_size,
            memory_usage_percent=round(memory_usage, 1),
            performance_tier=performance_tier,
            notes=notes
        )

    def _estimate_total_layers(self, model_size_gb: float) -> int:
        """Estimate total layers based on model size"""
        # Rough estimates based on common model architectures
        if model_size_gb <= 1:
            return 12  # Small models
        elif model_size_gb <= 3:
            return 24  # 1-3B models
        elif model_size_gb <= 7:
            return 32  # 7B models
        elif model_size_gb <= 15:
            return 40  # 13B models
        elif model_size_gb <= 35:
            return 60  # 30B models
        else:
            return 80  # 70B+ models

    def _get_performance_tier(self, model_name: str) -> str:
        """Get performance tier for a GPU model"""
        for tier, models in self.PERFORMANCE_TIERS.items():
            if model_name in models:
                return tier
        return "UNKNOWN"

    def _generate_optimization_notes(self, specs: GPUSpecs, tier: str, memory_usage: float) -> str:
        """Generate optimization notes based on GPU specs and usage"""
        notes = []
        
        # Performance tier specific notes
        if tier == "FLAGSHIP":
            notes.append("Flagship GPU - Excellent for large models and high batch sizes")
        elif tier == "HIGH_END":
            notes.append("High-end GPU - Great for most AI workloads")
        elif tier == "UPPER_MID":
            notes.append("Upper mid-range GPU - Good for medium-sized models")
        elif tier == "MID_RANGE":
            notes.append("Mid-range GPU - Suitable for smaller models")
        elif tier == "ENTRY_LEVEL":
            notes.append("Entry-level GPU - Best for inference and small models")
        elif tier == "LEGACY_HIGH":
            notes.append("Legacy high-end GPU - Still capable but consider upgrading")
        
        # Memory usage warnings
        if memory_usage > 90:
            notes.append("⚠️ High memory usage - consider reducing batch size")
        elif memory_usage > 75:
            notes.append("⚠️ Moderate memory usage - monitor for stability")
        
        # VRAM specific recommendations
        if specs.vram_gb >= 24:
            notes.append("✅ Excellent VRAM capacity for large models")
        elif specs.vram_gb >= 16:
            notes.append("✅ Good VRAM capacity for most models")
        elif specs.vram_gb >= 12:
            notes.append("✅ Adequate VRAM for medium models")
        elif specs.vram_gb >= 8:
            notes.append("⚠️ Limited VRAM - consider model quantization")
        else:
            notes.append("⚠️ Low VRAM - use smaller models or aggressive quantization")
        
        # Architecture specific optimizations
        if specs.series in [RTXSeries.RTX_40, RTXSeries.RTX_50]:
            notes.append("✅ Modern architecture with excellent AI performance")
        elif specs.series == RTXSeries.RTX_30:
            notes.append("✅ Good architecture for AI workloads")
        elif specs.series == RTXSeries.RTX_20:
            notes.append("⚠️ Older architecture - consider FP16 for better performance")
        
        return " | ".join(notes)

    def get_optimal_settings(self, model_size_gb: float = 7.0) -> Dict:
        """Get optimal settings for the primary GPU"""
        if not self.primary_gpu:
            return {"error": "No GPU detected"}
        
        recommendations = self.get_layer_recommendations(model_size_gb)
        primary_rec = recommendations.get(self.primary_gpu['name'])
        
        if not primary_rec:
            return {"error": "Could not generate recommendations"}
        
        return {
            "gpu_name": self.primary_gpu['name'],
            "gpu_specs": self.primary_gpu['specs'].__dict__,
            "recommended_layers": primary_rec.recommended_layers,
            "max_layers": primary_rec.max_layers,
            "optimal_batch_size": primary_rec.optimal_batch_size,
            "memory_usage_percent": primary_rec.memory_usage_percent,
            "performance_tier": primary_rec.performance_tier,
            "optimization_notes": primary_rec.notes,
            "vram_gb": self.primary_gpu['specs'].vram_gb,
            "memory_type": self.primary_gpu['specs'].memory_type
        }

    def print_gpu_report(self) -> None:
        """Print a comprehensive GPU detection report"""
        print("\n" + "="*80)
        print("NVIDIA RTX GPU DETECTION REPORT")
        print("="*80)
        
        if not self.detected_gpus:
            print("❌ No NVIDIA RTX GPUs detected")
            return
        
        for i, gpu in enumerate(self.detected_gpus):
            specs = gpu['specs']
            print(f"\n🎮 GPU {i+1}: {gpu['name']}")
            print(f"   Series: {specs.series.value}")
            print(f"   VRAM: {specs.vram_gb} GB {specs.memory_type}")
            print(f"   Memory Bus: {specs.memory_bus}-bit")
            print(f"   Bandwidth: {specs.bandwidth_gbps} GB/s")
            print(f"   CUDA Cores: {specs.cuda_cores:,}")
            if specs.tensor_cores:
                print(f"   Tensor Cores: {specs.tensor_cores}")
            if specs.rt_cores:
                print(f"   RT Cores: {specs.rt_cores}")
            print(f"   Performance Tier: {self._get_performance_tier(specs.model)}")
        
        # Show recommendations for 7B model
        print(f"\n📊 LAYER ALLOCATION RECOMMENDATIONS (7B Model)")
        print("-" * 60)
        recommendations = self.get_layer_recommendations(7.0)
        
        for gpu_name, rec in recommendations.items():
            print(f"\n{gpu_name}:")
            print(f"   Recommended Layers: {rec.recommended_layers}")
            print(f"   Maximum Layers: {rec.max_layers}")
            print(f"   Optimal Batch Size: {rec.optimal_batch_size}")
            print(f"   Memory Usage: {rec.memory_usage_percent}%")
            print(f"   Notes: {rec.notes}")

def detect_rtx_gpu() -> RTXGPUDetector:
    """Convenience function to detect RTX GPUs"""
    return RTXGPUDetector()

def get_gpu_recommendations(model_size_gb: float = 7.0) -> Dict:
    """Convenience function to get GPU recommendations"""
    detector = RTXGPUDetector()
    return detector.get_optimal_settings(model_size_gb)

# Self-check functionality
if __name__ == "__main__":
    print("🔍 Running RTX GPU Detection System...")
    detector = RTXGPUDetector()
    detector.print_gpu_report()
    
    print(f"\n🎯 OPTIMAL SETTINGS FOR 7B MODEL:")
    print("-" * 50)
    settings = detector.get_optimal_settings(7.0)
    for key, value in settings.items():
        if key != "gpu_specs":
            print(f"{key}: {value}")
