"""
Configuration management for the Dubbing Pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
import os
import torch

@dataclass
class PipelineConfig:
    # Model settings
    whisper_model_size: str = "medium"  # 'base', 'small', 'medium', 'large-v3'
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") or (hasattr(torch, 'cuda') and torch.cuda.is_available()) else "cpu"
    compute_type: str = "float16" if os.environ.get("CUDA_VISIBLE_DEVICES") else "int8"
    beam_size: int = 5

    # Voice / TTS settings
    default_voice: str = "en-US-ChristopherNeural"
    voice_map: dict = field(default_factory=lambda: {
        "male_us": "en-US-ChristopherNeural",
        "female_us": "en-US-JennyNeural",
        "male_indian": "en-IN-PrabhatNeural",
        "female_indian": "en-IN-NeerjaNeural",
        "male_uk": "en-GB-RyanNeural",
        "female_uk": "en-GB-SoniaNeural"
    })

    # Alignment & Audio Stretching parameters
    max_speedup_ratio: float = 1.30     # Maximum speed up factor to maintain voice naturalness
    min_slowdown_ratio: float = 0.85    # Minimum slow down factor
    tolerance_threshold_sec: float = 0.08  # Tolerated drift window

    # Paths
    temp_dir: Path = Path("./dubbing_workspace")
    output_dir: Path = Path("./output")

    def __post_init__(self):
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
