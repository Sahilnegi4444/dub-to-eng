"""
Core components of the Automated Video Dubbing System.
"""

from .downloader import YouTubeDownloader
from .transcriber import WhisperTranscriber
from .translator import SpeechTranslator
from .synthesizer import EdgeSpeechSynthesizer
from .aligner import AudioTimelineAligner
from .remuxer import FFmpegRemuxer

__all__ = [
    "YouTubeDownloader",
    "WhisperTranscriber",
    "SpeechTranslator",
    "EdgeSpeechSynthesizer",
    "AudioTimelineAligner",
    "FFmpegRemuxer",
]
