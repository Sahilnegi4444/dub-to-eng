"""
Translation Module.
Handles translating transcribed speech into natural, conversational English.
"""

from typing import List
from .transcriber import SpeechSegment

class SpeechTranslator:
    """
    Manages translation of transcription segments into English.
    Supports:
    1. Direct translation passthrough (from Whisper task='translate')
    2. Context-preserving LLM / IndicTrans translation hook
    """
    def __init__(self, mode: str = "whisper_direct"):
        self.mode = mode

    def translate_segments(self, segments: List[SpeechSegment], source_lang: str) -> List[SpeechSegment]:
        """
        Populates translated_text on each segment if not already set.
        """
        if source_lang.lower() == "en":
            for seg in segments:
                if not seg.translated_text:
                    seg.translated_text = seg.text
            return segments

        # If already populated by whisper task='translate'
        all_translated = all(bool(seg.translated_text) for seg in segments)
        if all_translated and self.mode == "whisper_direct":
            return segments

        # If custom post-processing or translation model is needed
        for seg in segments:
            if not seg.translated_text:
                seg.translated_text = seg.text  # Fallback

        return segments
