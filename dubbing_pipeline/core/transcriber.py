"""
Transcription & Speech-to-Text Module using faster-whisper.
"""

from pathlib import Path
from dataclasses import dataclass
from faster_whisper import WhisperModel

@dataclass
class SpeechSegment:
    id: int
    start: float
    end: float
    text: str
    duration: float
    translated_text: str = ""
    tts_audio_path: Path | None = None

class WhisperTranscriber:
    """
    Transcribes speech using faster-whisper with Silero-VAD filtering.
    """
    def __init__(self, model_size: str = "medium", device: str = "cpu", compute_type: str = "int8"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model = None

    def load_model(self):
        if self.model is None:
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type
            )

    def transcribe(self, audio_path: Path, task: str = "transcribe") -> tuple[list[SpeechSegment], str, float]:
        """
        Transcribes (or directly translates) audio to speech segments.
        task can be 'transcribe' or 'translate'.
        Returns:
            (segments, detected_language, language_probability)
        """
        self.load_model()

        raw_segments, info = self.model.transcribe(
            str(audio_path),
            task=task,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=400),
            beam_size=5
        )

        segments: list[SpeechSegment] = []
        for s in raw_segments:
            text = s.text.strip()
            if not text:
                continue
            seg = SpeechSegment(
                id=s.id,
                start=round(s.start, 3),
                end=round(s.end, 3),
                text=text,
                duration=round(s.end - s.start, 3)
            )
            if task == "translate":
                seg.translated_text = text
            segments.append(seg)

        return segments, info.language, info.language_probability
