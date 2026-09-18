"""
Audio Timeline Aligner & Speed Normalizer Module.
Prevents audio drift by aligning synthesized speech to the exact original timestamps.
"""

from pathlib import Path
import subprocess
from typing import List
from pydub import AudioSegment
from .transcriber import SpeechSegment

class AudioTimelineAligner:
    """
    Aligns synthesized audio segments to the original video timeline.
    Uses atempo time-stretching to fit segments within original speech boundaries,
    and overlays segments onto a continuous master timeline.
    """
    def __init__(self, max_speedup: float = 1.30, min_slowdown: float = 0.85):
        self.max_speedup = max_speedup
        self.min_slowdown = min_slowdown

    def fit_segment_duration(self, input_audio: Path, target_duration_sec: float, output_audio: Path):
        """
        Adjusts segment playback speed using FFmpeg's atempo filter if it exceeds the slot.
        """
        sound = AudioSegment.from_file(input_audio)
        current_duration_sec = len(sound) / 1000.0

        if target_duration_sec <= 0.1:
            sound.export(output_audio, format="wav")
            return

        ratio = current_duration_sec / target_duration_sec

        # If speech exceeds slot by more than 5%, speed it up
        if ratio > 1.05:
            speed = min(ratio, self.max_speedup)
            cmd = [
                "ffmpeg", "-y", "-i", str(input_audio),
                "-filter:a", f"atempo={speed:.3f}",
                "-vn", str(output_audio)
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            # Fits comfortably; export directly
            sound.export(output_audio, format="wav")

    def build_synchronized_master_track(
        self,
        segments: List[SpeechSegment],
        total_duration_sec: float,
        output_master_wav: Path,
        temp_dir: Path
    ) -> Path:
        """
        Assembles all aligned segments into a master audio track with zero timing drift.
        """
        temp_dir.mkdir(parents=True, exist_ok=True)
        aligned_dir = temp_dir / "aligned_segments"
        aligned_dir.mkdir(parents=True, exist_ok=True)

        # Create silent base track matching total original video duration
        master_duration_ms = int(total_duration_sec * 1000) + 1000
        master_track = AudioSegment.silent(duration=master_duration_ms)

        for seg in segments:
            if not seg.tts_audio_path or not seg.tts_audio_path.exists():
                continue

            aligned_path = aligned_dir / f"aligned_{seg.id:04d}.wav"
            self.fit_segment_duration(seg.tts_audio_path, seg.duration, aligned_path)

            if aligned_path.exists():
                seg_sound = AudioSegment.from_file(aligned_path)
                start_ms = int(seg.start * 1000)
                # Overlay at precise start timestamp
                master_track = master_track.overlay(seg_sound, position=start_ms)

        # Trim master track to exact video duration
        master_track = master_track[:int(total_duration_sec * 1000)]
        master_track.export(output_master_wav, format="wav")

        return output_master_wav
