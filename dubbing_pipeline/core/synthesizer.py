"""
Speech Synthesizer Module using edge-tts.
Generates human-like, expressive English speech.
"""

from pathlib import Path
import asyncio
import concurrent.futures
from typing import List
import edge_tts
from .transcriber import SpeechSegment

class EdgeSpeechSynthesizer:
    """
    Synthesizes speech for translated segments using Microsoft Edge Neural TTS.
    """
    def __init__(self, voice: str = "en-US-ChristopherNeural", rate: str = "+0%", pitch: str = "+0Hz"):
        self.voice = voice
        self.rate = rate
        self.pitch = pitch

    async def _synthesize_single(self, text: str, output_path: Path):
        communicate = edge_tts.Communicate(text, self.voice, rate=self.rate, pitch=self.pitch)
        await communicate.save(str(output_path))

    async def synthesize_segments(self, segments: List[SpeechSegment], output_dir: Path, batch_size: int = 10) -> List[SpeechSegment]:
        """
        Synthesizes audio for all segments asynchronously in batches.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        tasks = []
        for seg in segments:
            tts_path = output_dir / f"seg_{seg.id:04d}.mp3"
            seg.tts_audio_path = tts_path
            # Synthesize translated_text or fallback to original text
            text_to_speak = seg.translated_text if seg.translated_text else seg.text
            tasks.append(self._synthesize_single(text_to_speak, tts_path))

        # Batch execution to avoid connection timeouts or rate limiting
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            await asyncio.gather(*batch)

        return segments

    def run_synthesis(self, segments: List[SpeechSegment], output_dir: Path) -> List[SpeechSegment]:
        """
        Synchronous wrapper for synthesize_segments, safe for both CLI and Jupyter / Colab notebooks.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        coro = self.synthesize_segments(segments, output_dir)
        if loop and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()
        else:
            return asyncio.run(coro)
