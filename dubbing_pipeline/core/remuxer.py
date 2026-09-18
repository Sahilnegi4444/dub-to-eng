"""
Remuxer Module using FFmpeg.
Replaces the original video audio track with the dubbed audio track using lossless stream copying.
"""

from pathlib import Path
import subprocess

class FFmpegRemuxer:
    """
    Handles multiplexing video and audio streams using FFmpeg with -c:v copy.
    """
    def remux(self, video_input: Path, audio_input: Path, output_file: Path) -> Path:
        """
        Merges video and audio streams without re-encoding the video.
        """
        output_file.parent.mkdir(parents=True, exist_ok=True)
        if output_file.exists():
            output_file.unlink()

        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_input),
            "-i", str(audio_input),
            "-map", "0:v:0",       # First video stream from input 0
            "-map", "1:a:0",       # First audio stream from input 1
            "-c:v", "copy",        # Zero re-encoding for visuals (lossless & fast)
            "-c:a", "aac",         # High-fidelity AAC audio
            "-b:a", "192k",        # Crisp bitrate
            "-shortest",           # Align video/audio end
            str(output_file)
        ]

        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return output_file
