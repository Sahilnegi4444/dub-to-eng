"""
YouTube Downloader Module using yt-dlp.
Downloads video and extracts optimized audio for speech processing.
"""

from pathlib import Path
import time
import yt_dlp

class YouTubeDownloader:
    """
    Handles downloading YouTube media streams and converting audio to speech-ready format.
    """
    def __init__(self, workspace_dir: Path):
        self.workspace_dir = workspace_dir
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

    def download(self, url: str) -> tuple[Path, Path]:
        """
        Downloads the video stream and a 16kHz mono WAV audio file.
        Returns:
            (video_path, audio_path)
        """
        video_path = self.workspace_dir / "downloaded_video.mp4"
        audio_path = self.workspace_dir / "extracted_audio.wav"

        # Clear any previous runs
        if video_path.exists():
            video_path.unlink()
        if audio_path.exists():
            audio_path.unlink()

        # yt-dlp options for video stream (up to 1080p for fast download, preserving crisp visuals)
        ydl_opts_video = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': str(video_path),
            'merge_output_format': 'mp4',
            'quiet': True,
            'no_warnings': True,
        }

        # yt-dlp options for 16kHz mono WAV for Whisper
        temp_audio_template = str(self.workspace_dir / "temp_audio.%(ext)s")
        ydl_opts_audio = {
            'format': 'bestaudio/best',
            'outtmpl': temp_audio_template,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'postprocessor_args': [
                '-ac', '1',       # mono channel
                '-ar', '16000'    # 16 kHz sample rate (optimal for Whisper)
            ],
            'quiet': True,
            'no_warnings': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts_video) as ydl:
            ydl.download([url])

        with yt_dlp.YoutubeDL(ydl_opts_audio) as ydl:
            ydl.download([url])

        extracted = list(self.workspace_dir.glob("temp_audio*.wav"))
        if extracted:
            extracted[0].rename(audio_path)
        elif not audio_path.exists():
            raise FileNotFoundError("Failed to extract audio from downloaded media.")

        return video_path, audio_path
