"""
Pipeline Orchestrator Module.
Manages the end-to-end flow with stage timers, rich terminal logging, and benchmark reporting.
"""

from pathlib import Path
import time
from pydub import AudioSegment
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from .config import PipelineConfig
from .core.downloader import YouTubeDownloader
from .core.transcriber import WhisperTranscriber
from .core.translator import SpeechTranslator
from .core.synthesizer import EdgeSpeechSynthesizer
from .core.aligner import AudioTimelineAligner
from .core.remuxer import FFmpegRemuxer

console = Console()

class DubbingPipeline:
    """
    Coordinates end-to-end video dubbing workflow.
    """
    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.downloader = YouTubeDownloader(self.config.temp_dir)
        self.transcriber = WhisperTranscriber(
            model_size=self.config.whisper_model_size,
            device=self.config.device,
            compute_type=self.config.compute_type
        )
        self.translator = SpeechTranslator()
        self.synthesizer = EdgeSpeechSynthesizer(voice=self.config.default_voice)
        self.aligner = AudioTimelineAligner(
            max_speedup=self.config.max_speedup_ratio,
            min_slowdown=self.config.min_slowdown_ratio
        )
        self.remuxer = FFmpegRemuxer()

    def process(self, youtube_url: str, output_path: Path | None = None, voice: str | None = None) -> dict:
        """
        Runs the automated dubbing pipeline on the specified YouTube URL.
        """
        if voice:
            self.synthesizer.voice = voice

        benchmarks = {}
        pipeline_start = time.time()

        console.print(Panel.fit(
            f"[bold cyan]Automated Video Dubbing System[/bold cyan]\n"
            f"[yellow]Target URL:[/yellow] {youtube_url}\n"
            f"[yellow]Voice Model:[/yellow] {self.synthesizer.voice}\n"
            f"[yellow]Inference Device:[/yellow] {self.config.device} ({self.config.compute_type})",
            title="[bold green]IDEALABS DIGITAL PIPELINE[/bold green]"
        ))

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:

            # Stage 1: Download
            t0 = time.time()
            task_dl = progress.add_task("[cyan]Downloading media streams via yt-dlp...", total=100)
            video_file, audio_file = self.downloader.download(youtube_url)
            progress.update(task_dl, completed=100)
            benchmarks["Download"] = time.time() - t0

            # Inspect media duration
            audio_probe = AudioSegment.from_file(audio_file)
            media_duration_sec = len(audio_probe) / 1000.0
            benchmarks["Media Duration (s)"] = media_duration_sec

            # Stage 2: Transcribe & Translate
            t0 = time.time()
            task_trans = progress.add_task("[magenta]Transcribing & translating with Whisper VAD...", total=100)
            segments, detected_lang, prob = self.transcriber.transcribe(audio_file, task="translate")
            progress.update(task_trans, completed=100)
            benchmarks["Transcription & Translation"] = time.time() - t0
            benchmarks["Detected Language"] = f"{detected_lang.upper()} ({prob:.1%})"
            benchmarks["Segment Count"] = len(segments)

            # Stage 3: Synthesize Speech
            t0 = time.time()
            task_synth = progress.add_task(f"[green]Synthesizing English speech ({len(segments)} segments)...", total=100)
            tts_dir = self.config.temp_dir / "tts_segments"
            self.synthesizer.run_synthesis(segments, tts_dir)
            progress.update(task_synth, completed=100)
            benchmarks["Speech Synthesis"] = time.time() - t0

            # Stage 4: Align Audio Timeline (Anti-drift)
            t0 = time.time()
            task_align = progress.add_task("[yellow]Aligning audio timeline & applying atempo...", total=100)
            master_dubbed_wav = self.config.temp_dir / "dubbed_master.wav"
            self.aligner.build_synchronized_master_track(
                segments, media_duration_sec, master_dubbed_wav, self.config.temp_dir
            )
            progress.update(task_align, completed=100)
            benchmarks["Timeline Alignment"] = time.time() - t0

            # Stage 5: FFmpeg Stream Remuxing
            t0 = time.time()
            task_remux = progress.add_task("[blue]Multiplexing video & audio (-c:v copy)...", total=100)
            if not output_path:
                output_path = self.config.output_dir / "dubbed_video_output.mp4"
            final_video = self.remuxer.remux(video_file, master_dubbed_wav, output_path)
            progress.update(task_remux, completed=100)
            benchmarks["FFmpeg Remux"] = time.time() - t0

        total_elapsed = time.time() - pipeline_start
        benchmarks["Total Processing Time (s)"] = total_elapsed
        benchmarks["Speedup Multiplier"] = f"{media_duration_sec / max(total_elapsed, 0.001):.2f}x"
        benchmarks["Output Video"] = str(final_video.resolve())

        # Render Benchmark Summary Table
        self._print_summary_table(benchmarks)

        return benchmarks

    def _print_summary_table(self, benchmarks: dict):
        table = Table(title="Execution Benchmark Summary (IDEALABS Evaluation)")
        table.add_column("Pipeline Stage / Metric", style="cyan", no_wrap=True)
        table.add_column("Measurement", style="bold green")

        for k, v in benchmarks.items():
            if isinstance(v, float):
                formatted = f"{v:.2f} s ({v/60:.2f} min)" if "s" in k or "Time" in k else f"{v:.2f}"
            else:
                formatted = str(v)
            table.add_row(k, formatted)

        console.print(table)
