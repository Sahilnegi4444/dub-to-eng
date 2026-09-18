"""
Automated Video Dubbing System - CLI Entry Point.
"""

import argparse
import sys
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(
        description="Automated Video Dubbing System (Idealabs Digital Assessment)"
    )
    parser.add_argument(
        "--url", "-u",
        type=str,
        help="YouTube video URL to dub"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./output/dubbed_video.mp4",
        help="Path for output dubbed MP4 file (default: ./output/dubbed_video.mp4)"
    )
    parser.add_argument(
        "--voice", "-v",
        type=str,
        default="en-US-ChristopherNeural",
        help="TTS Voice name (e.g., en-US-ChristopherNeural, en-IN-PrabhatNeural, en-GB-SoniaNeural)"
    )
    parser.add_argument(
        "--model-size", "-m",
        type=str,
        default="medium",
        choices=["base", "small", "medium", "large-v3"],
        help="Whisper model size (default: medium)"
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Execution device ('cuda' or 'cpu'). Auto-detected if not specified."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    url = args.url
    if not url:
        # If no CLI argument passed, prompt interactively
        print("==================================================")
        print("       Automated Video Dubbing System")
        print("==================================================")
        url = input("Enter YouTube video URL: ").strip()

    if not url:
        print("[Error] A valid YouTube URL is required.")
        sys.exit(1)

    try:
        from dubbing_pipeline.pipeline import DubbingPipeline
        from dubbing_pipeline.config import PipelineConfig
    except ImportError as err:
        print(f"\n[Missing Dependency] {err}")
        print("Please install the required packages using: pip install -r requirements.txt")
        sys.exit(1)

    config = PipelineConfig(
        whisper_model_size=args.model_size
    )
    if args.device:
        config.device = args.device
        config.compute_type = "float16" if args.device == "cuda" else "int8"

    pipeline = DubbingPipeline(config=config)
    output_path = Path(args.output)

    try:
        pipeline.process(
            youtube_url=url,
            output_path=output_path,
            voice=args.voice
        )
    except KeyboardInterrupt:
        print("\n[Aborted] Process interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n[Execution Error]: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
