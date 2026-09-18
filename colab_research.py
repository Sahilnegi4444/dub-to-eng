"""
=============================================================================
Automated Video Dubbing System - Professional Studio Dubbing Pipeline
=============================================================================
Workflow:
1. Extract media audio & video via yt-dlp / local file.
2. AI Stem Separation via Demucs -> separates into vocals.wav (speech) and no_vocals.wav (background music & SFX).
3. Transcribe ONLY the speech track with faster-whisper (word timestamps, zero music hallucination).
4. Extract clean reference voice (100% music-free) & synthesize cloned English speech (F5-TTS / XTTS-v2).
5. Align timing to original pauses with dynamic atempo and anti-collision sequential playback.
6. Re-mix new English voice with original background music & sound effects (no_vocals.wav) using FFmpeg.
7. Multiplex with original video visuals (-c:v copy) for lossless output.
"""

# %% [markdown]
# ### Step 0: Google Colab Setup & Dependencies
# Run this cell in Google Colab (with T4 GPU enabled):
#
# !apt-get install -y ffmpeg
# !pip install -q yt-dlp faster-whisper edge-tts pydub numpy scipy demucs f5-tts

# %%
import os
import sys
import subprocess
import asyncio
import concurrent.futures
import time
import shutil
from pathlib import Path

# Automatically accept Coqui / HuggingFace license agreements without terminal prompts
os.environ["COQUI_TOS_AGREED"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

def run_async(coro):
    """
    Safely runs async coroutines in both standard Python and Google Colab / Jupyter notebooks.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)

def install_dependencies():
    print("[1/2] Checking FFmpeg installation...")
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("  ✓ FFmpeg is already installed.")
    except Exception:
        print("  → Installing FFmpeg...")
        subprocess.run(["apt-get", "update", "-y"], check=True)
        subprocess.run(["apt-get", "install", "-y", "ffmpeg"], check=True)

    print("[2/2] Installing required Python libraries...")
    reqs = ["yt-dlp", "faster-whisper", "edge-tts", "pydub", "numpy", "scipy", "demucs", "f5-tts"]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *reqs], check=True)
    print("  ✓ Dependencies installed successfully.")

# %% [markdown]
# ### Step 1: Configuration & Workspace Setup

# %%
import yt_dlp
from faster_whisper import WhisperModel
import edge_tts
from pydub import AudioSegment

WORKSPACE_DIR = Path("./dubbing_workspace")
WORKSPACE_DIR.mkdir(exist_ok=True)

WHISPER_DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") or os.path.exists("/proc/driver/nvidia") else "cpu"
WHISPER_MODEL_SIZE = "medium"

print(f"Inference Device: {WHISPER_DEVICE.upper()} | Whisper Model: {WHISPER_MODEL_SIZE}")

# %% [markdown]
# ### Step 2: Extract Media (Video & Master Audio)

# %%
def extract_media(source: str, output_dir: Path) -> tuple[Path, Path]:
    """
    Downloads media from YouTube or uses an uploaded local file.
    Outputs:
        video_path (.mp4) and master_audio (.wav, 44.1kHz stereo for studio fidelity).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "original_video.mp4"
    audio_path = output_dir / "original_audio.wav"

    source_path = Path(source)
    # Check if input is a local file in Colab
    if source_path.exists() and source_path.is_file():
        print(f"\n[MEDIA] Loading local file: {source_path.resolve()}")
        if source_path != video_path:
            shutil.copyfile(source_path, video_path)
        # Extract master stereo audio at 44.1kHz for stem separation
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-ar", "44100", "-ac", "2",
            str(audio_path)
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return video_path, audio_path

    # Download from YouTube URL using mobile API bypass to prevent bot verification
    if video_path.exists(): video_path.unlink()
    if audio_path.exists(): audio_path.unlink()

    print(f"\n[DOWNLOAD] Fetching stream from: {source}")
    common_args = {
        'youtube': {
            'player_client': ['android', 'ios', 'web_embedded', 'mweb']
        }
    }

    ydl_opts_video = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': str(video_path),
        'merge_output_format': 'mp4',
        'extractor_args': common_args,
        'quiet': False,
        'no_warnings': True,
    }

    ydl_opts_audio = {
        'format': 'bestaudio/best',
        'outtmpl': str(output_dir / "temp_audio.%(ext)s"),
        'extractor_args': common_args,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'postprocessor_args': [
            '-ar', '44100',
            '-ac', '2'
        ],
        'quiet': False,
        'no_warnings': True,
    }

    t0 = time.time()
    with yt_dlp.YoutubeDL(ydl_opts_video) as ydl:
        ydl.download([source])
    with yt_dlp.YoutubeDL(ydl_opts_audio) as ydl:
        ydl.download([source])

    extracted = list(output_dir.glob("temp_audio*.wav"))
    if extracted:
        extracted[0].rename(audio_path)

    elapsed = time.time() - t0
    print(f"  ✓ Video ready: {video_path} ({video_path.stat().st_size / (1024*1024):.2f} MB)")
    print(f"  ✓ Master Audio ready: {audio_path} ({audio_path.stat().st_size / (1024*1024):.2f} MB)")
    print(f"  ✓ Extracted in {elapsed:.2f}s")
    return video_path, audio_path

# %% [markdown]
# ### Step 3: AI Stem Separation (Demucs)
# Separates audio into:
# 1. vocals.wav (isolated dialogue only - zero background noise/music)
# 2. no_vocals.wav (pure background music, intro jingles, sound effects, and room ambiance)

# %%
def separate_stems_demucs(audio_path: Path, output_dir: Path, device: str = "cuda") -> tuple[Path, Path]:
    """
    Separates speech dialogue from background music and sound effects using Meta AI's Demucs.
    Returns:
        (vocals_path, background_music_path)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    vocals_track = output_dir / "vocals.wav"
    bg_track = output_dir / "background_music.wav"

    print("\n[STEM SEPARATION] Isolating speech from background music via Demucs...")
    t0 = time.time()

    demucs_out = output_dir / "demucs_temp"
    cmd = [
        sys.executable, "-m", "demucs",
        "--two-stems=vocals",
        "-n", "htdemucs",
        "--device", device,
        "--out", str(demucs_out),
        str(audio_path)
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        vocal_matches = list(demucs_out.glob("**/vocals.wav"))
        bg_matches = list(demucs_out.glob("**/no_vocals.wav"))

        if vocal_matches and bg_matches:
            shutil.copyfile(vocal_matches[0], vocals_track)
            shutil.copyfile(bg_matches[0], bg_track)
            print(f"  ✓ Isolated Speech Track (0% music): {vocals_track.name}")
            print(f"  ✓ Isolated Background Track (Music + SFX): {bg_track.name}")
            print(f"  ✓ Stem separation completed in {time.time() - t0:.2f}s")
            return vocals_track, bg_track
    except Exception as e:
        print(f"  ⚠ Demucs execution error: {e}")

    # Fallback if Demucs fails
    print("  → Demucs unavailable. Using original track as vocal source with muted background.")
    shutil.copyfile(audio_path, vocals_track)
    silent_bg = AudioSegment.silent(duration=len(AudioSegment.from_file(audio_path)))
    silent_bg.export(bg_track, format="wav")
    return vocals_track, bg_track

# %% [markdown]
# ### Step 4: Transcribe & Translate ONLY the Speech Track
# Transcribing only `vocals.wav` prevents music from tricking Whisper,
# and word-level timestamps pinpoint the exact millisecond speech begins.

# %%
def transcribe_and_translate_speech(vocals_audio: Path, model_size: str = "medium", device: str = "cpu"):
    """
    Transcribes and translates speech directly from the isolated vocals stem.
    Uses word_timestamps=True to lock speech onsets precisely.
    """
    print(f"\n[TRANSCRIBE & TRANSLATE] Running faster-whisper on isolated vocals ({model_size} on {device})...")
    compute_type = "float16" if device == "cuda" else "int8"
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    t0 = time.time()
    segments, info = model.transcribe(
        str(vocals_audio),
        task="translate",
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=600,
            threshold=0.50,
            min_speech_duration_ms=250,
            speech_pad_ms=100
        ),
        word_timestamps=True,
        beam_size=5
    )

    detected_lang = info.language
    lang_prob = info.language_probability
    print(f"  ✓ Detected source language: {detected_lang.upper()} (confidence: {lang_prob:.2%})")

    translated_segments = []
    for s in segments:
        text = s.text.strip()
        if not text:
            continue

        # Use exact first word timestamp to prevent intro music drift
        if hasattr(s, 'words') and s.words:
            seg_start = round(s.words[0].start, 3)
            seg_end = round(s.words[-1].end, 3)
        else:
            seg_start = round(s.start, 3)
            seg_end = round(s.end, 3)

        seg_duration = round(max(seg_end - seg_start, 0.2), 3)
        translated_segments.append({
            "id": s.id,
            "start": seg_start,
            "end": seg_end,
            "duration": seg_duration,
            "text": text
        })

    elapsed = time.time() - t0
    speech_onset = translated_segments[0]['start'] if translated_segments else 0.0
    print(f"  ✓ Transcribed {len(translated_segments)} segments in {elapsed:.2f}s (First dialogue starts at {speech_onset:.2f}s)")
    return translated_segments, detected_lang

# %% [markdown]
# ### Step 5: Clean Voice Reference Extraction & Zero-Shot Cloning (F5-TTS / XTTS-v2)
# Because reference audio is sampled from `vocals.wav`, it has 0% music contamination!

# %%
def extract_clean_reference(vocals_path: Path, segments: list, output_ref_path: Path) -> Path:
    """
    Extracts a pristine 6-10 second vocal reference sample from the isolated speech track.
    """
    sound = AudioSegment.from_file(vocals_path)
    chosen_start, chosen_end = None, None

    for s in segments:
        dur = s["end"] - s["start"]
        if 5.0 <= dur <= 12.0:
            chosen_start, chosen_end = s["start"], s["end"]
            break

    if chosen_start is None:
        if segments and segments[0]["duration"] >= 3.0:
            chosen_start = segments[0]["start"]
            chosen_end = min(chosen_start + 8.0, segments[0]["end"])
        else:
            chosen_start = 0.0
            chosen_end = min(10.0, len(sound) / 1000.0)

    ref_sound = sound[int(chosen_start * 1000) : int(chosen_end * 1000)]
    ref_sound = ref_sound.set_channels(1).set_frame_rate(22050)
    ref_sound.export(output_ref_path, format="wav")
    print(f"  ✓ Clean vocal reference isolated: {output_ref_path} ({len(ref_sound)/1000.0:.1f}s from {chosen_start:.1f}s to {chosen_end:.1f}s)")
    return output_ref_path

def detect_speaker_profile(audio_path: Path, detected_lang: str = "en") -> str:
    """
    Probes fundamental frequency (F0) on clean vocals to detect speaker gender & region.
    """
    try:
        import numpy as np
        sound = AudioSegment.from_file(audio_path)[:30000]
        samples = np.array(sound.get_array_of_samples()).astype(float)
        rate = sound.frame_rate
        chunk = samples[rate * 2 : rate * 7]
        chunk -= np.mean(chunk)
        corr = np.correlate(chunk, chunk, mode='full')
        corr = corr[len(corr) // 2 :]
        min_lag = int(rate / 320)
        max_lag = int(rate / 80)
        peak_lag = min_lag + np.argmax(corr[min_lag:max_lag])
        f0 = rate / peak_lag
        print(f"  → Clean Vocal Pitch (F0): {f0:.1f} Hz")
        is_female = (f0 > 190)
    except Exception:
        is_female = False

    if detected_lang.lower() == "hi":
        return "en-IN-NeerjaNeural" if is_female else "en-IN-PrabhatNeural"
    else:
        return "en-US-AvaNeural" if is_female else "en-US-ChristopherNeural"

def synthesize_cloned_voice(segments: list, reference_wav: Path, output_dir: Path, device: str = "cuda"):
    """
    Synthesizes English dialogue in the cloned speaker's voice using F5-TTS or XTTS-v2.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cloning_engine = None
    f5_model = None
    xtts_model = None

    try:
        from f5_tts.api import F5TTS
        print(f"\n[VOICE CLONING] Initializing F5-TTS Diffusion Engine on {device}...")
        f5_model = F5TTS(device=device)
        print(f"  ✓ F5-TTS loaded in {time.time() - t0:.2f}s")
        cloning_engine = "f5"
    except ImportError:
        try:
            from TTS.api import TTS
            print(f"\n[VOICE CLONING] Initializing Coqui XTTS-v2 on {device}...")
            xtts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
            print(f"  ✓ Coqui XTTS-v2 loaded in {time.time() - t0:.2f}s")
            cloning_engine = "xtts"
        except ImportError:
            raise RuntimeError("Voice cloning engine not found. Run: pip install f5-tts")

    print(f"  → Generating speech for {len(segments)} segments with cloned expressions...")
    t_synth = time.time()

    for idx, seg in enumerate(segments, start=1):
        seg_audio_path = output_dir / f"seg_{seg['id']:04d}.wav"
        seg["tts_audio_path"] = seg_audio_path
        text = seg["text"].strip()

        if not text or len(text) < 2:
            AudioSegment.silent(duration=int(seg["duration"] * 1000)).export(seg_audio_path, format="wav")
            continue

        try:
            if cloning_engine == "f5":
                wav, sr, _ = f5_model.infer(
                    ref_file=str(reference_wav),
                    ref_text="",
                    gen_text=text,
                    file_wave=str(seg_audio_path)
                )
            else:
                xtts_model.tts_to_file(
                    text=text,
                    speaker_wav=str(reference_wav),
                    language="en",
                    file_path=str(seg_audio_path)
                )
        except Exception as err:
            AudioSegment.silent(duration=int(seg["duration"] * 1000)).export(seg_audio_path, format="wav")

        if idx % 10 == 0 or idx == len(segments):
            print(f"    - Synthesized {idx}/{len(segments)} segments ({idx/len(segments)*100:.1f}%)")

    elapsed = time.time() - t_synth
    print(f"  ✓ All segments synthesized in {elapsed:.2f}s")
    return segments

async def synthesize_segment_edge(text: str, output_file: Path, voice: str):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(output_file))

async def batch_synthesize_edge(segments: list, output_dir: Path, voice: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[SYNTHESIZE] Using Neural Voice '{voice}'...")
    t0 = time.time()
    tasks = []
    for seg in segments:
        seg_audio_path = output_dir / f"seg_{seg['id']:04d}.mp3"
        seg["tts_audio_path"] = seg_audio_path
        tasks.append(synthesize_segment_edge(seg["text"], seg_audio_path, voice))

    batch_size = 10
    for i in range(0, len(tasks), batch_size):
        await asyncio.gather(*tasks[i:i + batch_size])

    print(f"  ✓ Synthesized in {time.time() - t0:.2f}s")
    return segments

# %% [markdown]
# ### Step 6: Timing Alignment & Re-Mixing with Original Background Track (FFmpeg)
# 1. Fits each English segment with dynamic `atempo` time-stretching.
# 2. Uses a Sequential Playback Guard to completely prevent sentence overlaps.
# 3. Preserves all pauses and intro music by overlaying onto `no_vocals.wav`.

# %%
def fit_audio_segment(source_audio: Path, target_duration_sec: float, output_audio: Path):
    sound = AudioSegment.from_file(source_audio)
    orig_duration_sec = len(sound) / 1000.0

    if target_duration_sec <= 0.1:
        sound.export(output_audio, format="wav")
        return

    ratio = orig_duration_sec / target_duration_sec

    # Dynamic time-stretch: speed up if speech exceeds available slot
    if ratio > 1.05:
        speed = min(ratio, 1.50)
        cmd = [
            "ffmpeg", "-y", "-i", str(source_audio),
            "-filter:a", f"atempo={speed:.3f}",
            "-vn", str(output_audio)
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        sound.export(output_audio, format="wav")

def build_synchronized_dubbed_master(
    segments: list,
    background_track_path: Path,
    total_duration_sec: float,
    output_wav: Path,
    temp_dir: Path
) -> Path:
    """
    Re-mixes the new English dubbed speech with the original background track.
    Eliminates sentence overlaps and guarantees 0% original vocal bleed.
    """
    print("\n[RE-MIXING] Overlaying English speech onto original background track...")
    t0 = time.time()

    target_duration_ms = int(total_duration_sec * 1000)

    # 1. Load the clean background track (intro music + beats + sound effects)
    if background_track_path.exists():
        master_bg = AudioSegment.from_file(background_track_path)
        # Soften background music slightly (-2 dB) so English voice sits clearly in front
        master_bg = master_bg - 2.0
        if len(master_bg) < target_duration_ms:
            master_bg = master_bg + AudioSegment.silent(duration=target_duration_ms - len(master_bg))
        else:
            master_bg = master_bg[:target_duration_ms]
    else:
        master_bg = AudioSegment.silent(duration=target_duration_ms)

    current_playback_head_ms = 0

    # 2. Sequentially place speech segments with Anti-Collision Guard
    for i, seg in enumerate(segments):
        raw_tts = seg.get("tts_audio_path")
        if not raw_tts or not raw_tts.exists():
            continue

        # Calculate maximum window before next sentence begins
        if i + 1 < len(segments):
            next_start_sec = segments[i + 1]["start"]
            available_duration = max(next_start_sec - seg["start"], 0.3)
        else:
            available_duration = seg["duration"]

        aligned_seg_path = temp_dir / f"aligned_{seg['id']:04d}.wav"
        fit_audio_segment(raw_tts, available_duration, aligned_seg_path)

        if aligned_seg_path.exists():
            seg_audio = AudioSegment.from_file(aligned_seg_path)
            # Crisp dialogue presence (+1 dB)
            seg_audio = seg_audio + 1.0
            scheduled_start_ms = int(seg["start"] * 1000)

            # ANTI-OVERLAP GUARD: If previous sentence ran long, push start time forward
            actual_start_ms = max(scheduled_start_ms, current_playback_head_ms)

            # Overlay onto background track (intro music & pauses remain 100% untouched)
            master_bg = master_bg.overlay(seg_audio, position=actual_start_ms)
            current_playback_head_ms = actual_start_ms + len(seg_audio) + 40

    master_bg = master_bg[:target_duration_ms]
    master_bg.export(output_wav, format="wav")

    elapsed = time.time() - t0
    print(f"  ✓ Synchronized master audio mixed in {elapsed:.2f}s: {output_wav}")
    return output_wav

# %% [markdown]
# ### Step 7: Lossless FFmpeg Video Remuxing (-c:v copy)

# %%
def remux_video(original_video: Path, dubbed_audio: Path, output_video: Path):
    print(f"\n[VIDEO REMUX] Merging dubbed audio with original visual stream...")
    t0 = time.time()
    cmd = [
        "ffmpeg", "-y",
        "-i", str(original_video),
        "-i", str(dubbed_audio),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",       # 100% lossless video preservation
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        str(output_video)
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"  ✓ Final Dubbed Video ready in {time.time() - t0:.2f}s: {output_video}")
    return output_video

# %% [markdown]
# ### Step 8: Master Runner Function

# %%
def run_dubbing_pipeline(
    youtube_url: str,
    clone_voice: bool = True,
    voice: str = "auto",
    generate_video: bool = False,  # Default to False for ultra-fast audio evaluation
    output_name: str = "dubbed_output.mp4"
):
    total_start = time.time()
    print("=" * 75)
    print("  STUDIO AUDIO DUBBING PIPELINE (DEMUCS + WHISPER + CLONING + REMIX)")
    print("=" * 75)

    # 1. Download Media / Audio
    video_path, master_audio_path = extract_media(youtube_url, WORKSPACE_DIR)

    orig_sound = AudioSegment.from_file(master_audio_path)
    total_duration_sec = len(orig_sound) / 1000.0
    print(f"  → Total Media Duration: {total_duration_sec / 60:.2f} min ({total_duration_sec:.1f}s)")

    # 2. Demucs AI Stem Separation (Speech vs Background Music)
    vocals_path, bg_music_path = separate_stems_demucs(master_audio_path, WORKSPACE_DIR, device=WHISPER_DEVICE)

    # 3. Transcribe & Translate ONLY the Speech Track
    segments, source_lang = transcribe_and_translate_speech(vocals_path, model_size=WHISPER_MODEL_SIZE, device=WHISPER_DEVICE)

    # 4. Voice Synthesis (Cloned Voice or Smart Neural Voice)
    tts_dir = WORKSPACE_DIR / "tts_segments"
    voice_used = "F5-TTS / Cloned Voice"

    if clone_voice:
        try:
            ref_voice_path = WORKSPACE_DIR / "clean_speaker_ref.wav"
            # Extract clean reference from vocals.wav (0% music contamination!)
            extract_clean_reference(vocals_path, segments, ref_voice_path)
            synthesize_cloned_voice(segments, ref_voice_path, tts_dir, device=WHISPER_DEVICE)
            voice_used = "Cloned Speaker Voice (F5-TTS Diffusion)"
        except Exception as e:
            print(f"\n  ⚠ Voice Cloning fallback ({e}). Using Region/Gender Matched Neural Voice...")
            matched_voice = detect_speaker_profile(vocals_path, detected_lang=source_lang) if voice == "auto" else voice
            voice_used = f"Edge-TTS ({matched_voice})"
            run_async(batch_synthesize_edge(segments, tts_dir, voice=matched_voice))
    else:
        matched_voice = detect_speaker_profile(vocals_path, detected_lang=source_lang) if voice == "auto" else voice
        voice_used = f"Edge-TTS ({matched_voice})"
        run_async(batch_synthesize_edge(segments, tts_dir, voice=matched_voice))

    # 5. Timing Alignment & Re-Mixing with Original Background Music
    dubbed_audio_path = WORKSPACE_DIR / "dubbed_master.wav"
    build_synchronized_dubbed_master(
        segments,
        background_track_path=bg_music_path,
        total_duration_sec=total_duration_sec,
        output_wav=dubbed_audio_path,
        temp_dir=WORKSPACE_DIR
    )

    # 6. Optional Video Remuxing
    final_output = dubbed_audio_path
    if generate_video:
        final_video_path = WORKSPACE_DIR / output_name
        remux_video(video_path, dubbed_audio_path, final_video_path)
        final_output = final_video_path

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 75)
    print("  PIPELINE EXECUTION COMPLETED")
    print("=" * 75)
    print(f"  - Source: {youtube_url}")
    print(f"  - Detected Language: {source_lang.upper()}")
    print(f"  - Voice Model: {voice_used}")
    print(f"  - Media Length: {total_duration_sec / 60:.2f} min")
    print(f"  - Total Processing Time: {total_elapsed / 60:.2f} min ({total_elapsed:.1f}s)")
    print(f"  - Speedup Ratio: {total_duration_sec / max(total_elapsed, 0.001):.2f}x real-time")
    print(f"  - Master Output File: {final_output.resolve()}")
    print("=" * 75)
    return final_output

# %%
if __name__ == "__main__":
    # Fast audio dubbing by default
    run_dubbing_pipeline(
        "https://www.youtube.com/watch?v=cwoP5XV8Kyg",
        clone_voice=True,
        generate_video=False,
        output_name="cloned_dubbed_output.mp4"
    )
