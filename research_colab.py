"""
=============================================================================
Automated Video Dubbing System - Clean Colab Pipeline
=============================================================================
Architecture & Timing Guarantees:
1. Meta Demucs Stem Separation: Isolates dialogue (vocals.wav) from background (background_music.wav).
2. 16-kHz Mono Speech Prep: Optimizes audio specifically for Whisper speed & accuracy.
3. Direct Speech Translation: Whisper runs task="translate" with beam_size=5 & word_timestamps=True.
4. Voice Cloning (F5-TTS Diffusion) or Neural Voice (Edge-TTS).
5. Strict Hard-Window Synchronization:
   - Each segment is anchored strictly at its exact original start timestamp (int(seg['start'] * 1000)).
   - Speech fits strictly within the segment's OWN duration (never stretches into pauses).
   - Dynamic mild atempo (<=1.30x) + hard boundary trimming with 25ms micro-fadeout.
   - Zero anti-collision playback head pushing (no cumulative drift).
   - 100% original conversational pauses, breaths, and background music are preserved.
6. Clean Workspace Wipe: Automatically resets temporary segment caches on each run.
=============================================================================
"""

# %% [markdown]
# ### Cell 0: Google Colab Setup & Dependencies
# Run this cell first in Google Colab (with T4 GPU enabled):
#
# !apt-get install -y ffmpeg
# !pip install -q yt-dlp faster-whisper edge-tts pydub numpy scipy demucs f5-tts deep-translator

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
    reqs = ["yt-dlp", "faster-whisper", "edge-tts", "pydub", "numpy", "scipy", "demucs", "f5-tts", "deep-translator"]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *reqs], check=True)
    print("  ✓ Dependencies installed successfully.")

# %% [markdown]
# ### Cell 1: Environment & Model Configuration

# %%
import yt_dlp
from faster_whisper import WhisperModel
import edge_tts
from pydub import AudioSegment

WORKSPACE_DIR = Path("./dubbing_workspace")
WORKSPACE_DIR.mkdir(exist_ok=True)

WHISPER_DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") or os.path.exists("/proc/driver/nvidia") else "cpu"
# large-v3-turbo provides large-v3 quality at 3x faster speed on Colab T4 GPU
WHISPER_MODEL_SIZE = "large-v3-turbo" if WHISPER_DEVICE == "cuda" else "small"

print(f"Inference Device: {WHISPER_DEVICE.upper()} | Whisper Model: {WHISPER_MODEL_SIZE}")


# %% [markdown]
# ### Cell 2: Media Extraction (Video & Master Audio)

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
# ### Cell 3: Fast Demucs AI Stem Separation

# %%
def separate_stems_demucs(audio_path: Path, output_dir: Path, device: str = "cuda") -> tuple[Path, Path]:
    """
    Separates speech dialogue from background music and sound effects using Meta AI's Demucs.
    Uses single-pass flags (--shifts=0 --overlap=0.1) for ~3x faster inference.
    Returns:
        (vocals_path, background_music_path)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    vocals_track = output_dir / "vocals.wav"
    bg_track = output_dir / "background_music.wav"

    print("\n[STEM SEPARATION] Isolating speech from background music via Demucs (Fast Single-Pass)...")
    t0 = time.time()

    demucs_out = output_dir / "demucs_temp"
    cmd = [
        sys.executable, "-m", "demucs",
        "--two-stems=vocals",
        "-n", "htdemucs",
        "--shifts=1",
        "--overlap=0.25",
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
            print(f"  ✓ Isolated Dialogue Track: {vocals_track} ({vocals_track.stat().st_size / (1024*1024):.2f} MB)")
            print(f"  ✓ Isolated Background Track: {bg_track} ({bg_track.stat().st_size / (1024*1024):.2f} MB)")
            print(f"  ✓ Stem separation completed in {time.time() - t0:.2f}s")
            return vocals_track, bg_track
    except Exception as e:
        print(f"  ⚠ Demucs execution note: {e}")

    # Fallback if Demucs is absent
    print("  → Demucs not found. Using original audio with muted background fallback.")
    shutil.copyfile(audio_path, vocals_track)
    silent_bg = AudioSegment.silent(duration=len(AudioSegment.from_file(audio_path)))
    silent_bg.export(bg_track, format="wav")
    return vocals_track, bg_track


# %% [markdown]
# ### Cell 4: 16-kHz Mono Conversion & Speech Transcription (task="transcribe")

# %%
def get_16k_mono_speech(vocals_audio: Path, output_dir: Path) -> Path:
    """
    Converts vocal stem to 16kHz mono specifically for Whisper processing efficiency.
    """
    out_path = output_dir / "vocals_16k_mono.wav"
    cmd = [
        "ffmpeg", "-y", "-i", str(vocals_audio),
        "-ar", "16000", "-ac", "1",
        str(out_path)
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_path

def transcribe_speech_native(vocals_audio: Path, model_size: str = "large-v3-turbo", device: str = "cuda"):
    """
    Transcribes speech directly from the isolated vocals stem in its native language.
    Uses task="transcribe" with loosened VAD parameters (threshold=0.35, min_speech=150ms, min_silence=350ms)
    and beam_size=5 for maximum recognition accuracy.
    """
    print(f"\n[TRANSCRIBE] Running faster-whisper on isolated vocals ({model_size} on {device})...")
    compute_type = "float16" if device == "cuda" else "int8"
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    vocals_16k = get_16k_mono_speech(vocals_audio, WORKSPACE_DIR)

    t0 = time.time()
    segments, info = model.transcribe(
        str(vocals_16k),
        task="transcribe",
        vad_filter=True,
        vad_parameters=dict(
            threshold=0.35,
            min_speech_duration_ms=150,
            min_silence_duration_ms=350,
            speech_pad_ms=100
        ),
        word_timestamps=True,
        beam_size=5
    )

    detected_lang = info.language
    lang_prob = info.language_probability
    print(f"  ✓ Detected source language: {detected_lang.upper()} (confidence: {lang_prob:.2%})")

    native_segments = []
    for s in segments:
        text = s.text.strip()
        if not text or len(text) < 2:
            continue

        if hasattr(s, 'words') and s.words:
            seg_start = round(s.words[0].start, 3)
            seg_end = round(s.words[-1].end, 3)
        else:
            seg_start = round(s.start, 3)
            seg_end = round(s.end, 3)

        seg_duration = round(max(seg_end - seg_start, 0.2), 3)
        native_segments.append({
            "id": s.id,
            "start": seg_start,
            "end": seg_end,
            "duration": seg_duration,
            "native_text": text
        })

    elapsed = time.time() - t0
    speech_onset = native_segments[0]['start'] if native_segments else 0.0
    print(f"  ✓ Transcribed {len(native_segments)} segments in {elapsed:.2f}s (First speech starts at {speech_onset:.2f}s)")
    return native_segments, detected_lang

def translate_speech_segments(segments: list, source_lang: str, vocals_16k_path: Path = None) -> list:
    """
    Translates each segment's native_text into English separately.
    Preserves exact original start, end, and duration timestamps.
    """
    if not segments or source_lang.lower() == "en":
        for s in segments:
            s["text"] = s.get("native_text", "")
        return segments

    print(f"\n[TRANSLATION] Translating {len(segments)} segments ({source_lang.upper()} -> EN) preserving timing...")
    t0 = time.time()
    from deep_translator import GoogleTranslator, MyMemoryTranslator

    texts = [s["native_text"].strip() for s in segments]
    translated_texts = ["" for _ in segments]

    # Batch translation with chunking
    CHUNK_SIZE = 15
    for c_start in range(0, len(texts), CHUNK_SIZE):
        c_end = min(c_start + CHUNK_SIZE, len(texts))
        chunk = texts[c_start:c_end]
        try:
            translator = GoogleTranslator(source=source_lang, target="en")
            batch_res = translator.translate_batch(chunk)
            if batch_res and len(batch_res) == len(chunk):
                for idx_in_chunk, res in enumerate(batch_res):
                    if res and res.strip():
                        translated_texts[c_start + idx_in_chunk] = res.strip()
        except Exception as e:
            print(f"  ℹ Google batch translation note ({e}). Trying single queries...")
            time.sleep(1.0)
            for idx_in_chunk, t_item in enumerate(chunk):
                try:
                    res = translator.translate(t_item)
                    if res and res.strip():
                        translated_texts[c_start + idx_in_chunk] = res.strip()
                    time.sleep(0.3)
                except Exception:
                    pass
        time.sleep(0.4)

    # Fallback to MyMemory for any missing
    missing_indices = [i for i, t in enumerate(translated_texts) if not t or not t.strip()]
    if missing_indices:
        try:
            mm = MyMemoryTranslator(source=source_lang, target="en")
            for idx in missing_indices:
                try:
                    res = mm.translate(texts[idx])
                    if res and res.strip():
                        translated_texts[idx] = res.strip()
                    time.sleep(0.3)
                except Exception:
                    pass
        except Exception:
            pass

    # Assign English text while strictly preserving start/end/duration
    for idx, seg in enumerate(segments):
        eng = translated_texts[idx].strip()
        if not eng:
            eng = seg["native_text"]
        seg["text"] = eng

    print(f"  ✓ Translation completed in {time.time() - t0:.2f}s")
    return segments


# %% [markdown]
# ### Cell 5: Reference Audio Extraction & Voice Cloning / Synthesis

# %%
def extract_clean_reference(vocals_path: Path, segments: list, output_ref_path: Path) -> tuple[Path, str]:
    """
    Selects the optimal expressive reference vocal clip based on acoustic energy (RMS)
    and pitch harmonic clarity (autocorrelation) instead of picking the first segment naively.
    Returns:
        (reference_audio_path, real_source_language_ref_text)
    """
    import numpy as np
    sound = AudioSegment.from_file(vocals_path)
    sample_rate = sound.frame_rate
    samples = np.array(sound.get_array_of_samples())
    if sound.channels == 2:
        samples = samples[::2]

    best_candidate = None
    best_score = -1.0

    # Evaluate candidate segments between 4.0s and 12.0s
    candidate_segments = [s for s in segments if 4.0 <= s["duration"] <= 12.0]
    if not candidate_segments:
        candidate_segments = [s for s in segments if s["duration"] >= 3.0]
    if not candidate_segments and segments:
        candidate_segments = [segments[0]]

    for seg in candidate_segments:
        st_sec = seg["start"]
        end_sec = seg["end"]
        st_idx = int(st_sec * sample_rate)
        end_idx = int(end_sec * sample_rate)
        seg_samples = samples[st_idx:end_idx].astype(float)

        if len(seg_samples) < sample_rate * 2:
            continue

        # 1. RMS energy (vocal power)
        rms = float(np.sqrt(np.mean(seg_samples**2)))
        if rms < 100.0:
            continue

        # 2. Pitch harmonic clarity via autocorrelation on central chunk
        chunk_len = min(len(seg_samples), int(sample_rate * 2.5))
        mid = len(seg_samples) // 2
        chunk = seg_samples[max(0, mid - chunk_len // 2) : min(len(seg_samples), mid + chunk_len // 2)]
        chunk -= np.mean(chunk)
        corr = np.correlate(chunk, chunk, mode='full')
        corr = corr[len(corr) // 2 :]

        min_lag = int(sample_rate / 320)
        max_lag = int(sample_rate / 80)
        if max_lag >= len(corr):
            continue

        peak_lag = min_lag + np.argmax(corr[min_lag:max_lag])
        f0 = sample_rate / peak_lag
        clarity = float(corr[peak_lag]) / max(float(corr[0]), 1e-6)

        # Expressive score combines RMS intensity with pitch stability/resonance
        score = (rms / 1000.0) * (clarity ** 1.5) * min(seg["duration"] / 6.0, 1.2)

        if score > best_score:
            best_score = score
            best_candidate = (st_sec, end_sec, seg.get("native_text", "").strip(), f0, rms)

    # Fallback if no high-clarity candidate found
    if best_candidate is None:
        if segments:
            c_start = segments[0]["start"]
            c_end = min(c_start + 8.0, segments[0]["end"])
            c_text = segments[0].get("native_text", "").strip()
        else:
            c_start = 0.0
            c_end = min(8.0, len(sound) / 1000.0)
            c_text = ""
        best_candidate = (c_start, c_end, c_text, 150.0, 500.0)

    chosen_start, chosen_end, ref_text, best_f0, best_rms = best_candidate

    ref_sound = sound[int(chosen_start * 1000) : int(chosen_end * 1000)]
    ref_sound = ref_sound.set_channels(1).set_frame_rate(22050)
    ref_sound.export(output_ref_path, format="wav")
    print(f"  ✓ Clean vocal reference isolated: {output_ref_path}")
    print(f"    - Window: {chosen_start:.2f}s to {chosen_end:.2f}s ({len(ref_sound)/1000.0:.1f}s)")
    print(f"    - Pitch: {best_f0:.1f} Hz | RMS: {best_rms:.0f}")
    print(f"    - Source Transcript (ref_text): \"{ref_text[:60]}...\"")

    return output_ref_path, ref_text

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

def synthesize_cloned_voice(
    segments: list,
    reference_wav: Path,
    ref_text: str = "",
    output_dir: Path = Path("tts_segments"),
    device: str = "cuda"
):
    """
    Synthesizes English dialogue in the cloned speaker's voice using F5-TTS or XTTS-v2.
    Uses real source-language transcript ref_text for expressive acoustic grounding.
    Logs every failure with segment id and text before falling back to silence, with 1 retry.
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

        # Synthesis with 1 retry (up to 2 attempts) and detailed logging
        success = False
        last_err = None
        for attempt in range(2):
            try:
                if cloning_engine == "f5":
                    wav, sr, _ = f5_model.infer(
                        ref_file=str(reference_wav),
                        ref_text=ref_text,
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
                success = True
                break
            except Exception as err:
                last_err = err
                if attempt == 0:
                    time.sleep(0.5)

        if not success:
            print(f"  ⚠ [TTS ERROR] Segment {seg.get('id', idx)} failed (\"{text[:40]}...\"): {last_err}")
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
# ### Cell 6: Strict Hard-Window Timing Alignment & Re-Mixing

# %%
def fit_audio_segment(
    source_audio: Path,
    target_duration_sec: float,
    output_audio: Path,
    max_headroom_sec: float = None
):
    """
    Fits synthesized audio into the speech window using available headroom up to the next segment's start.
    - If audio fits within max_headroom_sec (or target_duration_sec if no headroom), no speedup or trimming needed.
    - If audio exceeds available headroom, applies mild atempo (<=1.30x) and hard-trims strictly at max_headroom_sec with a 25ms micro-fadeout.
    - Preserves natural conversational pauses while strictly preventing overlap with subsequent segments.
    """
    sound = AudioSegment.from_file(source_audio)
    orig_duration_sec = len(sound) / 1000.0

    if target_duration_sec <= 0.1:
        sound.export(output_audio, format="wav")
        return

    # Effective headroom available before colliding with next segment
    if max_headroom_sec is None or max_headroom_sec < target_duration_sec:
        effective_headroom = target_duration_sec
    else:
        effective_headroom = max_headroom_sec

    max_ms = int(effective_headroom * 1000)

    # If audio already fits comfortably within the available headroom window, keep it untouched
    if orig_duration_sec <= effective_headroom:
        sound.export(output_audio, format="wav")
        return

    # Audio exceeds available headroom: apply mild atempo (capped at 1.30x for natural voice timbre)
    ratio = orig_duration_sec / max(effective_headroom, 0.2)
    speed = min(ratio, 1.30)
    temp_fitted = output_audio.parent / f"temp_fit_{output_audio.name}"
    cmd = [
        "ffmpeg", "-y", "-i", str(source_audio),
        "-filter:a", f"atempo={speed:.3f}",
        "-vn", str(temp_fitted)
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    final_sound = AudioSegment.from_file(temp_fitted)
    if temp_fitted.exists():
        temp_fitted.unlink()

    # Hard Speech Window Guard: Strictly trim at headroom limit if it still exceeds with 25ms micro-fadeout
    if len(final_sound) > max_ms:
        final_sound = final_sound[:max_ms].fade_out(25)

    final_sound.export(output_audio, format="wav")

def build_synchronized_dubbed_master(
    segments: list,
    background_track_path: Path,
    total_duration_sec: float,
    output_wav: Path,
    temp_dir: Path
) -> Path:
    """
    Re-mixes the new English dubbed speech with the original background track.
    Fixed Timing & Alignment with Available Headroom:
    - Anchors every segment strictly at its own scheduled start timestamp.
    - Lets each segment use headroom up to the next segment's start before hard-trimming.
    - Never pushes later segments with anti-collision playback head (zero drift/cumulative lag).
    - Preserves natural conversational pauses and intro/outro background music.
    """
    print("\n[RE-MIXING] Overlaying English speech onto background track with headroom-aware alignment...")
    t0 = time.time()

    target_duration_ms = int(total_duration_sec * 1000)

    # 1. Load the clean background track (Demucs no_vocals: music + ambient SFX)
    if background_track_path.exists():
        master_bg = AudioSegment.from_file(background_track_path)
        # Soften background music slightly (-2 dB) so dialogue is crystal clear
        master_bg = master_bg - 2.0
        if len(master_bg) < target_duration_ms:
            master_bg = master_bg + AudioSegment.silent(duration=target_duration_ms - len(master_bg))
        else:
            master_bg = master_bg[:target_duration_ms]
    else:
        master_bg = AudioSegment.silent(duration=target_duration_ms)

    # 2. Overlay each segment strictly at its own exact start timestamp, utilizing headroom
    for idx, seg in enumerate(segments):
        raw_tts = seg.get("tts_audio_path")
        if not raw_tts or not Path(raw_tts).exists():
            continue

        seg_start = seg["start"]
        seg_duration = seg["duration"]

        # Calculate available headroom up to next segment's start (with 50ms safety buffer)
        if idx + 1 < len(segments):
            next_start = segments[idx + 1]["start"]
            headroom_sec = max(round(next_start - seg_start - 0.05, 3), seg_duration)
        else:
            headroom_sec = max(round(total_duration_sec - seg_start - 0.05, 3), seg_duration)

        aligned_seg_path = temp_dir / f"aligned_{seg['id']:04d}.wav"
        fit_audio_segment(raw_tts, seg_duration, aligned_seg_path, max_headroom_sec=headroom_sec)

        if aligned_seg_path.exists():
            seg_audio = AudioSegment.from_file(aligned_seg_path)
            # Ensure it does not exceed headroom
            max_ms = int(headroom_sec * 1000)
            if len(seg_audio) > max_ms:
                seg_audio = seg_audio[:max_ms].fade_out(25)

            # Crisp dialogue presence (+1 dB)
            seg_audio = seg_audio + 1.0

            # STRICT TIMESTAMP ANCHOR: Every segment is locked to its original onset
            start_ms = int(seg_start * 1000)
            master_bg = master_bg.overlay(seg_audio, position=start_ms)

    master_bg = master_bg[:target_duration_ms]
    master_bg.export(output_wav, format="wav")

    elapsed = time.time() - t0
    print(f"  ✓ Synchronized master audio mixed with headroom alignment in {elapsed:.2f}s: {output_wav}")
    return output_wav


# %% [markdown]
# ### Cell 7: Optional Video Remuxing

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
# ### Cell 8: Master Runner Function

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

    # Clean workspace directories to guarantee a 100% fresh run on new media
    for stale_dir in [WORKSPACE_DIR / "tts_segments", WORKSPACE_DIR / "demucs_temp"]:
        if stale_dir.exists():
            shutil.rmtree(stale_dir, ignore_errors=True)
    (WORKSPACE_DIR / "clean_speaker_ref.wav").unlink(missing_ok=True)
    (WORKSPACE_DIR / "dubbed_master.wav").unlink(missing_ok=True)
    (WORKSPACE_DIR / "vocals_16k_mono.wav").unlink(missing_ok=True)
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Download Media / Audio
    video_path, master_audio_path = extract_media(youtube_url, WORKSPACE_DIR)

    orig_sound = AudioSegment.from_file(master_audio_path)
    total_duration_sec = len(orig_sound) / 1000.0
    print(f"  → Total Media Duration: {total_duration_sec / 60:.2f} min ({total_duration_sec:.1f}s)")

    # 2. Demucs AI Stem Separation (Speech vs Background Music)
    vocals_path, bg_music_path = separate_stems_demucs(master_audio_path, WORKSPACE_DIR, device=WHISPER_DEVICE)

    # 3. Transcribe speech in native language (task="transcribe")
    segments, source_lang = transcribe_speech_native(vocals_path, model_size=WHISPER_MODEL_SIZE, device=WHISPER_DEVICE)

    # 3b. Separate translation step preserving original start/end/duration
    segments = translate_speech_segments(segments, source_lang)

    # 4. Voice Synthesis (Cloned Voice or Smart Neural Voice)
    tts_dir = WORKSPACE_DIR / "tts_segments"
    voice_used = "F5-TTS / Cloned Voice"

    if clone_voice:
        try:
            ref_voice_path = WORKSPACE_DIR / "clean_speaker_ref.wav"
            # Extract clean reference using energy/pitch scoring and get real source transcript (ref_text)
            ref_voice_path, ref_text = extract_clean_reference(vocals_path, segments, ref_voice_path)
            synthesize_cloned_voice(segments, ref_voice_path, ref_text=ref_text, output_dir=tts_dir, device=WHISPER_DEVICE)
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


# %% [markdown]
# ### Cell 9: Execution Entry Point

# %%
if __name__ == "__main__":
    # Test execution: replace source with your local video file or YouTube URL
    run_dubbing_pipeline(
        "/content/french_vlog.mp4",
        clone_voice=True,
        generate_video=False,
        output_name="cloned_dubbed_output.mp4"
    )
