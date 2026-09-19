"""
=============================================================================
Automated Video Dubbing System - Optimized Studio Pipeline
=============================================================================
Key Architecture:
1. Fast Demucs Stem Separation (single-pass --shifts=0 --overlap=0.1) -> vocals.wav & no_vocals.wav.
2. Lightweight 16-kHz mono extraction for Whisper.
3. Decoupled STT: Whisper runs task="transcribe" with word_timestamps=True and beam_size=5 for maximum accuracy.
4. Sentence & Clause Grouping: Groups Whisper fragments by punctuation (।, ., ?, !) and pauses (>=0.6s).
5. Dynamic Translation Caching & Semantic Rewriter: Cached by transcript SHA-256 hash; retries without Hindi fallback; semantic contractions/compressions.
6. Closed-Loop TTS Alignment: Synthesizes -> measures duration -> concise semantic rewrite if needed -> mild atempo (<=1.25x).
6. Strict Hard Speech Windows: Each segment is placed strictly at its own start timestamp; gaps are preserved as silence/music.
7. Stage Checkpointing: Stems, transcripts, translations, and TTS segments are cached to disk to prevent re-computation.
8. Background Preservation: Clean Demucs no_vocals.wav is used as the backing track with zero generic noise reduction.
"""

# %% [markdown]
# ### Step 0: Google Colab Setup & Dependencies
# Run this cell in Google Colab (with T4 GPU enabled):
#
# !apt-get install -y ffmpeg
# !pip install -q yt-dlp faster-whisper edge-tts pydub numpy scipy demucs f5-tts deep-translator

# %%
import os
import sys
import json
import subprocess
import asyncio
import concurrent.futures
import time
import shutil
import re
import hashlib
from pathlib import Path

# Automatically accept licenses without terminal prompts
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
# ### Step 1: Configuration & Workspace Setup

# %%
import yt_dlp
from faster_whisper import WhisperModel
import edge_tts
from pydub import AudioSegment
from deep_translator import GoogleTranslator

WORKSPACE_DIR = Path("./dubbing_workspace")
CACHE_DIR = WORKSPACE_DIR / "cache"
WORKSPACE_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

WHISPER_DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") or os.path.exists("/proc/driver/nvidia") else "cpu"
# large-v3-turbo provides large-v3 accuracy at 3x the speed of medium
WHISPER_MODEL_SIZE = "large-v3-turbo" if WHISPER_DEVICE == "cuda" else "small"

print(f"Inference Device: {WHISPER_DEVICE.upper()} | Whisper Model: {WHISPER_MODEL_SIZE}")

# Singleton model caches to ensure models load only once in memory
GLOBAL_MODELS = {
    "whisper": None,
    "f5tts": None
}

def get_whisper_model(model_size: str = WHISPER_MODEL_SIZE, device: str = WHISPER_DEVICE):
    if GLOBAL_MODELS["whisper"] is None:
        print(f"\n[MODEL INIT] Loading faster-whisper ({model_size} on {device})...")
        compute_type = "float16" if device == "cuda" else "int8"
        GLOBAL_MODELS["whisper"] = WhisperModel(model_size, device=device, compute_type=compute_type)
        print("  ✓ Whisper model loaded into memory.")
    return GLOBAL_MODELS["whisper"]

def get_f5tts_model(device: str = WHISPER_DEVICE):
    if GLOBAL_MODELS["f5tts"] is None:
        from f5_tts.api import F5TTS
        print(f"\n[MODEL INIT] Loading F5-TTS Diffusion Engine on {device}...")
        t0 = time.time()
        GLOBAL_MODELS["f5tts"] = F5TTS(device=device)
        print(f"  ✓ F5-TTS loaded in {time.time() - t0:.2f}s")
    return GLOBAL_MODELS["f5tts"]

# %% [markdown]
# ### Step 2: Extract Media (Video & Master Audio)

# %%
def extract_media(source: str, output_dir: Path) -> tuple[Path, Path]:
    """
    Downloads media from YouTube or loads a local file.
    Outputs:
        video_path (.mp4) and master_audio (.wav, 44.1kHz stereo).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "original_video.mp4"
    audio_path = output_dir / "original_audio.wav"

    source_path = Path(source)
    if source_path.exists() and source_path.is_file():
        print(f"\n[MEDIA] Loading local file: {source_path.resolve()}")
        if source_path != video_path:
            shutil.copyfile(source_path, video_path)
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-ar", "44100", "-ac", "2",
            str(audio_path)
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return video_path, audio_path

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
        'postprocessor_args': ['-ar', '44100', '-ac', '2'],
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
# ### Step 3: Fast AI Stem Separation (Demucs - Single Pass)
# Separates speech dialogue (vocals.wav) from the accompaniment (no_vocals.wav).
# Uses single-pass flags (--shifts=0 --overlap=0.1) for ~3x speedup. Caches results to disk.

# %%
def separate_stems_demucs(audio_path: Path, output_dir: Path, device: str = "cuda") -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stems_cache_dir = output_dir / "stems_cache"
    stems_cache_dir.mkdir(parents=True, exist_ok=True)

    vocals_track = stems_cache_dir / "vocals.wav"
    bg_track = stems_cache_dir / "background_music.wav"

    # Check disk cache
    if vocals_track.exists() and bg_track.exists() and vocals_track.stat().st_size > 1000:
        print(f"\n[STEM SEPARATION] Using cached stems from: {stems_cache_dir}")
        return vocals_track, bg_track

    print("\n[STEM SEPARATION] Isolating speech & background via Demucs (Fast Single-Pass)...")
    t0 = time.time()

    demucs_out = output_dir / "demucs_temp"
    # --shifts=0 and --overlap=0.1 run in a single high-speed forward pass
    cmd = [
        sys.executable, "-m", "demucs",
        "--two-stems=vocals",
        "-n", "htdemucs",
        "--shifts=0",
        "--overlap=0.1",
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
            print(f"  ✓ Isolated Speech Track (vocals.wav): {vocals_track.stat().st_size / (1024*1024):.2f} MB")
            print(f"  ✓ Isolated Background Track (no_vocals.wav): {bg_track.stat().st_size / (1024*1024):.2f} MB")
            print(f"  ✓ Stem separation completed in {time.time() - t0:.2f}s")
            return vocals_track, bg_track
    except Exception as e:
        print(f"  ⚠ Demucs execution note: {e}")

    # Fallback
    print("  → Demucs not installed. Using original audio with muted background fallback.")
    shutil.copyfile(audio_path, vocals_track)
    silent_bg = AudioSegment.silent(duration=len(AudioSegment.from_file(audio_path)))
    silent_bg.export(bg_track, format="wav")
    return vocals_track, bg_track

# %% [markdown]
# ### Step 4: Decoupled Speech Transcription (Whisper task="transcribe")
# Runs on a 16-kHz mono speech track with greedy decoding (beam_size=1) for maximum speed.
# Produces native words & timestamps; results are cached to transcript.json.

# %%
def get_16k_mono_speech(vocals_audio: Path, output_dir: Path) -> Path:
    """
    Converts vocal stem to 16kHz mono specifically for Whisper processing efficiency.
    """
    out_path = output_dir / "vocals_16k_mono.wav"
    if out_path.exists() and out_path.stat().st_size > 1000:
        return out_path
    cmd = [
        "ffmpeg", "-y", "-i", str(vocals_audio),
        "-ar", "16000", "-ac", "1",
        str(out_path)
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_path

def transcribe_speech_native(vocals_16k_audio: Path, cache_dir: Path) -> tuple[list, str]:
    """
    Transcribes the isolated vocals stem in its native language with word timestamps.
    Uses beam_size=5 for maximum accuracy on fast speech and regional accents.
    Caches output keyed by audio file signature to prevent cross-video cache reuse.
    """
    # Key transcript cache by audio file signature (size + mtime + name)
    try:
        audio_stat = vocals_16k_audio.stat()
        audio_sig = f"{vocals_16k_audio.name}_{audio_stat.st_size}_{int(audio_stat.st_mtime)}"
        audio_hash = hashlib.sha256(audio_sig.encode()).hexdigest()[:12]
    except Exception:
        audio_hash = "default"

    transcript_cache_file = cache_dir / f"transcript_native_{audio_hash}.json"
    if transcript_cache_file.exists():
        with open(transcript_cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            print(f"\n[TRANSCRIBE] Loaded {len(data['segments'])} cached segments from: {transcript_cache_file.name}")
            return data["segments"], data["language"]

    print("\n[TRANSCRIBE] Running faster-whisper on 16kHz vocal stem (task='transcribe', beam_size=5)...")
    t0 = time.time()
    model = get_whisper_model()

    segments, info = model.transcribe(
        str(vocals_16k_audio),
        task="transcribe",
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=600,
            threshold=0.50,
            min_speech_duration_ms=250,
            speech_pad_ms=100
        ),
        word_timestamps=True,
        beam_size=5  # High-accuracy beam search (beam_size=5)
    )

    detected_lang = info.language
    lang_prob = info.language_probability
    print(f"  ✓ Detected source language: {detected_lang.upper()} (confidence: {lang_prob:.2%})")

    native_segments = []
    for s in segments:
        text = s.text.strip()
        if not text:
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
    print(f"  ✓ Transcribed {len(native_segments)} raw fragments in {elapsed:.2f}s (First speech starts at {native_segments[0]['start']:.2f}s)")

    # Save to disk cache
    with open(transcript_cache_file, "w", encoding="utf-8") as f:
        json.dump({"language": detected_lang, "segments": native_segments}, f, indent=2, ensure_ascii=False)

    return native_segments, detected_lang

# %% [markdown]
# ### Step 4b: Sentence & Clause Grouping
# Groups consecutive Whisper fragments into complete semantic sentences based on
# punctuation (।, ॥, ., ?, !) and conversational pauses (~0.5-0.8s).
# The translated English is mapped back to the group's original start/end timestamps.

# %%
def group_speech_segments(
    segments: list,
    pause_threshold: float = 0.6,
    max_duration: float = 12.0
) -> list:
    """
    Groups consecutive Whisper fragment segments into coherent sentences/clauses.
    A group boundary is formed when:
    1. A segment ends with sentence-ending punctuation (Hindi danda '।', '॥', '.', '?', '!').
    2. The pause between the end of the current segment and the start of the next segment exceeds `pause_threshold` (~0.5-0.8s).
    3. The accumulated group duration would exceed `max_duration` (preventing over-long segments).

    The resulting grouped segment spans [group[0]['start'] -> group[-1]['end']] exactly,
    preserving natural speech windows and ambient pauses on the timeline.
    """
    if not segments:
        return []

    print(f"\n[CLAUSE GROUPING] Grouping {len(segments)} fragments into complete sentences (pause_gap >= {pause_threshold}s)...")
    grouped = []
    current_group = []

    # Universal multilingual sentence terminators:
    # Western: . ? ! | Indic: । (U+0964), ॥ (U+0965)
    # CJK: 。 (U+3002), ！ (U+FF01), ？ (U+FF1F) | Arabic/Persian: ؟ (U+061F), ۔ (U+06D4)
    SENTENCE_END_REGEX = re.compile(r"[।॥.?!。！？\u061F\u06D4](\s*[\"'\)\]»›”’\s]*)?$")

    for i, seg in enumerate(segments):
        current_group.append(seg)
        text = seg["native_text"].strip()
        ends_with_punct = bool(SENTENCE_END_REGEX.search(text))

        # Check pause gap to the following segment
        has_pause = False
        if i < len(segments) - 1:
            gap = segments[i + 1]["start"] - seg["end"]
            if gap >= pause_threshold:
                has_pause = True
        else:
            has_pause = True  # Final segment finishes the group

        group_duration = current_group[-1]["end"] - current_group[0]["start"]
        duration_exceeded = group_duration >= max_duration

        if ends_with_punct or has_pause or duration_exceeded:
            g_start = current_group[0]["start"]
            g_end = current_group[-1]["end"]
            g_text = " ".join(s["native_text"].strip() for s in current_group if s["native_text"].strip()).strip()

            grouped.append({
                "id": len(grouped) + 1,
                "start": g_start,
                "end": g_end,
                "duration": round(max(g_end - g_start, 0.2), 3),
                "native_text": g_text,
                "fragment_count": len(current_group)
            })
            current_group = []

    if current_group:
        g_start = current_group[0]["start"]
        g_end = current_group[-1]["end"]
        g_text = " ".join(s["native_text"].strip() for s in current_group if s["native_text"].strip()).strip()
        grouped.append({
            "id": len(grouped) + 1,
            "start": g_start,
            "end": g_end,
            "duration": round(max(g_end - g_start, 0.2), 3),
            "native_text": g_text,
            "fragment_count": len(current_group)
        })

    print(f"  ✓ Merged into {len(grouped)} complete sentence clauses (avg {len(segments)/max(len(grouped), 1):.1f} fragments/clause)")
    return grouped

# %% [markdown]
# ### Step 5: Decoupled Translation with Semantic-Preserving Rewriting
# Translates grouped sentence clauses as unified semantic units into English.
# Replaces naive filler removal with semantic-preserving contractions and concise phrasing.
# Keys translation cache by SHA-256 hash of the transcript to prevent cross-video cache reuse.
# Retries translation and NEVER falls back to Hindi text as English.

# %%
def semantic_concise_rewrite(text: str) -> str:
    """
    Produces a semantic-preserving concise rewrite of English text without altering meaning.
    Applies spoken contractions and compresses wordy periphrastic phrasing into direct equivalents.
    NOTE: NEVER deletes modifier/emphasis words like 'really', 'just', 'actually' as that distorts semantics.
    """
    if not text:
        return ""

    # 1. Natural spoken contractions (100% semantic identity, reduces syllable count)
    contractions = {
        r"\bI am\b": "I'm",
        r"\bdo not\b": "don't",
        r"\bcannot\b": "can't",
        r"\bcan not\b": "can't",
        r"\bwill not\b": "won't",
        r"\bit is\b": "it's",
        r"\bthat is\b": "that's",
        r"\bthere is\b": "there's",
        r"\byou are\b": "you're",
        r"\bthey are\b": "they're",
        r"\bwe are\b": "we're",
        r"\bhave not\b": "haven't",
        r"\bhas not\b": "hasn't",
        r"\bwould not\b": "wouldn't",
        r"\bshould not\b": "shouldn't",
        r"\bcould not\b": "couldn't",
        r"\bwe will\b": "we'll",
        r"\bthey will\b": "they'll",
        r"\byou will\b": "you'll",
        r"\bI will\b": "I'll",
        r"\bI have\b": "I've",
        r"\byou have\b": "you've",
        r"\bwe have\b": "we've",
        r"\bthey have\b": "they've"
    }
    shortened = text
    for pattern, repl in contractions.items():
        shortened = re.sub(pattern, repl, shortened, flags=re.IGNORECASE)

    # 2. Semantic periphrastic compressions (concise synonyms preserving exact nuance)
    semantic_compressions = [
        (r"\bin order to\b", "to"),
        (r"\bdue to the fact that\b", "because"),
        (r"\bat this point in time\b", "now"),
        (r"\bat the present moment\b", "currently"),
        (r"\ba large number of\b", "many"),
        (r"\bfor the purpose of\b", "for"),
        (r"\bin the event that\b", "if"),
        (r"\bwith the exception of\b", "except"),
        (r"\bas a matter of fact\b", "in fact"),
        (r"\bmake a decision\b", "decide"),
        (r"\btake into consideration\b", "consider"),
        (r"\bgive an explanation\b", "explain"),
        (r"\bis able to\b", "can"),
        (r"\bare able to\b", "can"),
        (r"\bhas the ability to\b", "can"),
        (r"\bhave the ability to\b", "can"),
        (r"\bhas got to\b", "must"),
        (r"\bhave got to\b", "must"),
        (r"\bin spite of the fact that\b", "although"),
        (r"\buntil such time as\b", "until"),
        (r"\bprior to\b", "before"),
        (r"\bsubsequent to\b", "after"),
        (r"\ba sufficient amount of\b", "enough")
    ]
    for pattern, repl in semantic_compressions:
        shortened = re.sub(pattern, repl, shortened, flags=re.IGNORECASE)

    # Clean redundant spaces
    shortened = re.sub(r"\s+", " ", shortened).strip()
    return shortened

def translate_grouped_segments(
    segments: list,
    source_lang: str,
    cache_dir: Path,
    target_lang: str = "en"
) -> list:
    """
    Translates grouped sentence clauses to English (or target language) as unified units.
    - Keys cache by SHA-256 hash of (source_lang + target_lang + transcript_content).
    - Prevents any accidental cache reuse across videos or languages.
    - Retries translation with exponential backoff.
    - NEVER falls back to untranslated source text: marks failed segments to prevent foreign text in TTS.
    """
    if not segments:
        return []

    # Dynamic cache key derived from source_lang + target_lang + transcript content
    src = source_lang.lower().strip()
    tgt = target_lang.lower().strip()
    transcript_blob = "||".join(f"{s['id']}:{s['native_text']}" for s in segments)
    content_key = f"{src}->{tgt}||{transcript_blob}"
    transcript_hash = hashlib.sha256(content_key.encode("utf-8")).hexdigest()[:12]
    trans_cache_file = cache_dir / f"translations_{src}_to_{tgt}_{transcript_hash}.json"

    if trans_cache_file.exists():
        with open(trans_cache_file, "r", encoding="utf-8") as f:
            cached_segs = json.load(f)
            print(f"\n[TRANSLATION] Loaded {len(cached_segs)} cached translations ({src.upper()} -> {tgt.upper()}) from: {trans_cache_file.name}")
            return cached_segs

    print(f"\n[TRANSLATION] Translating {len(segments)} sentence clauses ({src.upper()} -> {tgt.upper()}) as coherent units...")
    t0 = time.time()
    translator = GoogleTranslator(source=src, target=tgt)

    for seg in segments:
        native = seg["native_text"].strip()
        english = ""
        max_retries = 3

        for attempt in range(1, max_retries + 1):
            try:
                res = translator.translate(native)
                if res and res.strip():
                    english = res.strip()
                    break
            except Exception as e:
                if attempt < max_retries:
                    time.sleep(0.7 * attempt)
                else:
                    print(f"  ⚠ Segment {seg['id']} translation attempt {attempt} failed: {e}")

        if english:
            seg["english_text"] = english
            seg["concise_text"] = semantic_concise_rewrite(english)
            seg["translation_failed"] = False
        else:
            # Strictly do NOT fall back to native source text!
            print(f"  ⚠ Translation permanently failed for clause {seg['id']}: '{native[:40]}...'. Marked as failed (zero untranslated text in TTS).")
            seg["english_text"] = ""
            seg["concise_text"] = ""
            seg["translation_failed"] = True

    elapsed = time.time() - t0
    print(f"  ✓ Translation completed in {elapsed:.2f}s (Cache saved: {trans_cache_file.name})")

    with open(trans_cache_file, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)

    return segments

# %% [markdown]
# ### Step 6: Closed-Loop TTS Alignment (Duration Feedback & Re-synthesis)
# 1. Synthesizes English segment.
# 2. Measures audio duration D_syn.
# 3. If D_syn > target_duration:
#    - Rewrites with condensed/shortened text.
#    - Re-synthesizes.
#    - Applies only mild atempo (<=1.25x) to seal remaining difference.
# 4. Trims strictly to target duration with a micro-fadeout (NEVER bleeds into next segment).

# %%
def extract_clean_reference(vocals_path: Path, segments: list, output_ref_path: Path) -> Path:
    sound = AudioSegment.from_file(vocals_path)
    chosen_start, chosen_end = None, None

    for s in segments:
        dur = s["duration"]
        if 5.0 <= dur <= 12.0:
            chosen_start, chosen_end = s["start"], s["end"]
            break

    if chosen_start is None:
        if segments and segments[0]["duration"] >= 3.0:
            chosen_start = segments[0]["start"]
            chosen_end = min(chosen_start + 8.0, segments[0]["end"])
        else:
            chosen_start = 0.0
            chosen_end = min(8.0, len(sound) / 1000.0)

    ref_sound = sound[int(chosen_start * 1000) : int(chosen_end * 1000)]
    ref_sound = ref_sound.set_channels(1).set_frame_rate(22050)
    ref_sound.export(output_ref_path, format="wav")
    print(f"  ✓ Clean vocal reference: {output_ref_path} ({len(ref_sound)/1000.0:.1f}s from {chosen_start:.1f}s to {chosen_end:.1f}s)")
    return output_ref_path

def detect_speaker_profile(audio_path: Path, detected_lang: str = "en") -> str:
    """
    Analyzes clean reference speech pitch (F0) and selects an appropriate regional English
    neural voice matching speaker gender and source accent profile.
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
        is_female = (f0 > 185)
    except Exception:
        is_female = False

    lang = (detected_lang or "en").lower().strip()

    # Regional accent-aware English voice mappings
    if lang in ["hi", "ur", "pa", "bn", "ta", "te", "mr", "gu", "kn", "ml"]:
        # South Asian languages -> Indian English Neural
        return "en-IN-NeerjaNeural" if is_female else "en-IN-PrabhatNeural"
    elif lang in ["de", "nl", "da", "sv", "no"]:
        # Germanic / Northern European -> British English (natural European pacing)
        return "en-GB-SoniaNeural" if is_female else "en-GB-RyanNeural"
    elif lang in ["fr", "it", "es", "pt"]:
        # Romance languages -> Expressive British/International English
        return "en-GB-LibbyNeural" if is_female else "en-GB-ThomasNeural"
    elif lang in ["ja", "zh", "ko"]:
        # East Asian -> Modern crisp neural English
        return "en-US-AvaNeural" if is_female else "en-US-ChristopherNeural"
    elif lang in ["ar", "fa", "he", "tr"]:
        # Middle Eastern / Mediterranean -> Warm neural English
        return "en-US-JennyNeural" if is_female else "en-US-GuyNeural"
    elif lang in ["ru", "uk", "pl", "cs", "ro"]:
        # Eastern European -> Clear standard neural English
        return "en-US-AvaNeural" if is_female else "en-US-ChristopherNeural"
    else:
        # Global Default -> High-fidelity US English
        return "en-US-AvaNeural" if is_female else "en-US-ChristopherNeural"

def synthesize_single_segment_tts(text: str, output_wav: Path, engine: str, voice_or_ref, device: str = "cuda"):
    """
    Synthesizes a single segment using F5-TTS or Edge-TTS.
    """
    if not text or len(text.strip()) < 2:
        AudioSegment.silent(duration=400).export(output_wav, format="wav")
        return

    if engine == "f5":
        f5_model = get_f5tts_model(device)
        f5_model.infer(
            ref_file=str(voice_or_ref),
            ref_text="",
            gen_text=text,
            file_wave=str(output_wav)
        )
    else:
        # Edge-TTS
        async def _run_edge():
            communicate = edge_tts.Communicate(text, str(voice_or_ref))
            await communicate.save(str(output_wav))
        run_async(_run_edge())

def fit_closed_loop_segment(
    seg: dict,
    output_final_wav: Path,
    engine: str,
    voice_or_ref,
    temp_dir: Path,
    device: str = "cuda"
):
    """
    Closed-Loop Duration Feedback:
    Synthesize -> measure duration -> concise semantic rewrite if necessary -> regenerate -> mild atempo (<=1.25x).
    Guarantees segment fits strictly into [start -> end] without bleeding into the next segment.
    If translation failed, outputs silence for the exact speech window to prevent foreign text corruption.
    """
    target_sec = seg["duration"]
    max_ms = int(target_sec * 1000)

    # 0. Translation failure guard: never speak Hindi in English TTS
    if seg.get("translation_failed") or not seg.get("english_text"):
        AudioSegment.silent(duration=max_ms).export(output_final_wav, format="wav")
        seg["tts_audio_path"] = output_final_wav
        seg["final_duration"] = target_sec
        return

    temp_wav_1 = temp_dir / f"raw1_{seg['id']:04d}.wav"

    # 1. First synthesis pass (standard translation)
    text_to_try = seg["english_text"]
    synthesize_single_segment_tts(text_to_try, temp_wav_1, engine, voice_or_ref, device)

    dur_1 = len(AudioSegment.from_file(temp_wav_1)) / 1000.0

    # 2. Closed-loop check: if speech is significantly longer than original slot (>12%)
    chosen_wav = temp_wav_1
    current_dur = dur_1

    concise_text = seg.get("concise_text", "")
    if dur_1 > target_sec * 1.12 and concise_text and concise_text != text_to_try:
        temp_wav_2 = temp_dir / f"raw2_{seg['id']:04d}.wav"
        # Re-synthesize with semantic concise rewrite
        synthesize_single_segment_tts(concise_text, temp_wav_2, engine, voice_or_ref, device)
        dur_2 = len(AudioSegment.from_file(temp_wav_2)) / 1000.0
        chosen_wav = temp_wav_2
        current_dur = dur_2

    # 3. Apply mild atempo if needed (capped at 1.25x to preserve natural vocal tone)
    ratio = current_dur / max(target_sec, 0.2)
    temp_fitted = temp_dir / f"fitted_{seg['id']:04d}.wav"

    if ratio > 1.05:
        speed = min(ratio, 1.25)  # Mild atempo (max 1.25x)
        cmd = [
            "ffmpeg", "-y", "-i", str(chosen_wav),
            "-filter:a", f"atempo={speed:.3f}",
            "-vn", str(temp_fitted)
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        final_sound = AudioSegment.from_file(temp_fitted)
    else:
        final_sound = AudioSegment.from_file(chosen_wav)

    # 4. Hard Speech Window Guard: Strictly trim if it still exceeds target_sec with a micro-fadeout
    if len(final_sound) > max_ms:
        final_sound = final_sound[:max_ms].fade_out(25)

    final_sound.export(output_final_wav, format="wav")
    seg["tts_audio_path"] = output_final_wav
    seg["final_duration"] = len(final_sound) / 1000.0

# %% [markdown]
# ### Step 7: Strict Hard-Window Timeline Alignment & Background Re-mixing
# Places each segment strictly at its own start timestamp on top of Demucs no_vocals.wav.
# Never pushes later segments forward; all original gaps and pauses are preserved as silence/music.

# %%
def build_hard_window_dubbed_master(
    segments: list,
    background_track_path: Path,
    total_duration_sec: float,
    output_wav: Path
) -> Path:
    print("\n[STRICT HARD-WINDOW ALIGNMENT] Anchoring speech strictly at original start timestamps...")
    t0 = time.time()

    target_duration_ms = int(total_duration_sec * 1000)

    # Load original Demucs accompaniment (0% vocals, 100% music/SFX)
    if background_track_path.exists():
        master_bg = AudioSegment.from_file(background_track_path)
        master_bg = master_bg - 1.5  # Subtle -1.5 dB ducking for dialogue clarity
        if len(master_bg) < target_duration_ms:
            master_bg = master_bg + AudioSegment.silent(duration=target_duration_ms - len(master_bg))
        else:
            master_bg = master_bg[:target_duration_ms]
    else:
        master_bg = AudioSegment.silent(duration=target_duration_ms)

    # Strict hard-window placement: every segment is overlaid at int(seg['start'] * 1000)
    for seg in segments:
        raw_audio = seg.get("tts_audio_path")
        if not raw_audio or not raw_audio.exists():
            continue

        seg_sound = AudioSegment.from_file(raw_audio)
        # Ensure speech does not exceed its own segment duration
        max_duration_ms = int(seg["duration"] * 1000)
        if len(seg_sound) > max_duration_ms:
            seg_sound = seg_sound[:max_duration_ms].fade_out(20)

        # Boost speech presence (+1 dB)
        seg_sound = seg_sound + 1.0

        # Hard timestamp anchor: NEVER pushes subsequent segments
        start_ms = int(seg["start"] * 1000)
        master_bg = master_bg.overlay(seg_sound, position=start_ms)

    master_bg = master_bg[:target_duration_ms]
    master_bg.export(output_wav, format="wav")

    print(f"  ✓ Master dubbed audio mixed with 100% pause preservation in {time.time() - t0:.2f}s: {output_wav.name}")
    return output_wav

def remux_video(original_video: Path, dubbed_audio: Path, output_video: Path):
    print(f"\n[VIDEO REMUX] Multiplexing dubbed audio with video track (-c:v copy)...")
    t0 = time.time()
    cmd = [
        "ffmpeg", "-y",
        "-i", str(original_video),
        "-i", str(dubbed_audio),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
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
    source: str,
    clone_voice: bool = True,
    voice: str = "auto",
    generate_video: bool = False,
    output_name: str = "dubbed_output.mp4"
):
    total_start = time.time()
    print("=" * 75)
    print("  OPTIMIZED AUDIO DUBBING PIPELINE (DECOUPLED STT + CLOSED-LOOP FIT)")
    print("=" * 75)

    # 1. Extract Media
    video_path, master_audio_path = extract_media(source, WORKSPACE_DIR)
    orig_sound = AudioSegment.from_file(master_audio_path)
    total_duration_sec = len(orig_sound) / 1000.0
    print(f"  → Total Media Duration: {total_duration_sec / 60:.2f} min ({total_duration_sec:.1f}s)")

    # 2. Fast Demucs Stem Separation (Cached)
    vocals_path, bg_music_path = separate_stems_demucs(master_audio_path, WORKSPACE_DIR, device=WHISPER_DEVICE)

    # 3. Lightweight 16kHz mono extraction for Whisper
    vocals_16k_path = get_16k_mono_speech(vocals_path, WORKSPACE_DIR)

    # 4. Decoupled STT: Native Transcription (task='transcribe', beam_size=5, word_timestamps=True)
    raw_segments, source_lang = transcribe_speech_native(vocals_16k_path, CACHE_DIR)

    # 4b. Sentence & Clause Grouping (Merge Whisper fragments by punctuation and 0.6s pauses)
    segments = group_speech_segments(raw_segments, pause_threshold=0.6, max_duration=12.0)

    # 5. Decoupled Translation with Dynamic Hash Caching & Semantic-Preserving Rewriter
    segments = translate_grouped_segments(segments, source_lang, CACHE_DIR)

    # 6. Closed-Loop Voice Synthesis (Measure -> Shorten -> Mild Atempo)
    tts_dir = WORKSPACE_DIR / "tts_segments"
    tts_dir.mkdir(parents=True, exist_ok=True)
    voice_used = "F5-TTS / Cloned Voice"
    engine = "f5"
    voice_or_ref = None

    if clone_voice:
        try:
            ref_voice_path = WORKSPACE_DIR / "clean_speaker_ref.wav"
            extract_clean_reference(vocals_path, segments, ref_voice_path)
            # Verify F5-TTS model is available
            get_f5tts_model(WHISPER_DEVICE)
            engine = "f5"
            voice_or_ref = ref_voice_path
            voice_used = "Cloned Speaker Voice (F5-TTS Diffusion)"
        except Exception as e:
            print(f"\n  ℹ Voice Cloning fallback ({e}). Using Regional Neural Voice...")
            matched_voice = detect_speaker_profile(vocals_path, detected_lang=source_lang) if voice == "auto" else voice
            engine = "edge"
            voice_or_ref = matched_voice
            voice_used = f"Edge-TTS ({matched_voice})"
    else:
        matched_voice = detect_speaker_profile(vocals_path, detected_lang=source_lang) if voice == "auto" else voice
        engine = "edge"
        voice_or_ref = matched_voice
        voice_used = f"Edge-TTS ({matched_voice})"

    print(f"\n[CLOSED-LOOP SYNTHESIS] Generating {len(segments)} segments with duration feedback...")
    t_synth = time.time()
    for idx, seg in enumerate(segments, start=1):
        final_seg_path = tts_dir / f"final_seg_{seg['id']:04d}.wav"
        # Check cache
        if final_seg_path.exists() and final_seg_path.stat().st_size > 500:
            seg["tts_audio_path"] = final_seg_path
            continue

        fit_closed_loop_segment(seg, final_seg_path, engine, voice_or_ref, WORKSPACE_DIR, device=WHISPER_DEVICE)
        if idx % 10 == 0 or idx == len(segments):
            print(f"    - Processed {idx}/{len(segments)} segments ({idx/len(segments)*100:.1f}%)")

    print(f"  ✓ Closed-loop synthesis completed in {time.time() - t_synth:.2f}s")

    # 7. Strict Hard-Window Alignment & Re-Mixing onto Demucs no_vocals.wav
    dubbed_audio_path = WORKSPACE_DIR / "dubbed_master.wav"
    build_hard_window_dubbed_master(
        segments,
        background_track_path=bg_music_path,
        total_duration_sec=total_duration_sec,
        output_wav=dubbed_audio_path
    )

    # 8. Optional Video Remuxing
    final_output = dubbed_audio_path
    if generate_video:
        final_video_path = WORKSPACE_DIR / output_name
        remux_video(video_path, dubbed_audio_path, final_video_path)
        final_output = final_video_path

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 75)
    print("  PIPELINE EXECUTION COMPLETED")
    print("=" * 75)
    print(f"  - Source: {source}")
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
    # Fast audio dubbing by default with closed-loop fitting and hard-window alignment
    run_dubbing_pipeline(
        "https://www.youtube.com/watch?v=cwoP5XV8Kyg",
        clone_voice=True,
        generate_video=False,
        output_name="cloned_dubbed_output.mp4"
    )
