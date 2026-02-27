import time, os, subprocess, sys, torch, gc, requests, logging
import librosa, soundfile as sf, noisereduce as nr
from faster_whisper import WhisperModel

# ─── CONFIG ────────────────────────────────────────────────────────
BASE_DIR    = r"C:\FMPC_Scribe"
INBOX       = os.path.join(BASE_DIR, "INBOX")
TRANSCRIPTS = os.path.join(BASE_DIR, "NOTES_Transcripts")
ARCHIVE     = r"D:\FMPC_Audio_Archive\Medical"
LOG_PATH    = os.path.join(BASE_DIR, "scribe_log.txt")
OLLAMA_URL  = "http://localhost:11434/api/generate"
TMP_WAV     = os.path.join(BASE_DIR, "temp_processing.wav")

# Noise reduction tuned for noisy lecture halls.
# Increase toward 0.85 for very loud rooms. Decrease to 0.50 for clean rooms.
NOISE_REDUCTION = 0.65

# Pause threshold for paragraph grouping (seconds).
# 2.5s = new paragraph when lecturer pauses between topics.
# Increase to 4.0 if paragraphs are too fragmented.
PARAGRAPH_PAUSE = 2.5

# Medical glossary — extend with your professors' specific terms.
INITIAL_PROMPT = (
    "Medical lecture, French and English. Terms: NGS, CRISPR, PCR, mRNA, "
    "Myocarde, Histologie, Pathophysiology, Apoptosis, Cardiomyopathy, "
    "Endocarditis, Tachycardia, Bradycardia, Fibrillation, Angioplasty, "
    "Epithelium, Mitochondria, Ribosomes, Endoplasmic reticulum, Golgi."
)

# ─── SETUP ─────────────────────────────────────────────────────────
for d in [INBOX, TRANSCRIPTS, ARCHIVE]:
    os.makedirs(d, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger()

# ─── FIX 2: STARTUP CLEANUP ────────────────────────────────────────
# Remove any leftover temp file from a previous crash or power cut.
if os.path.exists(TMP_WAV):
    os.remove(TMP_WAV)
    log.warning("Startup: Deleted stale temp_processing.wav from previous crash.")

# ─── VRAM MANAGEMENT ───────────────────────────────────────────────
def clear_vram():
    torch.cuda.empty_cache()
    gc.collect()
    log.info(f"VRAM freed. Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# ─── FIX 3: PARAGRAPH GROUPING ─────────────────────────────────────
def segments_to_paragraphs(segments, pause_threshold=PARAGRAPH_PAUSE):
    """
    Group Whisper segments into readable paragraphs.
    A new paragraph begins when the inter-segment gap exceeds pause_threshold.
    Returns list of (start_time, end_time, paragraph_text) tuples.
    """
    paragraphs = []
    current_text = []
    current_start = None
    prev_end = None

    for seg in segments:
        if current_start is None:
            current_start = seg.start
        if prev_end is not None and (seg.start - prev_end) > pause_threshold:
            if current_text:
                paragraphs.append((current_start, prev_end, ' '.join(current_text)))
            current_text = []
            current_start = seg.start
        current_text.append(seg.text.strip())
        prev_end = seg.end

    if current_text:
        paragraphs.append((current_start, prev_end, ' '.join(current_text)))
    return paragraphs

def format_time(seconds):
    """Convert float seconds to MM:SS string for transcript headers."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"

# ─── MAIN PROCESSOR ────────────────────────────────────────────────
def process_file(file_path, name):
    log.info(f"─── BEGIN: {name} ───────────────────────────")
    base_name = os.path.splitext(name)[0]
    txt_path  = os.path.join(TRANSCRIPTS, base_name + ".txt")
    sum_path  = os.path.join(TRANSCRIPTS, base_name + "_summary.txt")

    # ── STEP 1: AUDIO CLEANING ──────────────────────────────────
    log.info("Step 1/4: Cleaning audio...")
    y, sr = librosa.load(file_path, sr=None, mono=True)
    clean_y = nr.reduce_noise(y=y, sr=sr, prop_decrease=NOISE_REDUCTION)
    sf.write(TMP_WAV, clean_y, sr)
    log.info(f"  Duration: {len(y)/sr:.1f}s | Sample rate: {sr}Hz")

    # ── STEP 2: TRANSCRIPTION ────────────────────────────────────
    log.info("Step 2/4: Loading Whisper Large-v3...")
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    log.info(f"  VRAM after load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    segments, info = model.transcribe(
        TMP_WAV,
        beam_size=5,
        language=None,
        initial_prompt=INITIAL_PROMPT,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=700),
    )
    log.info(f"  Language: {info.language} ({info.language_probability:.0%})")

    # Collect segments into a list (generator exhausts on first pass)
    segment_list = list(segments)
    log.info(f"  Segments collected: {len(segment_list)}")

    # ── FIX 3: WRITE PARAGRAPHS ──────────────────────────────────
    paragraphs = segments_to_paragraphs(segment_list)
    log.info(f"  Grouped into {len(paragraphs)} paragraphs (threshold: {PARAGRAPH_PAUSE}s)")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"TRANSCRIPT: {base_name}\n")
        f.write(f"Language: {info.language} | Segments: {len(segment_list)} | Paragraphs: {len(paragraphs)}\n")
        f.write("=" * 60 + "\n\n")
        for (start, end, text) in paragraphs:
            f.write(f"[{format_time(start)} - {format_time(end)}]\n")
            f.write(f"{text}\n\n")
    log.info(f"  Transcript saved: {txt_path}")

    # ── VRAM HANDOVER ────────────────────────────────────────────
    del model
    clear_vram()
    log.info("  Whisper unloaded. VRAM cleared for Llama 3.")

    # ── STEP 3: SUMMARIZATION ────────────────────────────────────
    log.info("Step 3/4: Running Llama 3 summarization...")
    with open(txt_path, "r", encoding="utf-8") as f:
        transcript = f.read()

    prompt = (
        "Tu es un assistant medical expert. Voici la transcription d'un cours.\n"
        "Genere un resume structure avec:\n"
        "1. Points cles (bullet points)\n"
        "2. Termes medicaux importants et leurs definitions\n"
        "3. Concepts a retenir pour l'examen\n\n"
        f"TRANSCRIPTION:\n{transcript}"
    )
    try:
        res = requests.post(OLLAMA_URL, json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }, timeout=600)
        summary = res.json().get("response", "[No response from model]")
        with open(sum_path, "w", encoding="utf-8") as f:
            f.write(summary)
        log.info(f"  Summary saved: {sum_path}")
    except Exception as e:
        log.error(f"  LLM summarization failed: {e}")

    # ── STEP 4: ARCHIVE ──────────────────────────────────────────
    log.info("Step 4/4: Archiving at 48kbps Opus...")
    out_opus = os.path.join(ARCHIVE, base_name + ".opus")
    result = subprocess.run([
        "ffmpeg", "-i", file_path,
        "-c:a", "libopus", "-b:a", "48k",
        "-application", "voip",
        "-y", out_opus
    ], capture_output=True)
    if result.returncode != 0:
        log.error(f"  FFmpeg error: {result.stderr.decode()}")
    else:
        orig_mb = os.path.getsize(file_path) / 1e6
        opus_mb = os.path.getsize(out_opus)  / 1e6
        log.info(f"  {orig_mb:.1f}MB -> {opus_mb:.1f}MB ({100*(1-opus_mb/orig_mb):.0f}% saved)")

    # ── CLEANUP ──────────────────────────────────────────────────
    os.remove(file_path)
    if os.path.exists(TMP_WAV):
        os.remove(TMP_WAV)
    log.info(f"─── COMPLETE: {base_name} ────────────────────────")

# ─── WATCHDOG LOOP ─────────────────────────────────────────────────
SUPPORTED = ('.m4a', '.mp3', '.wav', '.aac', '.ogg', '.flac')

log.info("=" * 55)
log.info("FMPC SCRIBE ENGINE v1.1 — STARTED")
log.info(f"Watching: {INBOX}")
log.info(f"Paragraph pause threshold: {PARAGRAPH_PAUSE}s")
log.info(f"Noise reduction: {NOISE_REDUCTION}")
log.info(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT FOUND'}")
log.info("=" * 55)

while True:
    try:
        files = [f for f in os.listdir(INBOX) if f.lower().endswith(SUPPORTED)]
        for fname in files:
            fpath = os.path.join(INBOX, fname)
            try:
                process_file(fpath, fname)
            except Exception as e:
                log.error(f"FAILED [{fname}]: {e}", exc_info=True)
    except Exception as e:
        log.error(f"Watchdog error: {e}")
    time.sleep(10)
