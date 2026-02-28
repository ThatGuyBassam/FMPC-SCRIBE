import time, os, subprocess, sys, torch, gc, requests, logging, shutil, ctypes
import librosa, soundfile as sf, noisereduce as nr
from faster_whisper import WhisperModel
from datetime import datetime

# ─── CONFIG ────────────────────────────────────────────────────────
BASE_DIR    = r"C:\FMPC_Scribe"
INBOX       = os.path.join(BASE_DIR, "INBOX")
TRANSCRIPTS = os.path.join(BASE_DIR, "NOTES_Transcripts")
ARCHIVE_HDD = r"D:\FMPC_Audio_Archive\Medical"
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

# Minutes of empty INBOX before auto-shutdown check begins.
SHUTDOWN_AFTER_MINUTES = 30

# Medical glossary — extend with your professors' specific terms.
INITIAL_PROMPT = (
    "Medical lecture, French and English. Terms: NGS, CRISPR, PCR, mRNA, "
    "Myocarde, Histologie, Pathophysiology, Apoptosis, Cardiomyopathy, "
    "Endocarditis, Tachycardia, Bradycardia, Fibrillation, Angioplasty, "
    "Epithelium, Mitochondria, Ribosomes, Endoplasmic reticulum, Golgi."
)

# ─── SETUP ─────────────────────────────────────────────────────────
for d in [INBOX, TRANSCRIPTS, ARCHIVE_HDD]:
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

# ─── STARTUP CLEANUP ───────────────────────────────────────────────
# Remove any leftover temp file from a previous crash or power cut.
if os.path.exists(TMP_WAV):
    os.remove(TMP_WAV)
    log.warning("Startup: Deleted stale temp_processing.wav from previous crash.")

# ─── VRAM MANAGEMENT ───────────────────────────────────────────────
def clear_vram():
    torch.cuda.empty_cache()
    gc.collect()
    log.info(f"VRAM freed. Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# ─── USER ACTIVITY CHECK ───────────────────────────────────────────
def get_idle_seconds():
    """
    Returns seconds since last mouse movement or keypress.
    Uses Windows GetLastInputInfo — no external libraries needed.
    """
    class LASTINPUTINFO(ctypes.Structure):
        _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]

    lii = LASTINPUTINFO()
    lii.cbSize = ctypes.sizeof(lii)
    ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lii))
    millis = ctypes.windll.kernel32.GetTickCount() - lii.dwTime
    return millis / 1000.0

# ─── PARAGRAPH GROUPING ────────────────────────────────────────────
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

# ─── OLLAMA LABELING ───────────────────────────────────────────────
def generate_label(transcript_text):
    """
    Ask Ollama to identify the course name and main topic from the transcript.
    Returns a filename-safe label in the format: coursename_maintopic_YYYY-MM-DD
    """
    today = datetime.now().strftime("%Y-%m-%d")
    prompt = (
        "You are a medical lecture labeler. Read the following transcript excerpt "
        "and respond with ONLY a filename label in this exact format:\n"
        "coursename_maintopic_" + today + "\n\n"
        "Rules:\n"
        "- coursename: single word, lowercase, no spaces (e.g. hematology, histology, cardiology)\n"
        "- maintopic: 1-3 words max, lowercase, hyphenated (e.g. erythrocyte-disorders, cardiac-cycle)\n"
        "- Do not add any explanation, just the label\n\n"
        "Transcript excerpt:\n" + transcript_text[:2000]
    )
    try:
        res = requests.post(OLLAMA_URL, json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }, timeout=60)
        label = res.json().get("response", "").strip().lower()
        label = "".join(c for c in label if c.isalnum() or c in "-_")
        if not label:
            label = f"lecture_{today}"
        log.info(f"  Generated label: {label}")
        return label
    except Exception as e:
        log.error(f"  Labeling failed: {e}. Using fallback label.")
        return f"lecture_{today}"

# ─── MAIN PROCESSOR ────────────────────────────────────────────────
def process_file(file_path, name):
    log.info(f"─── BEGIN: {name} ───────────────────────────")

    # ── STEP 1: AUDIO CLEANING ──────────────────────────────────
    log.info("Step 1/3: Cleaning audio...")
    y, sr = librosa.load(file_path, sr=None, mono=True)
    clean_y = nr.reduce_noise(y=y, sr=sr, prop_decrease=NOISE_REDUCTION)
    sf.write(TMP_WAV, clean_y, sr)
    log.info(f"  Duration: {len(y)/sr:.1f}s | Sample rate: {sr}Hz")

    # ── STEP 2: TRANSCRIPTION ────────────────────────────────────
    log.info("Step 2/3: Loading Whisper Large-v3...")
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

    segment_list = list(segments)
    log.info(f"  Segments collected: {len(segment_list)}")

    paragraphs = segments_to_paragraphs(segment_list)
    log.info(f"  Grouped into {len(paragraphs)} paragraphs (threshold: {PARAGRAPH_PAUSE}s)")

    # Build raw transcript text
    transcript_lines = []
    for (start, end, text) in paragraphs:
        transcript_lines.append(f"[{format_time(start)} - {format_time(end)}]")
        transcript_lines.append(f"{text}\n")
    transcript_text = "\n".join(transcript_lines)

    del model
    clear_vram()
    log.info("  Whisper unloaded. VRAM cleared.")

    # ── STEP 3: OLLAMA LABELING ──────────────────────────────────
    log.info("Step 3/3: Generating label with Ollama...")
    label = generate_label(transcript_text)

    # ── SAVE TRANSCRIPT ──────────────────────────────────────────
    txt_filename = label + ".txt"
    txt_ssd_path = os.path.join(TRANSCRIPTS, txt_filename)
    txt_hdd_path = os.path.join(ARCHIVE_HDD, txt_filename)

    header = (
        f"TRANSCRIPT: {label}\n"
        f"Language: {info.language} | Segments: {len(segment_list)} | Paragraphs: {len(paragraphs)}\n"
        + "=" * 60 + "\n\n"
    )

    with open(txt_ssd_path, "w", encoding="utf-8") as f:
        f.write(header + transcript_text)
    log.info(f"  Transcript saved to SSD: {txt_ssd_path}")

    shutil.copy2(txt_ssd_path, txt_hdd_path)
    log.info(f"  Transcript copied to HDD: {txt_hdd_path}")

    # ── ARCHIVE AUDIO ────────────────────────────────────────────
    out_opus = os.path.join(ARCHIVE_HDD, label + ".opus")
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
        opus_mb = os.path.getsize(out_opus) / 1e6
        log.info(f"  Audio archived: {orig_mb:.1f}MB -> {opus_mb:.1f}MB ({100*(1-opus_mb/orig_mb):.0f}% saved)")

    # ── CLEANUP ──────────────────────────────────────────────────
    os.remove(file_path)
    if os.path.exists(TMP_WAV):
        os.remove(TMP_WAV)
    log.info(f"─── COMPLETE: {label} ────────────────────────")

# ─── WATCHDOG LOOP ─────────────────────────────────────────────────
SUPPORTED = ('.m4a', '.mp3', '.wav', '.aac', '.ogg', '.flac')

log.info("=" * 55)
log.info("FMPC SCRIBE ENGINE v2.0 — STARTED")
log.info(f"Watching: {INBOX}")
log.info(f"Paragraph pause threshold: {PARAGRAPH_PAUSE}s")
log.info(f"Noise reduction: {NOISE_REDUCTION}")
log.info(f"Auto-shutdown after: {SHUTDOWN_AFTER_MINUTES} minutes idle")
log.info(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT FOUND'}")
log.info("=" * 55)

idle_since = None

while True:
    try:
        files = [f for f in os.listdir(INBOX) if f.lower().endswith(SUPPORTED)]

        if files:
            idle_since = None
            for fname in files:
                fpath = os.path.join(INBOX, fname)
                try:
                    process_file(fpath, fname)
                except Exception as e:
                    log.error(f"FAILED [{fname}]: {e}", exc_info=True)
        else:
            if idle_since is None:
                idle_since = time.time()

            idle_minutes = (time.time() - idle_since) / 60

            if idle_minutes >= SHUTDOWN_AFTER_MINUTES:
                user_idle_seconds = get_idle_seconds()

                if user_idle_seconds >= 600:  # 10 minutes inactive
                    log.info("INBOX empty and user inactive for 10+ min. Triggering shutdown...")

                    # Visible popup on all Windows 11 installs
                    os.system(
                        'powershell -command "Add-Type -AssemblyName System.Windows.Forms; '
                        '[System.Windows.Forms.MessageBox]::Show('
                        "'FMPC Scribe: INBOX empty and you have been idle for 10+ minutes. "
                        "PC is shutting down in 60 seconds. To cancel: open CMD and type shutdown /a'"
                        ')"'
                    )

                    log.info("Shutdown popup shown. To cancel: open CMD and type shutdown /a")
                    os.system("shutdown /s /t 60")
                    break

                else:
                    log.info(
                        f"INBOX empty but user active "
                        f"({user_idle_seconds:.0f}s since last input). "
                        f"Checking again in 1 hour."
                    )
                    idle_since = time.time() - ((SHUTDOWN_AFTER_MINUTES - 60) * 60)

    except Exception as e:
        log.error(f"Watchdog error: {e}")

    time.sleep(10)
