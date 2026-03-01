import time, os, sys, logging, ctypes, subprocess
from datetime import datetime

# ─── CONFIG ────────────────────────────────────────────────────────
BASE_DIR    = r"C:\FMPC_Scribe"
INBOX       = os.path.join(BASE_DIR, "INBOX")
TRANSCRIPTS = os.path.join(BASE_DIR, "NOTES_Transcripts")
ARCHIVE_HDD = r"D:\FMPC_Audio_Archive\Medical"
LOG_PATH    = os.path.join(BASE_DIR, "scribe_log.txt")
TMP_WAV     = os.path.join(BASE_DIR, "temp_processing.wav")
TMP_RESULT  = os.path.join(BASE_DIR, "temp_result.json")

PYTHON      = r"C:\Users\GAMER\AppData\Local\Programs\Python\Python311\python.exe"
TRANSCRIBER = os.path.join(BASE_DIR, "transcriber.py")
LABELER     = os.path.join(BASE_DIR, "labeler.py")

SHUTDOWN_AFTER_MINUTES = 30
SUPPORTED = ('.m4a', '.mp3', '.wav', '.aac', '.ogg', '.flac')

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

# Startup cleanup
for tmp in [TMP_WAV, TMP_RESULT]:
    if os.path.exists(tmp):
        os.remove(tmp)
        log.warning(f"Startup: Deleted stale {os.path.basename(tmp)} from previous crash.")

# ─── USER ACTIVITY CHECK ───────────────────────────────────────────
def get_idle_seconds():
    class LASTINPUTINFO(ctypes.Structure):
        _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]
    lii = LASTINPUTINFO()
    lii.cbSize = ctypes.sizeof(lii)
    ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lii))
    millis = ctypes.windll.kernel32.GetTickCount() - lii.dwTime
    return millis / 1000.0

# ─── PROCESS FILE ──────────────────────────────────────────────────
def process_file(file_path, name):
    log.info(f"─── BEGIN: {name} ───────────────────────────")

    # Run transcriber in its own process
    log.info("Launching transcriber process...")
    result = subprocess.run(
        [PYTHON, TRANSCRIBER, file_path],
        capture_output=False  # prints directly to CMD for visibility
    )

    if result.returncode != 0:
        log.error(f"Transcriber failed with exit code {result.returncode}")
        return

    if not os.path.exists(TMP_RESULT):
        log.error("Transcriber finished but temp_result.json not found.")
        return

    log.info("Transcriber complete. Launching labeler process...")

    # Run labeler in its own process — no GPU needed
    result2 = subprocess.run(
        [PYTHON, LABELER],
        capture_output=False
    )

    if result2.returncode != 0:
        log.error(f"Labeler failed with exit code {result2.returncode}")
        return

    log.info(f"─── COMPLETE: {name} ───────────────────────────")

# ─── WATCHDOG LOOP ─────────────────────────────────────────────────

log.info("=" * 55)
log.info("FMPC SCRIBE ENGINE v3.0 — STARTED")
log.info(f"Watching: {INBOX}")
log.info(f"Auto-shutdown after: {SHUTDOWN_AFTER_MINUTES} minutes idle")
log.info(f"Python: {PYTHON}")
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

                if user_idle_seconds >= 600:
                    log.info("INBOX empty and user inactive for 10+ min. Triggering shutdown...")
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

