# transcriber.py — FMPC Scribe v3.5
# Engine: Qwen3-ASR-1.7B
# Runs as an isolated subprocess via scribe_engine.py.
# MUST end with os._exit(0) — bypasses Python/CUDA cleanup to prevent
# the 0xC0000005 Access Violation crash on Windows. DO NOT REMOVE.

import os
import sys
import json
import re
import gc
import time
import torch
import librosa
import subprocess
import noisereduce as nr
import soundfile as sf
from qwen_asr import Qwen3ASRModel
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TRANSFORMERS_OFFLINE"] = "1"  # Skip HuggingFace network check

# ─── CONFIG ────────────────────────────────────────────────────────
TMP_WAV            = r"C:\FMPC_Scribe\temp_processing.wav"
TMP_RESULT         = r"C:\FMPC_Scribe\temp_result.tmp"
TARGET_SR          = 16000
NOISE_REDUCTION    = 0.75
PARAGRAPH_PAUSE    = 2.5
MIN_CHUNK_SEC      = 0.5
MAX_CHUNK_SEC      = 60.0  # Split any chunk longer than this into equal pieces
BATCH_SIZE         = 2     # Try 4 if VRAM holds — 4x speedup vs original
MODEL_RELOAD_EVERY = 15    # Fewer reloads now that batch size is higher

MEDICAL_PROMPT = (
    "French. Cours de médecine, FMPC Maroc. Accent marocain, ignorez la darija. "
    "ATTENTION: Cette liste d'exemples n'est pas exhaustive. Elle sert uniquement "
    "à calibrer votre phonétique. Utilisez vos connaissances médicales pour "
    "transcrire tout autre terme non listé. Exemples: "
    "Anatomie: aorte, artère, péritoine, péricarde, plèvre, médiastin, diaphragme, aponévrose, plexus, moelle épinière, encéphale. "
    "Histologie: épithélium, tissu conjonctif, mitochondrie, réticulum endoplasmique, appareil de Golgi, lysosome, chromatine, desmosome. "
    "Physiologie: homéostasie, métabolisme, dépolarisation, repolarisation, synapse, neurotransmetteur, systole, diastole, clairance. "
    "Biochimie: protéine, enzyme, ADN, ARN, ARNm, ATP, glycolyse, phosphorylation oxydative, lipide. "
    "Hématologie: érythrocyte, leucocyte, thrombocyte, hématopoïèse, moelle osseuse, hémoglobine, coagulation, fibrine. "
    "Bactériologie: staphylocoque, streptocoque, pneumocoque, bacille, spirochète, anaérobie, antigène, anticorps. "
    "Embryologie: fœtus, zygote, blastocyste, gastrulation, organogenèse, notochorde. "
    "Pathologie: œdème, ischémie, infarctus, thrombose, embolie, nécrose, apoptose, carcinome, métastase."
)

# ─── ARGUMENT CHECK ────────────────────────────────────────────────
if len(sys.argv) < 2:
    print("[TRANSCRIBER] ERROR: No audio file provided.")
    os._exit(1)

AUDIO_FILE = sys.argv[1]

if not os.path.exists(AUDIO_FILE):
    print(f"[TRANSCRIBER] ERROR: File not found: {AUDIO_FILE}")
    os._exit(1)

print(f"[TRANSCRIBER] Starting — {AUDIO_FILE}")

# ─── STEP 1: AUDIO CLEANING ────────────────────────────────────────
print("[TRANSCRIBER] Step 1/3: Converting and cleaning audio...")

if os.path.exists(TMP_WAV):
    os.remove(TMP_WAV)

subprocess.run([
    "ffmpeg", "-y", "-i", AUDIO_FILE,
    "-ar", str(TARGET_SR), "-ac", "1", TMP_WAV
], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

audio, _ = librosa.load(TMP_WAV, sr=TARGET_SR, mono=True)
duration = len(audio) / TARGET_SR
print(f"[TRANSCRIBER] Duration: {duration:.1f}s | Sample rate: {TARGET_SR}Hz")

audio_clean = nr.reduce_noise(y=audio, sr=TARGET_SR, prop_decrease=NOISE_REDUCTION)
audio_clean = librosa.util.normalize(audio_clean)
sf.write(TMP_WAV, audio_clean, TARGET_SR)
print("[TRANSCRIBER] Clean WAV saved.")

# ─── STEP 2: PARAGRAPH BOUNDARY DETECTION ──────────────────────────
print("[TRANSCRIBER] Step 2/3: Detecting paragraph boundaries...")

intervals = librosa.effects.split(
    audio_clean,
    top_db=40,
    frame_length=2048,
    hop_length=512,
)

chunk_bounds = []
chunk_start  = 0

for i in range(1, len(intervals)):
    gap_start = intervals[i - 1][1]
    gap_end   = intervals[i][0]
    gap_sec   = (gap_end - gap_start) / TARGET_SR

    if gap_sec >= PARAGRAPH_PAUSE:
        chunk_end = gap_start
        chunk_dur = (chunk_end - chunk_start) / TARGET_SR
        if chunk_dur >= MIN_CHUNK_SEC:
            chunk_bounds.append((chunk_start, chunk_end))
        chunk_start = gap_end

final_dur = (len(audio_clean) - chunk_start) / TARGET_SR
if final_dur >= MIN_CHUNK_SEC:
    chunk_bounds.append((chunk_start, len(audio_clean)))

# Split any chunk that exceeds MAX_CHUNK_SEC into equal sub-chunks
split_bounds = []
for start, end in chunk_bounds:
    dur = (end - start) / TARGET_SR
    if dur > MAX_CHUNK_SEC:
        n_parts = int(dur / MAX_CHUNK_SEC) + 1
        part_len = (end - start) // n_parts
        for p in range(n_parts):
            p_start = start + p * part_len
            p_end   = start + (p + 1) * part_len if p < n_parts - 1 else end
            split_bounds.append((p_start, p_end))
    else:
        split_bounds.append((start, end))

chunk_bounds = split_bounds
print(f"[TRANSCRIBER] Found {len(chunk_bounds)} paragraph chunks (after splitting long segments).")

# ─── REPETITION FILTER ─────────────────────────────────────────────
def filter_repetitions(text: str) -> str:
    if not text:
        return text
    cleaned = re.sub(r'\b(\w[\w\'-]*[.,!?]?)\s+(?:\1\s*){3,}', r'\1 ', text, flags=re.IGNORECASE | re.UNICODE)
    cleaned = re.sub(r'(.{8,80}?)\s*(?:\1\s*){2,}', r'\1 ', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'  +', ' ', cleaned).strip()
    if any(phrase in cleaned for phrase in ["calibrer", "phonétique", "FMPC Maroc", "darija", "exhaustive"]):
        return ""
    words = cleaned.lower().split()
    if len(words) > 5:
        most_common = max(words.count(w) for w in set(words))
        if most_common / len(words) > 0.70:
            return ""
    return cleaned

# ─── VRAM CLEANUP ──────────────────────────────────────────────────
def purge_vram(model):
    """Aggressively free all VRAM before reloading the model."""
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(1)  # Give CUDA a moment to fully release
    gc.collect()
    torch.cuda.empty_cache()

# ─── MODEL LOADER ──────────────────────────────────────────────────
def load_model():
    print("[TRANSCRIBER] Loading Qwen3-ASR-1.7B on CUDA...")
    m = Qwen3ASRModel.from_pretrained(
        "Qwen/Qwen3-ASR-1.7B",
        dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    print("[TRANSCRIBER] Model loaded.")
    return m

# ─── STEP 3: TRANSCRIPTION ─────────────────────────────────────────
print("[TRANSCRIBER] Step 3/3: Transcribing...")

paragraphs        = []
language_detected = "French"
total_chunks      = len(chunk_bounds)
model             = None

for batch_start in range(0, total_chunks, BATCH_SIZE):
    batch_num     = batch_start // BATCH_SIZE + 1
    total_batches = (total_chunks + BATCH_SIZE - 1) // BATCH_SIZE

    # Reload model every MODEL_RELOAD_EVERY batches
    if model is None or batch_num % MODEL_RELOAD_EVERY == 1:
        purge_vram(model)
        model = load_model()

    batch_indices = chunk_bounds[batch_start : batch_start + BATCH_SIZE]
    print(f"[TRANSCRIBER]   Batch {batch_num}/{total_batches} ({len(batch_indices)} chunks)...")

    batch_audio = [(audio_clean[s:e], TARGET_SR) for s, e in batch_indices]

    # ── Try with medical context ──
    try:
        results = model.transcribe(
            audio=batch_audio,
            language="French",
            context=MEDICAL_PROMPT,
        )
    except torch.cuda.OutOfMemoryError as e:
        print(f"[TRANSCRIBER]   OOM on batch {batch_num} — purging VRAM and retrying...")
        purge_vram(model)
        model = load_model()
        # If batch had multiple chunks, retry them one at a time
        if len(batch_audio) > 1:
            print(f"[TRANSCRIBER]   Retrying {len(batch_audio)} chunks individually...")
            results = []
            for single_audio in batch_audio:
                try:
                    r = model.transcribe(audio=[(single_audio[0], single_audio[1])], language="French", context=MEDICAL_PROMPT)
                    results.extend(r)
                except torch.cuda.OutOfMemoryError:
                    try:
                        r = model.transcribe(audio=[(single_audio[0], single_audio[1])], language="French")
                        results.extend(r)
                    except Exception:
                        torch.cuda.empty_cache()
                        continue
                torch.cuda.empty_cache()
        else:
            try:
                results = model.transcribe(audio=batch_audio, language="French", context=MEDICAL_PROMPT)
            except torch.cuda.OutOfMemoryError:
                print(f"[TRANSCRIBER]   OOM again — trying without context...")
                try:
                    results = model.transcribe(audio=batch_audio, language="French")
                except Exception as e3:
                    print(f"[TRANSCRIBER]   Batch {batch_num} failed completely — skipping.")
                    torch.cuda.empty_cache()
                    continue
    except Exception as e:
        print(f"[TRANSCRIBER]   Batch {batch_num} failed — {e}. Trying without context...")
        try:
            results = model.transcribe(
                audio=batch_audio,
                language="French",
            )
        except Exception as e2:
            print(f"[TRANSCRIBER]   Batch {batch_num} failed completely — skipping.")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            continue

    for result in results:
        raw_text = result.text.strip() if result.text else ""
        if not raw_text:
            continue
        clean_text = filter_repetitions(raw_text)
        if clean_text:
            paragraphs.append(clean_text)
        if hasattr(result, "language") and result.language:
            language_detected = result.language

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

print(f"[TRANSCRIBER] Transcription complete — {len(paragraphs)} paragraphs kept.")

# ─── ASSEMBLE TRANSCRIPT ───────────────────────────────────────────
transcript = "\n\n".join(paragraphs)

if not transcript.strip():
    print("[TRANSCRIBER] WARNING: Transcript is empty — check audio quality.")

# ─── SAVE temp_result.tmp ─────────────────────────────────────────
result_payload = {
    "transcript":           transcript,
    "language":             language_detected,
    "language_probability": 1.0,
    "segment_count":        len(paragraphs),
    "paragraph_count":      len(paragraphs),
    "file_path":            AUDIO_FILE,
}

with open(TMP_RESULT, "w", encoding="utf-8") as f:
    json.dump(result_payload, f, ensure_ascii=False, indent=2)

print(f"[TRANSCRIBER] Result saved → {TMP_RESULT}")
print(f"[TRANSCRIBER] Done. {len(paragraphs)} paragraphs / {len(transcript)} chars.")

# ─── HARD EXIT ─────────────────────────────────────────────────────
# DO NOT replace with sys.exit() — os._exit() bypasses Python/CUDA
# cleanup and prevents the 0xC0000005 Access Violation on Windows.
os._exit(0)

