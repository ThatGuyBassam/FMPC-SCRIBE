# transcriber.py — FMPC Scribe v3.4
# Engine: Qwen3-ASR-1.7B
# Runs as an isolated subprocess via scribe_engine.py.
# MUST end with os._exit(0) — bypasses Python/CUDA cleanup to prevent
# the 0xC0000005 Access Violation crash on Windows. DO NOT REMOVE.

import os
import sys
import json
import re
import torch
import librosa
import subprocess
import noisereduce as nr
import soundfile as sf
from qwen_asr import Qwen3ASRModel
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ─── CONFIG ────────────────────────────────────────────────────────
TMP_WAV         = r"C:\FMPC_Scribe\temp_processing.wav"
TMP_RESULT      = r"C:\FMPC_Scribe\temp_result.json"
TARGET_SR       = 16000
NOISE_REDUCTION = 0.75
PARAGRAPH_PAUSE = 2.5
MIN_CHUNK_SEC   = 0.5
BATCH_SIZE      = 1

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
chunk_start = 0

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

print(f"[TRANSCRIBER] Found {len(chunk_bounds)} paragraph chunks.")

# ─── REPETITION FILTER ─────────────────────────────────────────────
def filter_repetitions(text: str) -> str:
    if not text:
        return text
    # Single word repetition
    cleaned = re.sub(r'\b(\w[\w\'-]*[.,!?]?)\s+(?:\1\s*){3,}', r'\1 ', text, flags=re.IGNORECASE | re.UNICODE)
    # Multi-word phrase repetition e.g. "Sous-titrage MFP. Sous-titrage MFP..."
    cleaned = re.sub(r'(.{8,80}?)\s*(?:\1\s*){2,}', r'\1 ', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'  +', ' ', cleaned).strip()
    words = cleaned.lower().split()
    if len(words) > 5:
        most_common = max(words.count(w) for w in set(words))
        if most_common / len(words) > 0.70:
            return ""
    return cleaned
# ─── STEP 3: TRANSCRIPTION ─────────────────────────────────────────
print("[TRANSCRIBER] Step 3/3: Loading Qwen3-ASR-1.7B on CUDA...")

model = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-1.7B",
    dtype=torch.bfloat16,
    device_map="cuda:0",
)
print("[TRANSCRIBER] Model loaded. Transcribing...")

paragraphs        = []
language_detected = "French"
total_chunks      = len(chunk_bounds)

for batch_start in range(0, total_chunks, BATCH_SIZE):
    batch_indices = chunk_bounds[batch_start : batch_start + BATCH_SIZE]
    batch_num     = batch_start // BATCH_SIZE + 1
    total_batches = (total_chunks + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"[TRANSCRIBER]   Batch {batch_num}/{total_batches} "
          f"({len(batch_indices)} chunks)...")

    batch_audio = [(audio_clean[s:e], TARGET_SR) for s, e in batch_indices]

    try:
        results = model.transcribe(
            audio=batch_audio,
            language="French",
            context=MEDICAL_PROMPT,
        )
    except Exception as e:
        print(f"[TRANSCRIBER]   WARNING: Batch {batch_num} failed — {e}. Trying without context...")
        try:
            results = model.transcribe(
                audio=batch_audio,
                language="French",
            )
        except Exception as e2:
            print(f"[TRANSCRIBER]   WARNING: Batch {batch_num} failed completely — {e2}. Skipping.")
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

print(f"[TRANSCRIBER] Transcription complete — {len(paragraphs)} paragraphs kept.")

# ─── ASSEMBLE TRANSCRIPT ───────────────────────────────────────────
transcript = "\n\n".join(paragraphs)

if not transcript.strip():
    print("[TRANSCRIBER] WARNING: Transcript is empty — check audio quality.")

# ─── SAVE temp_result.json ─────────────────────────────────────────
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
