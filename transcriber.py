```python
# transcriber.py — FMPC Scribe v3.0
# Engine: Qwen3-ASR-1.7B
# Runs as an isolated subprocess via scribe_engine.py.
# MUST end with os._exit(0) — bypasses Python/CUDA cleanup to prevent
# the 0xC0000005 Access Violation crash on Windows. DO NOT REMOVE.
#
# Install before first run:
#   py -3.11 -m pip install -U qwen-asr noisereduce librosa soundfile torch

import os
import sys
import json
import re
import torch
import librosa
import noisereduce as nr
import soundfile as sf
from qwen_asr import Qwen3ASRModel

# ─── CONFIG ────────────────────────────────────────────────────────
TMP_WAV         = r"C:\FMPC_Scribe\temp_processing.wav"
TMP_RESULT      = r"C:\FMPC_Scribe\temp_result.json"
TARGET_SR       = 16000
NOISE_REDUCTION = 0.75
PARAGRAPH_PAUSE = 2.5
MIN_CHUNK_SEC   = 0.5
BATCH_SIZE      = 4

MEDICAL_PROMPT = (
    MEDICAL_PROMPT = (
    "Cours magistral de médecine en français, enregistré au Maroc. "
    "Le professeur a un accent marocain. "
    "Ignorer les passages en darija marocain entre le professeur et les étudiants. "
    "Disciplines et termes attendus: "
    # Anatomie
    "anatomie, aorte, artère, veine, capillaire, vaisseau, "
    "cavité abdominale, cavité thoracique, cavité pelvienne, cavité pleurale, "
    "péritoine, péricarde, plèvre, médiastin, diaphragme, "
    "sternum, clavicule, scapula, humérus, radius, ulna, fémur, tibia, fibula, "
    "vertèbre, vertébral, sacrum, coccyx, bassin, colonne vertébrale, "
    "muscle, tendon, ligament, fascia, aponévrose, "
    "nerf, plexus, ganglion, moelle épinière, encéphale, cerveau, cervelet, "
    "crâne, mandibule, maxillaire, orbite, sinus, "
    "vue antérieure, vue postérieure, vue supérieure, vue inférieure, "
    "médial, latéral, proximal, distal, superficiel, profond, "
    # Histologie
    "histologie, cytologie, tissu, cellule, noyau, cytoplasme, membrane, "
    "épithélium, tissu conjonctif, tissu musculaire, tissu nerveux, "
    "mitochondrie, réticulum endoplasmique, appareil de Golgi, lysosome, "
    "ribosome, centrosome, vacuole, chromatine, nucléole, "
    "collagène, élastine, fibronectine, laminine, matrice extracellulaire, "
    "jonction serrée, jonction communicante, desmosome, hémidesmosome, "
    # Physiologie
    "physiologie, homéostasie, métabolisme, catabolisme, anabolisme, "
    "potentiel d'action, dépolarisation, repolarisation, synapse, neurotransmetteur, "
    "fréquence cardiaque, débit cardiaque, pression artérielle, systole, diastole, "
    "ventilation, diffusion, perfusion, compliance, résistance, "
    "filtration glomérulaire, réabsorption, sécrétion, clairance, "
    "hormone, récepteur, rétrocontrôle, homéostasie, "
    # Biochimie
    "biochimie, protéine, acide aminé, enzyme, substrat, cofacteur, "
    "ADN, ARN, ARNm, ARNt, ARNr, réplication, transcription, traduction, "
    "glucose, glycolyse, cycle de Krebs, phosphorylation oxydative, ATP, "
    "lipide, acide gras, triglycéride, phospholipide, cholestérol, "
    "glucide, saccharide, glycogène, amidon, "
    "pH, tampon, acide, base, osmolarité, "
    # Hématologie
    "hématologie, sang, plasma, sérum, hématocrite, "
    "érythrocyte, leucocyte, thrombocyte, plaquette, "
    "hémoglobine, hématopoïèse, moelle osseuse, "
    "neutrophile, éosinophile, basophile, lymphocyte, monocyte, "
    "coagulation, fibrine, fibrinogène, thrombine, "
    "groupe sanguin, rhésus, transfusion, "
    "anémie, leucémie, thrombopénie, polyglobulie, "
    # Bactériologie
    "bactériologie, bactérie, virus, champignon, parasite, "
    "gram positif, gram négatif, culture, milieu, colonies, "
    "antibiotique, résistance, antibiogramme, CMI, "
    "staphylocoque, streptocoque, pneumocoque, méningocoque, "
    "bacille, coque, spirochète, anaérobie, aérobie, "
    "infection, septicémie, bactériémie, virémie, "
    "vaccin, immunité, anticorps, antigène, "
    # Embryologie
    "embryologie, embryon, fœtus, zygote, blastocyste, "
    "gastrulation, neurulation, organogenèse, "
    "ectoderme, mésoderme, endoderme, "
    "placenta, cordon ombilical, liquide amniotique, "
    "implantation, fécondation, nidation, "
    "somite, notochorde, tube neural, "
    # Pathologie générale
    "inflammation, nécrose, apoptose, fibrose, cicatrice, "
    "tumeur, bénin, malin, métastase, carcinome, sarcome, "
    "œdème, ischémie, infarctus, thrombose, embolie, "
    "hypertrophie, hyperplasie, atrophie, dysplasie, "
    # Termes de cours généraux
    "schéma, coupe, vue, plan, région, compartiment, "
    "professeur, cours, chapitre, définition, classification, "
    "gauche, droite, antérieur, postérieur, supérieur, inférieur."
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
print("[TRANSCRIBER] Step 1/3: Cleaning and normalizing audio...")

audio, _ = librosa.load(AUDIO_FILE, sr=TARGET_SR, mono=True)
duration = len(audio) / TARGET_SR
print(f"[TRANSCRIBER] Duration: {duration:.1f}s | Resampled to: {TARGET_SR}Hz")

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
    pattern = r'\b(\w[\w\'-]*[.,!?]?)\s+(?:\1\s*){3,}'
    cleaned = re.sub(pattern, r'\1 ', text, flags=re.IGNORECASE | re.UNICODE)
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
            prompt=MEDICAL_PROMPT,
            max_new_tokens=1024,
        )
    except TypeError as e:
        if "prompt" in str(e):
            print("[TRANSCRIBER]   prompt failed, retrying with initial_prompt...")
            try:
                results = model.transcribe(
                    audio=batch_audio,
                    language="French",
                    initial_prompt=MEDICAL_PROMPT,
                    max_new_tokens=1024,
                )
            except Exception as e2:
                print(f"[TRANSCRIBER]   WARNING: Batch {batch_num} failed — {e2}. Skipping.")
                torch.cuda.empty_cache()
                continue
        else:
            print(f"[TRANSCRIBER]   WARNING: Batch {batch_num} failed — {e}. Skipping.")
            torch.cuda.empty_cache()
            continue
    except Exception as e:
        print(f"[TRANSCRIBER]   WARNING: Batch {batch_num} failed — {e}. Skipping.")
        torch.cuda.empty_cache()
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
