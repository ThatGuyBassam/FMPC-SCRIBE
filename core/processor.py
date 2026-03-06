import sys, json, os, shutil, subprocess, requests
from datetime import datetime, timedelta

# ─── CONFIG ────────────────────────────────────────────────────────
TRANSCRIPTS   = r"C:\FMPC_Scribe\NOTES_Transcripts"
ARCHIVE_HDD   = r"D:\FMPC_Audio_Archive\Medical"
TMP_RESULT    = r"C:\FMPC_Scribe\temp_result.tmp"
TMP_WAV       = r"C:\FMPC_Scribe\temp_processing.wav"
SCHEDULE_FILE = r"C:\FMPC_Scribe\schedule.json"
OLLAMA_URL    = "http://localhost:11434/api/generate"
OLLAMA_MODEL  = "qwen2.5:7b"
MAX_CHUNK_CHARS = 4000

# ─── SCHEDULE LOOKUP ───────────────────────────────────────────────
def load_schedule():
    with open(SCHEDULE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def get_candidates_from_schedule(file_path):
    """
    Find all possible courses scheduled at the recording time,
    across both sections, deduplicated by discipline.
    """
    try:
        sched = load_schedule()
        all_slots = sched["schedule"]

        # Try to read recording date from audio metadata via FFmpeg
        # Falls back to file creation time if metadata is unavailable
        file_dt = None
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", file_path],
                capture_output=True, timeout=10
            )
            info = json.loads(result.stdout.decode("utf-8", errors="ignore"))
            tags = info.get("format", {}).get("tags", {})
            # Phone recordings typically store date in these tags
            for key in ["creation_time", "date", "DATE", "CREATION_TIME"]:
                if key in tags:
                    raw = tags[key]
                    # Handle formats: "2026-03-02T08:30:00.000000Z" or "2026-03-02"
                    raw = raw[:19].replace("T", " ")
                    file_dt = datetime.strptime(raw[:16], "%Y-%m-%d %H:%M")
                    print(f"[PROCESSOR] Recording date from metadata: {file_dt.strftime('%Y-%m-%d %H:%M')}")
                    break
        except Exception as e:
            print(f"[PROCESSOR] Metadata read failed ({e}), using file creation time.")

        if file_dt is None:
            file_ctime = os.path.getctime(file_path)
            file_dt = datetime.fromtimestamp(file_ctime)
            print(f"[PROCESSOR] Audio recorded (file ctime): {file_dt.strftime('%Y-%m-%d %H:%M')}")

        file_date_str = file_dt.strftime("%Y-%m-%d")
        print(f"[PROCESSOR] Using date: {file_date_str}")

        day_slots = [s for s in all_slots if s["date"] == file_date_str]
        if not day_slots:
            print(f"[PROCESSOR] No schedule entries for {file_date_str}.")
            return []

        matching = []
        seen_disciplines = set()

        for slot in day_slots:
            slot_start   = datetime.strptime(f"{slot['date']} {slot['start']}", "%Y-%m-%d %H:%M")
            slot_end     = datetime.strptime(f"{slot['date']} {slot['end']}",   "%Y-%m-%d %H:%M")
            window_start = slot_start - timedelta(minutes=30)
            window_end   = slot_end   + timedelta(minutes=30)

            if window_start <= file_dt <= window_end:
                if slot["discipline"] not in seen_disciplines:
                    seen_disciplines.add(slot["discipline"])
                    matching.append(slot)

        if matching:
            print(f"[PROCESSOR] Candidate courses: {[s['discipline'] for s in matching]}")
        else:
            print(f"[PROCESSOR] No slot matches recording time {file_dt.strftime('%H:%M')}.")

        return matching

    except Exception as e:
        print(f"[PROCESSOR] Schedule lookup failed: {e}")
        return []

# ─── DISCIPLINE DETECTION ──────────────────────────────────────────
def detect_discipline(transcript_text, candidates):
    """
    1 candidate  → return directly, no LLM needed
    2+ candidates → Qwen2.5 reads transcript and picks
    0 candidates  → Qwen2.5 identifies from scratch
    """
    if len(candidates) == 1:
        slot = candidates[0]
        print(f"[PROCESSOR] Single candidate: {slot['discipline']}")
        return slot["discipline"], slot["professor"]

    if len(candidates) > 1:
        discipline_list = ", ".join([c["discipline"] for c in candidates])
        print(f"[PROCESSOR] Multiple candidates — asking Qwen2.5 to classify: {discipline_list}")
        prompt = (
            "Tu es un classificateur de cours de médecine. "
            "Lis l'extrait de cours suivant et détermine à quelle matière il appartient "
            "parmi les choix suivants: " + discipline_list + "\n\n"
            "RÈGLE STRICTE: Réponds UNIQUEMENT avec le nom exact d'une des matières listées. "
            "Un seul mot. Aucune explication.\n\n"
            "Extrait:\n" + transcript_text[:3000]
        )
        try:
            res = requests.post(OLLAMA_URL, json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            }, timeout=60)
            answer = res.json().get("response", "").strip().lower()
            answer = answer.split("\n")[0].strip()
            answer = "".join(c for c in answer if c.isalnum())

            for c in candidates:
                if c["discipline"].lower() in answer or answer in c["discipline"].lower():
                    print(f"[PROCESSOR] Qwen2.5 classified as: {c['discipline']}")
                    return c["discipline"], c["professor"]

            print(f"[PROCESSOR] Could not parse answer '{answer}', defaulting to first candidate.")
            return candidates[0]["discipline"], candidates[0]["professor"]

        except Exception as e:
            print(f"[PROCESSOR] Classification failed: {e}. Using first candidate.")
            return candidates[0]["discipline"], candidates[0]["professor"]

    # No candidates — full AI fallback
    print("[PROCESSOR] No schedule match. Qwen2.5 identifying discipline from scratch...")
    prompt = (
        "Tu es un classificateur de cours de médecine FMPC. "
        "Lis l'extrait suivant et réponds UNIQUEMENT avec la matière correspondante "
        "en un seul mot minuscule parmi: anatomie, bacteriologie, histologie, hematologie, "
        "embryologie, immunologie, physiologie, biochimie.\n\n"
        "Extrait:\n" + transcript_text[:3000]
    )
    try:
        res = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        }, timeout=60)
        discipline = res.json().get("response", "").strip().lower()
        discipline = discipline.split("\n")[0]
        discipline = "".join(c for c in discipline if c.isalnum())
        if not discipline:
            discipline = "medecine"
        print(f"[PROCESSOR] AI identified discipline: {discipline}")
        return discipline, None
    except Exception as e:
        print(f"[PROCESSOR] Full fallback failed: {e}")
        return "medecine", None

# ─── TOPIC LABEL ───────────────────────────────────────────────────
def generate_topic(transcript_text, discipline):
    prompt = (
        "Tu es un assistant qui nomme des fichiers de cours de médecine. "
        "La matière est: " + discipline + "\n"
        "Lis l'extrait suivant et réponds UNIQUEMENT avec 1 à 3 mots décrivant "
        "le sujet principal, en minuscules, séparés par des tirets "
        "(ex: cavite-abdominale, coloration-gram, structure-bacterienne).\n"
        "Aucune explication. Juste les mots.\n\n"
        "Extrait:\n" + transcript_text[:2000]
    )
    try:
        res = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        }, timeout=60)
        topic = res.json().get("response", "").strip().lower()
        topic = topic.split("\n")[0]
        topic = "".join(c for c in topic if c.isalnum() or c == "-")
        return topic if topic else "cours"
    except Exception as e:
        print(f"[PROCESSOR] Topic generation failed: {e}")
        return "cours"

def build_label(file_path, transcript_text):
    today = datetime.now().strftime("%Y-%m-%d")
    candidates = get_candidates_from_schedule(file_path)
    discipline, professor = detect_discipline(transcript_text, candidates)
    topic = generate_topic(transcript_text, discipline)
    label = f"{discipline}_{topic}_{today}"
    print(f"[PROCESSOR] Final label: {label}")
    return label, discipline, professor

# ─── SHARED CHUNKER ───────────────────────────────────────────────
def _chunk_transcript(text):
    paragraphs = text.split("\n\n")
    chunks, current, current_len = [], [], 0
    for p in paragraphs:
        if not p.strip():
            continue
        if current_len + len(p) > MAX_CHUNK_CHARS and current:
            chunks.append("\n\n".join(current))
            current, current_len = [p], len(p)
        else:
            current.append(p)
            current_len += len(p)
    if current:
        chunks.append("\n\n".join(current))
    return chunks

# ─── PASS 1: PHONETIC CORRECTION ───────────────────────────────────
def clean_transcript_chunked(transcript_text):
    chunks = _chunk_transcript(transcript_text)
    total  = len(chunks)
    print(f"[PROCESSOR] Correcting transcript across {total} chunks...")
    result = []
    for i, chunk in enumerate(chunks, 1):
        print(f"[PROCESSOR]   Correcting chunk {i}/{total}...")
        prompt = (
            "Tu es un correcteur orthographique médical expert pour la FMPC. "
            "Corrige UNIQUEMENT les erreurs phonétiques et fautes d'orthographe "
            "des termes médicaux du texte suivant. "
            "RÈGLES STRICTES : Ne résume pas. Ne modifie pas le style du professeur. "
            "N'ajoute AUCUNE phrase d'introduction. "
            "Renvoie EXCLUSIVEMENT le texte corrigé.\n\n"
            "Texte à corriger :\n" + chunk
        )
        try:
            res = requests.post(OLLAMA_URL, json={
                "model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
                "options": {"num_predict": 2048}
            }, timeout=150)
            cleaned = res.json().get("response", "").strip()
            if cleaned.lower().startswith("voici"):
                cleaned = cleaned.split(":", 1)[-1].strip()
            result.append(cleaned if cleaned else chunk)
        except Exception as e:
            print(f"[PROCESSOR]   WARNING: Chunk {i} failed ({e}). Keeping original.")
            result.append(chunk)
    return "\n\n".join(result)

# ─── PASS 2: MEDICAL CONTENT FILTER ────────────────────────────────
def filter_transcript_chunked(transcript_text):
    chunks = _chunk_transcript(transcript_text)
    total  = len(chunks)
    print(f"[PROCESSOR] Filtering medical content across {total} chunks...")
    result = []
    for i, chunk in enumerate(chunks, 1):
        print(f"[PROCESSOR]   Filtering chunk {i}/{total}...")
        prompt = (
            "Tu es un filtre de contenu médical pour la FMPC. "
            "Supprime UNIQUEMENT les passages non médicaux : bavardages, organisation du semestre, "
            "blagues, échanges avec les étudiants, commentaires personnels du professeur. "
            "Conserve TOUT le contenu médical : définitions, anatomie, physiologie, "
            "histologie, pathologie, biochimie, embryologie, bactériologie, immunologie, hématologie. "
            "Ne résume pas. Ne reformule pas. Ne rajoute rien. "
            "N'ajoute AUCUNE phrase d'introduction. "
            "Si un paragraphe entier est non médical, supprime-le complètement. "
            "Renvoie EXCLUSIVEMENT le texte filtré.\n\n"
            "Texte à filtrer :\n" + chunk
        )
        try:
            res = requests.post(OLLAMA_URL, json={
                "model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
                "options": {"num_predict": 2048}
            }, timeout=150)
            filtered = res.json().get("response", "").strip()
            if filtered.lower().startswith("voici"):
                filtered = filtered.split(":", 1)[-1].strip()
            if filtered:
                result.append(filtered)
        except Exception as e:
            print(f"[PROCESSOR]   WARNING: Chunk {i} failed ({e}). Keeping original.")
            result.append(chunk)
    return "\n\n".join(result)

# ─── MAIN ──────────────────────────────────────────────────────────
def main():
    print("[PROCESSOR] Starting...")

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("[PROCESSOR] VRAM cleared.")
    except Exception:
        pass

    with open(TMP_RESULT, "r", encoding="utf-8") as f:
        result = json.load(f)

    raw_transcript  = result["transcript"]
    language        = result["language"]
    segment_count   = result["segment_count"]
    paragraph_count = result["paragraph_count"]
    file_path       = result["file_path"]

    print(f"[PROCESSOR] Loaded raw transcript — {segment_count} segments, {paragraph_count} paragraphs")

    # STEP 1: PHONETIC CORRECTION
    print("[PROCESSOR] Step 1/5: Correcting phonetic medical errors...")
    cleaned_transcript = clean_transcript_chunked(raw_transcript)

    # STEP 2: MEDICAL CONTENT FILTER
    print("[PROCESSOR] Step 2/5: Filtering to medical content only...")
    filtered_transcript = filter_transcript_chunked(cleaned_transcript)

    # STEP 3: BUILD LABEL
    print("[PROCESSOR] Step 3/5: Building label...")
    label, discipline, professor = build_label(file_path, filtered_transcript)

    # STEP 4: SAVE BOTH VERSIONS
    print("[PROCESSOR] Step 4/5: Saving transcripts...")

    header_full = (
        f"TRANSCRIPT (FULL): {label}\n"
        f"Language: {language} | Segments: {segment_count} | Paragraphs: {paragraph_count}\n"
        f"Discipline: {discipline or 'unknown'} | Professor: {professor or 'unknown'}\n"
        f"AI Corrector: {OLLAMA_MODEL}\n"
        + "=" * 60 + "\n\n"
    )
    header_filtered = (
        f"TRANSCRIPT (MEDICAL ONLY): {label}\n"
        f"Language: {language} | Segments: {segment_count} | Paragraphs: {paragraph_count}\n"
        f"Discipline: {discipline or 'unknown'} | Professor: {professor or 'unknown'}\n"
        f"AI Corrector: {OLLAMA_MODEL} | Filter: medical-only\n"
        + "=" * 60 + "\n\n"
    )

    full_ssd  = os.path.join(TRANSCRIPTS, label + ".txt")
    full_hdd  = os.path.join(ARCHIVE_HDD, label + ".txt")
    with open(full_ssd, "w", encoding="utf-8") as f:
        f.write(header_full + cleaned_transcript)
    shutil.copy2(full_ssd, full_hdd)
    print(f"[PROCESSOR] Full transcript → SSD + HDD")

    clean_ssd = os.path.join(TRANSCRIPTS, label + "_clean.txt")
    clean_hdd = os.path.join(ARCHIVE_HDD, label + "_clean.txt")
    with open(clean_ssd, "w", encoding="utf-8") as f:
        f.write(header_filtered + filtered_transcript)
    shutil.copy2(clean_ssd, clean_hdd)
    print(f"[PROCESSOR] Filtered transcript → SSD + HDD")

    # STEP 5: AUTO-INGEST TRANSCRIPT INTO CHROMADB
    print("[PROCESSOR] Step 5a/5: Ingesting transcript into ChromaDB...")
    try:
        from ingest_slides import ingest_transcript
        ingest_transcript(full_ssd, discipline or "unknown", label, professor)
    except Exception as e:
        print(f"[PROCESSOR] WARNING: ChromaDB ingest failed ({e}). Transcript saved but not indexed.")
        print("[PROCESSOR] Run ingest_slides.py manually to index later.")

    # STEP 6: ARCHIVE AUDIO AS OPUS
    print("[PROCESSOR] Step 5b/5: Archiving audio...")
    out_opus = os.path.join(ARCHIVE_HDD, label + ".opus")
    result_ffmpeg = subprocess.run([
        "ffmpeg", "-i", file_path,
        "-c:a", "libopus", "-b:a", "48k",
        "-application", "voip",
        "-y", out_opus
    ], capture_output=True)

    if result_ffmpeg.returncode != 0:
        print(f"[PROCESSOR] FFmpeg error: {result_ffmpeg.stderr.decode()}")
    else:
        orig_mb = os.path.getsize(file_path) / 1e6
        opus_mb = os.path.getsize(out_opus) / 1e6
        print(f"[PROCESSOR] Audio archived: {orig_mb:.1f}MB → {opus_mb:.1f}MB ({100*(1-opus_mb/orig_mb):.0f}% saved)")

    # CLEANUP
    os.remove(file_path)
    if os.path.exists(TMP_WAV):
        os.remove(TMP_WAV)
    if os.path.exists(TMP_RESULT):
        os.remove(TMP_RESULT)
    print(f"[PROCESSOR] ─── COMPLETE: {label} ───")

if __name__ == "__main__":
    main()
