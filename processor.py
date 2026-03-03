import sys, json, os, shutil, subprocess, requests
from datetime import datetime

# ─── CONFIG ────────────────────────────────────────────────────────
TRANSCRIPTS = r"C:\FMPC_Scribe\NOTES_Transcripts"
ARCHIVE_HDD = r"D:\FMPC_Audio_Archive\Medical"
TMP_RESULT  = r"C:\FMPC_Scribe\temp_result.json"
TMP_WAV     = r"C:\FMPC_Scribe\temp_processing.wav"
OLLAMA_URL  = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b"
MAX_CHUNK_CHARS = 2500

def generate_label(transcript_text):
    today = datetime.now().strftime("%Y-%m-%d")
    prompt = (
        "Tu es un assistant chargé de nommer des fichiers de cours de médecine. Lis l'extrait suivant "
        "et réponds UNIQUEMENT avec un nom de fichier respectant ce format exact :\n"
        "coursename_maintopic_" + today + "\n\n"
        "Règles :\n"
        "- coursename: un seul mot, minuscules, sans espaces (ex: anatomie, histologie, physiologie)\n"
        "- maintopic: 1-3 mots max, minuscules, séparés par des tirets (ex: cavite-abdominale, cycle-cardiaque)\n"
        "- N'ajoute aucune explication, aucune introduction ni aucune ponctuation. Juste le nom du fichier.\n\n"
        "Extrait :\n" + transcript_text[:2000]
    )
    try:
        res = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        }, timeout=60)
        label = res.json().get("response", "").strip().lower()
        label = label.split("\n")[0].split("_202")[0].split("_201")[0].split("_200")[0]
        label = "".join(c for c in label if c.isalnum() or c in "-_")
        if not label:
            label = "lecture"
        return f"{label}_{today}"
    except Exception as e:
        print(f"[PROCESSOR] Labeling failed: {e}. Using fallback.")
        return f"lecture_{today}"

def clean_transcript_chunked(transcript_text):
    paragraphs = transcript_text.split('\n\n')
    chunks_to_process = []
    current_chunk = []
    current_len = 0

    for p in paragraphs:
        if not p.strip():
            continue
        if current_len + len(p) > MAX_CHUNK_CHARS and current_chunk:
            chunks_to_process.append("\n\n".join(current_chunk))
            current_chunk = [p]
            current_len = len(p)
        else:
            current_chunk.append(p)
            current_len += len(p)

    if current_chunk:
        chunks_to_process.append("\n\n".join(current_chunk))

    total_chunks = len(chunks_to_process)
    print(f"[PROCESSOR] Sliced transcript into {total_chunks} chunks.")

    cleaned_paragraphs = []
    for i, chunk in enumerate(chunks_to_process, 1):
        print(f"[PROCESSOR]   Correcting chunk {i}/{total_chunks}...")
        prompt = (
            "Tu es un correcteur orthographique médical expert pour la FMPC. "
            "Corrige UNIQUEMENT les erreurs phonétiques et fautes d'orthographe "
            "des termes médicaux du texte suivant. "
            "RÈGLES STRICTES : Ne résume pas. Ne modifie pas le style du professeur. "
            "N'ajoute AUCUNE phrase d'introduction comme 'Voici le texte corrigé'. "
            "Renvoie EXCLUSIVEMENT le texte corrigé.\n\n"
            "Texte à corriger :\n" + chunk
        )
        try:
            res = requests.post(OLLAMA_URL, json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            }, timeout=120)
            cleaned = res.json().get("response", "").strip()
            if cleaned.lower().startswith("voici"):
                cleaned = cleaned.split(":", 1)[-1].strip()
            cleaned_paragraphs.append(cleaned)
        except Exception as e:
            print(f"[PROCESSOR]   WARNING: Chunk {i} failed ({e}). Keeping original.")
            cleaned_paragraphs.append(chunk)

    return "\n\n".join(cleaned_paragraphs)

def filter_transcript_chunked(transcript_text):
    """Second pass — keep only actual medical content, strip professor small talk."""
    paragraphs = transcript_text.split('\n\n')
    chunks_to_process = []
    current_chunk = []
    current_len = 0

    for p in paragraphs:
        if not p.strip():
            continue
        if current_len + len(p) > MAX_CHUNK_CHARS and current_chunk:
            chunks_to_process.append("\n\n".join(current_chunk))
            current_chunk = [p]
            current_len = len(p)
        else:
            current_chunk.append(p)
            current_len += len(p)

    if current_chunk:
        chunks_to_process.append("\n\n".join(current_chunk))

    total_chunks = len(chunks_to_process)
    print(f"[PROCESSOR] Filtering medical content across {total_chunks} chunks...")

    filtered_paragraphs = []
    for i, chunk in enumerate(chunks_to_process, 1):
        print(f"[PROCESSOR]   Filtering chunk {i}/{total_chunks}...")
        prompt = (
            "Tu es un filtre de contenu médical pour la FMPC. "
            "Voici un extrait de cours de médecine transcrit. "
            "Supprime UNIQUEMENT les passages non médicaux : bavardages, organisation du semestre, "
            "blagues, échanges avec les étudiants, commentaires personnels du professeur. "
            "Conserve TOUT le contenu médical : définitions, schémas, anatomie, physiologie, "
            "histologie, pathologie, biochimie, embryologie, bactériologie. "
            "Ne résume pas. Ne reformule pas. Ne rajoute rien. "
            "N'ajoute AUCUNE phrase d'introduction. "
            "Si un paragraphe entier est non médical, supprime-le complètement. "
            "Renvoie EXCLUSIVEMENT le texte filtré.\n\n"
            "Texte à filtrer :\n" + chunk
        )
        try:
            res = requests.post(OLLAMA_URL, json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            }, timeout=120)
            filtered = res.json().get("response", "").strip()
            if filtered.lower().startswith("voici"):
                filtered = filtered.split(":", 1)[-1].strip()
            if filtered:
                filtered_paragraphs.append(filtered)
        except Exception as e:
            print(f"[PROCESSOR]   WARNING: Filter chunk {i} failed ({e}). Keeping original.")
            filtered_paragraphs.append(chunk)

    return "\n\n".join(filtered_paragraphs)

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

    raw_transcript   = result["transcript"]
    language         = result["language"]
    segment_count    = result["segment_count"]
    paragraph_count  = result["paragraph_count"]
    file_path        = result["file_path"]

    print(f"[PROCESSOR] Loaded raw transcript — {segment_count} segments, {paragraph_count} paragraphs")

    # STEP 1: SPELLING + PHONETIC CORRECTION
    print("[PROCESSOR] Step 1/5: Correcting phonetic medical errors with Qwen2.5:7b...")
    cleaned_transcript = clean_transcript_chunked(raw_transcript)

    # STEP 2: MEDICAL CONTENT FILTER
    print("[PROCESSOR] Step 2/5: Filtering to medical content only...")
    filtered_transcript = filter_transcript_chunked(cleaned_transcript)

    # STEP 3: GENERATE LABEL (from filtered version — cleaner signal)
    print("[PROCESSOR] Step 3/5: Generating label...")
    label = generate_label(filtered_transcript)
    print(f"[PROCESSOR] Generated label: {label}")

    # STEP 4: SAVE BOTH VERSIONS
    print("[PROCESSOR] Step 4/5: Saving transcripts...")

    header_full = (
        f"TRANSCRIPT (FULL): {label}\n"
        f"Language: {language} | Segments: {segment_count} | Paragraphs: {paragraph_count}\n"
        f"AI Corrector: {OLLAMA_MODEL}\n"
        + "=" * 60 + "\n\n"
    )
    header_filtered = (
        f"TRANSCRIPT (MEDICAL ONLY): {label}\n"
        f"Language: {language} | Segments: {segment_count} | Paragraphs: {paragraph_count}\n"
        f"AI Corrector: {OLLAMA_MODEL} | Filter: medical-only\n"
        + "=" * 60 + "\n\n"
    )

    # Full corrected transcript
    full_ssd  = os.path.join(TRANSCRIPTS, label + ".txt")
    full_hdd  = os.path.join(ARCHIVE_HDD, label + ".txt")
    with open(full_ssd, "w", encoding="utf-8") as f:
        f.write(header_full + cleaned_transcript)
    shutil.copy2(full_ssd, full_hdd)
    print(f"[PROCESSOR] Full transcript saved: {full_ssd}")

    # Filtered medical-only transcript
    clean_ssd = os.path.join(TRANSCRIPTS, label + "_clean.txt")
    clean_hdd = os.path.join(ARCHIVE_HDD, label + "_clean.txt")
    with open(clean_ssd, "w", encoding="utf-8") as f:
        f.write(header_filtered + filtered_transcript)
    shutil.copy2(clean_ssd, clean_hdd)
    print(f"[PROCESSOR] Filtered transcript saved: {clean_ssd}")

    # STEP 5: ARCHIVE AUDIO AS OPUS
    print("[PROCESSOR] Step 5/5: Archiving audio...")
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
        print(f"[PROCESSOR] Audio archived: {orig_mb:.1f}MB -> {opus_mb:.1f}MB ({100*(1-opus_mb/orig_mb):.0f}% saved)")

    # CLEANUP
    os.remove(file_path)
    if os.path.exists(TMP_WAV):
        os.remove(TMP_WAV)
    if os.path.exists(TMP_RESULT):
        os.remove(TMP_RESULT)
    print(f"[PROCESSOR] ─── COMPLETE: {label} ───")

if __name__ == "__main__":
    main()
