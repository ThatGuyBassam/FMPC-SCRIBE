import sys, json, os, shutil, subprocess, requests
from datetime import datetime

# ─── CONFIG ────────────────────────────────────────────────────────
TRANSCRIPTS = r"C:\FMPC_Scribe\NOTES_Transcripts"
ARCHIVE_HDD = r"D:\FMPC_Audio_Archive\Medical"
TMP_RESULT  = r"C:\FMPC_Scribe\temp_result.json"
TMP_WAV     = r"C:\FMPC_Scribe\temp_processing.wav"
OLLAMA_URL  = "http://localhost:11434/api/generate"

def generate_label(transcript_text):
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
        # Remove any date Llama may have added — we append our own
        label = label.split("_202")[0].split("_201")[0].split("_200")[0]
        label = "".join(c for c in label if c.isalnum() or c in "-_")
        if not label:
            label = "lecture"
        label = f"{label}_{today}"

        print(f"[LABELER] Generated label: {label}")
        return label
    except Exception as e:
        print(f"[LABELER] Labeling failed: {e}. Using fallback.")
        return f"lecture_{today}"

def main():
    print("[LABELER] Starting...")

    # READ TRANSCRIPT FROM TEMP FILE
    with open(TMP_RESULT, "r", encoding="utf-8") as f:
        result = json.load(f)

    transcript_text = result["transcript"]
    language = result["language"]
    segment_count = result["segment_count"]
    paragraph_count = result["paragraph_count"]
    file_path = result["file_path"]

    print(f"[LABELER] Loaded transcript — {segment_count} segments, {paragraph_count} paragraphs")

    # GENERATE LABEL
    print("[LABELER] Step 1/3: Generating label with Ollama...")
    label = generate_label(transcript_text)

    # SAVE TRANSCRIPT TO SSD
    print("[LABELER] Step 2/3: Saving transcript...")
    txt_filename = label + ".txt"
    txt_ssd_path = os.path.join(TRANSCRIPTS, txt_filename)
    txt_hdd_path = os.path.join(ARCHIVE_HDD, txt_filename)

    header = (
        f"TRANSCRIPT: {label}\n"
        f"Language: {language} | Segments: {segment_count} | Paragraphs: {paragraph_count}\n"
        + "=" * 60 + "\n\n"
    )

    with open(txt_ssd_path, "w", encoding="utf-8") as f:
        f.write(header + transcript_text)
    print(f"[LABELER] Transcript saved to SSD: {txt_ssd_path}")

    shutil.copy2(txt_ssd_path, txt_hdd_path)
    print(f"[LABELER] Transcript copied to HDD: {txt_hdd_path}")

    # ARCHIVE AUDIO AS OPUS
    print("[LABELER] Step 3/3: Archiving audio...")
    out_opus = os.path.join(ARCHIVE_HDD, label + ".opus")
    result_ffmpeg = subprocess.run([
        "ffmpeg", "-i", file_path,
        "-c:a", "libopus", "-b:a", "48k",
        "-application", "voip",
        "-y", out_opus
    ], capture_output=True)

    if result_ffmpeg.returncode != 0:
        print(f"[LABELER] FFmpeg error: {result_ffmpeg.stderr.decode()}")
    else:
        orig_mb = os.path.getsize(file_path) / 1e6
        opus_mb = os.path.getsize(out_opus) / 1e6
        print(f"[LABELER] Audio archived: {orig_mb:.1f}MB -> {opus_mb:.1f}MB ({100*(1-opus_mb/orig_mb):.0f}% saved)")

    # CLEANUP
    os.remove(file_path)
    if os.path.exists(TMP_WAV):
        os.remove(TMP_WAV)
    if os.path.exists(TMP_RESULT):
        os.remove(TMP_RESULT)
    print(f"[LABELER] ─── COMPLETE: {label} ───")

if __name__ == "__main__":
    main()

