import sys, json, torch, gc, os
import librosa, soundfile as sf, noisereduce as nr
from faster_whisper import WhisperModel

# ─── CONFIG ────────────────────────────────────────────────────────
TMP_WAV     = r"C:\FMPC_Scribe\temp_processing.wav"
TMP_RESULT  = r"C:\FMPC_Scribe\temp_result.json"
NOISE_REDUCTION = 0.65
PARAGRAPH_PAUSE = 2.5

INITIAL_PROMPT = (
    "Medical lecture, French and English. Terms: NGS, CRISPR, PCR, mRNA, "
    "Myocarde, Histologie, Pathophysiology, Apoptosis, Cardiomyopathy, "
    "Endocarditis, Tachycardia, Bradycardia, Fibrillation, Angioplasty, "
    "Epithelium, Mitochondria, Ribosomes, Endoplasmic reticulum, Golgi."
)

def format_time(seconds):
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"

def segments_to_paragraphs(segments, pause_threshold=PARAGRAPH_PAUSE):
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

def main():
    file_path = sys.argv[1]
    print(f"[TRANSCRIBER] Processing: {file_path}")

    # STEP 1: AUDIO CLEANING
    print("[TRANSCRIBER] Step 1/2: Cleaning audio...")
    y, sr = librosa.load(file_path, sr=None, mono=True)
    clean_y = nr.reduce_noise(y=y, sr=sr, prop_decrease=NOISE_REDUCTION)
    sf.write(TMP_WAV, clean_y, sr)
    print(f"[TRANSCRIBER] Duration: {len(y)/sr:.1f}s | Sample rate: {sr}Hz")

    # STEP 2: TRANSCRIPTION
    print("[TRANSCRIBER] Step 2/2: Loading Whisper Large-v3...")
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")

    segments, info = model.transcribe(
        TMP_WAV,
        beam_size=5,
        language=None,
        initial_prompt=INITIAL_PROMPT,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=700),
    )

    print(f"[TRANSCRIBER] Language: {info.language} ({info.language_probability:.0%})")

    segment_list = list(segments)
    print(f"[TRANSCRIBER] Segments: {len(segment_list)}")

    paragraphs = segments_to_paragraphs(segment_list)
    print(f"[TRANSCRIBER] Paragraphs: {len(paragraphs)}")

    transcript_lines = []
    for (start, end, text) in paragraphs:
        transcript_lines.append(f"[{format_time(start)} - {format_time(end)}]")
        transcript_lines.append(f"{text}\n")
    transcript_text = "\n".join(transcript_lines)

    # SAVE RESULT TO TEMP JSON — labeler.py reads this
    result = {
        "transcript": transcript_text,
        "language": info.language,
        "language_probability": info.language_probability,
        "segment_count": len(segment_list),
        "paragraph_count": len(paragraphs),
        "file_path": file_path
    }

    with open(TMP_RESULT, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)

    print(f"[TRANSCRIBER] Result saved to {TMP_RESULT}")
    print("[TRANSCRIBER] DONE — exiting cleanly.")
    os._exit(0)
    # Process exits here — CUDA context released naturally by OS

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
