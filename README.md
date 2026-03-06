# FMPC Scribe

Personal project I built to stop wasting time relistening to 2-hour lectures — records, transcribes, cleans, archives, and indexes medical lectures fully locally on my computer, woken remotely via a Raspberry Pi from my phone. No cloud, no subscriptions, just my GPU doing the work while I focus on other things.

## How it works

1. Record lecture on phone (any format — M4A, MP3, WAV)
2. Open Solid Explorer on phone, upload audio to the Pi's staging folder (`~/fmpc_staging`)
3. Pi detects the file automatically, sends a Magic Packet to wake the PC, waits for Windows to boot, refreshes the SMB mount, and moves the file into the PC's INBOX
4. Windows Task Scheduler runs `scribe_engine.py` automatically on boot — it orchestrates two sub-processes: `transcriber.py` handles Qwen3-ASR on GPU, `processor.py` handles Qwen2.5 cleanup, labeling, ChromaDB indexing, and archiving
5. Engine cleans audio, transcribes with Qwen3-ASR-1.7B, corrects medical terminology errors with Qwen2.5, identifies discipline from schedule + transcript content, saves full and medical-only versions, indexes into ChromaDB, archives transcript and audio
6. PC shuts itself down automatically once INBOX has been empty for 30 minutes and the computer isn't in use

You upload the file and walk away. Everything else is automatic.

## Architecture

- **PC (Brain):** RTX 4060 8GB, Windows 11, Qwen3-ASR + Ollama
- **Raspberry Pi 4 (Gateway):** 24/7 uptime, automatic WoL sender, file relay, always watching the staging folder
- **Network:** Direct Cat6 Ethernet bridge (192.168.2.x) between Pi and PC for file transfer, home Wi-Fi for internet/Tailscale
- **Remote access:** Tailscale mesh VPN — upload from anywhere on any network via Solid Explorer SFTP

## File flow

```
Phone (Solid Explorer SFTP)
    ↓ upload
Pi staging folder: ~/fmpc_staging/
    ↓ pi_watchdog.py detects file → sends Magic Packet → waits for boot → refreshes mount → moves file
PC INBOX: C:\FMPC_Scribe\INBOX\
    ↓ scribe_engine.py detects file → cleans audio → transcribes → corrects → labels → indexes → archives
SSD: C:\FMPC_Scribe\NOTES_Transcripts\course_topic_date.txt + course_topic_date_clean.txt
HDD: D:\FMPC_Audio_Archive\Medical\course_topic_date.opus + .txt copies
ChromaDB: C:\FMPC_Scribe\chroma_db\ (transcripts + slides + exam_questions)
    ↓ PC shuts down automatically when idle
```

## Key technical decisions

- **Qwen3-ASR-1.7B** — ASR-LLM hybrid released January 2026, handles Moroccan French accent and medical terminology significantly better than Whisper Large-v3 on noisy lecture hall recordings
- **142-word medical context prompt** — injected via the `context` parameter to bias the model's phonetic decoding toward FMPC terminology
- **60-second chunk cap** — audio split at 2.5s silence gaps, any chunk over 60s automatically sub-divided before hitting the model. Prevents CUDA OOM on 8GB VRAM during 2-hour lectures
- **BATCH_SIZE = 3** — transcriber processes 3 audio chunks simultaneously. Runs at ~70% VRAM, 3x throughput vs the original single-chunk approach. OOM retry falls back to individual chunks automatically
- **MODEL_RELOAD_EVERY = 15** — model reloads every 15 batches instead of 5, reducing overhead on long lectures. VRAM purge uses double gc.collect() + empty_cache() + 1s sleep to prevent fragmentation
- **Qwen2.5:7b cleanup pass** — second LLM pass corrects phonetic errors in medical terms. Chosen over Llama 3 for native French precision and strict instruction-following
- **Two-pass processing** — Pass 1 corrects spelling/phonetics. Pass 2 filters non-medical content. Both saved: full for RAG, filtered for review. Shared `_chunk_transcript()` helper, MAX_CHUNK_CHARS = 4000, num_predict = 2048 on all calls
- **Ollama retry wrapper** — `ollama_with_retry()` retries failed correction/filter calls 3 times with 5s backoff before falling back to original text. Prevents silent corruption from transient Ollama failures
- **Audio deletion guard** — original recording only deleted after confirmed Opus archive. If FFmpeg fails, original is preserved and a warning is printed
- **keep_alive: 0 after processor** — forces Ollama to unload the 7B model from VRAM immediately after processor.py finishes. Prevents OOM when a second file arrives before the default 5-minute keepalive expires
- **Subprocess timeouts** — scribe_engine.py enforces 3-hour timeout on transcriber, 1-hour on processor. Prevents infinite hangs on CUDA deadlock or Ollama freeze
- **Gap markers** — failed transcription batches insert `[SEGMENT MANQUANT — ...]` into the transcript instead of silently dropping content
- **Schedule-based labeling** — `schedule.json` maps day/time to discipline. On ambiguous slots Qwen2.5 reads transcript content to decide. Falls back to pure AI identification if no schedule entry
- **Three-script isolation** — `scribe_engine.py` orchestrates, `transcriber.py` handles GPU, `processor.py` handles everything after. Fully isolated subprocesses, no shared memory
- **os._exit(0) hard kill** — `transcriber.py` exits with `os._exit(0)` to bypass Python and CUDA cleanup and prevent a Windows-specific ctranslate2 Access Violation (0xC0000005) that crashes silently after every transcription
- **paraphrase-multilingual-MiniLM-L12-v2** — embedding model for ChromaDB, native French support, CPU-only so it doesn't compete with Qwen3-ASR for VRAM
- **FastAPI + vanilla HTML/CSS/JS** — replaced Streamlit entirely. Full design freedom, dark mode, instant UI response. Backend serves frontend at `localhost:8000`
- **Question-boundary chunking for exam ingestion** — exam PDFs split at regex-detected question markers before sending to Qwen2.5, capped at 2200 chars per chunk. Prevents context overflow and JSON truncation on 60-question exams
- **exam_questions collection** — past exams parsed at ingest time into structured question objects (question + 5 choices + correct answers + explanation). Loaded directly by the study interface — no generation at query time
- **Two-step QCM generation** — claims extracted from professor material once before the generation loop, then all question batches generate strictly from those claims. Prevents Qwen from drawing on general medical knowledge instead of course content
- **Explanation cache** — professor-grounded corrections cached to `explain_cache.json` after first generation. Repeat views return instantly without an Ollama call
- **QCM bank** — generated QCM sessions saved to `qcm_bank.json`. Bank scanned before every generation to prevent repeats across sessions
- **OCR fallback in slide ingestion** — pdfplumber extracts embedded text first. If fewer than 10% of pages yield text, automatically switches to PyMuPDF (300 DPI) + Tesseract OCR with French language model

## The engineering battles

**Whisper Large-v3 was unusable on this audio.** "Vertèbre lombaire" → "vertèbre romaine", "aorte abdominale" → "Hortes, Amérique", 27-minute segment → "Péritoine." repeated in a loop. Moroccan French accent + lecture hall reverb + medical vocabulary was outside Whisper's range. Switched to Qwen3-ASR-1.7B.

**The qwen-asr wrapper exposed no prompt parameter.** Every attempt — `prompt`, `initial_prompt`, `context`, `system_prompt` — threw `TypeError`. A sys.modules patch also failed because the wrapper calls `.capitalize()` on the language argument before validation, destroying the injected prompt. Fixed by editing the wrapper source directly: two lines in `utils.py` replaced with `pass` to disable validation, then `context` works as intended.

**Llama 3 anglicized French medical text.** Kept turning "érythrocyte" into "erythrocyte" and rewriting sentences. Replaced with Qwen2.5:7b.

**VRAM fragmentation on long lectures.** Batches 31 and 41 of a 2-hour lecture consistently hit CUDA OOM. `torch.cuda.empty_cache()` alone wasn't enough — PyTorch holds fragmented reserved memory. Fixed with double `gc.collect()` + `empty_cache()` + 1s sleep before reload, plus 60s chunk cap.

**Windows Defender silently deleted temp_result.json.** Defender quarantined the transcriber output file between write and read, causing consistent `FileNotFoundError`. Fixed by renaming to `.tmp` and adding the working directory to Defender exclusions.

**ffprobe UnicodeDecodeError on Windows.** `processor.py` called ffprobe with `text=True`, letting Windows use cp1252. Some audio metadata bytes are undefined in cp1252. Fixed by reading raw bytes and decoding explicitly as UTF-8 with `errors="ignore"`.

**OCR on scanned exam PDFs returned zero text.** pdfplumber extracts embedded text only. Added automatic scanned PDF detection: if zero pages return text, switches to PyMuPDF (300 DPI render) + Tesseract OCR with French language model. PyMuPDF chosen over pdf2image — no external binary dependencies.

**Qwen2.5 truncating exam JSON mid-output.** Model hit default output token limit on 60-question exams, returning malformed JSON. Fixed with `num_predict: 4096` on all exam parsing calls and one retry on JSON parse failure before skipping.

**Histologie exam timeout — 19,000-char chunk.** Boundary detector found only 12 markers in a 17-page exam, collapsing content into one massive chunk that timed out at 300s. Fixed with a hard 2200-char cap that sub-divides any oversized chunk at the nearest newline.

**QCM corrections were hallucinating anatomy.** Qwen used retrieved chunks as topic inspiration but then drew on general medical knowledge to write questions and explanations, producing anatomically incorrect distractors. Fixed with two-step generation: extract verbatim claims from professor material first, then generate strictly from those claims. Each correction now cites the exact course sentence it's based on.

**JS crash on load — unescaped French apostrophes.** French contractions (`l'analyse`, `d'abord`) inside single-quoted JS strings broke the entire script block, preventing anything from loading. Fixed by replacing with HTML entities (`&#39;`) in all affected innerHTML assignments.

## Study interface

FastAPI backend + single-page HTML/CSS/JS frontend. Dark mode ("Obsidian"). Served at `http://localhost:8000`. Academic year 2025–2026.

```
cd C:\FMPC_Scribe
py -3.11 -m uvicorn api:app --host 0.0.0.0 --port 8000
```

### 💬 Assistant RAG

Ask any question, get an answer grounded in professor material. Retrieves top chunks from both transcript and slide collections, Qwen2.5 answers in French. Contradictions between sources flagged. Source chips show discipline, filename, similarity score.

### 📝 QCM — 15 Questions

Generates 15 multiple-choice questions strictly from professor course material.

- **Two-step generation** — claims extracted from course text first, questions generated only from those claims. No general medical knowledge
- **Course selector** — filter by discipline, then pick a specific course file to restrict generation to that exact lecture
- **Topic field** — optional. Specify a subject or leave blank for random coverage
- **TOR scoring** — all-or-nothing. Must select every correct answer and no wrong ones to score the point
- **Per-question correction** — after Valider: colour-coded answer states + explanation citing exact course material + distractor explanations
- **Confidence checkbox** — mark uncertainty before validating. Feeds into weak point tracking
- **Live score bar** — answered/total, correct count, percentage, progress fill
- **💾 Sauvegarder** — saves session to bank. Future generations scan bank to avoid repeats

### 🗃️ Banque QCM

All saved QCM sessions listed with discipline, topic, question count, date. Click **Pratiquer** to redo any session with full TOR scoring and corrections. Delete with ✕. Stored in `qcm_bank.json`.

### 📚 Examens

Practice from real past exam PDFs.

- Select block → year → file → Commencer
- All questions shown at once with checkboxes, no hints
- **Corriger** per question: locks choices, shows answer states, loads professor-grounded correction inline
- **Soumettre** at end: scores everything, reuses already-corrected cards, fetches remaining explanations
- Score card: X/N, percentage, RÉUSSI (≥50%) or ÉCHOUÉ
- Corrections grounded in professor material, cached after first generation

Exam blocks:
```
microbiologie           → bacteriologie + virologie
anatomie                → anatomie
histologie-embryologie  → histologie + embryologie
hematologie-immunologie → hematologie + immunologie
physiologie             → physiologie
```

### 🎯 Points Faibles

Tracks every answer attempt silently. Surfaces questions ranked by weakness score: wrong answers weight 3×, correct-but-uncertain weight 1.5×, both decay over 30 days. Click **S'entraîner** to drill exactly those questions. Stored in `answer_history.json`.

### 🔍 Timeline

Search any concept across all transcripts. Returns every chunk where the professor mentioned it, with similarity score, discipline, filename, and date. Search term highlighted in results.

### ⚡ Lacunes

Compares slides vs transcripts for a discipline. Identifies concepts in slides not mentioned in lectures, concepts mentioned in lectures not in slides, and contradictions between sources. Returns a coverage score.

### 🏆 Pré-Examen

Generates targeted study sessions weighted toward upcoming exam blocks. Pulls previously-wrong questions first (marked ↺), fills the rest with new generated questions. Blocks ranked by urgency based on practice history.

## Exam ingestion

```
cd C:\FMPC_Scribe
py -3.11 ingest_exams.py                    # watch mode
py -3.11 ingest_exams.py list               # show all ingested exams
py -3.11 ingest_exams.py delete "filename"  # remove one exam
```

Folder structure:
```
C:\FMPC_Scribe\exams\
    microbiologie\examen_bacterio_virologie_2025.pdf
    anatomie\examen_anatomie_2025.pdf
    histologie-embryologie\examen_histologie_2025.pdf
    hematologie-immunologie\examen_hemato_immuno_2025.pdf
    physiologie\examen_physiologie_2024.pdf
```

Filename must contain a 4-digit year for auto-detection. If not, pass it manually:
```
py -3.11 ingest_exams.py "exams\microbiologie\examen.pdf" microbiologie 2025
```

OCR dependencies (scanned PDFs only):
```
py -3.11 -m pip install pymupdf pytesseract pillow
```
Plus Tesseract from https://github.com/UB-Mannheim/tesseract/wiki — check "French language data".

## PC-side scripts

| File | Role |
|------|------|
| `scribe_engine.py` | Watchdog and orchestrator, runs permanently. Subprocess timeouts: 3h transcriber, 1h processor |
| `transcriber.py` | Isolated GPU process — Qwen3-ASR batch=3, exits with `os._exit(0)` |
| `processor.py` | Qwen2.5 two-pass cleanup, discipline labeling, ChromaDB indexing, audio archiving |
| `ingest_slides.py` | PDF/PPTX watcher and slide ingestion, OCR fallback for scanned PDFs |
| `ingest_exams.py` | Exam PDF ingestion — OCR + Qwen2.5 question parsing |
| `api.py` | FastAPI backend — RAG, QCM, exam loading, QCM bank, history, gaps, timeline, pre-exam |
| `index.html` | Single-page study interface — dark mode, TOR scoring, 7 panels |
| `anki_generator.py` | Anki flashcard generation via AnkiConnect |

## Storage structure

```
SSD — C:\FMPC_Scribe\NOTES_Transcripts\
HDD — D:\FMPC_Audio_Archive\Medical\

ChromaDB — C:\FMPC_Scribe\chroma_db\
    transcripts       (chunked lecture transcripts)
    slides            (chunked PDF/PPTX content)
    exam_questions    (structured past exam questions)

Data files — C:\FMPC_Scribe\
    qcm_bank.json        (saved QCM sessions)
    explain_cache.json   (cached professor-grounded corrections)
    answer_history.json  (per-question answer tracking for weak points)
    schedule.json        (timetable — not in repo)
```

## Configuration

**schedule.json** — not committed. Create `C:\FMPC_Scribe\schedule.json`:

```json
{
  "schedule": [
    {"date": "2026-03-02", "start": "08:30", "end": "10:15", "discipline": "anatomie", "professor": "Pr. X", "section": 1},
    {"date": "2026-03-02", "start": "10:15", "end": "12:00", "discipline": "bacteriologie", "professor": "Pr. Y", "section": 1}
  ]
}
```

## Stack

- Qwen3-ASR-1.7B (CUDA, transformers)
- Qwen2.5:7b via Ollama (cleanup, labeling, QCM generation, exam parsing, Anki cards)
- paraphrase-multilingual-MiniLM-L12-v2 (ChromaDB embeddings, CPU)
- ChromaDB (local vector store — 3 collections)
- FastAPI + Uvicorn (study interface backend)
- Vanilla HTML/CSS/JS (study interface — dark mode, no framework)
- AnkiConnect (Anki flashcard push)
- FFmpeg (Opus 48kbps compression)
- Noisereduce + Librosa (audio cleaning)
- pdfplumber + python-pptx (text extraction)
- PyMuPDF + Tesseract (OCR for scanned PDFs)
- watchdog (folder monitoring)
- Tailscale (VPN)
- etherwake (Wake-on-LAN)
- SMB/CIFS (Pi → PC file transfer)
- systemd (Pi watchdog service)

## Launch

```
start_all.bat
```

Starts three windows: Slide Watcher, RAG Assistant (uvicorn on port 8000), Scribe Engine. Open `http://localhost:8000`.

## Status

Full pipeline operational. Transcription stable on 2-hour lectures at batch size 3 (~70% VRAM). Study interface on FastAPI — RAG assistant, grounded QCM generation with claim extraction, real exam practice with TOR scoring and professor-grounded corrections, weak point tracking, lecture timeline search, gap detection, and pre-exam mode. Academic year 2025–2026.

## Why I built this

I got tired of relistening to 2-hour lectures to find one concept I missed. I'd rather spend that time building the tool that does it for me.
