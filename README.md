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
ChromaDB: C:\FMPC_Scribe\chroma_db\ (transcripts + slides indexed for RAG)
    ↓ PC shuts down automatically when idle
```

## Key technical decisions

- **Qwen3-ASR-1.7B** — ASR-LLM hybrid released January 2026, handles Moroccan French accent and medical terminology significantly better than Whisper Large-v3 on noisy lecture hall recordings
- **142-word medical context prompt** — injected via the `context` parameter to bias the model's phonetic decoding toward FMPC terminology (anatomie, histologie, physiologie, biochimie, hématologie, bactériologie, embryologie, pathologie)
- **60-second chunk cap** — audio is split at 2.5s silence gaps, but any resulting chunk longer than 60 seconds is automatically sub-divided before hitting the model. Prevents CUDA OOM on 8GB VRAM during 2-hour lectures regardless of how long a professor talks without pausing
- **Qwen2.5:7b cleanup pass** — after transcription, a second LLM pass through Ollama corrects phonetic errors in medical terms chunk by chunk. Qwen2.5 was chosen over Llama 3 specifically for its native French precision and strict instruction-following — Llama 3 kept anglicizing the text
- **Two-pass processing** — Pass 1 corrects spelling/phonetics. Pass 2 filters non-medical content (banter, admin, personal comments). Both saved: full corrected version for RAG, filtered version for personal review
- **Schedule-based labeling** — `schedule.json` maps each day/time to a discipline. On ambiguous slots (two sections with swapped timetables), Qwen2.5 reads the transcript and picks the right discipline by content. Falls back to pure AI identification if no schedule entry exists
- **Three-script isolation** — `scribe_engine.py` orchestrates, `transcriber.py` handles GPU transcription, `processor.py` handles everything after. Each runs as a fully isolated subprocess with no shared memory
- **os._exit(0) hard kill** — `transcriber.py` ends with `os._exit(0)` instead of `sys.exit()`. This bypasses all Python and CUDA cleanup and prevents a Windows-specific ctranslate2 Access Violation (exit code 0xC0000005) that crashes the process silently after every transcription. Non-negotiable — removing it brings the crash back immediately
- **Paragraph grouping by silence** — audio split at 2.5s silence gaps rather than line by line, output reads like a document
- **Noise reduction at 0.75** — tuned upward from 0.65 for lecture hall reverberation and background noise
- **Opus 48kbps** — chosen over 32kbps to preserve high-frequency consonant information in medical terminology for future re-processing
- **No summarization** — raw transcript preserved exactly as spoken for maximum RAG accuracy
- **Dual storage** — transcript saved to SSD for fast RAG access, copied to HDD alongside compressed audio for long-term archive
- **paraphrase-multilingual-MiniLM-L12-v2** — embedding model for ChromaDB, chosen for native French support and CPU-only operation so it doesn't compete with Qwen3-ASR for VRAM

## The engineering battles

Things that didn't go smoothly and the workarounds that fixed them.

**Whisper Large-v3 was unusable on this audio.** "Vertèbre lombaire" came out as "vertèbre romaine", "aorte abdominale" as "Hortes, Amérique", and a 27-minute segment produced nothing but "Péritoine. Péritoine. Péritoine." in a loop. The combination of Moroccan French accent, lecture hall reverb, and dense medical vocabulary was outside Whisper's reliable range. Switched to Qwen3-ASR-1.7B, which handles all three natively.

**The qwen-asr Python wrapper exposed no prompt parameter.** The whole point of Qwen3-ASR is prompt biasing. After installation, every attempt — `prompt`, `initial_prompt`, `context`, `system_prompt` — threw `TypeError: unexpected keyword argument`. A runtime hack that patched `sys.modules` to inject the prompt into the language validation list also failed because the wrapper silently calls `.capitalize()` on the language argument before checking it, which lowercased the entire injected 500-word prompt and broke the string match. The fix was to just edit the wrapper source directly — two lines in `utils.py` were replaced with `pass` to disable validation entirely, then `context` works as intended.

**Llama 3 anglicized French medical text.** The original cleanup pass used Llama 3, which kept turning "érythrocyte" into "erythrocyte" and rewriting the professor's sentences. Replaced with Qwen2.5:7b, which is multilingual by design and actually follows the "don't rewrite anything" instruction.

**VRAM fragmentation on long lectures.** Even with periodic model reloads, batches 31 and 41 of a 2-hour lecture consistently hit CUDA OOM on 8GB VRAM. Standard `torch.cuda.empty_cache()` wasn't enough — PyTorch holds fragmented reserved memory that doesn't get released. Fixed with a double `gc.collect()` + `empty_cache()` + 1 second sleep before reload, plus automatic chunk splitting to cap individual audio segments at 60 seconds regardless of silence gap detection.

**Windows Defender silently deleted temp_result.json.** The transcriber writes a JSON file that the processor reads 2 seconds later. Defender was quarantining it in between, causing a consistent `FileNotFoundError` that looked like a double-trigger bug. Fixed by renaming the temp file to `.tmp` and adding the working directory to Defender exclusions.

## RAG system

After each transcription, the transcript is automatically indexed into ChromaDB alongside any professor slides you've ingested. A Streamlit interface lets you ask questions and get answers grounded in what your professor actually said.

**Slide ingestion** — drop any PDF or PPTX into `slides/discipline/` and a file watcher ingests it automatically. Discipline is inferred from the subfolder name. Supports `.pdf` and `.pptx`. Runs as a background process alongside the scribe engine.

**Query flow** — question gets embedded → top chunks retrieved from both transcript and slide collections → Qwen2.5 answers in French with source citations → contradictions between transcript and slides flagged explicitly.

**Anki integration** — `anki_generator.py` reads ingested slides from ChromaDB, batches them through Qwen2.5 with a flashcard prompt, and pushes cards directly into Anki via AnkiConnect. Duplicate-safe. Deck structure: `FMPC → discipline → filename`.

## Pi scripts

**pi_watchdog.py** — permanent background daemon, starts on Pi boot via systemd. Watches `~/fmpc_staging`, automatically wakes PC when a file arrives, waits for boot, refreshes SMB mount, moves file to INBOX. This is the script that makes the daily workflow fully hands-off.

**wake.sh** — manual override script. Used when you want to wake the PC without uploading a file. Not needed for normal daily use.

**PC-side scripts** — five files working together:
- `scribe_engine.py` — watchdog and orchestrator, runs permanently, launches the other two
- `transcriber.py` — isolated GPU process, runs Qwen3-ASR-1.7B, saves result to temp file, and exits with `os._exit(0)`
- `processor.py` — reads transcript, runs Qwen2.5 cleanup in chunks, identifies discipline, saves to SSD and HDD, indexes into ChromaDB, archives audio as Opus 48kbps
- `ingest_slides.py` — PDF/PPTX watcher and ingestion pipeline, runs as background process
- `rag_engine.py` — Streamlit RAG interface, runs as background process

## Storage structure

```
SSD — C:\FMPC_Scribe\NOTES_Transcripts\
    anatomie_cavite-abdominale_2026-03-02.txt
    anatomie_cavite-abdominale_2026-03-02_clean.txt
    histologie_tissu-conjonctif_2026-03-05.txt
    histologie_tissu-conjonctif_2026-03-05_clean.txt

HDD — D:\FMPC_Audio_Archive\Medical\
    anatomie_cavite-abdominale_2026-03-02.txt   (transcript copy)
    anatomie_cavite-abdominale_2026-03-02.opus  (compressed audio)
    histologie_tissu-conjonctif_2026-03-05.txt
    histologie_tissu-conjonctif_2026-03-05.opus

ChromaDB — C:\FMPC_Scribe\chroma_db\
    transcripts collection  (chunked lecture transcripts)
    slides collection       (chunked PDF/PPTX slide content)
```

## Configuration

**schedule.json** — not committed to the repo. Create `C:\FMPC_Scribe\schedule.json` with your semester timetable:

```json
{
  "schedule": [
    {"date": "2026-03-02", "start": "08:30", "end": "10:15", "discipline": "anatomie", "professor": "Pr. Fadili", "section": 1},
    {"date": "2026-03-02", "start": "10:15", "end": "12:00", "discipline": "bacteriologie", "professor": "Pr. El Kettani", "section": 1}
  ]
}
```

Both sections of the same day go in the same file. On ambiguous slots, Qwen2.5 reads the transcript content to decide.

## Stack

- Qwen3-ASR-1.7B (CUDA, transformers backend)
- Qwen2.5:7b via Ollama (cleanup, labeling, RAG generation, Anki card generation)
- paraphrase-multilingual-MiniLM-L12-v2 (ChromaDB embeddings, CPU)
- ChromaDB (local vector store)
- Streamlit (RAG interface)
- AnkiConnect (Anki flashcard push)
- FFmpeg (Opus 48kbps compression)
- Noisereduce + Librosa (audio cleaning + normalization)
- pdfplumber + python-pptx (slide text extraction)
- watchdog (folder monitoring)
- Tailscale (VPN)
- etherwake (Wake-on-LAN)
- SMB/CIFS (Pi to PC file transfer)
- systemd (Pi watchdog service)
- Three-process GPU isolation — Windows ctranslate2 CUDA crash fix

## Status

RAG system operational. Anki integration is complete. The transcription pipeline is stable for 2-hour lectures. Raspberry Pi assembly in progress.

## Why I built this

I got tired of re-listening to 2-hour lectures to find one concept I missed. I'd rather spend that time building the tool that does it for me.
