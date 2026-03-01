# FMPC Scribe

Personal project I built to stop wasting time relistening to 2-hour lectures — records, transcribes, and archives medical lectures fully locally on my computer, woken remotely via a Raspberry Pi from my phone. No cloud, no subscriptions, just my GPU doing the work while I live.

## How it works

1. Record lecture on phone (any format — M4A, MP3, WAV)
2. Open Solid Explorer on phone, upload audio to the Pi's staging folder (`~/fmpc_staging`)
3. Pi detects the file automatically, sends Magic Packet to wake the PC, waits for Windows to boot, refreshes the SMB mount, moves the file into the PC's INBOX
4. Windows Task Scheduler runs `scribe_engine.py` automatically on boot — it orchestrates two sub-processes: `transcriber.py` handles Whisper on GPU, `labeler.py` handles Ollama labeling and archiving
5. Engine cleans audio, transcribes with Whisper Large-v3, labels automatically with Llama 3, archives transcript and audio
6. PC shuts itself down automatically once INBOX has been empty for 30 minutes and you are not actively using it

You upload the file and walk away. Everything else is automatic.

## Architecture

- **PC (Brain):** RTX 4060 8GB, Windows 11, Faster-Whisper + Ollama
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
    ↓ scribe_engine.py detects file → cleans → transcribes → labels → archives
SSD: C:\FMPC_Scribe\NOTES_Transcripts\course_topic_date.txt
HDD: D:\FMPC_Audio_Archive\Medical\course_topic_date.opus + .txt copy
    ↓ PC shuts down automatically when idle
```

## Key technical decisions

- **Whisper Large-v3 with language=None** — handles French/English/Darija code-switching with a medical glossary initial prompt to prevent phonetic Frenchification of English terms
- **Sequential VRAM handover** — Whisper (~4GB) fully unloads before Llama 3 (~5GB) loads, the only way to run both on 8GB VRAM without crashing
- **Opus 48kbps** — chosen over 32kbps specifically to preserve high-frequency consonant information in medical terminology for future re-processing
- **Paragraph grouping** — segments grouped by silence gaps rather than written line by line, output reads like a document not a log
- **Noise reduction at 0.65** — tuned for noisy lecture hall environments
- **No summarization** — raw transcript preserved exactly as spoken for maximum RAG accuracy. Ask Ollama directly when you want a summary
- **Automatic labeling** — Ollama reads the transcript and generates a structured filename (course_topic_date) automatically, no manual naming ever
- **Dual storage** — transcript saved to SSD for fast RAG access, copied to HDD alongside compressed audio for long-term archive
- **Fully automated wake** — pi_watchdog.py runs as a systemd service on Pi boot, no manual SSH or wake.sh needed for daily use
- **Upload completion check** — Pi watchdog verifies file size is stable before moving to INBOX, preventing partial upload processing

## Pi scripts

**pi_watchdog.py** — permanent background daemon, starts on Pi boot via systemd. Watches `~/fmpc_staging`, automatically wakes PC when a file arrives, waits for boot, refreshes SMB mount, moves file to INBOX. This is the script that makes the daily workflow fully hands-off.

**wake.sh** — manual override script. Use this when you want to wake the PC without uploading a file (remote desktop, checking something, etc). Not needed for normal daily use.

**PC-side scripts** — three files working together:
- `scribe_engine.py` — watchdog and orchestrator, runs permanently, launches the other two
- `transcriber.py` — isolated GPU process, runs Whisper Large-v3, saves result to temp JSON and exits. Isolated to avoid a Windows/ctranslate2 CUDA cleanup bug (exit code 0xC0000005)
- `labeler.py` — CPU-only process, reads transcript JSON, calls Ollama for labeling, saves to SSD and HDD, archives audio as Opus 48kbps

## Storage structure

```
SSD — C:\FMPC_Scribe\NOTES_Transcripts\
    hematology_erythrocyte-disorders_2025-10-15.txt
    cardiology_cardiac-cycle_2025-10-17.txt

HDD — D:\FMPC_Audio_Archive\Medical\
    hematology_erythrocyte-disorders_2025-10-15.txt  (transcript copy)
    hematology_erythrocyte-disorders_2025-10-15.opus (compressed audio)
    cardiology_cardiac-cycle_2025-10-17.txt
    cardiology_cardiac-cycle_2025-10-17.opus
```

## Stack

- Faster-Whisper (Large-v3, CUDA 12.1)
- Ollama + Llama 3 (local, labeling only)
- FFmpeg (Opus 48kbps compression)
- Noisereduce + Librosa (audio cleaning)
- Tailscale (VPN)
- etherwake (Wake-on-LAN)
- SMB/CIFS (Pi to PC file transfer)
- systemd (Pi watchdog service)
- Two-process GPU isolation (transcriber + labeler) — Windows ctranslate2 CUDA fix

## Planned features

- **Anki card generation** — Llama 3 generates front/back flashcards from transcript + lecture PDF, pushed directly into Anki via AnkiConnect
- **RAG system** — ChromaDB indexes raw transcripts and slide PDFs (not summaries, to preserve accuracy). When asking questions, Llama 3 retrieves the most relevant chunks from what my professor actually said, following a strict priority hierarchy: transcribed lecture first, slides second, general knowledge as last resort fallback. Any contradictions detected between the transcript and the slides get flagged explicitly in the response. Built from the transcripts generated by this engine
- **MCQ interface** — Streamlit-based interface on top of the RAG system for practicing past exam questions. Inaccuracies in existing platforms have been a known issue — this version grounds every explanation directly in your own lecture material, sourced from what your professor actually taught

## Status

Hardware assembly in progress. Code complete and documented.

## Why I built this

I got tired of relistening to 2-hour lectures to find one concept I missed. I'd rather spend that time building the tool that does it for me.

