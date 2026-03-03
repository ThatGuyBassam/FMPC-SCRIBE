# FMPC Scribe

Personal project I built to stop wasting time relistening to 2-hour lectures — records, transcribes, cleans, and archives medical lectures fully locally on my computer, woken remotely via a Raspberry Pi from my phone. No cloud, no subscriptions, just my GPU doing the work while I do other things.

## How it works

1. Record lecture on phone (any format — M4A, MP3, WAV)
2. Open Solid Explorer on phone, upload audio to the Pi's staging folder (`~/fmpc_staging`)
3. Pi detects the file automatically, sends a Magic Packet to wake the PC, waits for Windows to boot, refreshes the SMB mount, and moves the file into the PC's INBOX
4. Windows Task Scheduler runs `scribe_engine.py` automatically on boot — it orchestrates two sub-processes: `transcriber.py` handles Qwen3-ASR on GPU, `processor.py` handles Qwen2.5 cleanup, labeling, and archiving
5. Engine cleans audio, transcribes with Qwen3-ASR-1.7B, corrects medical terminology errors with Qwen2.5, labels automatically, archives transcript and audio
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
    ↓ scribe_engine.py detects file → cleans audio → transcribes → corrects → labels → archives
SSD: C:\FMPC_Scribe\NOTES_Transcripts\course_topic_date.txt
HDD: D:\FMPC_Audio_Archive\Medical\course_topic_date.opus + .txt copy
    ↓ PC shuts down automatically when idle
```

## Key technical decisions

- **Qwen3-ASR-1.7B** — ASR-LLM hybrid released January 2026, handles Moroccan French accent and medical terminology significantly better than Whisper Large-v3 on noisy lecture hall recordings
- **142-word medical context prompt** — injected via the `context` parameter to bias the model's phonetic decoding toward FMPC terminology (anatomie, histologie, physiologie, biochimie, hématologie, bactériologie, embryologie, pathologie)
- **Qwen2.5:7b cleanup pass** — after transcription, a second LLM pass through Ollama corrects phonetic errors in medical terms chunk by chunk. Qwen2.5 was chosen over Llama 3 specifically for its native French precision and strict instruction-following — Llama 3 kept anglicizing the text
- **Three-script isolation** — `scribe_engine.py` orchestrates, `transcriber.py` handles GPU transcription, `processor.py` handles everything after. Each runs as a fully isolated subprocess with no shared memory
- **os._exit(0) hard kill** — `transcriber.py` ends with `os._exit(0)` instead of `sys.exit()`. This bypasses all Python and CUDA cleanup and prevents a Windows-specific ctranslate2 Access Violation (exit code 0xC0000005) that crashes the process silently after every transcription. Non-negotiable — removing it brings the crash back immediately
- **Paragraph grouping by silence** — audio split at 2.5s silence gaps rather than line by line, output reads like a document
- **Noise reduction at 0.75** — tuned upward from 0.65 for lecture hall reverberation and background noise
- **Opus 48kbps** — chosen over 32kbps to preserve high-frequency consonant information in medical terminology for future re-processing
- **No summarization** — raw transcript preserved exactly as spoken for maximum RAG accuracy
- **Automatic labeling** — Qwen2.5 reads the cleaned transcript and generates a structured French filename (discipline_sujet_date) automatically, no manual naming ever
- **Dual storage** — transcript saved to SSD for fast RAG access, copied to HDD alongside compressed audio for long-term archive

## The engineering battles

Things that didn't go smoothly and the workarounds that fixed them.

**Whisper Large-v3 was unusable on this audio.** "Vertèbre lombaire" came out as "vertèbre romaine", "aorte abdominale" as "Hortes, Amérique", and a 27-minute segment produced nothing but "Péritoine. Péritoine. Péritoine." in a loop. The combination of Moroccan French accent, lecture hall reverb, and dense medical vocabulary was outside Whisper's reliable range. Switched to Qwen3-ASR-1.7B, which handles all three natively.

**The qwen-asr Python wrapper exposed no prompt parameter.** The whole point of Qwen3-ASR is prompt biasing. After installation, every attempt — `prompt`, `initial_prompt`, `context`, `system_prompt` — threw `TypeError: unexpected keyword argument`. A runtime hack that patched `sys.modules` to inject the prompt into the language validation list also failed because the wrapper silently calls `.capitalize()` on the language argument before checking it, which lowercased the entire injected 500-word prompt and broke the string match. The fix was to just edit the wrapper source directly — two lines in `utils.py` were replaced with `pass` to disable validation entirely, then `context` works as intended.

**Llama 3 anglicized French medical text.** The original cleanup pass used Llama 3, which kept turning "érythrocyte" into "erythrocyte" and rewriting the professor's sentences. Replaced with Qwen2.5:7b, which is multilingual by design and actually follows the "don't rewrite anything" instruction.

## Pi scripts

**pi_watchdog.py** — permanent background daemon, starts on Pi boot via systemd. Watches `~/fmpc_staging`, automatically wakes PC when a file arrives, waits for boot, refreshes SMB mount, moves file to INBOX. This is the script that makes the daily workflow fully hands-off.

**wake.sh** — manual override script. Used when you want to wake the PC without uploading a file. Not needed for normal daily use.

**PC-side scripts** — three files working together:
- `scribe_engine.py` — watchdog and orchestrator, runs permanently, launches the other two
- `transcriber.py` — isolated GPU process, runs Qwen3-ASR-1.7B, saves result to temp JSON, and exits with `os._exit(0)`
- `processor.py` — reads transcript JSON, runs Qwen2.5 cleanup in chunks, generates label, saves to SSD and HDD, archives audio as Opus 48kbps

## Storage structure

```
SSD — C:\FMPC_Scribe\NOTES_Transcripts\
    anatomie_cavite-abdominale_2026-03-02.txt
    histologie_tissu-conjonctif_2026-03-05.txt

HDD — D:\FMPC_Audio_Archive\Medical\
    anatomie_cavite-abdominale_2026-03-02.txt   (transcript copy)
    anatomie_cavite-abdominale_2026-03-02.opus  (compressed audio)
    histologie_tissu-conjonctif_2026-03-05.txt
    histologie_tissu-conjonctif_2026-03-05.opus
```

## Stack

- Qwen3-ASR-1.7B (CUDA, transformers backend)
- Qwen2.5:7b via Ollama (cleanup + labeling)
- FFmpeg (Opus 48kbps compression)
- Noisereduce + Librosa (audio cleaning + normalization)
- Tailscale (VPN)
- etherwake (Wake-on-LAN)
- SMB/CIFS (Pi to PC file transfer)
- systemd (Pi watchdog service)
- Three-process GPU isolation — Windows ctranslate2 CUDA crash fix

## Planned features

- **Anki card generation** — Qwen2.5 generates front/back flashcards from transcript + lecture PDF, pushed directly into Anki via AnkiConnect
- **RAG system** — ChromaDB indexes raw transcripts and slide PDFs. When asking questions, the model retrieves the most relevant chunks from what the professor actually said, following a strict priority hierarchy: transcribed lecture first, slides second, general knowledge as a last resort. Contradictions between the transcript and the professor's courses get flagged explicitly
- **MCQ interface** — Streamlit-based interface on top of the RAG system for practicing past exam questions, grounded by the professor's own lecture material

## Status

Hardware assembly in progress. Code complete and tested on real lectures.

## Why I built this

I got tired of relistening to 2-hour lectures to find one concept I missed. I'd rather spend that time building the tool that does it for me.

