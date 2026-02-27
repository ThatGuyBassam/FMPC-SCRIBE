# FMPC Scribe

A fully local, automated medical lecture transcription and summarization engine I built for myself as a first-year medical student at FMPC Casablanca.

## The problem it solves

2-hour medical lectures in French, English, and Darija. Dense terminology. No time to relisten. I wanted to drop an audio file on my phone and wake up with a clean transcript, a structured summary, and my audio archived — without paying for cloud APIs or sending patient-adjacent data anywhere.

## How it works

1. Record lecture on phone (any format — M4A, MP3, WAV)
2. SSH into Raspberry Pi via Tailscale from anywhere
3. Run `wake.sh` — Pi sends a Magic Packet over a direct Ethernet bridge to wake the PC
4. Upload audio via SFTP to the Pi, which drops it into the PC's INBOX via a mounted SMB share
5. Windows Task Scheduler runs `scribe_engine.py` automatically on boot
6. Engine cleans audio, transcribes with Whisper Large-v3, summarizes with Llama 3, archives to Opus 48kbps

Everything runs locally on an RTX 4060. No cloud, no subscriptions, no data leaving the machine.

## Architecture

- **PC (Brain):** RTX 4060 8GB, Windows 11, Faster-Whisper + Ollama
- **Raspberry Pi 4 (Gateway):** 24/7 uptime, Wake-on-LAN sender, file relay
- **Network:** Direct Cat6 Ethernet bridge (192.168.2.x) between Pi and PC for file transfer, home Wi-Fi for internet/Tailscale
- **Remote access:** Tailscale mesh VPN — works from anywhere on any network

## Key technical decisions

- **Whisper Large-v3 with language=None** — handles French/English code-switching with a medical glossary initial prompt to prevent phonetic Frenchification of English terms
- **Sequential VRAM handover** — Whisper (~4GB) fully unloads before Llama 3 (~5GB) loads, the only way to run both on 8GB VRAM without crashing
- **Opus 48kbps** — chosen over 32kbps specifically to preserve high-frequency consonant information in medical terminology for future re-processing
- **Paragraph grouping** — segments are grouped by silence gaps rather than written line by line, output reads like a document not a log
- **Noise reduction at 0.65** — tuned for noisy lecture hall environments

## Stack

- Faster-Whisper (Large-v3, CUDA 12.1)
- Ollama + Llama 3 (local)
- FFmpeg (Opus compression)
- Noisereduce + Librosa (audio cleaning)
- Tailscale (VPN)
- etherwake (Wake-on-LAN)
- SMB/CIFS (Pi to PC file transfer)

## Status

Hardware assembly in progress. Code complete and documented.

## Why I built this

I got tired of relistening to 2-hour lectures to find one concept I missed. I'd rather spend that time building the tool that does it for me.
