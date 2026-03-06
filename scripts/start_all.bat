@echo off
title FMPC Scribe Launcher
cd C:\FMPC_Scribe

echo Starting FMPC Scribe components...
echo.

:: Slide Watcher
start "FMPC - Slide Watcher" cmd /k "cd C:\FMPC_Scribe && py -3.11 ingest_slides.py watch"

:: RAG Engine
start "FMPC - RAG Assistant" cmd /k "cd C:\FMPC_Scribe && py -3.11 -m uvicorn api:app --host 0.0.0.0 --port 8000"

:: Scribe Engine (last — heaviest, starts after others are up)
timeout /t 3 /nobreak >nul
start "FMPC - Scribe Engine" cmd /k "cd C:\FMPC_Scribe && py -3.11 scribe_engine.py"

echo.
echo All components launched in separate windows.
echo.
echo   [Scribe Engine]  Drop audio in C:\FMPC_Scribe\INBOX
echo   [Slide Watcher]  Drop slides in C:\FMPC_Scribe\slides\discipline\
echo   [RAG Assistant]  Open browser at http://localhost:8000
echo.
pause
