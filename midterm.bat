@echo off
setlocal enableextensions
title VoxScript - midterm
cd /d "%~dp0"

rem --- Configurable paths ---
set "INPUT=E:\S01_E01_Pilot.mp4"
set "SUBTITLE=E:\S01_E01_Pilot.ass"
set "LANGUAGE=en"
set "OUTPUT=E:\\"

rem --- Hugging Face mirror (used when model needs downloading) ---
set "HF_ENDPOINT=https://hf-mirror.com"

rem --- Validate input files ---
if not exist "%INPUT%" (
    echo [ERROR] Input file not found: "%INPUT%"
    goto :fail
)
if not exist "%SUBTITLE%" (
    echo [ERROR] Subtitle file not found: "%SUBTITLE%"
    goto :fail
)

rem --- Check uv availability ---
where uv >nul 2>&1 || (
    echo [ERROR] "uv" not found on PATH. Install from https://docs.astral.sh/uv/
    goto :fail
)

rem --- Run ---
echo Running VoxScript...
uv run starter.py -i "%INPUT%" -s "%SUBTITLE%" -l "%LANGUAGE%" -o "%OUTPUT%" -t
set "EXIT_CODE=%errorlevel%"
if not "%EXIT_CODE%"=="0" (
    echo [ERROR] VoxScript exited with code %EXIT_CODE%.
    goto :fail
)
echo Midterm Test Done.
exit /b 0

:fail
pause
exit /b 1
