@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo [GSoC Phase 1] Starting ML Data Augmentation Pipeline...
echo ---------------------------------------
echo This script will take a complete artifact, digitally shatter it into
echo random chunks, and scramble them to create a Machine Learning dataset.
echo.

if exist "C:\Python\python.exe" (
    "C:\Python\python.exe" src\augment_data.py
) else (
    py src\augment_data.py
)

echo.
echo Augmentation finished.
pause
