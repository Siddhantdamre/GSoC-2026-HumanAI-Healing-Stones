@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo Starting Fragment Alignment Pipeline (ICP)...
echo ---------------------------------------
echo This tool will align Fragment 01 and Fragment 02 and show them snapped together.
echo.
echo Please wait, feature matching takes a moment...
echo.

if exist "C:\Python\python.exe" (
    "C:\Python\python.exe" src\align_fragments.py
) else (
    py src\align_fragments.py
)

echo.
echo Alignment finished.
pause
