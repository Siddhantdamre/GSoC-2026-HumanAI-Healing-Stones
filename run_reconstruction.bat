@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo Starting Global Reconstruction (22 Fragments)...
echo ---------------------------------------
echo This tool will iteratively align and snap all 22 pieces together.
echo This can take 5-10 minutes depending on your CPU/RAM.
echo Please wait...
echo.

if exist "C:\Python\python.exe" (
    "C:\Python\python.exe" src\reconstruct_mayan_stone.py
) else (
    py src\reconstruct_mayan_stone.py
)

echo.
echo Reconstruction finished.
pause
