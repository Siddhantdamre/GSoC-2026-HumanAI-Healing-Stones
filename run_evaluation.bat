@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo [GSoC Phase 3] Starting ML Evaluation Metrics...
echo ---------------------------------------
echo This script loads the trained weights, performs inference,
echo and outputs the 'Integrity of generated data' metrics required.
echo.
echo Installing Matplotlib if missing...
if exist "C:\Python\python.exe" (
    "C:\Python\python.exe" -m pip install matplotlib --quiet
    "C:\Python\python.exe" src\evaluate.py
) else (
    py -m pip install matplotlib --quiet
    py src\evaluate.py
)

echo.
echo Evaluation finished. Check models/training_loss_curve.png for the plot.
pause
