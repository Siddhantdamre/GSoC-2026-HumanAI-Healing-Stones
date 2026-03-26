@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo [GSoC Phase 2] Starting ML Model Training (PyTorch)...
echo ---------------------------------------
echo This script defines the PointNet architecture and backpropagates
echo against the shattered 3D artifacts dataset to learn transformations.
echo.
echo Installing PyTorch if missing...
if exist "C:\Python\python.exe" (
    "C:\Python\python.exe" -m pip install torch torchvision --quiet
    "C:\Python\python.exe" src\train_model.py
) else (
    py -m pip install torch torchvision --quiet
    py src\train_model.py
)

echo.
echo Training finished.
pause
