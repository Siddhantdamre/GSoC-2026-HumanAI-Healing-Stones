@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ========================================================
echo   GSoC 2026: HumanAI Healing Stones - MASTER PIPELINE
echo ========================================================
echo.

:: Detect Python
set PYTHON_CMD=
if exist "C:\Python\python.exe" (
    set PYTHON_CMD="C:\Python\python.exe"
) else (
    where py >nul 2>nul
    if !ERRORLEVEL! equ 0 (
        set PYTHON_CMD=py
    ) else (
        where python >nul 2>nul
        if !ERRORLEVEL! equ 0 (
            set PYTHON_CMD=python
        ) else (
            echo ERROR: Could not find Python.
            pause
            exit /b 1
        )
    )
)

echo [1/4] Starting Synthetic Data Augmentation...
!PYTHON_CMD! src\augment_data.py
if %ERRORLEVEL% NEQ 0 (
    echo Error during Data Augmentation.
    pause
    exit /b 1
)

echo.
echo [2/4] Training PointNet SE(3) Deep Learning Model (100 Epochs)...
!PYTHON_CMD! src\train_model.py
if %ERRORLEVEL% NEQ 0 (
    echo Error during Deep Learning Training.
    pause
    exit /b 1
)

echo.
echo [3/4] Generating AI Evaluation Integrity Metrics + Loss Curve...
!PYTHON_CMD! src\evaluate.py
if %ERRORLEVEL% NEQ 0 (
    echo Error during Evaluation.
    pause
    exit /b 1
)

echo.
echo [4/4] Running Geometric Fragment Alignment (FPFH + RANSAC + ICP)...
echo       Auto-scanning all fragments to find best adjacent pair...
!PYTHON_CMD! src\align_fragments.py --best
if %ERRORLEVEL% NEQ 0 (
    echo Error during Alignment.
    pause
    exit /b 1
)

echo.
echo ========================================================
echo  PIPELINE COMPLETE - ALL SYSTEMS GO
echo ========================================================
pause
