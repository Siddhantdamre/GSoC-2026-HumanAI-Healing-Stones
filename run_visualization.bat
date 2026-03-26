@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo Starting Mayan Fragment Visualizer...
echo ---------------------------------------
echo This script will:
echo 1. Load 2 fragments and save a combined file.
echo 2. Downsample them to save memory
echo 3. Save a combined model: combined_mayan_stone.ply
echo 4. Open the 3D viewer
echo.
echo Please wait, this will take several minutes...
echo.

:: Use the verified Python installation
if exist "C:\Python\python.exe" (
    echo Using verified Python at C:\Python...
    "C:\Python\python.exe" src\visualize_fragments.py
) else (
    :: Fallback to PATH commands
    where py >nul 2>nul
    if %ERRORLEVEL% equ 0 (
        py src\visualize_fragments.py
    ) else (
        where python >nul 2>nul
        if %ERRORLEVEL% equ 0 (
            python src\visualize_fragments.py
        ) else (
            echo.
            echo ERROR: Could not find Python. 
            echo Please restart your computer for the PATH changes to take effect.
        )
    )
)

echo.
echo Execution finished.
pause
