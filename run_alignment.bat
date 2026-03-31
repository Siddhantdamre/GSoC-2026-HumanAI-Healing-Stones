@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ========================================================
echo   Healing Stones - Fragment Alignment (Best Pair Finder)
echo ========================================================
echo.

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

!PYTHON_CMD! src\align_fragments.py --best

echo.
echo Alignment finished.
pause
