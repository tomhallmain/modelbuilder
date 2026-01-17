@echo off
REM Batch file wrapper for Model Builder CLI
REM This allows calling 'mb' as an executable on Windows

REM Set Python executable path
set PYTHON_EXE=C:\Users\tehal\miniconda\envs\modelbuilder\python.exe

REM Check if Python executable exists
if not exist "%PYTHON_EXE%" (
    echo ERROR: Python executable not found at: %PYTHON_EXE%
    echo Please update PYTHON_EXE in mb.bat to point to your Python installation
    exit /b 1
)

REM Call mb CLI module
"%PYTHON_EXE%" -m mb.cli %*
