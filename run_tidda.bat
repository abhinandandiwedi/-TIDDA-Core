@echo off
setlocal enabledelayedexpansion
title TIDDA C2 Launcher
color 0A

:: ---- Configuration ----
set "ROOT=%~dp0"
set "VENV=%ROOT%.venv"
set "PYTHON=%VENV%\Scripts\python.exe"
set "SERVER=%ROOT%tidda_c2.py"
set "FRONTEND=%ROOT%html\TIDDA_GCS_real h ai wala.html"
set "PORT=8000"

echo.
echo ============================================
echo   TIDDA C2 - One Click Launcher
echo ============================================
echo.
echo   Project : %ROOT%
echo   Python  : %PYTHON%
echo   Server  : %SERVER%
echo   Frontend: %FRONTEND%
echo.

:: ---- Verify venv exists ----
if not exist "%PYTHON%" (
    echo [ERROR] python.exe not found at:
    echo         %PYTHON%
    echo.
    echo   Make sure you have a .venv folder with:
    echo     python -m venv .venv
    echo.
    pause
    exit /b 1
)
echo [OK] Found venv python: %PYTHON%

:: ---- Verify server script exists ----
if not exist "%SERVER%" (
    echo [ERROR] tidda_c2.py not found at:
    echo         %SERVER%
    pause
    exit /b 1
)
echo [OK] Found server script: %SERVER%

:: ---- Check for port conflict on 8000 ----
echo.
echo   Checking port %PORT%...
set "PORT_BUSY=0"
for /f "tokens=5" %%P in ('netstat -aon 2^>nul ^| findstr ":%PORT% " ^| findstr "LISTENING"') do (
    set "PORT_BUSY=1"
    set "BUSY_PID=%%P"
)

if "!PORT_BUSY!"=="1" (
    echo [WARNING] Port %PORT% is already in use by PID !BUSY_PID!
    echo.
    set /p "KILL_CHOICE=   Kill existing process? (Y/N): "
    if /i "!KILL_CHOICE!"=="Y" (
        taskkill /F /PID !BUSY_PID! >nul 2>&1
        echo   [OK] Killed PID !BUSY_PID!
        timeout /t 2 /nobreak >nul
    ) else (
        echo   [ABORT] Cannot start -- port %PORT% in use.
        pause
        exit /b 1
    )
)
echo [OK] Port %PORT% is free.

:: ---- Activate venv ----
echo.
echo   Activating virtual environment...
call "%VENV%\Scripts\activate.bat"
if errorlevel 1 (
    echo [ERROR] Failed to activate venv.
    pause
    exit /b 1
)
echo [OK] venv activated.

:: ---- Launch frontend in default browser ----
echo.
echo   Opening GCS Frontend in browser...
if exist "%FRONTEND%" (
    start "" "%FRONTEND%"
    echo [OK] Frontend launched in browser.
) else (
    echo [WARNING] Frontend HTML not found at:
    echo          %FRONTEND%
    echo          Continuing with backend only...
)

:: ---- Start backend server ----
echo.
echo ============================================
echo   Starting TIDDA C2 Server on port %PORT%
echo   Press Ctrl+C to stop the server.
echo ============================================
echo.

"%PYTHON%" "%SERVER%"

:: ---- Server stopped ----
echo.
echo ============================================
echo   Server has stopped.
echo ============================================
pause
