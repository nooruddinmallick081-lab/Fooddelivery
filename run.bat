@echo off
REM ============================================================================
REM  FeedForward launcher for Windows.
REM  Just double-click this file. First run sets up a small local toolbox
REM  (a Python "venv") and installs the 4 packages the app needs, then opens
REM  it in your browser. Every run after that is fast (the toolbox is reused).
REM ============================================================================

setlocal
cd /d "%~dp0"

echo.
echo  FeedForward - Smart Delivery ETA
echo  ----------------------------------
echo.

REM --- find a Python we can use -----------------------------------------------
where py >nul 2>nul
if %errorlevel%==0 (
    set PYLAUNCHER=py
) else (
    where python >nul 2>nul
    if %errorlevel%==0 (
        set PYLAUNCHER=python
    ) else (
        echo [ERROR] Python was not found on this computer.
        echo Please install Python 3.10+ from https://www.python.org/downloads/
        echo ^(tick "Add python.exe to PATH" during install^) and run this again.
        pause
        exit /b 1
    )
)

REM --- create the venv if it doesn't exist yet --------------------------------
if not exist ".venv\Scripts\python.exe" (
    echo [1/3] First time setup - creating a local Python toolbox, one moment...
    %PYLAUNCHER% -m venv .venv
    if not exist ".venv\Scripts\python.exe" (
        echo [ERROR] Could not create the virtual environment. Is Python installed correctly?
        pause
        exit /b 1
    )
)

REM --- install requirements (marker file avoids reinstalling every run) ------
if not exist ".venv\feedforward_installed.marker" (
    echo [2/3] Installing app requirements ^(needs internet, only happens once^)...
    ".venv\Scripts\python.exe" -m pip install --quiet --upgrade pip
    ".venv\Scripts\python.exe" -m pip install --quiet -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Installing requirements failed - check your internet connection.
        pause
        exit /b 1
    )
    echo done > ".venv\feedforward_installed.marker"
) else (
    echo [2/3] Requirements already installed, skipping.
)

REM --- skip Streamlit's first-run "enter your email" prompt ------------------
if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
    if not exist "%USERPROFILE%\.streamlit" mkdir "%USERPROFILE%\.streamlit"
    echo [general] > "%USERPROFILE%\.streamlit\credentials.toml"
    echo email = "" >> "%USERPROFILE%\.streamlit\credentials.toml"
)

REM --- launch --------------------------------------------------------------
echo [3/3] Starting FeedForward... a browser tab will open automatically.
echo        ^(close this window to stop the app^)
echo.
".venv\Scripts\python.exe" -m streamlit run "frontend\app.py" --browser.gatherUsageStats false

pause
