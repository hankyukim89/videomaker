@echo off
setlocal
cd /d "%~dp0"
set PY=python
python --version >nul 2>nul
if errorlevel 1 set PY=py
%PY% --version >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install it from https://www.python.org/downloads/
  echo and tick "Add python.exe to PATH" during install, then run this again.
  pause
  exit /b 1
)
if not exist .venv (
  echo First run: setting up environment, this takes a minute or two...
  %PY% -m venv .venv
  if errorlevel 1 goto fail
)
call .venv\Scripts\pip install -q -r requirements.txt
if errorlevel 1 goto fail
echo.
echo Starting AI Video Maker at http://127.0.0.1:8765
echo Keep this window open while you use the app. Close it to stop.
echo.
.venv\Scripts\python server.py
echo.
echo Server stopped.
pause
exit /b 0
:fail
echo.
echo Setup failed - see the messages above.
pause
