@echo off
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" "src\15_check_refinement.py"
if errorlevel 1 (
  echo VALIDATION CHECK FAILED. Send a screenshot of this window.
)
pause
