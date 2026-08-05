@echo off
setlocal
cd /d "%~dp0"
set HF_HUB_OFFLINE=1
".venv\Scripts\python.exe" "src\13_evaluate_grouped_ensemble.py" --config "config_grouped_fold0.json"
if errorlevel 1 (
  echo EVALUATION STOPPED. Send a screenshot of this window.
) else (
  echo EVALUATION COMPLETE. The final line says whether BOTH 85 percent targets passed.
)
pause
