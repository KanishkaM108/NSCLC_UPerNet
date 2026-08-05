@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo ERROR: Run this file from the NSCLC_UPerNet project folder.
  pause
  exit /b 1
)
set HF_HUB_OFFLINE=1
for %%F in (0 1 2 3 4) do (
  echo ============================================================
  echo TRAINING DIRECT FIVE-CLASS MODEL - FOLD %%F OF 4
  echo ============================================================
  ".venv\Scripts\python.exe" "src\12_train_grouped.py" --config "config_grouped_fold%%F.json"
  if errorlevel 1 (
    echo TRAINING STOPPED ON FOLD %%F. Send a screenshot of this window.
    pause
    exit /b 1
  )
)
echo ALL FIVE GROUPED MODELS FINISHED SUCCESSFULLY.
pause
