@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo ERROR: Extract this patch inside the NSCLC_UPerNet project folder.
  pause
  exit /b 1
)
set HF_HUB_OFFLINE=1
for %%F in (0 1 2 3 4) do (
  echo ============================================================
  echo VALIDATION-CONTROLLED RARE-TISSUE REFINEMENT - FOLD %%F OF 4
  echo ============================================================
  ".venv\Scripts\python.exe" "src\14_refine_grouped.py" --config "config_refined_fold%%F.json"
  if errorlevel 1 (
    echo REFINEMENT STOPPED ON FOLD %%F. Send a screenshot of this window.
    pause
    exit /b 1
  )
)
echo ALL FIVE REFINED MODELS FINISHED SUCCESSFULLY.
pause
