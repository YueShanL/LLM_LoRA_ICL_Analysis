@echo off
setlocal EnableDelayedExpansion

set ROOT=experiments\lora_selected_tasks_instruct_rawchat_r8_20260709
set PY=venv\Scripts\python.exe
set ARGS=--max-samples 16 --seed 13 --dtype bfloat16 --device cuda

for /d %%D in ("%ROOT%\*") do (
  if exist "%%~fD\config.json" (
    echo === START %%~nxD RQ2 !DATE! !TIME! ===
    "%PY%" -u -m lora_instruction_analysis.experiment.run_rq2 --run-dir "%%~fD" %ARGS%
    if errorlevel 1 (
      echo === FAILED %%~nxD RQ2 exit=!ERRORLEVEL! !DATE! !TIME! ===
      exit /b !ERRORLEVEL!
    )

    echo === START %%~nxD RQ2.1 !DATE! !TIME! ===
    "%PY%" -u -m lora_instruction_analysis.experiment.run_rq21 --run-dir "%%~fD" %ARGS%
    if errorlevel 1 (
      echo === FAILED %%~nxD RQ2.1 exit=!ERRORLEVEL! !DATE! !TIME! ===
      exit /b !ERRORLEVEL!
    )

    echo === DONE %%~nxD RQ2/RQ2.1 !DATE! !TIME! ===
  )
)

echo === ALL DONE !DATE! !TIME! ===
