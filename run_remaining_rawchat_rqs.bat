@echo off
setlocal EnableDelayedExpansion

set ROOT=experiments\lora_selected_tasks_instruct_rawchat_r8_20260709
set PY=venv\Scripts\python.exe
set ARGS=--max-samples 16 --seed 13 --dtype bfloat16 --device cuda

for %%T in (
  formal_language_a_n_b_n
  has_repeated_word
  reverse_words
  uppercase_last_word
  words_containing_bigram_qu
  words_starting_with_letter
) do (
  echo === START %%T !DATE! !TIME! ===
  "%PY%" -u -m lora_instruction_analysis.experiment.run_all_rqs --run-dir "%ROOT%\%%T" %ARGS%
  if errorlevel 1 (
    echo === FAILED %%T exit=!ERRORLEVEL! !DATE! !TIME! ===
    exit /b !ERRORLEVEL!
  )
  echo === DONE %%T !DATE! !TIME! ===
)

echo === ALL DONE !DATE! !TIME! ===
