Set-Location "E:\IdeaProject\Pycharm\LoRA_Instruction_analysis"
& ".\venv\Scripts\Activate.ps1"
python "scripts\hpc_task_pipeline.py" --config "configs\gemma4_e4b_icl8_pipeline.json" 1> "build\gemma4_e4b_icl8_pipeline.out.log" 2> "build\gemma4_e4b_icl8_pipeline.err.log"
