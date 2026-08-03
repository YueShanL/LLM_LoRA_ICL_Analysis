import subprocess, sys
from pathlib import Path
root = Path('experiments/lora_selected_tasks_instruct_rawchat_r8_20260709')
runs = [p/'states'/'rq1' for p in root.iterdir() if (p/'states'/'rq1'/'metrics.jsonl').exists()]
sae_dir = Path('experiments/sae_llama32_3b_instruct/residual')
cmd = [sys.executable, '-m', 'lora_instruction_analysis.model.sae_fit']
for run in runs:
    cmd += ['--run', str(run)]
cmd += ['--output-dir', str(sae_dir), '--mode', 'residual', '--features', '1024', '--max-vectors', '20000', '--epochs', '5', '--batch-size', '256', '--device', 'cuda']
subprocess.check_call(cmd)
for run in runs:
    out = run.parents[1] / 'plots' / 'rq1_sae'
    subprocess.check_call([sys.executable, '-m', 'lora_instruction_analysis.model.sae_analysis', '--run', str(run), '--sae-path', str(sae_dir/'sae.pt'), '--output-dir', str(out), '--mode', 'residual', '--top-k', '20'])
print('residual SAE pipeline complete')
