import subprocess, sys
from pathlib import Path
root = Path('experiments/lora_selected_tasks_instruct_rawchat_r8_20260709')
runs = [p/'states'/'rq21' for p in root.iterdir() if (p/'states'/'rq21'/'metrics.jsonl').exists()]
configs = [
    ('attention_outputs', Path('experiments/sae_llama32_3b_instruct/attention_outputs'), '512', 'attention_outputs_sae'),
    ('attention_post_o_proj_outputs', Path('experiments/sae_llama32_3b_instruct/attention_post_o_proj_outputs'), '1024', 'attention_post_o_proj_outputs_sae'),
]
for mode, sae_dir, features, plot_name in configs:
    cmd = [sys.executable, '-m', 'lora_instruction_analysis.model.sae_fit']
    for run in runs:
        cmd += ['--run', str(run)]
    cmd += ['--output-dir', str(sae_dir), '--mode', mode, '--features', features, '--max-vectors', '20000', '--epochs', '5', '--batch-size', '256', '--device', 'cuda']
    subprocess.check_call(cmd)
    for run in runs:
        out = run.parents[1] / 'plots' / 'rq21' / plot_name
        subprocess.check_call([sys.executable, '-m', 'lora_instruction_analysis.model.sae_analysis', '--run', str(run), '--sae-path', str(sae_dir/'sae.pt'), '--output-dir', str(out), '--mode', mode, '--top-k', '20'])
print('attention SAE pipeline complete')
