import re
import subprocess
import time


def run_on_slurm(python_cmd: str, gpu: int, attach: bool, job_name: str = "bash"):
    gpu_types = {0: "rtx2080ti", 1: "rtx3090"}
    gpu = f'--gres=gpu:{gpu_types[gpu]}:1'
    wrap_cmd = f'--wrap="{python_cmd}"'
    slurm_cmd = f'sbatch -p ls6 -J "{job_name}" {gpu} {wrap_cmd} -o "logs/slurm-%j.out"'

    result = subprocess.run(slurm_cmd, shell=True, capture_output=True, text=True)

    if result.stdout:
        match = re.search(r"Submitted batch job (\d+)", result.stdout)
        if match:
            job_id = match.group(1)
            print(f"Slurm job ID: {job_id}")

            if attach:
                print("Attaching to log file... (waiting 5s)")
                time.sleep(5)
                tail_cmd = f"tail -f logs/slurm-{job_id}.out"
                subprocess.run(tail_cmd, shell=True)
        else:
            print("Failed to submit job to Slurm or parse job ID.")
