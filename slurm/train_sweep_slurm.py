import argparse
import re
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description='Run a W&B sweep on SLURM.')
    parser.add_argument('sweep_id', type=str, help='ID of a W&B sweep (e.g. mvrcii_/SEER/pc6pxow4)')
    parser.add_argument('gpu', type=int, choices=[0, 1], help="Choose GPU: 0=rtx2080ti, 1=rtx3090")
    parser.add_argument('--a', action='store_true', help="Attach to log output.")
    args = parser.parse_args()

    gpu_types = {0: "rtx2080ti", 1: "rtx3090"}
    gpu = f'--gres=gpu:{gpu_types[args.gpu]}:1'
    command = f'--wrap="wandb agent {args.sweep_id}"'

    slurm_cmd = f'sbatch -p ls6 -J "sweep" {gpu} {command} -o "logs/slurm-%j.out"'

    result = subprocess.run(slurm_cmd, shell=True, capture_output=True, text=True)

    if result.stdout:
        match = re.search(r"Submitted batch job (\d+)", result.stdout)
        if match:
            job_id = match.group(1)
            print(f"Slurm job ID: {job_id}")

            if args.a:
                print("Attaching to log file... (waiting 10s)")
                time.sleep(10)
                tail_cmd = f"tail -f logs/slurm-{job_id}.out"
                subprocess.run(tail_cmd, shell=True)
        else:
            print("Failed to submit job to Slurm or parse job ID.")


if __name__ == "__main__":
    main()
