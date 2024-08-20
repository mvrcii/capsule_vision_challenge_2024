import argparse
import re
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description='Run a single W&B training on SLURM.')
    parser.add_argument('config', type=str, help='Path to configuration file (e.g. configs/runs/worthy-sweep-8.yaml)')
    parser.add_argument('gpu', type=int, choices=[0, 1], help="Choose GPU: 0=rtx2080ti, 1=rtx3090")
    parser.add_argument('-a', '--attach', action='store_true', help="Attach to log output.")
    args = parser.parse_args()

    gpu_types = {0: "rtx2080ti", 1: "rtx3090"}
    gpu = f'--gres=gpu:{gpu_types[args.gpu]}:1'

    command = f'--wrap="python train.py --config {args.config}"'

    slurm_cmd = f'sbatch -p ls6 -J "train" {gpu} {command} -o "logs/slurm-%j.out"'
    # sbatch -p ls6 -J "e37fd6" --gres=gpu:rtx3090:1 --wrap="python train.py --train_bs 64 --val_bs 64 --seed 42 --model_arch 'regnety_640.seer' --fold_id 3 --max_epochs 100" -o "logs/slurm-%j.out"
    result = subprocess.run(slurm_cmd, shell=True, capture_output=True, text=True)

    if result.stdout:
        match = re.search(r"Submitted batch job (\d+)", result.stdout)
        if match:
            job_id = match.group(1)
            print(f"Slurm job ID: {job_id}")

            if args.attach:
                print("Attaching to log file... (waiting 5s)")
                time.sleep(5)
                tail_cmd = f"tail -f logs/slurm-{job_id}.out"
                subprocess.run(tail_cmd, shell=True)
        else:
            print("Failed to submit job to Slurm or parse job ID.")


if __name__ == "__main__":
    main()
