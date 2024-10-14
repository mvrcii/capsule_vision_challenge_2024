import argparse
import re
import subprocess
import time


def train_sweep():
    """
    Parses command-line arguments to submit a W&B (Weights & Biases) sweep job to a SLURM cluster and optionally
    attaches to the job's log file in real-time.

    This function submits a SLURM job based on the W&B sweep ID provided by the user and assigns the job to a GPU
    resource on the cluster. After submitting the job, the function optionally waits for 10 seconds and then
    tails the job's log file if the `-a` argument is supplied.

    Command-line arguments:
        - sweep_id (str): Required. The W&B sweep ID to run. Example: mvrcii_/SEER/pc6pxow4.
        - gpu (int): Required. Selects which GPU to use: 0 for rtx2080ti, 1 for rtx3090.
        - name (str): Optional. Custom job name for the SLURM scheduler. Default is 'sweep'.
        - a: Optional. Attaches to the log output of the job. Waits 10 seconds before tailing the log.

    Example usage:
        $ python slurm/train_sweep.py <name>/<project>/<sweep_id> <gpu_type> -n <job_name> -a

    """
    parser = argparse.ArgumentParser(description='Run a W&B sweep on SLURM.')
    parser.add_argument('sweep_id', type=str, help='ID of a W&B sweep (e.g. mvrcii_/SEER/pc6pxow4)')
    parser.add_argument('gpu', type=int, choices=[0, 1], help="Choose GPU: 0=rtx2080ti, 1=rtx3090")
    parser.add_argument('-n', '--name', type=str, default='sweep', help="Name of the job.")
    parser.add_argument('-a', action='store_true', help="Attach to log output.")
    args = parser.parse_args()

    gpu_types = {0: "rtx2080ti", 1: "rtx3090"}
    gpu = f'--gres=gpu:{gpu_types[args.gpu]}:1'
    command = f'--wrap="wandb agent {args.sweep_id}"'

    slurm_cmd = f'sbatch -p ls6 -J "{args.name}" {gpu} {command} -o "logs/slurm-%j.out"'

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
    train_sweep()
