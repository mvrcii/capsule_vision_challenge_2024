import argparse
import re
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description='Run a multi-node, multi-GPU training session on SLURM.')
    parser.add_argument('config', type=str, help='Path to configuration file (e.g., configs/runs/worthy-sweep-8.yaml)')
    parser.add_argument('gpu', type=int, choices=[0, 1], help="Choose GPU type: 0=rtx2080ti, 1=rtx3090")
    parser.add_argument('--num_nodes', type=int, default=1, help="Number of nodes")
    parser.add_argument('-n', '--name', type=str, default='train', help="Name of the job.")
    parser.add_argument('-d', '--num_devices', type=int, default=1, help="Number of devices (GPUs) per node")
    parser.add_argument('-m', '--memory', type=str, default='0', help="Memory per node")
    parser.add_argument('-a', '--attach', action='store_true', help="Attach to log output.")
    args = parser.parse_args()

    gpu_types = {0: "rtx2080ti", 1: "rtx3090"}
    gpu = f'--gres=gpu:{gpu_types[args.gpu]}:{args.num_devices}'

    num_device_str = f'--num_devices={args.num_devices}' if args.num_devices > 1 else ''
    num_node_str = f'--num_nodes={args.num_nodes}' if args.num_nodes > 1 else ''

    wrap_command = f"python train.py --config {args.config}"

    # If additional SLURM configurations are needed, append them
    if num_node_str:
        wrap_command += " " + num_node_str
    if num_device_str:
        wrap_command += " " + num_device_str

    slurm_args = [
        f'--nodes={args.num_nodes}' if args.num_nodes > 1 else '',
        f'--ntasks-per-node={args.num_devices}' if args.num_devices > 1 else '',
        gpu,
        '-o "logs/slurm-%j.out"',
        f'--wrap="{wrap_command}"'
    ]

    slurm_cmd = f'sbatch -p ls6 -J "{args.name}" ' + ' '.join(slurm_args)
    # sbatch -p ls6 -J "e37fd6" --gres=gpu:rtx3090:1 --wrap="python train.py --train_bs 64 --val_bs 64 --seed 42 --model_arch 'regnety_640.seer' --fold_id 3 --max_epochs 100" -o "logs/slurm-%j.out"
    # sbatch -p ls6 -J sanity --gres=gpu:rtx4090:1 --wrap="python train.py " -o "logs/slurm-%j.out"
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
    else:
        print(f"Error submitting job: {result.stderr}")


if __name__ == "__main__":
    main()
