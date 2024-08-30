import argparse
import os
import re
from collections import namedtuple, Counter

CheckpointMetadata = namedtuple('CheckpointMetadata', ['rel_path', 'base_name', 'wandb_name', 'sweep_id'])


def find_ckpt_files(dir):
    ckpt_files = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith('.ckpt'):
                ckpt_files.append(os.path.join(root, file))
    return ckpt_files


def get_base_name(path):
    if path.startswith('checkpoints/sweep'):
        # Extract sweep ID
        pattern = r"sweep-(\w+)/"
        match = re.search(pattern, path)
        if match:
            sweep_id = match.group(1)
            print("Extracted Sweep ID:", sweep_id)
        else:
            print("No sweep id found for sweep run.")
            raise ValueError("Es knallt!")

        basename = os.path.basename(path)
        wandb_name = basename.split('_')[0]
        return CheckpointMetadata(path, basename, wandb_name, sweep_id)

    if path.startswith('checkpoints/run'):
        pattern = r"run-\d{8}_\d{6}-(\w+-\w+-\d+)"
        match = re.search(pattern, path)
        if match:
            wandb_name = match.group(1)
        else:
            print("No wandb name found for run.")
            raise ValueError("Es knallt!")
        basename = os.path.basename(path)
        return CheckpointMetadata(path, basename, wandb_name, None)
    else:
        raise ValueError("Es knallt gewaltig!")


def check_val_results(checkpoint_dir):
    ckpt_dirs = find_ckpt_files(checkpoint_dir)
    ckpt_infos = [get_base_name(ckpt) for ckpt in ckpt_dirs]
    print("\n".join(map(str, ckpt_infos)))

    x = set()
    for wandb_name, sweep_id in list(map(lambda _x: (_x.wandb_name, _x.sweep_id), ckpt_infos)):
        if (wandb_name, sweep_id) in x:
            print(f"Duplicate found: {wandb_name}, {sweep_id}")
        x.add((wandb_name, sweep_id))

if __name__ == '__main__':
    argparse = argparse.ArgumentParser(description='Check validation results.')
    argparse.add_argument('--checkpoint_dir', default="checkpoints", type=str,
                          help='Path to the checkpoint directory.')
    args = argparse.parse_args()

    check_val_results(args.checkpoint_dir)
