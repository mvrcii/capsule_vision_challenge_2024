import argparse
import os
import re
from collections import namedtuple, Counter

CheckpointMetadata = namedtuple('CheckpointMetadata', ['rel_path', 'base_name', 'wandb_name'])


def find_ckpt_files(dir):
    ckpt_files = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith('.ckpt'):
                ckpt_files.append(os.path.join(root, file))
    return ckpt_files


def get_base_name(path):
    if path.startswith('checkpoints/sweep'):
        basename = os.path.basename(path)
        wandb_name = basename.split('_')[0]
        return CheckpointMetadata(path, basename, wandb_name)
    if path.startswith('checkpoints/run'):
        pattern = r"run-\d{8}_\d{6}-(\w+-\w+-\d+)"
        match = re.search(pattern, path)
        if match:
            wandb_name = match.group(1)
        else:
            print("No match found for run.")
            raise ValueError("Es knallt!")
        basename = os.path.basename(path)
        return CheckpointMetadata(path, basename, wandb_name)
    else:
        raise ValueError("Es knallt gewaltig!")


def check_val_results(checkpoint_dir):
    ckpt_dirs = find_ckpt_files(checkpoint_dir)
    ckpt_infos = [get_base_name(ckpt) for ckpt in ckpt_dirs]
    print("\n".join(map(str, ckpt_infos)))
    assert Counter(map(lambda x: x.wandb_name, ckpt_infos)).most_common(1)[0][1] == 1, "Duplicate W&B names found!"


if __name__ == '__main__':
    argparse = argparse.ArgumentParser(description='Check validation results.')
    argparse.add_argument('--checkpoint_dir', default="checkpoints", type=str,
                          help='Path to the checkpoint directory.')
    args = argparse.parse_args()

    check_val_results(args.checkpoint_dir)
