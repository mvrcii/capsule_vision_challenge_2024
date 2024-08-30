import argparse
import logging
import os
import re
from collections import namedtuple, Counter

import wandb

CheckpointMetadata = namedtuple('CheckpointMetadata', ['rel_path', 'base_name', 'wandb_name', 'sweep_id', 'run_id'])


def find_files_with_ending(dir, ending='.ckpt'):
    ckpt_files = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(ending):
                ckpt_files.append(os.path.join(root, file))
    return ckpt_files


def get_base_name(path):
    if path.startswith('checkpoints/sweep'):
        # Extract sweep ID
        pattern = r"sweep-(\w+)/"
        match = re.search(pattern, path)
        if match:
            sweep_id = match.group(1)
        else:
            print("No sweep id found for sweep run.")
            raise ValueError("Es knallt!")

        basename = os.path.basename(path)
        wandb_name = basename.split('_')[0]
        return CheckpointMetadata(path, basename, wandb_name, sweep_id, None)

    if path.startswith('checkpoints/run'):
        pattern = r"run-\d{8}_\d{6}-(\w+-\w+-\d+)"
        match = re.search(pattern, path)
        if match:
            wandb_name = match.group(1)
        else:
            print("No wandb name found for run.")
            raise ValueError("Es knallt!")
        basename = os.path.basename(path)
        return CheckpointMetadata(path, basename, wandb_name, None, None)
    else:
        raise ValueError("Es knallt gewaltig!")


def check_duplicate_checkpoints(ckpt_infos):
    x = set()
    counts = 0
    for wandb_name, sweep_id in list(map(lambda _x: (_x.wandb_name, _x.sweep_id), ckpt_infos)):
        if (wandb_name, sweep_id) in x:
            print(f"Duplicate found: {wandb_name}, {sweep_id}")
            counts += 1
        x.add((wandb_name, sweep_id))
    if counts > 0:
        return False
    return True


def get_finished_wandb_runs():
    api = wandb.Api()
    runs = api.runs(f'wuesuv/CV2024')

    finished_wandb_runs, running_wandb_runs = [], []
    for run in runs:
        if run.state != 'running':
            sweep_id = run.sweep.id if run.sweep else None
            finished_wandb_runs.append((run.name, run.id, sweep_id))
        if run.state == 'running':
            print(f"Excluded running run: {run.name} ({run.id})")
    return finished_wandb_runs


def check_val_results(checkpoint_dir, result_dir):
    ckpt_dirs = find_files_with_ending(checkpoint_dir, ending='.ckpt')
    ckpt_infos = [get_base_name(ckpt) for ckpt in ckpt_dirs]

    # run_name, run_id, sweep_id
    finished_wandb_runs = get_finished_wandb_runs()

    result = []
    for ckpt_info in ckpt_infos:
        for run_name, run_id, sweep_id in finished_wandb_runs:
            if run_name in ckpt_info and sweep_id in ckpt_info:
                result.append(ckpt_info._replace(run_id=run_id))

    assert check_duplicate_checkpoints(result)

    print("\n".join(map(str, map(lambda x: x.wandb_name, result))))

    # val_metric_filenames = find_files_with_ending(result_dir, ending='.json')

    # Check for existing validation result files
    # 5afkssf_peaky-firefly-5.csv
    val_pred_filenames = find_files_with_ending(result_dir, ending='.csv')
    existing_pred_run_ids = set(filename.split('_')[0] for filename in val_pred_filenames)
    missing_pred_checkpoints = filter(lambda _x: _x.run_id not in existing_pred_run_ids, result)

    print(f"Found {len(result)} finished runs without validation results.")

    return missing_pred_checkpoints


if __name__ == '__main__':
    argparse = argparse.ArgumentParser(description='Check validation results.')
    argparse.add_argument('--checkpoint_dir', default="checkpoints", type=str,
                          help='Path to the checkpoint directory.')
    argparse.add_argument('--result_dir', default="ensemble/val_results", type=str,
                          help='Path to the model validation results.')
    args = argparse.parse_args()

    check_val_results(args.checkpoint_dir, args.result_dir)
