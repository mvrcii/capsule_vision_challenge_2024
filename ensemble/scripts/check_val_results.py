import argparse
import logging
import os
import re
from collections import namedtuple

import wandb

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
CheckpointMetadata = namedtuple('CheckpointMetadata', ['rel_path', 'base_name', 'wandb_name', 'sweep_id', 'run_id'])


def find_files_with_ending(dir, file_only=False, ending='.ckpt'):
    files = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(ending):
                files.append(os.path.join(root, file))
    logging.info(f"Found {len(files)} files with {ending} ending.")
    return files


def get_base_name(path):
    logging.debug(f"Processing path: {path}")
    if path.startswith('checkpoints/sweep'):
        # Extract sweep ID
        pattern = r"sweep-(\w+)/"
        match = re.search(pattern, path)
        if match:
            sweep_id = match.group(1)
        else:
            logging.info("No sweep id found for sweep run.")
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
            logging.info("No wandb name found for run.")
            raise ValueError("Es knallt!")
        basename = os.path.basename(path)
        return CheckpointMetadata(path, basename, wandb_name, None, None)

    logging.error("Invalid checkpoint path format.")
    raise ValueError("Invalid path format.")


def check_duplicate_checkpoints(ckpt_infos):
    seen = set()
    duplicate_count = 0
    for metadata in ckpt_infos:
        if (metadata.wandb_name, metadata.sweep_id) in seen:
            logging.warning(f"Duplicate found: {metadata.wandb_name}, {metadata.sweep_id}")
            duplicate_count += 1
        seen.add((metadata.wandb_name, metadata.sweep_id))
    logging.info(f"Found {duplicate_count} duplicates.")
    return duplicate_count == 0


def get_finished_wandb_runs():
    api = wandb.Api()
    runs = api.runs(f'wuesuv/CV2024')
    valid_runs = []
    for run in runs:
        if run.state == 'running':
            logging.info(f"Excluding running run: {run.name} ({run.id})")
        elif run.summary.get('transforms') is None:
            logging.info(f"Excluding run without transforms: {run.name} ({run.id})")
        else:
            sweep_id = run.sweep.id if run.sweep else None
            valid_runs.append((run.name, run.id, sweep_id))
    logging.info(f"Retrieved {len(valid_runs)} valid wandb runs.")
    return valid_runs


def check_val_results(checkpoint_dir, result_dir):
    ckpt_dirs = find_files_with_ending(checkpoint_dir, ending='.ckpt')
    ckpt_infos = [get_base_name(ckpt) for ckpt in ckpt_dirs]
    valid_wandb_runs = get_finished_wandb_runs()  # run_name, run_id, sweep_id
    result = []
    for ckpt_info in ckpt_infos:
        for run_name, run_id, sweep_id in valid_wandb_runs:
            if run_name == ckpt_info.wandb_name and (sweep_id == ckpt_info.sweep_id or ckpt_info.sweep_id is None):
                result.append(ckpt_info._replace(run_id=run_id))

    assert check_duplicate_checkpoints(result)

    # Proceed to filter out runs with already existing prediction files
    val_pred_filenames = find_files_with_ending(result_dir, file_only=True, ending='.csv')
    logging.info(val_pred_filenames)
    existing_pred_run_ids = set(filename.split('_')[0] for filename in val_pred_filenames)
    existing_runs_str = ", ".join(existing_pred_run_ids)
    logging.info(f"Existing predictions found for ID: {existing_runs_str}")

    missing_pred_checkpoints = list(filter(lambda _x: _x.run_id not in existing_pred_run_ids, result))

    logging.info(f"Total missing prediction checkpoints found: {len(missing_pred_checkpoints)}")
    return missing_pred_checkpoints


if __name__ == '__main__':
    argparse = argparse.ArgumentParser(description='Check validation results.')
    argparse.add_argument('--checkpoint_dir', default="checkpoints", type=str,
                          help='Path to the checkpoint directory.')
    argparse.add_argument('--result_dir', default="ensemble/val_results", type=str,
                          help='Path to the model validation results.')
    args = argparse.parse_args()

    check_val_results(args.checkpoint_dir, args.result_dir)
