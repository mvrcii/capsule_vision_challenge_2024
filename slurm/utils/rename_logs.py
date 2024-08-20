import argparse
import os
import re


def rename_slurm_logs(directory='logs'):
    # Regex to match SLURM log files
    filename_pattern = re.compile(r"slurm-(\d+).out")
    # Regex for extracting sweep IDs and run names from wandb URL lines
    sweep_url_pattern = re.compile(r"View sweep at https://wandb.ai/[\w\-_]+/[\w\-_]+/sweeps/([\w\-]+)")
    run_url_pattern = re.compile(r"View run at https://wandb.ai/[\w\-_]+/[\w\-_]+/runs/([\w\-]+)")

    files = os.listdir(directory)
    print(f"{len(files)} files found in '{directory}'")

    # Iterate over files in the specified directory
    for filename in files:
        if filename_pattern.match(filename):
            full_path = os.path.join(directory, filename)
            sweep_id, run_names = None, set()

            with open(full_path, 'r') as file:
                for line in file:
                    sweep_match = sweep_url_pattern.search(line)
                    if sweep_match:
                        sweep_id = sweep_match.group(1)

                    run_match = run_url_pattern.search(line)
                    if run_match:
                        run_names.add(run_match.group(1))

            new_filename = None
            if sweep_id is None and run_names:
                print(f"Run found: {', '.join(run_names)}")
                run_names_str = "-".join(sorted(run_names))
                new_filename = f"slurm-{filename_pattern.search(filename).group(1)}-runs-{run_names_str}.out"
            elif sweep_id and run_names:
                print(f"Sweep found: {sweep_id}, run name(s): {', '.join(run_names)}")
                run_names_str = "-".join(sorted(run_names))
                new_filename = f"slurm-{filename_pattern.search(filename).group(1)}-sweep-{sweep_id}-runs-{run_names_str}.out"

            if new_filename:
                new_full_path = os.path.join(directory, new_filename)
                os.rename(full_path, new_full_path)
                print(f"Renamed '{filename}' to '{new_filename}'")
        else:
            print(f"Skip: '{filename}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Rename SLURM log files based on Weights & Biases sweep IDs and run names.')
    parser.add_argument('--directory', type=str, default='logs', help='Directory containing SLURM log files.')
    args = parser.parse_args()
    rename_slurm_logs(args.directory)
