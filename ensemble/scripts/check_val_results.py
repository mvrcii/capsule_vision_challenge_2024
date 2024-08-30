import argparse
import os


def find_ckpt_files(dir):
    ckpt_files = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith('.ckpt'):
                ckpt_files.append(os.path.join(root, file))
    return ckpt_files


def check_val_results(checkpoint_dir):
    ckpt_dirs = find_ckpt_files(checkpoint_dir)
    print(ckpt_dirs)


if __name__ == '__main__':
    argparse = argparse.ArgumentParser(description='Check validation results.')
    argparse.add_argument('--checkpoint_dir', default="checkpoints", type=str,
                          help='Path to the checkpoint directory.')
    args = argparse.parse_args()

    check_val_results(args.checkpoint_dir)
