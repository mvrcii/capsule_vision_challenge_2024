import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb
from PIL import Image
from lightning import LightningDataModule, Trainer
from torch import softmax
from torch.utils.data import DataLoader, Dataset

from ensemble.scripts.check_val_results import CheckpointMetadata, check_val_results
from ensemble.scripts.utils.slurm_utils import run_on_slurm
from src.models.regnety.regnety import RegNetY
from src.utils.transform_utils import load_transforms

warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*tensorboardX.*")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class PredictImageDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = Path(self.df.iloc[idx]['frame_path'])
        image_path = image_path.as_posix()

        with Image.open(image_path) as img:
            image_np = np.array(img.convert('RGB'))

            if self.transform is not None:
                augmented = self.transform(image=image_np)
                image_tensor = augmented['image']

        return image_tensor


class PredictDataModule(LightningDataModule):
    def __init__(self, debug, transforms, dataset_path, dataset_csv_path, fold_idx=1, num_workers=8,
                 pred_bs=32):
        super().__init__()
        self.dataset_path = dataset_path
        self.dataset_csv_path = dataset_csv_path
        self.fold_idx = fold_idx
        self.pred_bs = pred_bs
        self.num_workers = num_workers
        self.train_transform, self.val_transform = transforms
        self.datasets = {}
        self.debug = debug

    def vectorized_path_update(self, dataset):
        dataset = dataset.copy()
        dataset.loc[:, 'frame_path'] = dataset['frame_path'].apply(
            lambda x: os.path.join(self.dataset_path, x).replace('\\', '/'))
        return dataset

    def __load_test_data(self):
        test_path = os.path.join(self.dataset_csv_path, 'test.csv')
        if os.path.exists(test_path):
            X_test = pd.read_csv(test_path)
            X_test = self.vectorized_path_update(X_test)
            return X_test
        return None

    def __load_data(self):
        train_val_path = os.path.join(self.dataset_csv_path, 'train_val.csv')

        X_test = self.__load_test_data()

        X_train_val = pd.read_csv(train_val_path)

        unique_fold_idcs = X_train_val['fold'].unique()
        val_fold_idx = self.fold_idx

        if val_fold_idx not in unique_fold_idcs:
            raise ValueError(f"Fold index {val_fold_idx} not found in the available folds: {unique_fold_idcs}")

        X_val = X_train_val[X_train_val['fold'] == self.fold_idx]

        logging.info(f"DataModule: Fold {val_fold_idx} with {len(X_val)} samples used for prediction")

        X_val = self.vectorized_path_update(X_val)

        return X_val, X_test

    def setup(self, stage=None):
        X_val, X_test = self.__load_data()

        if self.debug:
            X_val = X_val.sample(n=0.05)

        self.datasets = {
            'val': PredictImageDataset(X_val, self.val_transform),
        }

        if X_test is not None:
            self.datasets['test'] = PredictImageDataset(X_test, self.val_transform)

    def predict_dataloader(self):
        return DataLoader(self.datasets['val'], batch_size=self.pred_bs, num_workers=self.num_workers, shuffle=False)


def load_class_mapping(path):
    absolute_path = os.path.abspath(path)
    if not os.path.exists(absolute_path):
        raise FileNotFoundError(f"Class mapping file not found at {path}")

    with open(path, 'r') as f:
        class_mapping = json.load(f)
    return class_mapping


def save_predictions(preds, dataset_path, result_dir, ckpt_id, ckpt_run_name, class_mapping, dataloader):
    idx_to_class = {v: k for k, v in class_mapping.items()}

    # Collect data for DataFrame
    frame_paths = []
    probabilities_list = []
    predicted_classnames = []

    # Assume preds is a list of tensors
    for pred_batch in preds:
        for pred, data in zip(pred_batch, dataloader.dataset.df.iterrows()):
            frame_path = data[1]['frame_path']
            frame_paths.append(frame_path)

            # Softmax to compute probabilities
            probabilities = softmax(pred, dim=0).cpu().numpy()
            probabilities_list.append(probabilities)

            predicted_index = probabilities.argmax()

            predicted_classnames.append(idx_to_class[predicted_index])

    # Create DataFrame
    df = pd.DataFrame(probabilities_list, columns=[f'class{i}' for i in range(probabilities_list[0].shape[0])])
    df.insert(0, 'framepath', frame_paths)
    df['predicted_classname'] = predicted_classnames

    # Update frame paths to remove the base dataset path
    df['framepath'] = df['framepath'].apply(lambda x: frame_path.split(dataset_path)[1][1:])

    # Output path for the CSV
    output_path = os.path.join(result_dir, f"{ckpt_id}_{ckpt_run_name}.csv")
    df.to_csv(output_path, index=False)
    logging.info(f"Saved predictions to {output_path}")


def pred_checkpoint(debug, ckpt_id, ckpt_run_name, ckpt_path, result_dir, dataset_path, dataset_csv_path):
    logging.info(f"Connecting to W&B API to fetch run data for run: {ckpt_run_name}")
    api = wandb.Api()
    run = api.run(f'wuesuv/CV2024/{ckpt_id}')
    logging.info("Run data retrieved successfully.")

    config = argparse.Namespace(**run.config)
    class_mapping = load_class_mapping(os.path.join(dataset_csv_path, 'class_mapping.json'))
    transforms = load_transforms(img_size=config.img_size, transforms_string=run.summary.get('transforms'))

    model = RegNetY.load_from_checkpoint(checkpoint_path=ckpt_path, config=config, class_to_idx=class_mapping)
    model.to(torch.device('cuda'))
    model.eval()

    num_workers = 8
    pred_bs = 64
    if debug:
        pred_bs = 16
        num_workers = 0

    data_module = PredictDataModule(
        debug=debug,
        transforms=transforms,
        pred_bs=pred_bs,
        dataset_path=dataset_path,
        dataset_csv_path=dataset_csv_path,
        fold_idx=1,
        num_workers=num_workers
    )
    data_module.setup()

    logging.info("Pred/Val Images:", len(data_module.predict_dataloader().dataset))

    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision="16-mixed",
        gradient_clip_val=0.5,
        inference_mode=True
    )

    preds = trainer.predict(model, datamodule=data_module)

    save_predictions(preds, dataset_path, result_dir, ckpt_id, ckpt_run_name, class_mapping,
                     data_module.predict_dataloader())


def main(args):
    if 'SLURM_JOB_ID' in os.environ:
        logging.info("Detected SLURM environment with JOB ID: " + os.environ['SLURM_JOB_ID'])
    elif args.slurm:
        logging.info("SLURM submission flag detected. Preparing to submit the script to SLURM.")
        python_cmd = "python " + " ".join(sys.argv[0:1] + [arg for arg in sys.argv[1:] if arg != '--slurm'])
        run_on_slurm(python_cmd, args.gpu, args.attach, job_name="preds")
        sys.exit()
    else:
        logging.info("Running locally without SLURM submission.")

    logging.info("Loading checkpoint metadata from directory.")
    missing_pred_checkpoints: CheckpointMetadata = check_val_results(args.checkpoint_dir, args.result_dir)
    for checkpoint in missing_pred_checkpoints:
        logging.info(f"Processing checkpoint: {checkpoint.wandb_name}")
        ckpt_path = checkpoint.rel_path
        ckpt_run_name = checkpoint.wandb_name
        ckpt_id = checkpoint.run_id
        pred_checkpoint(
            debug=args.debug,
            ckpt_id=ckpt_id,
            ckpt_run_name=ckpt_run_name,
            ckpt_path=ckpt_path,
            result_dir=args.result_dir,
            dataset_path=args.dataset_path,
            dataset_csv_path=args.dataset_csv_path,
        )
        logging.info(f"Finished processing checkpoint: {ckpt_run_name}")
    logging.info("All checkpoints have been processed.")

# DEBUGGING
# def main(args):
#     ckpt_id = '4ema6q7u'
#     ckpt_run_name = 'sage-sweep-1'
#     ckpt_path = 'checkpoints/fast-sweep-2_epoch10_val_AUC_macro0.99.ckpt'
#
#     pred_checkpoint(
#         ckpt_id=ckpt_id,
#         ckpt_run_name=ckpt_run_name,
#         ckpt_path=ckpt_path,
#         result_dir=args.result_dir,
#         dataset_path=args.dataset_path,
#         dataset_csv_path=args.dataset_csv_path,
#     )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check validation results.')
    parser.add_argument('--checkpoint_dir', default="checkpoints", type=str,
                        help='Path to the checkpoint directory.')
    parser.add_argument('--result_dir', default="ensemble/val_results", type=str,
                        help='Path to the model validation results.')
    parser.add_argument('--dataset_path', default="../data/",
                        type=str, help='Path to the validation dataset.')
    parser.add_argument('--dataset_csv_path', default="dataset",
                        type=str, help='Path to the validation dataset.')
    parser.add_argument('--debug', action='store_true', help='Debug mode (default: False)')
    parser.add_argument('-s', '--slurm', action='store_true', help='Run on SLURM (default: False)')
    parser.add_argument('--gpu', type=int, choices=[0, 1], default=0, help="Choose GPU: 0=rtx2080ti, 1=rtx3090")
    parser.add_argument('-a', '--attach', action='store_false', help="Attach to log output (default: True)")
    args = parser.parse_args()

    logging.info("Starting script with the following configuration:")
    logging.info(f"Checkpoint Directory: {args.checkpoint_dir}")
    logging.info(f"Result Directory: {args.result_dir}")
    logging.info(f"Dataset Path: {args.dataset_path}")
    logging.info(f"Dataset CSV Path: {args.dataset_csv_path}")
    logging.info(f"Debug Mode: {'Enabled' if args.debug else 'Disabled'}")
    logging.info(f"Running on SLURM: {'Yes' if args.slurm else 'No'}")
    main(args)
