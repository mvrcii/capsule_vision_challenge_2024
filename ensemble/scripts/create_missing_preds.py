import argparse
import json
import logging
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb
from PIL import Image
from lightning import LightningDataModule, Trainer
from torch.utils.data import DataLoader, Dataset

from src.models.regnety.regnety import RegNetY
from src.utils.transform_utils import load_transforms

warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*tensorboardX.*")


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
    def __init__(self, transforms, dataset_path, dataset_csv_path, fold_idx=1, num_workers=8,
                 pred_bs=32):
        super().__init__()
        self.dataset_path = dataset_path
        self.dataset_csv_path = dataset_csv_path
        self.fold_idx = fold_idx
        self.pred_bs = pred_bs
        self.num_workers = num_workers
        self.train_transform, self.val_transform = transforms
        self.datasets = {}

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


def pred_checkpoint(ckpt_id, ckpt_run_name, ckpt_path, result_dir, dataset_path, dataset_csv_path):
    api = wandb.Api()
    run = api.run(f'wuesuv/CV2024/{ckpt_id}')

    config = argparse.Namespace(**run.config)
    class_mapping = load_class_mapping(os.path.join(dataset_csv_path, 'class_mapping.json'))
    transforms = load_transforms(img_size=config.img_size, transforms_string=run.summary.get('transforms'))

    model = RegNetY.load_from_checkpoint(checkpoint_path=ckpt_path, config=config, class_to_idx=class_mapping)
    model.to(torch.device('cuda'))
    model.eval()

    data_module = PredictDataModule(
        transforms=transforms,
        pred_bs=16,
        dataset_path=dataset_path,
        dataset_csv_path=dataset_csv_path,
        fold_idx=1,
        num_workers=0
    )
    data_module.setup()

    print("Pred/Val Images:", len(data_module.predict_dataloader().dataset))

    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision="16-mixed",
        gradient_clip_val=0.5,
        inference_mode=True
    )

    preds = trainer.predict(model, datamodule=data_module)

    print(preds)
    # output_path = os.path.join(result_dir, f"{ckpt_id}_{ckpt_run_name}.csv")


# def main(args):
#     missing_pred_checkpoints: CheckpointMetadata = check_val_results(args.checkpoint_dir, args.result_dir)
#
#     for checkpoint in missing_pred_checkpoints:
#         ckpt_path = checkpoint.rel_path
#         ckpt_run_name = checkpoint.wandb_name
#         ckpt_id = checkpoint.run_id
#
#         pred_checkpoint(
#             ckpt_id=ckpt_id,
#             ckpt_run_name=ckpt_run_name,
#             ckpt_path=ckpt_path,
#             result_dir=args.result_dir,
#             dataset_path=args.dataset_path,
#             dataset_csv_path=args.dataset_csv_path,
#         )


def main(args):
    ckpt_id = '4ema6q7u'
    ckpt_run_name = 'sage-sweep-1'
    ckpt_path = 'checkpoints/fast-sweep-2_epoch10_val_AUC_macro0.99.ckpt'

    pred_checkpoint(
        ckpt_id=ckpt_id,
        ckpt_run_name=ckpt_run_name,
        ckpt_path=ckpt_path,
        result_dir=args.result_dir,
        dataset_path=args.dataset_path,
        dataset_csv_path=args.dataset_csv_path,
    )


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
    args = parser.parse_args()
    main(args)
