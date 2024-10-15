import logging
import os
import warnings

import pandas as pd
import torch
from lightning import LightningDataModule
from torch.utils.data import WeightedRandomSampler, DataLoader

from src.data.dataset import ImageDataset

warnings.filterwarnings("ignore", ".*does not have many workers.*")


class DataModule(LightningDataModule):
    def __init__(self, class_mapping, transforms, train_bs, val_bs, dataset_path, dataset_csv_path: str, fold_idx=0,
                 num_workers=8, train_frac=1.0, val_frac=1.0, include_test_in_train=False):
        super().__init__()
        self.train_bs, self.val_bs = train_bs, val_bs
        self.include_test_in_train = include_test_in_train
        self.dataset_path = dataset_path
        self.dataset_csv_path = dataset_csv_path
        self.fold_idx = fold_idx
        self.num_workers = num_workers
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.train_transform, self.val_transform = transforms

        self.datasets = {}
        self.class_to_index = class_mapping
        self.sample_weights = []

    def __get_valid_csv_combination(self):
        file_paths = {
            'train.csv': os.path.join(self.dataset_csv_path, 'train.csv'),
            'val.csv': os.path.join(self.dataset_csv_path, 'val.csv'),
            'train_val.csv': os.path.join(self.dataset_csv_path, 'train_val.csv'),
            'test.csv': os.path.join(self.dataset_csv_path, 'test.csv')
        }

        existing_files = [file for file, path in file_paths.items() if os.path.exists(path)]

        supported_file_combinations = [
            ['train.csv'],
            ['train.csv', 'val.csv'],
            ['train_val.csv'],
        ]

        valid_combinations = supported_file_combinations + [combo + ['test.csv'] for combo in
                                                            supported_file_combinations]

        for combination in valid_combinations:
            if sorted(combination) == sorted(existing_files):
                return combination

        raise ValueError(f"Invalid file combination: {existing_files}. "
                         f"Supported combinations are {supported_file_combinations} with optional 'test.csv'.")

    def setup(self, stage=None):
        csv_files = self.__get_valid_csv_combination()

        datasets = self.__load_data(csv_files)

        X_train = datasets.get('train', None)
        if X_train is not None:
            self.calculate_inverse_weights(X_train)

        for key, value in datasets.items():
            transform = self.train_transform if key == 'train' else self.val_transform
            datasets[key] = ImageDataset(value, self.class_to_index, transform)

        self.datasets = datasets

        self.__setup_dataloader_args()

    def calculate_inverse_weights(self, df):
        class_counts = df['class'].value_counts()
        inverse_weights = 1 / class_counts
        self.sample_weights = torch.tensor(df['class'].map(inverse_weights).values, dtype=torch.double)

    def train_dataloader(self):
        if self.datasets.get('train', None) is None:
            logging.info("Training dataset was requested but is not available.")
            return None
        sampler = WeightedRandomSampler(self.sample_weights, num_samples=len(self.datasets['train']))
        return DataLoader(self.datasets['train'], batch_size=self.train_bs, sampler=sampler, drop_last=True,
                          prefetch_factor=self.prefetch_factor, persistent_workers=self.persistent_workers,
                          num_workers=self.num_workers, pin_memory=self.pin_memory)

    def val_dataloader(self):
        if self.datasets.get('val', None) is None:
            logging.info("Validation dataset was requested but is not available.")
            return None
        return DataLoader(self.datasets['val'], drop_last=True, batch_size=self.val_bs, num_workers=self.num_workers,
                          prefetch_factor=self.prefetch_factor, persistent_workers=self.persistent_workers,
                          pin_memory=self.pin_memory)

    def test_dataloader(self):
        if self.datasets.get('test', None) is None:
            logging.info("Test dataset was requested but is not available.")
            return None
        return DataLoader(self.datasets['test'], drop_last=True, batch_size=self.val_bs, num_workers=self.num_workers,
                          prefetch_factor=self.prefetch_factor, persistent_workers=self.persistent_workers,
                          pin_memory=self.pin_memory)

    def __setup_dataloader_args(self):
        on_local_device = self.num_workers == 0
        self.prefetch_factor = None if on_local_device else 2
        self.persistent_workers = False if on_local_device else True
        self.pin_memory = False if on_local_device else True

    def __load_data(self, csv_files):
        def vectorized_path_update(dataset):
            dataset = dataset.copy()
            dataset.loc[:, 'frame_path'] = dataset['frame_path'].apply(
                lambda x: os.path.join(self.dataset_path, x).replace('\\', '/'))
            return dataset

        if ('train.csv' in csv_files and 'train_val.csv' in csv_files) or (
                'val.csv' in csv_files and 'train_val.csv' in csv_files):
            raise ValueError("Either 'train.csv' or 'val.csv' cannot be used together with 'train_val.csv'")

        datasets = {}
        for filename in csv_files:
            data_path = os.path.join(self.dataset_csv_path, filename)
            data = pd.read_csv(data_path)

            if filename == 'train.csv':
                X_train = data.sample(frac=self.train_frac)
                X_train = vectorized_path_update(X_train)
                datasets['train'] = X_train
            elif filename == 'val.csv':
                X_val = data.sample(frac=self.val_frac)
                X_val = vectorized_path_update(X_val)
                datasets['val'] = X_val
            elif filename == 'train_val.csv':
                # Parse K-Fold Split
                if self.fold_idx is None:
                    raise ValueError("Fold index must be provided for train_val.csv")
                logging.info(f"Using fold {self.fold_idx} for validation.")
                X_train = data[data['fold'] != self.fold_idx]
                X_train = X_train.sample(frac=self.train_frac)
                X_val = data[data['fold'] == self.fold_idx]
                X_val = X_val.sample(frac=self.train_frac)
                datasets['train'] = X_train
                datasets['val'] = X_val
            elif filename == 'test.csv':
                X_test = data
                datasets['test'] = X_test
            else:
                raise ValueError(f"Invalid filename: {filename}")

        # Conditionally merge test data into train data
        if self.include_test_in_train and 'test' in datasets and 'train' in datasets:
            X_train = datasets['train']
            X_test = datasets['test']
            datasets['train'] = pd.concat([X_train, X_test])
            logging.info(f"Merged Test data into Train: New Train Size: {len(datasets['train'])}")
            del datasets['test']

        for key, value in datasets.items():
            datasets[key] = vectorized_path_update(value)
            logging.info(f"{key.capitalize()} Size: {len(value)}")

        return datasets
