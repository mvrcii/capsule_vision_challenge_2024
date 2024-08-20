import os

import pandas as pd
import torch
from lightning import LightningDataModule
from torch.utils.data import WeightedRandomSampler, DataLoader

from src.data.dataset import ImageDataset


class DataModule(LightningDataModule):
    def __init__(self, class_mapping, transforms, train_bs, val_bs, dataset_path, dataset_csv_path, fold_idx=0,
                 num_workers=8):
        super().__init__()
        self.train_bs, self.val_bs = train_bs, val_bs
        self.dataset_path = dataset_path
        self.dataset_csv_path = dataset_csv_path
        self.fold_idx = fold_idx
        self.num_workers = num_workers
        self.train_transform, self.val_transform = transforms

        self.datasets = {}
        self.class_to_index = class_mapping
        self.sample_weights = []

    def vectorized_path_update(self, dataset):
        dataset['frame_path'] = dataset['frame_path'].apply(
            lambda x: os.path.join(self.dataset_path, x).replace('\\', '/'))
        return dataset

    def __load_test_data(self):
        test_path = os.path.join(self.dataset_csv_path, 'test.csv')
        X_test = pd.read_csv(test_path)
        X_test = self.vectorized_path_update(X_test)
        return X_test

    @staticmethod
    def __balance_classes(df, random_state=42):
        df = df.copy()
        class_counts = df['class'].value_counts()
        # Determine the smallest number of instances among the classes, except the most dominant
        target_class_size = class_counts.sort_values(ascending=False)[1]  # Index 1 for the second most frequent class

        balanced_df = pd.DataFrame()

        for class_value in class_counts.index:
            class_subset = df[df['class'] == class_value]
            if class_counts[class_value] > target_class_size:
                class_subset = class_subset.sample(n=target_class_size, random_state=random_state)
            balanced_df = pd.concat([balanced_df, class_subset], axis=0)

        # Shuffle the DataFrame to avoid any ordering issues later on
        balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        return balanced_df

    def __load_data(self, test_set=False):
        train_val_path = os.path.join(self.dataset_csv_path, 'train_val.csv')

        if test_set:
            X_test = self.__load_test_data()
        else:
            X_test = None

        X_train_val = pd.read_csv(train_val_path)

        print(f"Using fold {self.fold_idx} for validation.")
        X_train = X_train_val[X_train_val['fold'] != self.fold_idx]
        X_val = X_train_val[X_train_val['fold'] == self.fold_idx]

        # Apply class balancing
        X_train = self.__balance_classes(X_train)
        # X_val = self.__balance_classes(X_val)
        print('Train Value Counts:', X_train['class'].value_counts())

        # Apply the optimized path update
        print(X_train['frame_path'][0])
        X_train = self.vectorized_path_update(X_train)
        print(X_train['frame_path'][0])
        X_val = self.vectorized_path_update(X_val)

        return X_train, X_val, X_test

    def setup(self, stage=None):
        X_train, X_val, X_test = self.__load_data()

        self.datasets = {
            'train': ImageDataset(X_train, self.class_to_index, self.train_transform),
            'val': ImageDataset(X_val, self.class_to_index, self.val_transform),
        }

        if X_test is not None:
            self.datasets['test'] = ImageDataset(X_test, self.class_to_index, self.val_transform)
        self.calculate_inverse_weights(X_train)

    def calculate_inverse_weights(self, df):
        class_counts = df['class'].value_counts()
        inverse_weights = 1 / class_counts
        self.sample_weights = torch.tensor(df['class'].map(inverse_weights).values, dtype=torch.double)

    def train_dataloader(self):
        sampler = WeightedRandomSampler(self.sample_weights, num_samples=len(self.datasets['train']))
        return DataLoader(self.datasets['train'], batch_size=self.train_bs, sampler=sampler, drop_last=True,
                          pin_memory=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.datasets['val'], drop_last=True, pin_memory=True, batch_size=self.val_bs,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.datasets['test'], drop_last=True, pin_memory=True, batch_size=self.val_bs,
                          num_workers=self.num_workers)
