import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, df, class_to_index, transform, multilabel=False):
        """
        df: Pandas DataFrame containing 'frame_path', and 'class' columns.
        class_to_index: A dictionary mapping class names to indices.
        transform: The transformations to be applied on images.
        multilabel: Whether the dataset is multilabel or not.
        """
        self.df = df
        self.class_to_index = class_to_index
        self.transform = transform
        self.multilabel = multilabel

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.df.iloc[idx]['frame_path']
        label = self.df.iloc[idx]['class']

        with Image.open(image_path) as img:
            image_np = np.array(img.convert('RGB'))

            if self.transform is not None:
                augmented = self.transform(image=image_np)
                image_tensor = augmented['image']

        if self.multilabel:
            label_indices = [self.class_to_index[cls] for cls in label.split(',')]
            label_tensor = torch.zeros(len(self.class_to_index))
            label_tensor[label_indices] = 1
        else:
            label_tensor = torch.tensor(self.class_to_index[label], dtype=torch.long)

        return image_tensor, label_tensor
