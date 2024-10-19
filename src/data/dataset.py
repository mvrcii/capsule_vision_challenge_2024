from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, df, transform, label_encoder=None, label_column='class', image_column='frame_path'):
        """
        Args:
            df (pd.DataFrame): DataFrame containing image paths and labels.
            transform (callable, optional): Transformations to apply to the images.
            label_encoder (LabelEncoder, optional): Encoder to convert string labels to integers.
            label_column (str): Name of the label column in df.
            image_column (str): Name of the image path column in df.
        """
        self.df = df.copy()
        self.transform = transform
        self.image_column = image_column

        if label_encoder is not None and label_column in df.columns:
            self.label_encoder = label_encoder
            # If labels are not yet encoded, encode them
            if not np.issubdtype(df[label_column].dtype, np.number):
                self.df['label_idx'] = self.label_encoder.fit_transform(self.df[label_column])
            else:
                self.df['label_idx'] = self.df[label_column]
            self.has_labels = True
        else:
            self.has_labels = False

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = Path(row[self.image_column]).as_posix()

        # Load and preprocess the image
        try:
            with Image.open(image_path) as img:
                image_np = np.array(img.convert('RGB'))

                if self.transform is not None:
                    augmented = self.transform(image=image_np)
                    image = augmented['image']

        except Exception as e:
            raise RuntimeError(f"Error loading image {image_path}: {e}")

        if self.has_labels:
            label = row['label_idx']
            label_tensor = torch.tensor(label, dtype=torch.long)
            return image, label_tensor
        else:
            return image
