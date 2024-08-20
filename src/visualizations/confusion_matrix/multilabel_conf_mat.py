import argparse
import os
from typing import Tuple

import albumentations as A
import pandas as pd
import seaborn as sns
import torch
import wandb
from albumentations.pytorch import ToTensorV2
from lightning import Trainer
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from torch.utils.data import DataLoader
from torchmetrics.classification import MultilabelConfusionMatrix

from src.data.dataset import ImageDataset
from src.models.enums.model_mode import ModelMode
from src.models.regnety.regnety import RegNetY
from src.utils.class_mapping import load_class_mapping


def plot(cm, class_names):
    sns.set_style("white")
    sns.set_context("paper")

    num_classes = len(class_names)

    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(14, 10), squeeze=False)
    axes = axes.flatten()

    for i in range(num_classes):
        thresh = 40
        ax = axes.flatten()[i]
        cm_label = cm[i]  # Get the 2x2 confusion matrix for label i

        cm_normalized = (cm_label.astype('float') / cm_label.sum()) * 100
        norm = TwoSlopeNorm(vmin=0, vcenter=thresh, vmax=100)
        cax = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues', norm=norm)

        # Set axes titles
        ax.set_title(class_names[i], fontsize=14, weight='bold', pad=10)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Pos', 'Neg'], fontsize=12)
        ax.set_yticklabels(['Pos', 'Neg'], fontsize=12, rotation=90, va='center')
        ax.tick_params(axis='x', which='major', pad=0)
        ax.tick_params(axis='y', which='major', pad=0)

        # Simplify annotations
        fmt = 'd'
        for j in range(2):
            for k in range(2):
                ax.text(k, j, format(cm_label[j, k], fmt),
                        ha="center", va="center",
                        color="white" if cm_normalized[j, k] > thresh else "black",
                        fontsize=10)

    # Global axes labels
    fig.text(0.52, 0.11, 'Prediction', ha='center', va='center', fontsize=20)
    fig.text(0.11, 0.52, 'Ground Truth', ha='center', va='center', rotation='vertical', fontsize=20)

    # Add a single color bar on the right side
    cbar_ax = fig.add_axes([0.92, 0.15, 0.03, 0.75])
    cbar = fig.colorbar(cax, cax=cbar_ax)
    cbar.set_label('Percentage', fontsize=20)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_ticks([0, 20, 40, 60, 80, 100])
    cbar.set_ticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])

    fig.suptitle(f'Confusion Matrix', fontsize=18, weight='bold')

    plt.subplots_adjust(left=0.15, right=0.88, bottom=0.15, top=0.9, wspace=0.4)
    plt.show()


def get_transforms(config) -> Tuple[A.Compose, A.Compose]:
    if config.img_size:
        img_size = config.img_size
    else:
        raise ValueError("Image size not provided")

    train_transforms = A.Compose([
        A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.08, 1.0), ratio=(0.75, 1.3333),
                            interpolation=2),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ColorJitter(p=0.5, brightness=(0.6, 1.4), contrast=(0.6, 1.4), saturation=(0.6, 1.4)),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    val_transforms = A.Compose([
        A.Resize(height=img_size, width=img_size, interpolation=2),
        A.CenterCrop(height=img_size, width=img_size, always_apply=True),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(always_apply=True)
    ])

    return train_transforms, val_transforms


def load_data(dataset_csv_path, dataset_path, csv_name='test.csv'):
    test_path = os.path.join(dataset_csv_path, csv_name)
    X_test = pd.read_csv(test_path)
    X_test['frame_path'] = dataset_path + X_test['frame_path']
    return X_test


def get_batch(dataset, start_idx, batch_size):
    max_idx = len(dataset)
    end_idx = min(start_idx + batch_size, max_idx)
    batch_images, batch_labels = [], []

    for i in range(start_idx, end_idx):
        image, label = dataset[i]
        batch_images.append(image)
        batch_labels.append(label)

    # Stack the list of images and labels into tensors
    batch_images = torch.stack(batch_images)

    if isinstance(batch_labels[0], torch.Tensor):
        batch_labels = torch.stack(batch_labels)
    else:
        batch_labels = torch.tensor(batch_labels)

    return batch_images.cuda(), batch_labels.cuda()


def main(args):
    api = wandb.Api()
    run = api.run(f'{args.entity}/{args.wandb_project}/{args.run_id}')
    config = argparse.Namespace(**run.config)

    class_mapping = load_class_mapping(os.path.join(config.dataset_csv_path, 'class_mapping.json'))
    num_classes = len(class_mapping)
    class_names = list(class_mapping.keys())
    is_multilabel = config.model_mode == ModelMode.MULTI_LABEL.value

    X_test = load_data(config.dataset_csv_path, config.dataset_path, csv_name='test.csv')
    _, val_transforms = get_transforms(config)

    dataset = ImageDataset(X_test, class_mapping, transform=val_transforms, multilabel=is_multilabel)

    model = RegNetY.load_from_checkpoint(checkpoint_path=args.ckpt_path, config=config, class_to_idx=class_mapping,
                                         device='cuda')
    model = model.eval()

    batch_size = 16
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    result = Trainer().predict(model, data_loader)

    preds = torch.cat([preds for preds, _ in result])
    labels = torch.cat([labels for _, labels in result])

    print(classification_report(labels.cpu().int().numpy(), preds.cpu().int().numpy(), target_names=class_names, zero_division=0))

    conf_mat = MultilabelConfusionMatrix(num_labels=num_classes).cuda()
    conf_mat(target=labels.cuda().int(), preds=preds.cuda().int())  # Shape: [num_labels, 2, 2]
    cm = conf_mat.compute().detach().cpu().numpy()

    plot(cm, class_names)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str,
                        default="manual_checkpoints/apricot-sweep-7/apricot-sweep-7_epoch75_val_mAP_weighted0.79.ckpt",
                        help="Path to the checkpoint file")
    parser.add_argument("--run_id", default="8jyg3sk9", type=str, help="Run ID to fetch")
    parser.add_argument("--wandb_project", default="SEER", type=str, help="Wandb project name")
    parser.add_argument("--entity", default="mvrcii_", type=str, help="Wandb entity name")
    parser.add_argument("--mode", choices=['val', 'test'], default='val', help="Mode to evaluate the model")
    arguments = parser.parse_args()

    main(arguments)
