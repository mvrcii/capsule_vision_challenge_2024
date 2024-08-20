import argparse
import os
from typing import Tuple

import albumentations as A
import numpy as np
import pandas as pd
import torch
from PIL import Image
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

import wandb
from src.data.dataset import ImageDataset
from src.models.regnety.regnety import RegNetY
from src.utils.class_mapping import load_class_mapping


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


def prepare_model(ckpt_path, config, class_to_idx):
    """Load the model from the checkpoint."""
    if not ckpt_path:
        raise ValueError("Checkpoint path not provided")
    print(f"Checkpoint path: {ckpt_path}")
    model = RegNetY.load_from_checkpoint(checkpoint_path=ckpt_path,
                                         output_layers=[5],
                                         config=config,
                                         class_to_idx=class_to_idx)
    if not model:
        raise ValueError("Failed to load model from checkpoint")
    return model


def visualize_cam(image, cam):
    cam_colormap = plt.cm.get_cmap('jet')(cam)  # Apply colormap (returns RGBA)
    cam_image = Image.fromarray((cam_colormap[:, :, :3] * 255).astype(np.uint8))  # Convert only RGB to image

    cam_resized = cam_image.resize(image.shape[:2], Image.Resampling.LANCZOS)
    cam_resized = np.array(cam_resized)

    # Assuming 'image' is a numpy array in HxWxC format and properly scaled for display (0 to 1 or 0 to 255)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')

    ax = plt.subplot(1, 2, 2)
    plt.imshow(image, alpha=0.8)  # Show the original image
    _ = plt.imshow(cam_resized, alpha=0.6)  # Overlay CAM with transparency
    plt.title('Class Activation Map')
    plt.axis('off')

    # Adding a color bar
    norm = Normalize(vmin=np.min(cam), vmax=np.max(cam))
    sm = plt.cm.ScalarMappable(cmap=plt.jet, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)

    plt.show()


def load_data(dataset_csv_path, dataset_path):
    test_path = os.path.join(dataset_csv_path, 'test.csv')
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
    batch_labels = torch.tensor(batch_labels)

    return batch_images, batch_labels


def unnormalize_image(image, mean, std):
    mean = np.array(mean)
    std = np.array(std)

    mean = mean.reshape(1, 1, 3)
    std = std.reshape(1, 1, 3)

    image = image * std + mean

    image = np.clip(image, 0, 1)
    return image


def main(args):
    api = wandb.Api()
    run = api.run(f'{args.entity}/{args.wandb_project}/{args.run_id}')
    config = argparse.Namespace(**run.config)

    class_mapping = load_class_mapping(os.path.join(config.dataset_csv_path, 'class_mapping.json'))

    X_test = load_data(config.dataset_csv_path, config.dataset_path)
    _, val_transforms = get_transforms(config)

    dataset = ImageDataset(X_test, class_mapping, transform=val_transforms)

    model = RegNetY.load_from_checkpoint(checkpoint_path=args.ckpt_path, config=config, class_to_idx=class_mapping)
    model = model.eval()

    batch_size = 8
    images, labels = get_batch(dataset, start_idx=3, batch_size=batch_size)

    if torch.cuda.is_available():
        images = images.cuda()
        labels = labels.cuda()
        model = model.cuda()

    with torch.no_grad():
        outputs, feature_maps = model.forward(images)

    predicted_label = torch.argmax(outputs, dim=1)

    for i in range(batch_size):
        print(f"Predicted: {predicted_label[i].item()}, Actual: {labels[i].item()}")

        cam_feature_maps = feature_maps['final_conv'][i].unsqueeze(0)
        cam = model.get_cam(cam_feature_maps, predicted_label[i].item())
        cam = cam.cpu().numpy()
        cam_resized = np.interp(cam, (cam.min(), cam.max()), (0, 1))

        orig_image = images[i].detach().cpu().numpy().transpose(1, 2, 0)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        orig_image = unnormalize_image(orig_image, mean, std)

        visualize_cam(orig_image, cam_resized)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str,
                        default="manual_checkpoints/autumn-sweep-2/autumn-sweep-2_epoch77_val_mAP_weighted0.82.ckpt",
                        help="Path to the checkpoint file")
    parser.add_argument("--run_id", default="texcr5p8", type=str, help="Run ID to fetch")
    parser.add_argument("--wandb_project", default="SEER", type=str, help="Wandb project name")
    parser.add_argument("--entity", default="mvrcii_", type=str, help="Wandb entity name")
    parser.add_argument("--mode", choices=['val', 'test'], default='val', help="Mode to evaluate the model")
    arguments = parser.parse_args()

    main(arguments)
