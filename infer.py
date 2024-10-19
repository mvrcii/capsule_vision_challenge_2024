import argparse
import logging
import os
import warnings

import numpy as np
import pandas as pd
import torch
import wandb
from matplotlib import pyplot as plt
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.data.dataset import ImageDataset
from src.models.timm.timm_model import TimmModel
from src.utils.class_mapping import load_class_mapping
from src.utils.transform_utils import load_transforms

warnings.filterwarnings("ignore", ".*A new version of Albumentations is*")


def save_predictions_to_excel(image_paths, y_pred, output_path):
    class_columns = ['Angioectasia', 'Bleeding', 'Erosion', 'Erythema', 'Foreign Body', 'Lymphangiectasia', 'Normal',
                     'Polyp', 'Ulcer', 'Worms']
    y_pred_classes = np.argmax(y_pred, axis=1)
    predicted_class_names = [class_columns[i] for i in y_pred_classes]
    df_prob = pd.DataFrame(y_pred, columns=class_columns)
    df_prob.insert(0, 'image_path', image_paths)
    df_class = pd.DataFrame({'image_path': image_paths, 'predicted_class': predicted_class_names})
    df_merged = pd.merge(df_prob, df_class, on='image_path')
    df_merged.to_excel(output_path, index=False)


def prepare_model(ckpt_path, config, class_to_idx):
    """Load the model from the checkpoint."""
    if not ckpt_path:
        raise ValueError("Checkpoint path not provided")
    print(f"Loading model from checkpoint: {ckpt_path}")
    model = TimmModel.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        config=config,
        class_to_idx=class_to_idx
    )
    if not model:
        raise ValueError("Failed to load model from checkpoint")
    model.eval()
    return model


def load_data(dataset_csv_path, dataset_path):
    """
    Load the test dataset CSV and prepend the dataset path to frame paths.

    Args:
        dataset_csv_path (str): Path to the directory containing CSV files.
        dataset_path (str): Path to the dataset directory.

    Returns:
        pd.DataFrame: DataFrame with updated frame paths.
    """
    test_path = os.path.join(dataset_csv_path, 'test.csv')
    X_test = pd.read_csv(test_path)
    X_test['frame_path'] = X_test['frame_path'].apply(lambda x: os.path.join(dataset_path, x))
    return X_test


def get_batch(dataset, start_idx, batch_size):
    """
    Retrieve a batch of images and labels from the dataset.

    Args:
        dataset (ImageDataset): The dataset to retrieve data from.
        start_idx (int): Starting index of the batch.
        batch_size (int): Number of samples in the batch.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Batch of images and labels.
    """
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


def get_image_batch(dataset, start_idx, batch_size):
    """
    Retrieve a batch of images from the dataset (no labels).

    Args:
        dataset (ImageDataset): The dataset to retrieve data from.
        start_idx (int): Starting index of the batch.
        batch_size (int): Number of samples in the batch.

    Returns:
        torch.Tensor: Batch of images.
    """
    max_idx = len(dataset)
    end_idx = min(start_idx + batch_size, max_idx)
    batch_images = []

    for i in range(start_idx, end_idx):
        # Assuming dataset returns only image when class_to_index is not provided
        image = dataset[i]
        batch_images.append(image)

    # Stack the list of images into tensors
    batch_images = torch.stack(batch_images)

    return batch_images


def unnormalize_image(image, mean, std):
    """
    Unnormalize a tensor image and convert to numpy array.

    Args:
        image (numpy.ndarray): Image array in [C, H, W].
        mean (list): Mean used for normalization.
        std (list): Std used for normalization.

    Returns:
        numpy.ndarray: Unnormalized image in [H, W, C].
    """
    mean = np.array(mean).reshape(1, 1, 3)
    std = np.array(std).reshape(1, 1, 3)

    image = image.transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
    image = image * std + mean
    image = np.clip(image, 0, 1)
    return image


def plot_batch_with_cam(images, predicted_labels, actual_labels=None, idx_to_class=None,
                        grad_cam=None, target_layers=None, batch_number=1,
                        save_plots=False, save_dir='plots'):
    """
    Plots a batch of images with predicted (and actual) labels and GradCAM overlays.

    Args:
        images (torch.Tensor): Batch of images [batch_size, C, H, W]
        predicted_labels (torch.Tensor): Predicted labels [batch_size]
        actual_labels (torch.Tensor, optional): Actual labels [batch_size]
        idx_to_class (dict, optional): Mapping from index to class name
        grad_cam (GradCAM): Initialized GradCAM object
        target_layers (list): Target layers for GradCAM
        batch_number (int): The current batch number for labeling/saving.
        save_plots (bool): Whether to save the plotted images to disk.
        save_dir (str): Directory to save the plots if save_plots is True.
    """
    batch_size = images.size(0)
    grid_size = int(np.ceil(np.sqrt(batch_size)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = axes.flatten()

    for i in range(batch_size):
        image = images[i].cpu().numpy()
        image_unnorm = unnormalize_image(image, mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        # Prepare image for GradCAM
        input_tensor = images[i].unsqueeze(0)  # Add batch dimension
        input_tensor = input_tensor.cuda() if torch.cuda.is_available() else input_tensor

        # Get the predicted class index
        pred_idx = predicted_labels[i].item()
        pred_class = idx_to_class[pred_idx] if idx_to_class else pred_idx

        # Define target for GradCAM
        target = ClassifierOutputTarget(pred_idx)

        # Compute GradCAM
        grayscale_cam = grad_cam(
            input_tensor=input_tensor,
            targets=[target]
        )
        grayscale_cam = grayscale_cam[0, :]  # Remove batch dimension

        # Overlay CAM on image
        cam_image = show_cam_on_image(image_unnorm, grayscale_cam, use_rgb=True, image_weight=0.5)

        ax = axes[i]
        ax.imshow(cam_image)
        if idx_to_class:
            if actual_labels is not None:
                actual_idx = actual_labels[i].item()
                actual_class = idx_to_class[actual_idx]
                title = f"P: {pred_class}\nA: {actual_class}"
            else:
                title = f"P: {pred_class}"
            ax.set_title(title, fontsize=12)
        ax.axis('off')

    # Hide any remaining subplots if batch_size is not a perfect square
    for i in range(batch_size, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    if save_plots:
        os.makedirs(save_dir, exist_ok=True)
        plot_path = os.path.join(save_dir, f'batch_{batch_number}.png')
        plt.savefig(plot_path, bbox_inches='tight')
        print(f"Saved plot to {plot_path}")
    plt.show()


def plot_batch_normal(images, predicted_labels, actual_labels=None, idx_to_class=None,
                      batch_number=1, save_plots=False, save_dir='plots'):
    """
    Plots a batch of images with predicted (and actual) labels.

    Args:
        images (torch.Tensor): Batch of images [batch_size, C, H, W]
        predicted_labels (torch.Tensor): Predicted labels [batch_size]
        actual_labels (torch.Tensor, optional): Actual labels [batch_size]
        idx_to_class (dict, optional): Mapping from index to class name
        batch_number (int): The current batch number for labeling/saving.
        save_plots (bool): Whether to save the plotted images to disk.
        save_dir (str): Directory to save the plots if save_plots is True.
    """
    batch_size = images.size(0)
    grid_size = int(np.ceil(np.sqrt(batch_size)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = axes.flatten()

    for i in range(batch_size):
        image = images[i].cpu().numpy()
        image_unnorm = unnormalize_image(image, mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        ax = axes[i]
        ax.imshow(image_unnorm)
        if idx_to_class:
            if actual_labels is not None:
                actual_idx = actual_labels[i].item()
                actual_class = idx_to_class[actual_idx]
                title = f"P: {idx_to_class[predicted_labels[i].item()]} \nA: {actual_class}"
            else:
                title = f"P: {idx_to_class[predicted_labels[i].item()]}"
            ax.set_title(title, fontsize=12)
        ax.axis('off')

    # Hide any remaining subplots if batch_size is not a perfect square
    for i in range(batch_size, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    if save_plots:
        os.makedirs(save_dir, exist_ok=True)
        plot_path = os.path.join(save_dir, f'batch_{batch_number}.png')
        plt.savefig(plot_path, bbox_inches='tight')
        print(f"Saved plot to {plot_path}")
    plt.show()


def main(args):
    # Initialize Weights & Biases API and fetch run config
    api = wandb.Api()
    run = api.run(f'{args.entity}/{args.wandb_project}/{args.run_id}')
    config = argparse.Namespace(**run.config)

    submission = args.submission
    mode = args.mode

    if args.dataset_path:
        config.dataset_path = args.dataset_path

    if args.dataset_csv_path:
        config.dataset_csv_path = args.dataset_csv_path

    class_mapping = load_class_mapping(os.path.join(config.dataset_csv_path, 'class_mapping.json'))
    idx_to_class = {v: k for k, v in class_mapping.items()}

    _, val_transforms = load_transforms(img_size=config.img_size, transform_path=config.transform_path)

    test_df = load_data(config.dataset_csv_path, config.dataset_path)

    # Initialize Dataset
    test_dataset = ImageDataset(test_df, transform=val_transforms)

    # Load Model
    model = prepare_model(ckpt_path=args.ckpt_path, config=config, class_to_idx=class_mapping)
    if torch.cuda.is_available():
        model = model.cuda()

    # Define GradCAM target layers if GradCAM is activated
    if args.use_gradcam and not submission:
        target_layers = [model.backbone.blocks[-1].attn]  # Adjust based on your model's architecture
        grad_cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
    else:
        grad_cam = None

    batch_size = 8
    num_batches = args.num_batches
    start_idx = args.start_idx

    # Total possible batches
    total_possible_batches = (len(test_dataset) - start_idx) // batch_size

    if submission:
        num_batches = total_possible_batches

    if num_batches > total_possible_batches:
        logging.warning(
            f"Requested {num_batches} batches, but only {total_possible_batches} are available. Adjusting to {total_possible_batches}.")
        num_batches = total_possible_batches

    for batch_num in range(1, num_batches + 1):
        current_start_idx = start_idx + (batch_num - 1) * batch_size

        if mode == 'test':
            images = get_image_batch(test_dataset, start_idx=current_start_idx, batch_size=batch_size)
            labels = None
        elif mode == 'val':
            images, labels = get_batch(test_dataset, start_idx=current_start_idx, batch_size=batch_size)
        else:
            raise ValueError("Invalid mode. Choose from 'val' or 'test'")

        if torch.cuda.is_available():
            images = images.cuda()
            if labels is not None:
                labels = labels.cuda()

        # Forward pass
        with torch.no_grad():
            # Assuming model.forward returns (logits, feature_maps)
            logits, _ = model.forward(images)

        pred_outputs = torch.softmax(logits, dim=1)

        if submission:
            # Save prediction probabilities to XLSX file
            os.makedirs('submission', exist_ok=True)
            img_paths = test_df['frame_path'].values[current_start_idx:current_start_idx + batch_size]
            save_predictions_to_excel(img_paths, )

        # Prediction
        predicted_labels = torch.argmax(pred_outputs, dim=1)

        # Print predictions
        for i in range(predicted_labels.size(0)):
            predicted_label = idx_to_class[predicted_labels[i].item()]
            if mode == 'val':
                actual_label = idx_to_class[labels[i].item()]
                print(f"Batch {batch_num}, Image {i + 1}: Predicted: {predicted_label}, Actual: {actual_label}")
            else:
                print(f"Batch {batch_num}, Image {i + 1}: Predicted: {predicted_label}")

        # Plotting the batch
        if args.use_gradcam and grad_cam is not None:
            plot_batch_with_cam(
                images,
                predicted_labels,
                actual_labels=labels if mode == 'val' else None,
                idx_to_class=idx_to_class,
                grad_cam=grad_cam,
                target_layers=None,  # Not needed here
                batch_number=batch_num,
                save_plots=args.save_plots,
                save_dir=args.save_dir
            )
        else:
            plot_batch_normal(
                images,
                predicted_labels,
                actual_labels=labels if mode == 'val' else None,
                idx_to_class=idx_to_class,
                batch_number=batch_num,
                save_plots=args.save_plots,
                save_dir=args.save_dir
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict and plot multiple batches of images with labels.")

    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the checkpoint file")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--dataset_csv_path", type=str,
                        help="Path to the dataset csv files. Allows for overriding if required. (Default is the path from the checkpoint config)")
    parser.add_argument("--run_id", type=str, required=True, help="Run ID to fetch")
    parser.add_argument("--wandb_project", default="SEER", type=str, help="Wandb project name")
    parser.add_argument("--entity", default="mvrcii_", type=str, help="Wandb entity name")
    parser.add_argument("--mode", choices=['val', 'test'], default='val', help="Mode to evaluate the model")
    parser.add_argument("--num_batches", type=int, default=1, help="Number of consecutive batches to process")
    parser.add_argument("--start_idx", type=int, default=0, help="Starting index for batch processing")
    parser.add_argument("--save_plots", action='store_true', help="Whether to save the plotted images to disk")
    parser.add_argument("--save_dir", type=str, default='plots', help="Directory to save plots if save_plots is True")

    arguments = parser.parse_args()

    main(arguments)
