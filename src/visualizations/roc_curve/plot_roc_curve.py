import argparse
import os
from typing import Tuple

import albumentations as A
import numpy as np
import seaborn as sns
import torch
import wandb
from albumentations.pytorch import ToTensorV2
from lightning import Trainer
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

from src.data.datamodule import DataModule
from src.models.enums.finetune_mode import FineTuneMode
from src.models.enums.model_mode import ModelMode
from src.models.regnety.regnety import RegNetY
from src.utils.class_mapping import load_class_mapping


def plot_roc_curve(save_dir: str, preds: np.array, labels: np.array, class_mapping):
    print("Creating ROC curve plot")
    sns.set(style="whitegrid", context="poster", palette="bright")
    sns.set_style('ticks')

    class_labels = list(class_mapping.keys())
    num_classes = len(class_labels)

    if labels.ndim == 1 or labels.shape[1] == 1:
        micro_labels = label_binarize(labels, classes=np.arange(num_classes)).ravel()
    else:
        micro_labels = labels.ravel()

    micro_preds = preds.ravel()

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []

    # Create a subplot figure with 1 row and 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(28, 10))
    fig.subplots_adjust(hspace=0.4, wspace=0.4, bottom=0.15)

    ax1 = axes[0]
    for i in range(num_classes):
        binary_labels = (labels == i)
        fpr, tpr, _ = roc_curve(binary_labels, preds[:, i])
        auc_score = auc(fpr, tpr)
        aucs.append(auc_score)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        tprs.append(interp_tpr)
        sns.lineplot(x=fpr, y=tpr, ax=ax1, label=f'{class_labels[i]} (AUC = {auc_score:.2f})')

    sns.lineplot(x=[0, 1], y=[0, 1], ax=ax1, color="gray", linestyle='--', label='Random Classifier')
    ax1.set_title('ROC Curves for Each Class')
    ax1.set_xlabel('False Positive Rate (Specificity)')
    ax1.set_ylabel('True Positive Rate (Sensitivity)')
    ax1.legend(loc='lower right', fontsize='small', title_fontsize='medium')

    # Calculate and plot the micro-average ROC curve
    micro_fpr, micro_tpr, _ = roc_curve(micro_labels, micro_preds)
    micro_auc = auc(micro_fpr, micro_tpr)
    sns.lineplot(x=micro_fpr, y=micro_tpr, ax=axes[1], color='blue', label=f'Micro-average ROC (AUC = {micro_auc:.2f})')

    # Macro-average ROC curve
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    mean_tpr[0] = 0.0
    sns.lineplot(x=mean_fpr, y=mean_tpr, ax=axes[1], color='red', label=f'Macro-average ROC (AUC = {mean_auc:.2f})')

    sns.lineplot(x=[0, 1], y=[0, 1], ax=axes[1], color="gray", linestyle='--', label='Random Classifier')
    axes[1].set_title('Macro and Micro-average ROC Curves')
    axes[1].set_xlabel('False Positive Rate (Specificity)')
    axes[1].set_ylabel('True Positive Rate (Sensitivity)')
    axes[1].legend(loc='lower right', fontsize='small', title_fontsize='medium')

    if not os.path.exists(save_dir):
        print(f"Skip saving the plot as the directory does not exist: {save_dir}")
        return

    plt.tight_layout()

    plt.savefig(os.path.join(save_dir, 'roc_curve.png'))
    print("Successfully saved ROC curve plot")

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


def prepare_datamodule(config, class_mapping, num_workers=8):
    return DataModule(
        class_mapping=class_mapping,
        transforms=get_transforms(config),
        train_bs=config.train_bs,
        val_bs=config.val_bs,
        dataset_path=config.dataset_path,
        dataset_csv_path=config.dataset_csv_path,
        fold_idx=config.fold_id,
        num_workers=num_workers,
        multilabel=True
    )


def prepare_trainer(config) -> Trainer:
    accelerator = "mps" if torch.backends.mps.is_available() else (
        "gpu" if torch.cuda.is_available() else "cpu")

    return Trainer(
        devices=1,
        max_epochs=config.max_epochs,
        accelerator=accelerator,
        precision="16-mixed",
        gradient_clip_val=0.5,
        enable_progress_bar=True,
        enable_model_summary=False,
        inference_mode=True
    )


def extract_results(results) -> Tuple[torch.Tensor, torch.Tensor]:
    print("Extracting prediction results")
    predictions_list = []
    labels_list = []
    for batch_results in results:
        logits, labels = batch_results[0], batch_results[1]
        probabilities = torch.softmax(logits.detach().cpu(), dim=1)
        predictions_list.append(probabilities)
        labels_list.append(labels.detach().cpu())

    predictions_tensor = torch.cat(predictions_list, dim=0)
    labels_tensor = torch.cat(labels_list, dim=0)

    return predictions_tensor, labels_tensor


def main(args):
    api = wandb.Api()
    run = api.run(f'{args.entity}/{args.wandb_project}/{args.run_id}')

    config = argparse.Namespace(**run.config)
    config.model_mode = ModelMode.MULTI_CLASS
    config.ft_mode = FineTuneMode.HEAD
    config.train_bs = 4
    config.val_bs = 4

    class_mapping = load_class_mapping(os.path.join(args.dataset_csv_path, args.class_mapping_filename))

    print(f"Checkpoint path: {args.ckpt_path}")
    if not args.ckpt_path:
        return None

    model = RegNetY.load_from_checkpoint(checkpoint_path=args.ckpt_path, config=config, class_to_idx=class_mapping)
    model = model.eval()

    trainer = prepare_trainer(config)
    data_module = prepare_datamodule(config, class_mapping=class_mapping)
    data_module.setup()
    inputs, labels = next(iter(data_module.train_dataloader()))

    results = trainer.predict(model, dataloaders=data_module.test_dataloader())

    preds, labels = extract_results(results)

    save_dir = os.path.dirname(args.ckpt_path)
    plot_roc_curve(save_dir, preds.numpy(), labels.numpy(), class_mapping)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_id", default="texcr5p8", type=str, help="Run ID to fetch")
    parser.add_argument("--wandb_project", default="SEER", type=str, help="Wandb project name")
    parser.add_argument("--entity", default="mvrcii_", type=str, help="Wandb entity name")

    parser.add_argument("--ckpt_path", type=str,
                        default="manual_checkpoints/autumn-sweep-2/autumn-sweep-2_epoch77_val_mAP_weighted0.82.ckpt",
                        help="Path to the checkpoint file")

    parser.add_argument("--dataset_csv_path", type=str, help="Path to the dataset CSV file")
    parser.add_argument("--class_mapping_filename", type=str, default="class_mapping.json")

    parser.add_argument("--mode", choices=['val', 'test'], default='val', help="Mode to evaluate the model")
    arguments = parser.parse_args()

    main(arguments)
