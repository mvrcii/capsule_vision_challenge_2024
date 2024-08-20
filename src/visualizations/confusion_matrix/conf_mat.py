import argparse
import os
from typing import Tuple

import albumentations as A
import torch
import wandb
from albumentations.pytorch import ToTensorV2
from lightning import Trainer
from matplotlib import pyplot as plt

from src.data.datamodule import DataModule
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


def main(args):
    api = wandb.Api()

    runs = []
    if args.sweep_id:
        sweep = api.sweep(f'{args.entity}/{args.wandb_project}/{args.sweep_id}')
        runs = sweep.runs
    elif args.run_id:
        runs = [api.run(f'{args.entity}/{args.wandb_project}/{args.run_id}')]

    for run in runs:
        print("Processing Run:", run.id, run.name)

        config = argparse.Namespace(**run.config)
        config.checkpoint_dir = "manual_checkpoints"

        device = "mps" if torch.backends.mps.is_available() else ("gpu" if torch.cuda.is_available() else "cpu")
        class_mapping = load_class_mapping(os.path.join(config.dataset_csv_path, 'class_mapping.json'))

        data_module = DataModule(
            transforms=get_transforms(config),
            train_bs=config.train_bs,
            val_bs=4,
            dataset_path=config.dataset_path,
            dataset_csv_path=config.dataset_csv_path,
            fold_idx=config.fold_id,
            num_workers=config.num_workers,
            class_mapping=class_mapping
        )
        data_module.setup()
        if args.mode == 'val':
            dataloader = data_module.val_dataloader()
        elif args.mode == 'test':
            dataloader = data_module.test_dataloader()
        else:
            raise ValueError("Invalid mode")

        trainer = Trainer(
            devices=1,
            max_epochs=config.max_epochs,
            accelerator=device,
            precision="16-mixed",
            gradient_clip_val=0.5,
            enable_progress_bar=True,
            enable_model_summary=False,
            inference_mode=True
        )

        ckpt_dir = f"manual_checkpoints/{run.name}"
        ckpt_path = None
        os.listdir(ckpt_dir)
        for file in os.listdir(ckpt_dir):
            if file.endswith(".ckpt"):
                ckpt_path = os.path.join(ckpt_dir, file)
        if not ckpt_path:
            raise FileNotFoundError("Checkpoint not found")

        model = RegNetY.load_from_checkpoint(config=config, class_to_idx=class_mapping, checkpoint_path=ckpt_path)
        model = model.eval()

        results = trainer.predict(model, dataloaders=dataloader)

        predictions_list = []
        labels_list = []
        for batch_results in results:
            predictions = batch_results[0]
            labels = batch_results[1]
            predictions_list.extend(predictions.cpu().numpy().tolist())
            labels_list.extend(labels.cpu().numpy().tolist())

        # wandb.init(
        #     project=run.project, entity=run.entity,
        #     config=run.config, resume='must', id=run.id
        # )
        #
        # fig = model.create_multiclass_conf_matrix(preds=predictions_list, labels=labels_list, epoch=epoch)
        # wandb.log({f"best_{args.mode}_conf_mat": wandb.Image(fig)})
        # plt.close()
        #
        # wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--sweep_id",
                        type=str, help="Sweep ID to fetch runs from")
    parser.add_argument("-r", "--run_id", default="8jyg3sk9",
                        type=str, help="Run ID to fetch the best model from")
    parser.add_argument("--mode", choices=['val', 'test'], default='val', help="Mode to evaluate the model")
    parser.add_argument("--wandb_project", default="SEER", type=str, help="Wandb project name")
    parser.add_argument("--entity", default="mvrcii_", type=str, help="Wandb entity name")
    arguments = parser.parse_args()

    main(arguments)
