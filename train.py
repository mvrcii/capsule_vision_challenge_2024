import argparse
import logging
import os
import warnings
from datetime import datetime
from typing import Tuple

import albumentations as A
import torch
import yaml
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, ModelSummary
from lightning.pytorch.loggers import WandbLogger

import wandb
from src.data.datamodule import DataModule
from src.models.enums.finetune_mode import FineTuneMode
from src.models.regnety.regnety import RegNetY
from src.models.vit.vit import ViT
from src.utils.class_mapping import load_class_mapping
from src.utils.transform_utils import load_transforms

warnings.filterwarnings("ignore", category=UserWarning, module='pydantic')

torch.set_float32_matmul_precision('medium')


class TrainHandler:
    def __init__(self, args) -> None:
        self.wandb_logger, self.callbacks = self.__preparations(args)
        self.data_module = TrainHandler.__prepare_data_module(args)
        self.class_mapping = self.data_module.class_to_index
        self.trainer = self.__prepare_trainer(args)
        self.model = self.__prepare_model(args)

    def train(self):
        self.trainer.fit(model=self.model, datamodule=self.data_module)

    def test(self):
        self.trainer.test(model=self.model, dataloaders=self.data_module.test_dataloader(), ckpt_path='best')

    @staticmethod
    def __preparations(config):
        seed_everything(config.seed)
        sweep_id = wandb.run.sweep_id
        metric = config.metric

        wandb_logger = WandbLogger(experiment=wandb.run)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = wandb_logger.experiment.name

        if sweep_id:
            # FORMAT: <checkpoint_dir>/sweep-<sweep_id>/
            directory_name = f"sweep-{sweep_id}"
            checkpoint_dir = os.path.join(config.checkpoint_dir, directory_name)
            os.makedirs(checkpoint_dir, exist_ok=True)
            filename = f"{run_name}_epoch{{epoch:02d}}_{metric}{{{metric}:.2f}}"
        else:
            # FORMAT: <checkpoint_dir>/run-<timestamp>-<run_name>/
            directory_name = f"run-{timestamp}-{run_name}"
            checkpoint_dir = os.path.join(config.checkpoint_dir, directory_name)
            os.makedirs(checkpoint_dir, exist_ok=True)
            filename = f"best_epoch{{epoch:02d}}_{metric}{{{metric}:.2f}}"

        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=filename,
            save_top_k=1,
            save_weights_only=False,
            mode='max',
            monitor=metric,
            verbose=config.verbose,
            auto_insert_metric_name=False
        )

        callbacks = [
            checkpoint_callback,
            ModelSummary(max_depth=0),
            LearningRateMonitor(logging_interval="epoch"),
        ]

        return wandb_logger, callbacks

    @staticmethod
    def __prepare_transforms(args) -> Tuple[A.Compose, A.Compose]:
        img_size = args.img_size
        logging.info(f"Using provided image size: {img_size}")

        train_transforms, val_transforms = load_transforms(img_size=img_size, transform_path=args.transform_path)

        with open(args.transform_path, 'r') as f:
            wandb.log({'transforms': f.read()})

        return train_transforms, val_transforms

    @staticmethod
    def __get_batch_size(args) -> Tuple[int, int]:
        # Detect GPU and VRAM
        if torch.cuda.is_available():
            total_vram = 0
            for i in range(torch.cuda.device_count()):
                vram = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3  # VRAM in GB
                total_vram += vram  # Summing up VRAM if multiple GPUs
                logging.info(f"GPU {i}: {torch.cuda.get_device_name(i)}, Total VRAM: {vram:.2f} GB")
        else:
            logging.info("No CUDA GPUs are available")

        train_bs = 64
        val_bs = train_bs

        # If given, override with the provided batch sizes
        if args.train_bs:
            train_bs = args.train_bs
        if args.val_bs:
            val_bs = args.val_bs

        logging.info(f"Batch sizes: Train: {train_bs}, Val: {val_bs}")
        wandb.config.update({"train_bs": train_bs, "val_bs": val_bs}, allow_val_change=True)

        return train_bs, val_bs

    @staticmethod
    def __prepare_data_module(args) -> DataModule:
        class_mapping = TrainHandler.__prepare_class_mapping(args)
        transforms = TrainHandler.__prepare_transforms(args)
        train_bs, val_bs = TrainHandler.__get_batch_size(args)

        return DataModule(
            class_mapping=class_mapping,
            transforms=transforms,
            train_bs=train_bs,
            val_bs=val_bs,
            dataset_path=args.dataset_path,
            dataset_csv_path=args.dataset_csv_path,
            fold_idx=args.fold_id,
            num_workers=args.num_workers,
            train_frac=args.train_frac,
            val_frac=args.val_frac
        )

    @staticmethod
    def __prepare_class_mapping(args):
        class_mapping_path = os.path.join(args.dataset_csv_path, args.class_mapping_filename)
        return load_class_mapping(class_mapping_path)

    def __prepare_trainer(self, args) -> Trainer:
        accelerator = "mps" if torch.backends.mps.is_available() else (
            "gpu" if torch.cuda.is_available() else "cpu")
        ddp_training = args.num_devices > 1 or args.num_nodes > 1
        devices = "auto" if args.num_devices == 1 else args.num_devices
        strategy = "ddp" if ddp_training else "auto"
        return Trainer(
            logger=self.wandb_logger,
            accelerator=accelerator,
            devices=devices,
            num_nodes=args.num_nodes,
            max_epochs=args.max_epochs,
            callbacks=self.callbacks,
            precision="16-mixed",
            gradient_clip_val=0.5,
            enable_model_summary=False,
            strategy=strategy
        )

    def __prepare_model(self, config):
        checkpoint_path = TrainHandler.__get_checkpoint_path(config)
        model_cls = TrainHandler.__get_model_cls(config.model_type)
        return model_cls(config=config, class_to_idx=self.class_mapping, checkpoint_path=checkpoint_path)

    @staticmethod
    def __get_model_cls(model_type):
        if model_type == "seer":
            return RegNetY
        elif model_type == "vit":
            return ViT
        else:
            raise ValueError(f"Model type {model_type} is not supported.")

    @staticmethod
    def __get_checkpoint_path(config):
        checkpoint_filename = getattr(config, "checkpoint_filename", None)
        checkpoint_dir = getattr(config, "pretrained_checkpoint_dir", None)
        checkpoint_path = None
        if checkpoint_filename and checkpoint_dir:
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
        return checkpoint_path


def main(args):
    config_args = {}
    project_conf_path = "configs/project.yaml"
    local_conf_path = "configs/local.yaml"

    # Override argparse arguments with project config
    if os.path.exists(project_conf_path):
        with open(project_conf_path, 'r') as f:
            project_conf_path = yaml.safe_load(f)
            for key, value in project_conf_path.items():
                setattr(args, key, value)

    # Read arguments from passed config
    if args.config:
        with open(args.config, 'r') as f:
            config_args = yaml.safe_load(f)
            for key, value in config_args.items():
                setattr(args, key, value)

    # Overwrite argparse arguments with local config
    if os.path.exists(local_conf_path):
        with open(local_conf_path, 'r') as f:
            local_config_args = yaml.safe_load(f)
            for key, value in local_config_args.items():
                setattr(args, key, value)

    # Ensure argparse arguments override config file arguments if provided
    args_dict = vars(args)
    for key, value in args_dict.items():
        if value is None and key in config_args:
            setattr(args, key, config_args[key])

    wandb.init(
        config=args,
        entity=args.entity,
        project=args.wandb_project,
        reinit=True
    )

    args = argparse.Namespace(**wandb.config)

    logging.basicConfig(
        format='%(levelname)s: %(message)s',
        level=logging.INFO
    )

    trainer = TrainHandler(args)
    trainer.train()
    # trainer.test()

    wandb.finish()


def arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--reduced_data_mode", action="store_true",
                        help="Enable reduced data mode to use less data for faster iteration")
    parser.add_argument("--wandb_project", default="SEER", type=str)
    parser.add_argument("--entity", default="mvrcii_", type=str)

    # === Paths ===
    parser.add_argument("--checkpoint_filename", type=str)
    parser.add_argument("--checkpoint_dir", default="checkpoints", type=str,
                        help="Directory to save checkpoints and logs")
    parser.add_argument("--pretrained_checkpoint_dir", type=str,
                        help="Directory to load pretrained checkpoints from")

    parser.add_argument("--dataset_path", default="../data/", type=str)
    parser.add_argument("--dataset_csv_path", default="../endoscopy/data_alpha", type=str)
    parser.add_argument("--class_mapping_filename", default="class_mapping.json", type=str)
    parser.add_argument("--transform_path", default="configs/transforms/base_transforms.py", type=str)

    # === Training ===
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--fold_id", default=0, type=int)
    parser.add_argument("--max_epochs", default=100, type=int)
    parser.add_argument("--train_bs", default=64, type=int)
    parser.add_argument("--val_bs", default=64, type=int)
    parser.add_argument("--num_workers", default=16, type=int)

    parser.add_argument("--num_nodes", default=1, type=int)
    parser.add_argument("--num_devices", default=1, type=int)

    # === Training Modes ===
    parser.add_argument("--ft_mode", type=str, choices=[mode.value for mode in FineTuneMode], default='head',
                        help="Fine-tune mode: 'head' only the head, 'backbone' only the backbone, or 'full' both head and backbone.")
    parser.add_argument("--metric", type=str, choices=['val_mAP_weighted', 'val_AUC_macro', 'val_f1_weighted'],
                        default='val_AUC_macro', help="Metric to optimize for during training.")
    parser.add_argument("--train_frac", type=float, default=1, help="Fraction of training data to use")
    parser.add_argument("--val_frac", type=float, default=1, help="Fraction of validation data to use")

    # === Model ===
    parser.add_argument("--model_arch", default="regnety_640.seer", type=str)
    parser.add_argument("--model_type", default="seer", type=str)
    parser.add_argument("--img_size", default=384, type=int)

    # === Optimizer ===
    parser.add_argument("--optimizer", default="adabelief", type=str)
    parser.add_argument("--lr", default=0.03, type=float)
    parser.add_argument("--weight_decay", default=2e-4, type=float)

    # === Scheduler ===
    parser.add_argument("--scheduler", default="cosine", type=str)
    parser.add_argument("--eta_min", type=float, default=0, help="Minimum learning rate for the scheduler")
    parser.add_argument("--lambda_factor", type=float, default=0.95, help="Lambda for the scheduler")

    return parser


if __name__ == '__main__':
    arguments = arg_parser().parse_args()
    main(arguments)
