import logging
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import wandb
from adabelief_pytorch import AdaBelief
from lightning import LightningModule
from matplotlib.colors import TwoSlopeNorm
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score, \
    MulticlassAveragePrecision, MulticlassAUROC, MulticlassConfusionMatrix, MultilabelF1Score, MultilabelPrecision, \
    MultilabelRecall, MultilabelAveragePrecision, MultilabelAUROC, MultilabelConfusionMatrix

from src.models.enums.finetune_mode import FineTuneMode
from src.models.enums.model_mode import ModelMode
from src.models.linear_classifier import LinearClassifier


class AbstractLightningModule(LightningModule, ABC):
    def __init__(self, config, class_to_idx, checkpoint_path=None):
        super().__init__()
        self.config = config
        self.model_arch = config.model_arch
        self.model_type = config.model_type
        self.lr = float(config.lr)
        self.weight_decay = float(config.weight_decay)

        self.checkpoint_path = checkpoint_path
        self.verbose = config.verbose

        self.class_names = list(class_to_idx.keys())
        self.num_classes = len(self.class_names)

        model_mode_str = getattr(config, 'model_mode', 'multi-class')
        self.model_mode = ModelMode(model_mode_str) if isinstance(model_mode_str, str) else model_mode_str

        self.val_labels = []
        self.val_preds = []
        self.test_labels = []
        self.test_preds = []

        # Used for postprocessing predictions
        self.predict_labels = []
        self.predict_preds = []

        self.backbone = self.init_backbone()
        self.classifier = self.init_classifier()

        self.__setup_model_fine_tuning()
        self.__setup_criterion()
        self.__setup_metrics()

        self.best_val_AUC_macro = float('-inf')

    @abstractmethod
    def init_backbone(self):
        """
        Initialize the classifier part of the model.
        Can be overridden by subclasses if necessary.
        """
        pass

    def init_classifier(self):
        return LinearClassifier(in_features=self.backbone.num_features, num_classes=self.num_classes)

    def configure_optimizers(self):
        trainable_params = list(filter(lambda p: p.requires_grad, self.backbone.parameters())) + \
                           list(filter(lambda p: p.requires_grad, self.classifier.parameters()))
        if self.config.optimizer == 'adabelief':
            optimizer = AdaBelief(trainable_params, lr=self.lr, eps=1e-16, betas=(0.9, 0.999), weight_decouple=True,
                                  rectify=False, weight_decay=self.weight_decay, print_change_log=False)
        elif self.config.optimizer == 'adamw':
            optimizer = torch.optim.Adam(trainable_params, lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Invalid optimizer: {self.config.optimizer}")

        if self.config.scheduler == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=self.config.eta_min)
        elif self.config.scheduler == 'linear':
            end_factor = self.config.eta_min / self.config.lr
            scheduler = LinearLR(optimizer, start_factor=1., end_factor=end_factor, total_iters=self.trainer.max_epochs)
        else:
            raise ValueError(f"Invalid scheduler: {self.config.scheduler}")

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_AUC_macro"}}

    def forward(self, x):
        """
        Forward pass of the model. Takes an input tensor and returns the output logits and predictions.
        :param x: Input tensor
        :return:  Tuple of logits and predictions
        """
        features = self.backbone(x)
        logits = self.classifier(features)

        if self.model_mode == ModelMode.MULTI_LABEL:
            preds = torch.sigmoid(logits)  # Apply sigmoid
            preds = (preds > 0.5).float()  # Apply thresholding
        elif self.model_mode == ModelMode.MULTI_CLASS:
            preds = logits
        else:
            raise ValueError(f"Unsupported model mode: {self.model_mode}")

        return logits, preds

    def predict_step(self, batch, batch_idx):
        imgs, labels = batch
        labels = labels.float() if self.model_mode == ModelMode.MULTI_LABEL else labels.long()
        _, preds = self.forward(imgs)

        return preds, labels

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        labels = labels.float() if self.model_mode == ModelMode.MULTI_LABEL else labels.long()
        logits, preds = self.forward(imgs)

        return self.__calc_loss(logits, labels, mode='train')

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        labels = labels.float() if self.model_mode == ModelMode.MULTI_LABEL else labels.long()
        logits, preds = self.forward(imgs)
        self.__calc_loss(logits, labels, mode='val')
        self.__log_step_metrics(preds=preds, labels=labels, mode='val')
        self.__store_step_preds(preds=preds, labels=labels, mode='val')

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        labels = labels.float() if self.model_mode == ModelMode.MULTI_LABEL else labels.long()
        logits, preds = self.forward(imgs)
        self.__calc_loss(logits, labels, mode='test')
        self.__log_step_metrics(preds=preds, labels=labels, mode='test')
        self.__store_step_preds(preds=preds, labels=labels, mode='test')

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return

        self.__log_epoch_metrics(all_labels=self.val_labels, all_preds=self.val_preds, mode='val')

        current_val_metric = self.trainer.logged_metrics.get('val_AUC_macro')
        if current_val_metric > self.best_val_AUC_macro:
            self.best_val_AUC_macro = current_val_metric
            self.__log_conf_matrix(all_labels=self.val_labels, all_preds=self.val_preds, mode='val')
            self.__log_roc_curve(mode='val')

    def on_test_epoch_end(self):
        if self.trainer.sanity_checking:
            return

        self.__log_epoch_metrics(all_labels=self.test_labels, all_preds=self.test_preds, mode='test')
        self.__log_conf_matrix(all_labels=self.test_labels, all_preds=self.test_preds, mode='test')

    def __setup_model_fine_tuning(self):
        def support_deprecated(config):
            ft_head = getattr(config, 'ft_head', False)
            ft_backbone = getattr(config, 'ft_backbone', False)
            if ft_head and ft_backbone:
                return FineTuneMode.FULL
            elif ft_head and not ft_backbone:
                return FineTuneMode.HEAD
            elif not ft_head and ft_backbone:
                return FineTuneMode.BACKBONE
            else:
                return None

        ft_mode_dep = support_deprecated(self.config)
        ft_mode = ft_mode_dep if ft_mode_dep else self.config.ft_mode

        for param in self.classifier.parameters():
            param.requires_grad = False
        for param in self.backbone.parameters():
            param.requires_grad = False

        if ft_mode == FineTuneMode.HEAD.value or ft_mode == FineTuneMode.FULL.value:
            for param in self.classifier.parameters():
                param.requires_grad = True

        if ft_mode == FineTuneMode.BACKBONE.value or ft_mode == FineTuneMode.FULL.value:
            for param in self.backbone.parameters():
                param.requires_grad = True

        if self.verbose:
            logging.info(f"Fine-tuning mode set to: {ft_mode}")

    def __setup_criterion(self):
        model_mode = self.model_mode
        loss_fn_map = {
            ModelMode.MULTI_LABEL: nn.BCEWithLogitsLoss(),
            ModelMode.MULTI_CLASS: nn.CrossEntropyLoss()
        }
        try:
            self.criterion = loss_fn_map[model_mode]
        except KeyError:
            raise ValueError(f"Unsupported model mode: {model_mode}")

    def __setup_metrics(self):
        if self.model_mode == ModelMode.MULTI_LABEL:
            self.__setup_multi_label_metrics()
        elif self.model_mode == ModelMode.MULTI_CLASS:
            self.__setup_multi_class_metrics()
        else:
            raise ValueError(f"Unsupported model mode: {self.model_mode}")

    def __setup_multi_class_metrics(self):
        # Weighted Metrics for Overall Performance Evaluation
        self.precision_weighted = MulticlassPrecision(num_classes=self.num_classes, average="weighted")
        self.recall_weighted = MulticlassRecall(num_classes=self.num_classes, average="weighted")
        self.f1_weighted = MulticlassF1Score(num_classes=self.num_classes, average="weighted")
        self.mAP_weighted = MulticlassAveragePrecision(num_classes=self.num_classes, average="weighted")
        self.AUC_weighted = MulticlassAUROC(num_classes=self.num_classes, average="weighted")

        # Macro Metrics for Fairness Across Classes
        self.precision_macro = MulticlassPrecision(num_classes=self.num_classes, average="macro")
        self.recall_macro = MulticlassRecall(num_classes=self.num_classes, average="macro")
        self.f1_macro = MulticlassF1Score(num_classes=self.num_classes, average="macro")
        self.mAP_macro = MulticlassAveragePrecision(num_classes=self.num_classes, average="macro")
        self.AUC_macro = MulticlassAUROC(num_classes=self.num_classes, average="macro")

        # Per Class Metrics
        self.f1_per_class = MulticlassF1Score(num_classes=self.num_classes, average=None)
        self.mAP_per_class = MulticlassAveragePrecision(num_classes=self.num_classes, average=None)
        self.AUC_per_class = MulticlassAUROC(num_classes=self.num_classes, average=None)

    def __setup_multi_label_metrics(self):
        # Weighted Metrics for Overall Performance Evaluation
        self.precision_weighted = MultilabelPrecision(num_labels=self.num_classes, average="weighted")
        self.recall_weighted = MultilabelRecall(num_labels=self.num_classes, average="weighted")
        self.f1_weighted = MultilabelF1Score(num_labels=self.num_classes, average="weighted")
        self.mAP_weighted = MultilabelAveragePrecision(num_labels=self.num_classes, average="weighted")
        self.AUC_weighted = MultilabelAUROC(num_labels=self.num_classes, average="weighted")

        # Macro Metrics for Fairness Across Classes
        self.precision_macro = MultilabelPrecision(num_labels=self.num_classes, average="macro")
        self.recall_macro = MultilabelRecall(num_labels=self.num_classes, average="macro")
        self.f1_macro = MultilabelF1Score(num_labels=self.num_classes, average="macro")
        self.mAP_macro = MultilabelAveragePrecision(num_labels=self.num_classes, average="macro")
        self.AUC_macro = MultilabelAUROC(num_labels=self.num_classes, average="macro")

        # Per Class Metrics
        self.f1_per_class = MultilabelF1Score(num_labels=self.num_classes, average=None)
        self.mAP_per_class = MultilabelAveragePrecision(num_labels=self.num_classes, average=None)
        self.AUC_per_class = MultilabelAUROC(num_labels=self.num_classes, average=None)

    def __calc_loss(self, logits, labels, mode='train'):
        loss = self.criterion(logits, labels)
        self.log(f'{mode}_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def __log_step_metrics(self, preds, labels, mode):
        if self.trainer.sanity_checking:
            return

        self.log(f'{mode}_precision_weighted', self.precision_weighted(preds, labels), on_epoch=True, prog_bar=False)
        self.log(f'{mode}_recall_weighted', self.recall_weighted(preds, labels), on_epoch=True, prog_bar=False)
        self.log(f'{mode}_f1_weighted', self.f1_weighted(preds, labels), on_epoch=True, prog_bar=True)

        self.log(f'{mode}_precision_macro', self.precision_macro(preds, labels), on_epoch=True, prog_bar=False)
        self.log(f'{mode}_recall_macro', self.recall_macro(preds, labels), on_epoch=True, prog_bar=False)
        self.log(f'{mode}_f1_macro', self.f1_macro(preds, labels), on_epoch=True, prog_bar=False)

    def __store_step_preds(self, preds, labels, mode):
        if mode == 'val' or mode == 'test':
            getattr(self, f"{mode}_labels").extend(labels.detach().cpu().tolist())
            getattr(self, f"{mode}_preds").extend(preds.detach().cpu().tolist())

    def __log_epoch_metrics(self, all_labels, all_preds, mode='val'):
        labels = torch.tensor(all_labels, dtype=torch.int8)
        preds = torch.tensor(all_preds, dtype=torch.float16)

        # Log weighted and macro metrics that are computed over the epoch
        self.log(f"{mode}_mAP_weighted", self.mAP_weighted(preds, labels), on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{mode}_AUC_weighted", self.AUC_weighted(preds, labels), on_step=False, on_epoch=True, prog_bar=True)

        self.log(f"{mode}_mAP_macro", self.mAP_macro(preds, labels), on_step=False, on_epoch=True, prog_bar=False)
        self.log(f"{mode}_AUC_macro", self.AUC_macro(preds, labels), on_step=False, on_epoch=True, prog_bar=False)

        # Per-Class metrics logging
        mAP_pc = torch.nan_to_num(self.mAP_per_class(preds, labels), 0.0)
        AUC_pc = torch.nan_to_num(self.AUC_per_class(preds, labels), 0.0)

        for i, score in enumerate(mAP_pc):
            class_label = self.class_names[i]
            self.log(f"{mode}_mAP_{class_label}", score.item(), on_step=False, on_epoch=True, prog_bar=False)

        for i, score in enumerate(AUC_pc):
            class_label = self.class_names[i]
            self.log(f"{mode}_AUC_{class_label}", score.item(), on_step=False, on_epoch=True, prog_bar=False)

    def create_multilabel_conf_matrix(self, labels, preds, val_AUC_macro=None, epoch=None):
        sns.set_style("white")
        sns.set_context("paper")

        conf_mat = MultilabelConfusionMatrix(num_labels=self.num_classes)
        all_preds, all_labels = torch.tensor(preds, dtype=torch.float), torch.tensor(labels, dtype=torch.int)
        conf_mat.update(preds=all_preds, target=all_labels)
        cm = conf_mat.compute().numpy()  # Shape: [num_labels, 2, 2]

        # Plot confusion matrices for each label
        fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(14, 10), squeeze=False)
        axes = axes.flatten()

        for i in range(self.num_classes):
            thresh = 40
            ax = axes.flatten()[i]
            cm_label = cm[i]  # Get the 2x2 confusion matrix for label i

            cm_normalized = (cm_label.astype('float') / cm_label.sum()) * 100
            norm = TwoSlopeNorm(vmin=0, vcenter=thresh, vmax=100)
            cax = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues', norm=norm)

            # Set axes titles
            ax.set_title(self.class_names[i], fontsize=14, weight='bold', pad=10)

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

        current_epoch = epoch if epoch else self.trainer.current_epoch
        val_AUC_macro = (val_AUC_macro if val_AUC_macro else self.best_val_AUC_macro) * 100
        fig.suptitle(f'Confusion Matrix (Epoch={current_epoch}; val_AUC_macro={val_AUC_macro:.2f})',
                     fontsize=18, weight='bold')

        plt.subplots_adjust(left=0.15, right=0.88, bottom=0.15, top=0.9, wspace=0.4)

        return fig

    def create_multiclass_conf_matrix(self, labels, preds, val_AUC_macro=None, epoch=None):
        sns.set_style("white")
        sns.set_style("ticks")

        conf_mat = MulticlassConfusionMatrix(num_classes=self.num_classes)
        all_preds, all_labels = torch.tensor(preds, dtype=torch.float16), torch.tensor(labels, dtype=torch.int8)
        conf_mat.update(preds=all_preds, target=all_labels)
        cm = conf_mat.compute().numpy()

        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        fig, ax = plt.subplots(figsize=(12, 8))
        norm = TwoSlopeNorm(vmin=0, vcenter=40, vmax=100)
        cax = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues', norm=norm)

        # Add color bar
        cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Percentage', fontsize=14)
        cbar.ax.tick_params(labelsize=12)
        cbar.set_ticks([0, 20, 40, 60, 80, 100])
        cbar.set_ticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])

        ax.grid(False)

        # Add the labels and tick marks
        ax.set_xticks(np.arange(self.num_classes))
        ax.set_yticks(np.arange(self.num_classes))
        ax.set_xticklabels(self.class_names, rotation=45, ha='right', fontsize=14)
        ax.set_yticklabels(self.class_names, rotation=0, fontsize=14)

        # Annotate the cells with integer counts
        fmt = 'd'
        thresh = cm_normalized.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm_normalized[i, j] > thresh else "black",
                        fontsize=10)

        # Annotate the cells with integer counts
        ax.set_xlabel('Predicted Label', fontsize=14, weight='bold')
        ax.set_ylabel('True Label', fontsize=14, weight='bold')
        current_epoch = epoch if epoch else self.trainer.current_epoch
        val_AUC_macro = (val_AUC_macro if val_AUC_macro else self.best_val_AUC_macro) * 100
        ax.set_title(f'Confusion Matrix (Epoch={current_epoch}; val_AUC_macro={val_AUC_macro:.2f})', fontsize=18,
                     weight='bold',
                     pad=15)

        if self.verbose:
            logging.info(
                f"Val AUC increased: Logging Confusion Matrix (Epoch={current_epoch}; val_AUC_macro={val_AUC_macro:.2f})")
        plt.tight_layout()

        return fig

    def __log_conf_matrix(self, all_labels, all_preds, mode='val'):
        if self.model_mode == ModelMode.MULTI_LABEL:
            fig = self.create_multilabel_conf_matrix(all_labels, all_preds)
        elif self.model_mode == ModelMode.MULTI_CLASS:
            fig = self.create_multiclass_conf_matrix(all_labels, all_preds)
        else:
            raise ValueError(f"Unsupported model mode: {self.model_mode}")

        wandb.log({f"best_{mode}_conf_mat": wandb.Image(fig)})
        plt.close()

        # Clear values
        if mode == 'val':
            self.val_labels.clear()
            self.val_preds.clear()
        if mode == 'test':
            self.test_labels.clear()
            self.test_preds.clear()

    def __log_roc_curve(self, mode='val'):
        if self.model_mode == ModelMode.MULTI_LABEL:
            return
        fig, ax = plt.subplots(figsize=(12, 8))
        self.AUC_weighted.plot(ax=ax)
        plt.title(f"{mode.capitalize()} ROC Curve", fontsize=18)
        plt.tight_layout()
        wandb.log({f"{mode}_roc_curve": wandb.Image(fig)})
        plt.close()