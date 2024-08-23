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
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score, \
    MulticlassAveragePrecision, MulticlassAUROC, MulticlassConfusionMatrix

from src.models.enums.finetune_mode import FineTuneMode
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

        self.val_labels = []
        self.val_preds = []
        self.test_labels = []
        self.test_preds = []

        # Used for postprocessing predictions
        self.predict_labels = []
        self.predict_preds = []

        self.backbone = self.init_backbone()
        self.classifier = self.init_classifier()

        if checkpoint_path:
            self.load_checkpoint_weights()

        self.criterion = nn.CrossEntropyLoss()

        self.__setup_model_fine_tuning()
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
        logging.info(f"Model: Classifier with {self.num_classes} classes built.")
        return LinearClassifier(in_features=self.backbone.num_features, num_classes=self.num_classes)

    def load_checkpoint_weights(self):
        logging.info(f"Loading weights from checkpoint: {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, weights_only=True)
        state_dict = checkpoint['state_dict']

        backbone_state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items() if 'backbone.' in k}
        self.backbone.load_state_dict(backbone_state_dict, strict=False)

        classifier_state_dict = {k.replace('classifier.', ''): v for k, v in state_dict.items() if 'classifier.' in k}
        self.classifier.load_state_dict(classifier_state_dict, strict=False)

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

        return logits

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        labels = labels.long()
        logits = self.forward(imgs)

        return self.__calc_loss(logits, labels, mode='train')

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        labels = labels.long()
        logits = self.forward(imgs)
        self.__calc_loss(logits, labels, mode='val')
        self.__log_step_metrics(preds=logits, labels=labels, mode='val')
        self.__store_step_preds(preds=logits, labels=labels, mode='val')

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        labels = labels.long()
        logits = self.forward(imgs)
        self.__calc_loss(logits, labels, mode='test')
        self.__log_step_metrics(preds=logits, labels=labels, mode='test')
        self.__store_step_preds(preds=logits, labels=labels, mode='test')

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return

        self.__log_epoch_metrics(all_labels=self.val_labels, all_preds=self.val_preds, mode='val')

        current_val_metric = self.trainer.logged_metrics.get('val_AUC_macro')
        if current_val_metric > self.best_val_AUC_macro:
            self.best_val_AUC_macro = current_val_metric

            labels = np.array(getattr(self, f"val_labels"))
            preds = np.array(getattr(self, f"val_preds"))

            self.__log_conf_matrix(preds=preds, labels=labels, mode='val')
            self.__log_roc_curve(preds=preds, labels=labels, mode='val')
        return np.array(getattr(self, f"val_labels")), np.array(getattr(self, f"val_preds"))

    def on_validation_epoch_start(self) -> None:
        self.val_labels.clear()
        self.val_preds.clear()

    def on_test_epoch_start(self) -> None:
        self.test_labels.clear()
        self.test_preds.clear()

    def on_test_epoch_end(self):
        if self.trainer.sanity_checking:
            return

        self.__log_epoch_metrics(all_labels=self.test_labels, all_preds=self.test_preds, mode='test')
        self.__log_conf_matrix(preds=self.test_preds, labels=self.test_labels, mode='test')

    def __setup_model_fine_tuning(self):
        ft_mode = self.config.ft_mode

        for param in self.classifier.parameters():
            param.requires_grad = False
        for param in self.backbone.parameters():
            param.requires_grad = False

        if ft_mode == FineTuneMode.HEAD.value or ft_mode == FineTuneMode.FULL.value:
            logging.info("Unfreezing head parameters for fine-tuning.")
            for param in self.classifier.parameters():
                param.requires_grad = True

        if ft_mode == FineTuneMode.BACKBONE.value or ft_mode == FineTuneMode.FULL.value:
            logging.info("Unfreezing backbone parameters for fine-tuning.")
            for param in self.backbone.parameters():
                param.requires_grad = True

        if self.verbose:
            logging.info(f"Fine-tuning mode set to: {ft_mode}")

    def __setup_metrics(self):
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

    def __calc_loss(self, logits, labels, mode='train'):
        loss = self.criterion(logits, labels)
        self.log(f'{mode}_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def __log_step_metrics(self, preds, labels, mode):
        if self.trainer.sanity_checking:
            return

        self.log(f'{mode}_precision_weighted', self.precision_weighted(preds, labels), on_epoch=True, prog_bar=False)
        self.log(f'{mode}_recall_weighted', self.recall_weighted(preds, labels), on_epoch=True, prog_bar=False)
        self.log(f'{mode}_f1_weighted', self.f1_weighted(preds, labels), on_epoch=True, prog_bar=False)

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
        self.log(f"{mode}_AUC_macro", self.AUC_macro(preds, labels), on_step=False, on_epoch=True, prog_bar=True)

        # Per-Class metrics logging
        mAP_pc = torch.nan_to_num(self.mAP_per_class(preds, labels), 0.0)
        AUC_pc = torch.nan_to_num(self.AUC_per_class(preds, labels), 0.0)

        for i, score in enumerate(mAP_pc):
            class_label = self.class_names[i]
            self.log(f"{mode}_mAP_{class_label}", score.item(), on_step=False, on_epoch=True, prog_bar=False)

        for i, score in enumerate(AUC_pc):
            class_label = self.class_names[i]
            self.log(f"{mode}_AUC_{class_label}", score.item(), on_step=False, on_epoch=True, prog_bar=False)

    def __log_conf_matrix(self, preds, labels, mode='val'):
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
        current_epoch = self.trainer.current_epoch
        val_AUC_macro = self.best_val_AUC_macro * 100
        ax.set_title(f'Confusion Matrix (Epoch={current_epoch}; val_AUC_macro={val_AUC_macro:.2f})', fontsize=18,
                     weight='bold',
                     pad=15)

        if self.verbose:
            logging.info(
                f"Val AUC increased: Logging Confusion Matrix (Epoch={current_epoch}; val_AUC_macro={val_AUC_macro:.2f})")
        plt.tight_layout()

        wandb.log({f"best_{mode}_conf_mat": wandb.Image(fig)})
        plt.close()

    def __log_roc_curve(self, preds, labels, mode='val'):
        # print("Creating ROC curve plot")
        sns.set(style="whitegrid", context="poster", palette="bright")
        sns.set_style('ticks')

        class_labels = self.class_names
        num_classes = self.num_classes

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
        sns.lineplot(x=micro_fpr, y=micro_tpr, ax=axes[1], color='blue',
                     label=f'Micro-average ROC (AUC = {micro_auc:.2f})')

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
        plt.tight_layout()
        wandb.log({f"best_{mode}_roc_curve": wandb.Image(fig)})
        plt.close()
